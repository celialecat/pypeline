# sbi_builder.py
"""
Reusable loaders and trainers for SBI on tSZ bandpowers.

Contents
--------
1) TszDataset  : loads pre-simulated (theta, x) tensors from .pt files
2) NPETrainer  : trains a direct posterior estimator q_phi(theta | x) and samples with DirectPosterior
3) NLETrainer  : trains a neural likelihood p_phi(x | theta) and samples with MCMCPosterior

Notes
-----
- Matches your original training loops (loss conventions, gradient clipping, schedulers).
- Minimal, dependency-light design; save/load checkpoints include hyperparameters.
- Device handling is automatic but overrideable via the constructor or fit(..., device="cuda").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# ---- sbi imports --------------------------------------------------------------
import sbi
from sbi.neural_nets import posterior_nn, likelihood_nn
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.utils import BoxUniform


Tensor = torch.Tensor
PathLike = Union[str, Path]


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _auto_device(explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_scheduler(optimizer: torch.optim.Optimizer,
                    step_size: int = 30,
                    gamma: float = 0.9):
    """StepLR, same defaults as your NPE script (NLE overrides step_size to 50)."""
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
    return {"scheduler": scheduler}


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    for pg in optimizer.param_groups:
        return float(pg["lr"])
    return float("nan")


def build_boxuniform_prior(theta_lower: Iterable[float],
                           theta_upper: Iterable[float],
                           device: Optional[str] = None) -> BoxUniform:
    """Convenience helper to create a BoxUniform prior on the chosen device."""
    device = _auto_device(device)
    low = torch.as_tensor(theta_lower, dtype=torch.get_default_dtype(), device=device)
    high = torch.as_tensor(theta_upper, dtype=torch.get_default_dtype(), device=device)
    return BoxUniform(low, high, device=device)


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class TszDataset(Dataset):
    """
    Dataset of *pre-simulated* tSZ band-power vectors and their θ-parameters.

    Each .pt in `theta_files` must align with the corresponding .pt in `x_files`.

    Parameters
    ----------
    theta_files : sequence of paths to (N_i, n_params) tensors
    x_files     : sequence of paths to (N_i, n_ell) tensors
    x_keep      : optional column indices / slice to keep subset of ℓ-bins
    max_samples : optional truncation for quick dev runs
    device      : map_location when loading the tensors ("cpu", "cuda", etc.)
    """

    def __init__(
        self,
        theta_files: Sequence[PathLike],
        x_files: Sequence[PathLike],
        *,
        x_keep: Optional[Union[slice, Sequence[int]]] = None,
        max_samples: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()

        theta_list: List[Tensor] = []
        x_list: List[Tensor] = []

        for p in theta_files:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(p)
            theta_list.append(torch.load(p, map_location=device))

        for p in x_files:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(p)
            x_list.append(torch.load(p, map_location=device))

        if len(theta_list) != len(x_list):
            raise ValueError(
                f"Number of θ files and x files must match (got {len(theta_list)} vs {len(x_list)})."
            )

        self.theta = torch.cat(theta_list, dim=0)
        self.x = torch.cat(x_list, dim=0)

        if x_keep is not None:
            self.x = self.x[:, x_keep]

        if max_samples is not None:
            self.theta = self.theta[: max_samples]
            self.x = self.x[: max_samples]

        if self.theta.shape[0] != self.x.shape[0]:
            raise RuntimeError(
                f"θ rows ({self.theta.shape[0]}) and x rows ({self.x.shape[0]}) differ!"
            )

        print(
            f"TszDataset loaded: {self.theta.shape[0]} samples | "
            f"θ dim = {self.theta.shape[1]} | x dim = {self.x.shape[1]}"
        )

    def __len__(self) -> int:
        return self.theta.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.theta[idx], self.x[idx]

    def get_dataloader(
        self,
        batch_size: int = 200,
        shuffle: bool = True,
        **dataloader_kwargs,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **dataloader_kwargs,
        )


# ------------------------------------------------------------------------------
# NPE: Direct Posterior Estimation q_phi(theta | x)
# ------------------------------------------------------------------------------

@dataclass
class NPEConfig:
    model: str = "maf"
    hidden_features: int = 50
    num_transforms: int = 5
    num_components: int = 5
    dropout_probability: float = 0.0
    activation: Callable[..., nn.Module] = nn.ReLU
    grad_clip: float = 5.0
    seed: Optional[int] = None


class NPETrainer:
    """
    Direct posterior estimator using sbi.neural_nets.posterior_nn + DirectPosterior.

    Training loss and dataloader flow matches your shared script.
    """

    def __init__(self, config: Optional[NPEConfig] = None, device: Optional[str] = None):
        self.config = config or NPEConfig()
        self.device = _auto_device(device)
        self.net: Optional[nn.Module] = None
        self.input_dims: Optional[Dict[str, int]] = None  # {"theta": d, "x": d}

    # --- model building ---------------------------------------------------------

    def _build_from_batch(self, theta: Tensor, x: Tensor) -> nn.Module:
        build_fn = posterior_nn(
            model=self.config.model,
            hidden_features=self.config.hidden_features,
            num_transforms=self.config.num_transforms,
            num_components=self.config.num_components,
            dropout_probability=self.config.dropout_probability,
            activation=self.config.activation,
        )
        net = build_fn(theta, x)
        self.input_dims = {"theta": theta.shape[-1], "x": x.shape[-1]}
        return net

    def build(self, sample_theta: Tensor, sample_x: Tensor) -> nn.Module:
        self.net = self._build_from_batch(sample_theta, sample_x)
        return self.net

    # --- training ---------------------------------------------------------------

    def fit(
        self,
        train: Dataset,
        val: Dataset,
        *,
        batch_size: int = 64,
        max_epochs: int = 100,
        initial_lr: float = 1e-3,
        scheduler_fn: Callable[..., Dict[str, Any]] = setup_scheduler,
        device: Optional[str] = None,
        verbose_every: int = 10,
    ) -> Dict[str, Any]:
        set_seed(self.config.seed)
        device = _auto_device(device) or self.device
        self.device = device

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

        # Build the network if needed from the first batch.
        if self.net is None:
            with torch.no_grad():
                th0, x0 = next(iter(train_loader))
            self.net = self._build_from_batch(th0, x0)

        net = self.net.to(device)
        opt = AdamW(net.parameters(), lr=initial_lr)
        sch = scheduler_fn(opt)["scheduler"]

        run: Dict[str, Any] = {"train_losses": [], "val_losses": []}

        print(f"[NPE] Starting training for {max_epochs} epochs on {device}...")
        try:
            for epoch in range(max_epochs):
                net.train()
                epoch_train_loss = 0.0
                n_batches = 0

                for theta, x in train_loader:
                    theta, x = theta.to(device), x.to(device)
                    opt.zero_grad()
                    loss = net.loss(theta, x).mean()
                    loss.backward()
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.config.grad_clip)
                    opt.step()

                    run["train_losses"].append(loss.item())
                    epoch_train_loss += loss.item()
                    n_batches += 1

                sch.step()

                net.eval()
                with torch.no_grad():
                    val_loss_total = 0.0
                    n_val = 0
                    for theta, x in val_loader:
                        theta, x = theta.to(device), x.to(device)
                        val_loss_total += net.loss(theta, x).sum().item()
                        n_val += theta.shape[0]
                    val_loss_avg = val_loss_total / len(val)
                    run["val_losses"].append(val_loss_avg)

                if (epoch % verbose_every == 0) or (epoch == max_epochs - 1):
                    print(
                        f"[NPE] Epoch {epoch+1:3d}/{max_epochs} | "
                        f"Train: {epoch_train_loss / max(1, n_batches):.4f} | "
                        f"Val: {val_loss_avg:.4f} | LR: {get_current_lr(opt):.6f}"
                    )

            run["density_estimator_state"] = net.state_dict()
            run["config"] = asdict(self.config)
            run["input_dims"] = self.input_dims
            return run

        except KeyboardInterrupt:
            print("[NPE] Training interrupted. Returning current state...")
            run["density_estimator_state"] = net.state_dict()
            run["config"] = asdict(self.config)
            run["input_dims"] = self.input_dims
            return run

    # --- inference --------------------------------------------------------------

    def sample_posterior(
        self,
        prior: torch.distributions.Distribution,
        observation: Tensor,
        N_samples: int = 100_000,
        show_progress_bars: bool = True,
    ) -> Tuple[Tensor, DirectPosterior]:
        if self.net is None:
            raise RuntimeError("NPETrainer: call build()/fit() before sampling.")
        posterior = DirectPosterior(self.net, prior)
        samples = posterior.sample((N_samples,), x=observation.to(self.device), show_progress_bars=show_progress_bars)
        return samples, posterior

    # --- I/O --------------------------------------------------------------------

    def save(self, path: PathLike) -> None:
        if self.net is None or self.input_dims is None:
            raise RuntimeError("NPETrainer.save(): nothing to save (build/fit first).")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self.net.state_dict(),
            "config": asdict(self.config),
            "input_dims": self.input_dims,
        }
        torch.save(ckpt, path)
        print(f"[NPE] Saved checkpoint to {path}")

    def load(self, path: PathLike, map_location: Optional[str] = None) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.config = NPEConfig(**ckpt["config"])
        self.input_dims = ckpt["input_dims"]
        # Rebuild from stored dims
        d_th = self.input_dims["theta"]
        d_x = self.input_dims["x"]
        dummy_theta = torch.zeros(2, d_th)
        dummy_x = torch.zeros(2, d_x)
        self.net = self._build_from_batch(dummy_theta, dummy_x).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        print(f"[NPE] Loaded checkpoint from {path}")


# ------------------------------------------------------------------------------
# NLE: Neural Likelihood Estimation p_phi(x | theta) + MCMC sampling
# ------------------------------------------------------------------------------

@dataclass
class NLEConfig:
    model: str = "maf"
    hidden_features: int = 50
    num_transforms: int = 5
    num_components: int = 5
    num_bins: int = 10
    dropout_probability: float = 0.0
    activation: Callable[..., nn.Module] = nn.ReLU
    grad_clip: float = 5.0
    # MCMC defaults
    mcmc_method: str = "slice_np_vectorized"
    num_chains: int = 20
    thin: int = 5
    seed: Optional[int] = None


class NLETrainer:
    """
    Neural likelihood estimator using sbi.neural_nets.likelihood_nn + MCMCPosterior.

    Matches your shared loop: net.loss(x, condition=theta), StepLR with step_size=50.
    """

    def __init__(self, config: Optional[NLEConfig] = None, device: Optional[str] = None):
        self.config = config or NLEConfig()
        self.device = _auto_device(device)
        self.net: Optional[nn.Module] = None
        self.input_dims: Optional[Dict[str, int]] = None

    # --- model building ---------------------------------------------------------

    def _build_from_batch(self, theta: Tensor, x: Tensor) -> nn.Module:
        build_fn = likelihood_nn(
            model=self.config.model,
            hidden_features=self.config.hidden_features,
            num_transforms=self.config.num_transforms,
            num_components=self.config.num_components,
            dropout_probability=self.config.dropout_probability,
            num_bins=self.config.num_bins,
            activation=self.config.activation,
        )
        net = build_fn(theta, x)
        self.input_dims = {"theta": theta.shape[-1], "x": x.shape[-1]}
        return net

    def build(self, sample_theta: Tensor, sample_x: Tensor) -> nn.Module:
        self.net = self._build_from_batch(sample_theta, sample_x)
        return self.net

    # --- training ---------------------------------------------------------------

    def fit(
        self,
        train: Dataset,
        val: Dataset,
        *,
        batch_size: int = 64,
        max_epochs: int = 100,
        initial_lr: float = 1e-3,
        scheduler_fn: Callable[..., Dict[str, Any]] = lambda opt: setup_scheduler(opt, step_size=50, gamma=0.9),
        device: Optional[str] = None,
        verbose_every: int = 10,
    ) -> Dict[str, Any]:
        set_seed(self.config.seed)
        device = _auto_device(device) or self.device
        self.device = device

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

        # Build the network if needed from the first batch.
        if self.net is None:
            with torch.no_grad():
                th0, x0 = next(iter(train_loader))
            self.net = self._build_from_batch(th0, x0)

        net = self.net.to(device)
        opt = AdamW(net.parameters(), lr=initial_lr)
        sch = scheduler_fn(opt)["scheduler"]

        run: Dict[str, Any] = {
            "train_losses": [],
            "val_losses": [],
            "train_log_probs": [],
            "val_log_probs": [],
            "simulation_count": [],
        }

        print(f"[NLE] Starting training for {max_epochs} epochs on {device}...")
        print(f"[NLE] Training on {len(train)} samples, validating on {len(val)} samples")

        try:
            for epoch in range(max_epochs):
                net.train()
                epoch_train_loss = 0.0
                n_batches = 0

                for theta, x in train_loader:
                    theta, x = theta.to(device), x.to(device)
                    opt.zero_grad()
                    # Per your script: loss tensor returned by net.loss(x, condition=theta)
                    log_probs = net.loss(x, condition=theta)
                    loss = torch.mean(log_probs)
                    loss.backward()
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.config.grad_clip)
                    opt.step()

                    run["train_losses"].append(loss.item())
                    epoch_train_loss += loss.item()
                    n_batches += 1

                sch.step()

                net.eval()
                with torch.no_grad():
                    val_loss_total = 0.0
                    for theta, x in val_loader:
                        theta, x = theta.to(device), x.to(device)
                        log_probs = net.loss(x, condition=theta)
                        val_loss_total += torch.sum(log_probs).item()
                    val_loss_avg = val_loss_total / len(val)
                    run["val_losses"].append(val_loss_avg)

                if (epoch % verbose_every == 0) or (epoch == max_epochs - 1):
                    print(
                        f"[NLE] Epoch {epoch+1:3d}/{max_epochs} | "
                        f"Train: {epoch_train_loss / max(1, n_batches):.4f} | "
                        f"Val: {val_loss_avg:.4f} | LR: {get_current_lr(opt):.6f}"
                    )

            run["density_estimator_state"] = net.state_dict()
            run["config"] = asdict(self.config)
            run["input_dims"] = self.input_dims
            return run

        except KeyboardInterrupt:
            print("[NLE] Training interrupted. Returning current state...")
            run["density_estimator_state"] = net.state_dict()
            run["config"] = asdict(self.config)
            run["input_dims"] = self.input_dims
            return run

    # --- inference --------------------------------------------------------------

    def sample_posterior(
        self,
        prior: torch.distributions.Distribution,
        observation: Tensor,
        N_samples: int = 100_000,
        *,
        mcmc_method: Optional[str] = None,
        num_chains: Optional[int] = None,
        thin: Optional[int] = None,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        show_progress_bars: bool = True,
    ) -> Tuple[Tensor, MCMCPosterior]:
        """
        Uses likelihood_estimator_based_potential(...) + MCMCPosterior.
        """
        if self.net is None:
            raise RuntimeError("NLETrainer: call build()/fit() before sampling.")

        if mcmc_kwargs is None:
            mcmc_kwargs = {}

        mcmc_method = mcmc_method or self.config.mcmc_method
        num_chains = num_chains or self.config.num_chains
        thin = thin or self.config.thin

        potential_fn, theta_transform = likelihood_estimator_based_potential(
            likelihood_estimator=self.net,
            prior=prior,
            x_o=observation.to(self.device),
        )

        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            method=mcmc_method,
            device=self.device,
            num_chains=num_chains,
            thin=thin,
            **mcmc_kwargs,
        )

        samples = posterior.sample((N_samples,), show_progress_bars=show_progress_bars)
        return samples, posterior

    # --- I/O --------------------------------------------------------------------

    def save(self, path: PathLike) -> None:
        if self.net is None or self.input_dims is None:
            raise RuntimeError("NLETrainer.save(): nothing to save (build/fit first).")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self.net.state_dict(),
            "config": asdict(self.config),
            "input_dims": self.input_dims,
        }
        torch.save(ckpt, path)
        print(f"[NLE] Saved checkpoint to {path}")

    def load(self, path: PathLike, map_location: Optional[str] = None) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.config = NLEConfig(**ckpt["config"])
        self.input_dims = ckpt["input_dims"]
        d_th = self.input_dims["theta"]
        d_x = self.input_dims["x"]
        dummy_theta = torch.zeros(2, d_th)
        dummy_x = torch.zeros(2, d_x)
        self.net = self._build_from_batch(dummy_theta, dummy_x).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        print(f"[NLE] Loaded checkpoint from {path}")


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

__all__ = [
    "TszDataset",
    "NPEConfig",
    "NPETrainer",
    "NLEConfig",
    "NLETrainer",
    "setup_scheduler",
    "get_current_lr",
    "set_seed",
    "build_boxuniform_prior",
]
