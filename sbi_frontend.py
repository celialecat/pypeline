from __future__ import annotations
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from joblib import Parallel, delayed

sbi_builder_path = "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_backend"
sys.path.append(str(sbi_builder_path))

# ---- import your reusable utilities
from sbi_backend import (
    TszDataset,
    NPEConfig,
    NPETrainer,
    setup_scheduler,
    set_seed,
    build_boxuniform_prior,
)

print("sbi_builder path:", sbi_builder_path)
print("Using torch", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)



# ----------------------------------------------------------------------------
# 1. Load the pre-simulated (theta, x) tensors and split into train/val
# ----------------------------------------------------------------------------
# Replace these paths with your actual data files
theta_files = [
    #"/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/theta_samples_paint_10000.pt",  # <-- update this path
    #"/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/theta_dell_5000.pt"
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/theta_wst.pt"    
]

x_files = [
    #"/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/wst_S0_S1_allmaps.pt",      # <-- update this path
    #"/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/dell_dataset_maxell10000.pt"
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/x_wst.pt"
]

# Number of samples to load and fraction for training
train_size = 5000
train_frac = 0.8
    
# Load the full dataset (concatenates all provided files)
full_ds = TszDataset(theta_files, x_files, device=device)

# Select the first `train_size` samples to speed up development
idx = torch.arange(min(train_size, len(full_ds)))
n_train = int(train_frac * len(idx))
train_idx = idx[:n_train]
val_idx   = idx[n_train:]

train_dataset = Subset(full_ds, train_idx)
val_dataset   = Subset(full_ds, val_idx)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# ----------------------------------------------------------------------------
# 3. Save the trained ensemble to disk
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 2. Configure and train an ensemble of NLE models
# ----------------------------------------------------------------------------

ensemble_size = 1  # number of independent NLEs to train
NUM_WORKERS   = 3           # adjust to CPU cores / available GPUs
nle_trainers = []   # will hold the trained NLETrainer instances
run_infos   = []    # optional: to store training curves and metrics

# Hyperparameters common to all ensemble members
hidden_features = 50
num_transforms  = 5
num_components  = 5
num_bins        = 3  # only used by some models, kept here for completeness
activation_fn   = nn.Tanh  # can also try nn.ReLU
max_epochs      = 2**20
stop_after_epoch = 200  # early stopping patience
batch_size      = 256
initial_lr      = 0.0004871
step_size      = 63   # decay LR every `step_size` epochs
gamma          = 0.8551  # multiply LR by `gamma` at each step (1.0 = no decay)
dropout_probability = 0.05


save_dir = "/rds/rds-clecat/pipeline_alina_full/alina_paper/ensemble_npe_paint_180k_wst"  # directory to store checkpoints
os.makedirs(save_dir, exist_ok=True)

# # Record hyperparameters for reproducibility
ensemble_hparams = {
    "model": "maf",
    "hidden_features": hidden_features,
    "num_transforms": num_transforms,
    "num_components": num_components,
    "num_bins": num_bins,
    "dropout_probability": dropout_probability, 
    "activation": activation_fn.__name__,
}
with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
    json.dump(ensemble_hparams, f, indent=2)

# # Save each trained estimator using its trainer's save() method
# for idx, trainer in enumerate(nle_trainers):
#     ckpt_path = os.path.join(save_dir, f"member_{idx:02d}.pt")
#     trainer.save(ckpt_path)
#     print(f"Saved ensemble member {idx} to {ckpt_path}")

# print("All ensemble members saved.")

def _train_and_save_member(i: int):
    # Rebuild datasets inside the worker process
    full_ds_local = TszDataset(theta_files, x_files, device=device)
    idx_local = torch.arange(min(train_size, len(full_ds_local)))
    n_train_local = int(train_frac * len(idx_local))
    train_idx_local = idx_local[:n_train_local]
    val_idx_local   = idx_local[n_train_local:]
    train_dataset_local = Subset(full_ds_local, train_idx_local)
    val_dataset_local   = Subset(full_ds_local, val_idx_local)

    # Per-member seed for diversity
    seed_i = 4 + i

    # Configure the NLE (same hyperparams you already set)
    npe_cfg_i = NPEConfig(
        model="maf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_components=num_components,
        dropout_probability=dropout_probability,
        activation=activation_fn,

        grad_clip=5.0,
        stop_after_epoch=stop_after_epoch,
        seed=seed_i,
    )
    trainer_i = NPETrainer(config=npe_cfg_i, device=device)
    run_info_i = trainer_i.fit(
        train=train_dataset_local,
        val=val_dataset_local,
        batch_size=batch_size,
        max_epochs=max_epochs,
        initial_lr=initial_lr,
        # keep your LR schedule; NPE’s default step_size is 30, you used 22 — both fine
        scheduler_fn=lambda opt: setup_scheduler(opt, step_size=step_size, gamma=gamma),
        device=device,
        verbose_every=10,
    )

    # Save member directly from the worker
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path_i = os.path.join(save_dir, f"member_{i}.pt")
    trainer_i.save(ckpt_path_i)

    # Return what the parent needs (path + minimal metrics)
    return ckpt_path_i, {
        "final_train_loss": run_info_i["train_losses"][-1],
        "final_val_loss": run_info_i["val_losses"][-1],
    }

# ----------------------------------------------------------------------------
# 2. Configure and train an ensemble of NLE models
# ----------------------------------------------------------------------------

print(f"Training {ensemble_size} ensemble members in parallel with {NUM_WORKERS} workers...")
results = Parallel(n_jobs=NUM_WORKERS, backend="loky", verbose=10)(
    delayed(_train_and_save_member)(i) for i in range(ensemble_size)
)

# Unpack results
ckpt_paths = [r[0] for r in results]
metrics    = [r[1] for r in results]
print("Finished training the ensemble.")
for p, m in zip(ckpt_paths, metrics):
    print(f"{p} | train={m['final_train_loss']:.4f} | val={m['final_val_loss']:.4f}")


print("Finished training the ensemble.")