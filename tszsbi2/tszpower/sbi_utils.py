import torch
from torch.distributions import Distribution, Uniform, Normal, Independent
from sbi.utils.torchutils import BoxUniform
from torch import Tensor
from typing import List, Dict, Union, Optional
import numpy as np

# Helper to process the device in the same style as BoxUniform.
def process_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    return device

class CustomFDistribution(Distribution):
    r"""
    Custom distribution for the F parameter using a histogram‐based density 
    approximation. This distribution computes a histogram from the provided
    F values and returns the corresponding (approximate) density on evaluation.
    
    Args:
        f_values (Tensor): 1D tensor of F values.
        bins (int): Number of bins used to compute the histogram.
    
    Example:
    --------
    ::
    
        f_vals = torch.tensor([...], dtype=torch.float32)
        f_dist = CustomFDistribution(f_vals, bins=50)
        sample = f_dist.sample((100,))
        logp = f_dist.log_prob(sample)
    """
    def __init__(self, f_values: Tensor, bins: int = 50):
        super().__init__()
        # Move f_values to cpu for histogram computation if necessary.
        f_np = f_values.cpu().numpy()
        hist, bin_edges = np.histogram(f_np, bins=bins, density=True)
        self.hist = torch.as_tensor(hist, dtype=torch.float32, device=f_values.device)
        self.bin_edges = torch.as_tensor(bin_edges, dtype=torch.float32, device=f_values.device)
        self.f_min = self.bin_edges[0].item()
        self.f_max = self.bin_edges[-1].item()
    
    def sample(self, sample_shape=torch.Size()):
        # A simple sampler that draws uniform samples on the support of F.
        # (For more accurate sampling, consider weighted or rejection sampling.)
        return torch.rand(sample_shape, device=self.hist.device) * (self.f_max - self.f_min) + self.f_min

    def log_prob(self, value: Tensor):
        # Compute the log probability density for input values based on the histogram.
        # For each value, determine which bin it falls in.
        indices = torch.sum(value.unsqueeze(-1) >= self.bin_edges[:-1], dim=-1) - 1
        indices = indices.clamp(0, self.hist.numel() - 1)
        prob_density = self.hist[indices]
        return torch.log(prob_density + 1e-8)

class CustomPrior(Distribution):
    r"""
    Custom joint prior over the parameter vector [F, param_2, param_3, ...] where:
    
      - The first parameter, F, uses a custom distribution built from histogram
        evaluation of provided F values.
      - The remaining parameters are distributed uniformly over a box.
    
    This prior is designed in a similar style to the BoxUniform class.
    
    Args:
        f_values (Union[Tensor, np.ndarray]): Values for computing the custom F distribution.
        low (Union[Tensor, np.ndarray]): Lower bounds for the uniform part (remaining parameters).
        high (Union[Tensor, np.ndarray]): Upper bounds for the uniform part.
        bins (int): Number of bins for the histogram of F values.
        reinterpreted_batch_ndims (int): Number of batch dims to reinterpret as event dims.
        device (Optional[Union[str, torch.device]]): Device to create the prior on.
    
    Example:
    --------
    ::
    
        import torch
        
        # Suppose f_values_tensor contains sampled F values.
        f_values_tensor = torch.tensor([...], dtype=torch.float32)
        low_rest = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        high_rest = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
    
        prior = CustomPrior(
            f_values=f_values_tensor,
            low=low_rest,
            high=high_rest,
            bins=50,
            reinterpreted_batch_ndims=1,
            device="cpu"
        )
    
        sample = prior.sample((100,))
        logp = prior.log_prob(sample)
    """
    def __init__(
        self,
        f_values: Union[Tensor, np.ndarray],
        low: Union[Tensor, np.ndarray],
        high: Union[Tensor, np.ndarray],
        bins: int = 50,
        reinterpreted_batch_ndims: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = torch.device("cpu")
        else:
            device = process_device(device)
        self.device = device
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        # Ensure f_values, low, and high are tensors.
        if not torch.is_tensor(f_values):
            f_values = torch.tensor(f_values, dtype=torch.float32, device=device)
        else:
            f_values = f_values.to(device=device)
            
        if not torch.is_tensor(low):
            low = torch.tensor(low, dtype=torch.float32, device=device)
        else:
            low = low.to(device=device)
            
        if not torch.is_tensor(high):
            high = torch.tensor(high, dtype=torch.float32, device=device)
        else:
            high = high.to(device=device)

        # Initialize the custom F distribution for the first parameter.
        self.f_dist = CustomFDistribution(f_values, bins=bins)
        # Initialize the uniform distribution over the box for the remaining parameters.
        self.uniform_dist = Independent(
            Uniform(low=low, high=high, validate_args=False),
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Samples a parameter vector where the first component is from the custom F distribution
        and the remaining components are from the uniform distribution.
        """
        f_sample = self.f_dist.sample(sample_shape).unsqueeze(-1)  # Ensure shape [..., 1]
        rest_sample = self.uniform_dist.sample(sample_shape)
        return torch.cat((f_sample, rest_sample), dim=-1)

    def log_prob(self, value: Tensor):
        """
        Evaluates the joint log-probability for a parameter vector, summing the contributions from
        the custom F distribution and the uniform distribution.
        
        Args:
            value (Tensor): A tensor of shape [..., 1 + n_rest], where the first column is F.
        """
        f_val = value[..., 0]
        rest_val = value[..., 1:]
        return self.f_dist.log_prob(f_val) + self.uniform_dist.log_prob(rest_val)

    def to(self, device: Union[str, torch.device]) -> "CustomPrior":
        """
        Moves the custom prior to the specified device in place.
        
        Args:
            device: Target device (e.g., "cpu", "cuda").
        
        Example:
        --------
        ::
        
            custom_prior.to("cuda")
        """
        self.device = process_device(device)
        # Move custom F distribution parameters.
        self.f_dist.hist = self.f_dist.hist.to(self.device)
        self.f_dist.bin_edges = self.f_dist.bin_edges.to(self.device)
        # Move uniform distribution parameters.
        self.uniform_dist.base_dist.low = self.uniform_dist.base_dist.low.to(self.device)
        self.uniform_dist.base_dist.high = self.uniform_dist.base_dist.high.to(self.device)
        return self



class MixedPrior(Distribution):
    """Mixed distribution of Uniform and Gaussian priors for parameters."""

    def __init__(
        self,
        prior_specs: List[Dict[str, Union[str, float]]],
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device if device else 'cpu'

        self.distributions = []
        for spec in prior_specs:
            prior_type = spec.get('type').lower()
            if prior_type == 'uniform':
                dist = Uniform(
                    torch.tensor(spec['low'], dtype=torch.float32, device=self.device),
                    torch.tensor(spec['high'], dtype=torch.float32, device=self.device),
                )
            elif prior_type == 'gaussian':
                dist = Normal(
                    torch.tensor(spec['mean'], dtype=torch.float32, device=self.device),
                    torch.tensor(spec['std'], dtype=torch.float32, device=self.device),
                )
            else:
                raise ValueError(f"Unsupported prior type: {prior_type}")
            self.distributions.append(dist)

    def sample(self, sample_shape=torch.Size()):
        samples = [dist.sample(sample_shape).unsqueeze(-1) for dist in self.distributions]
        return torch.cat(samples, dim=-1)

    def log_prob(self, value):
        log_probs = torch.stack([
            dist.log_prob(value[..., i]) for i, dist in enumerate(self.distributions)
        ], dim=-1)
        return log_probs.sum(dim=-1)

    def to(self, device):
        self.device = device
        for i, dist in enumerate(self.distributions):
            self.distributions[i] = type(dist)(
                dist.loc.to(device), dist.scale.to(device)
            ) if isinstance(dist, Normal) else type(dist)(
                dist.low.to(device), dist.high.to(device)
            )
        return self
    

class ConstrainedBoxUniform(Distribution):
    def __init__(
        self,
        low,
        high,
        foreground_data_directory="/home/lx256/tsz_project/tszpower/data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15_RC.txt",
        rc_data_directory="/home/lx256/tsz_project/tszpower/data/data_rc-ell-rc-errrc_backup.txt",
        obs_data_directory="/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15_RC.txt",
        A_cn_fixed=0.9033,
        num_cosmo=6,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.num_cosmo = num_cosmo
        self.A_cn_fixed = A_cn_fixed

        # Store BoxUniform as attribute for representation
        self.box_uniform = BoxUniform(self.low, self.high, device=device)
        self.box_cosmo = BoxUniform(self.low[:num_cosmo], self.high[:num_cosmo], device=device)
        self.low_amp = self.low[num_cosmo:]
        self.high_amp = self.high[num_cosmo:]

        # Load model data from files
        D_fg = np.loadtxt(foreground_data_directory)
        D_rc = np.loadtxt(rc_data_directory)
        D_obs = np.loadtxt(obs_data_directory)
        self.A_CIB_MODEL = torch.tensor(D_fg[:, 1][12:], dtype=torch.float32, device=device)
        self.A_RS_MODEL  = torch.tensor(D_fg[:, 2][12:], dtype=torch.float32, device=device)
        self.A_IR_MODEL  = torch.tensor(D_fg[:, 3][12:], dtype=torch.float32, device=device)
        self.A_CN_MODEL  = torch.tensor(D_fg[:, 4][12:], dtype=torch.float32, device=device)
        self.Dl_obs = torch.tensor(D_obs[:, 1][12:], dtype=torch.float32, device=device)
        self.Dl_rc  = torch.tensor(D_rc[:, 1][12:], dtype=torch.float32, device=device)

    def __repr__(self):
        # Mimic BoxUniform repr, but indicate constrained
        return f"ConstrainedBoxUniform({repr(self.box_uniform)}, constraint=foreground sum < Dl_obs-Dl_rc)"
    
    def sample(self, sample_shape=torch.Size()):
        samples = []
        n = int(torch.tensor(sample_shape).prod()) if len(sample_shape) > 0 else 1
        while len(samples) < n:
            # Sample cosmo params
            theta_cosmo = self.box_cosmo.sample().squeeze(0)
            # Foreground amplitudes: sample uniformly within their bounds
            A_cib = torch.empty(1).uniform_(float(self.low_amp[0]), float(self.high_amp[0])).item()
            A_rs  = torch.empty(1).uniform_(float(self.low_amp[1]), float(self.high_amp[1])).item()
            A_ir  = torch.empty(1).uniform_(float(self.low_amp[2]), float(self.high_amp[2])).item()
            A_cn  = self.A_cn_fixed

            fg_sum = (
                A_cib * self.A_CIB_MODEL +
                A_rs  * self.A_RS_MODEL +
                A_ir  * self.A_IR_MODEL +
                A_cn  * self.A_CN_MODEL
            )
            if torch.all(fg_sum < (self.Dl_obs - self.Dl_rc)):
                theta_full = torch.cat([
                    theta_cosmo,
                    torch.tensor([A_cib, A_rs, A_ir], device=self.device)
                ])
                samples.append(theta_full)
        return torch.stack(samples).reshape(sample_shape + (-1,))

    def log_prob(self, x):
        in_box = (x >= self.low) & (x <= self.high)
        in_box = in_box.all(-1)
        # Foregrounds: last three numbers
        A_cib, A_rs, A_ir = x[..., -3], x[..., -2], x[..., -1]
        A_cn = self.A_cn_fixed
        fg_sum = (
            A_cib.unsqueeze(-1) * self.A_CIB_MODEL +
            A_rs.unsqueeze(-1) * self.A_RS_MODEL +
            A_ir.unsqueeze(-1) * self.A_IR_MODEL +
            A_cn * self.A_CN_MODEL
        )
        allowed = torch.all(fg_sum < (self.Dl_obs - self.Dl_rc), dim=-1)
        total_in = in_box & allowed
        logprob = torch.full(x.shape[:-1], -float("inf"), device=x.device)
        logprob[total_in] = 0.0
        return logprob



# Example usage:
# prior_specs = [
#     {"type": "uniform", "low": 2.5, "high": 3.5},
#     {"type": "gaussian", "mean": 0.0225, "std": 0.001},
#     {"type": "uniform", "low": 0.11, "high": 0.13},
#     {"type": "gaussian", "mean": 72.5, "std": 10.0},
#     {"type": "uniform", "low": 0.94, "high": 1.0},
#     {"type": "uniform", "low": 1.0, "high": 2.0},
#     {"type": "gaussian", "mean": 2.5, "std": 1.0},
#     {"type": "uniform", "low": 0.0, "high": 5.0},
#     {"type": "uniform", "low": 0.0, "high": 5.0},
# ]

# mixed_prior = MixedPrior(prior_specs, device='cpu')
# samples = mixed_prior.sample((100,))
# log_probs = mixed_prior.log_prob(samples)