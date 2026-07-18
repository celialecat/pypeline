#!/usr/bin/env python
"""
Neural Posterior Estimation (NPE) with sbi on precomputed
Wavelet Scattering Transform (WST) coefficients.

- Parameters (theta):   cosmological parameters (logA, Ob0h2, Oc0h2, h, n_s, B)
- Data (x):             WST coefficients (S0, S1, S2) from CSV files.

The script:
1. Loads theta from a .pt file and builds a pandas DataFrame.
2. Loads all WST CSVs into a design matrix x (num_maps x num_features).
3. Optionally standardizes WST features.
4. Defines a BoxUniform prior over the 6 parameters.
5. Trains an NPE model (normalizing flow) with sbi.
6. Builds a DirectPosterior.
7. Shows how to sample from the posterior for a given observation.
8. Optionally visualizes marginals with sbi.analysis.pairplot.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.analysis import pairplot


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Reproducibility: random seed for both PyTorch and NumPy
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Path to cosmological parameters .pt file
THETA_PT_PATH = (
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/"
    "theta_samples_paint_10000.pt"
)

# Base directory containing WST coefficient CSVs.
WST_BASE_DIR = (
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/wst_full_coefs"
)
WST_FILE_TEMPLATE = "wst_map_{i}.csv"  # i from 0 to num_maps-1.

# Cosmological parameter names and dimensionality.
PARAM_NAMES = ["logA", "Ob0h2", "Oc0h2", "h", "n_s", "B"]
NUM_PARAMS = len(PARAM_NAMES)

# Whether to standardize WST features (recommended for NNs).
STANDARDIZE_WST = True

# Observation index for demonstration.
OBS_INDEX = 13

# Number of posterior samples for the demo.
NUM_POSTERIOR_SAMPLES = 1000


# ---------------------------------------------------------------------------
# 2. Helper functions for data loading & preprocessing
# ---------------------------------------------------------------------------

def load_df_from_pt(path: str) -> pd.DataFrame:
    """
    Load cosmological parameters from the .pt file and keep ONLY
    the first 6 columns: logA, Ob0h2, Oc0h2, h, n_s, B.

    The row index (0..N-1) is the map index used in wst_map_{i}.csv.
    """
    data = torch.load(path, map_location="cpu")  # shape [N, D]

    if data.ndim != 2:
        raise ValueError(f"Expected 2D tensor [N, D], got shape {data.shape}")

    # Keep only the first 6 columns, regardless of how many there are.
    data_6 = data[:, :6]  # shape [N, 6]

    # Set the column names exactly as desired.
    df = pd.DataFrame(
        data_6.numpy(),
        columns=["logA", "Ob0h2", "Oc0h2", "h", "n_s", "B"],
    )

    # Ensure that the index 0..N-1 matches the map index i in wst_map_{i}.csv
    df = df.reset_index(drop=True)

    print("Loaded cosmological parameters from .pt (first 6 columns only):")
    print("  shape:", df.shape)
    print(df.head())

    return df


def build_theta_from_df(df: pd.DataFrame, param_names: List[str]) -> torch.Tensor:
    """
    Extract theta (parameters) as a torch.FloatTensor of shape [num_maps, num_params].

    Args:
        df:           pandas DataFrame with columns matching `param_names`.
        param_names:  list of column names to extract, in desired order.

    Returns:
        theta: torch.FloatTensor of shape [num_maps, num_params].
    """
    # Ensure all requested parameters exist in the DataFrame.
    missing = [p for p in param_names if p not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    theta_np = df[param_names].values  # shape [num_maps, num_params]
    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    return theta


def _read_single_wst_file(
    file_path: str,
    base_kind_order: List[str],
    base_indices_per_kind: dict,
) -> np.ndarray:
    """
    Read a single WST CSV file and return a 1D numpy vector of features
    in a fixed, consistent order across all maps.

    Ordering convention:
        - For each kind in base_kind_order (e.g. ["S0", "S1", "S2"]):
          - values sorted by index in ascending order.

    Any missing / NaN values are filled with 0.0, but the feature dimension
    is kept identical for every map.

    Args:
        file_path:             path to CSV file (wst_map_{i}.csv).
        base_kind_order:       list of kinds (e.g. ["S0", "S1", "S2"]).
        base_indices_per_kind: dict mapping kind -> 1D array/list of indices that
                               define the global ordering for that kind.

    Returns:
        features: np.ndarray of shape [num_features].
    """
    # Read CSV, skipping lines starting with '#'
    df_wst = pd.read_csv(file_path, comment="#")

    # Filter to sample == 0 (we only use that one sample per map).
    df_wst = df_wst[df_wst["sample"] == 0]

    # Ensure numeric type for value column and handle NaNs later.
    df_wst["value"] = pd.to_numeric(df_wst["value"], errors="coerce")

    feature_list = []

    for kind in base_kind_order:
        # Subset to this kind.
        sub = df_wst[df_wst["kind"] == kind].copy()

        # Set index to 'index' column; this ensures we can reindex to base_indices_per_kind[kind].
        sub = sub.set_index("index")

        # Reindex to the full set of indices for this kind (ensuring fixed length).
        full_index = base_indices_per_kind[kind]
        sub = sub.reindex(full_index)

        # Extract the "value" column; any missing entries become NaN.
        values = sub["value"].to_numpy(dtype=np.float64)

        # Replace NaNs with 0.0. (You could choose a different strategy.)
        values = np.nan_to_num(values, nan=0.0)

        feature_list.append(values)

    # Concatenate across kinds: [S0_values, S1_values, S2_values, ...].
    features = np.concatenate(feature_list, axis=0)
    return features


def determine_wst_structure(
    example_file_path: str,
    kinds: Tuple[str, ...] = ("S0", "S1", "S2"),
) -> Tuple[List[str], dict]:
    """
    Determine the global ordering of WST coefficients from a single example CSV.

    We:
        - Read an example file.
        - Filter to `sample == 0`.
        - For each `kind` in `kinds`, collect and sort the unique `index` values.

    These indices define the consistent feature ordering across all maps.

    Args:
        example_file_path: path to a representative WST CSV file.
        kinds:             tuple of kinds to include (S0, S1, S2).

    Returns:
        base_kind_order:       list of kinds in the order they will be concatenated.
        base_indices_per_kind: dict mapping kind -> 1D np.ndarray of sorted indices.
    """
    df_wst = pd.read_csv(example_file_path, comment="#")
    df_wst = df_wst[df_wst["sample"] == 0]  # keep only sample=0

    base_kind_order = list(kinds)
    base_indices_per_kind = {}

    for kind in base_kind_order:
        sub = df_wst[df_wst["kind"] == kind]
        # Get sorted unique indices for this kind.
        indices = np.sort(sub["index"].unique())
        base_indices_per_kind[kind] = indices

    return base_kind_order, base_indices_per_kind


def build_wst_design_matrix(
    num_maps: int,
    wst_dir: str,
    file_template: str = "wst_map_{i}.csv",
) -> torch.Tensor:
    """
    Construct the WST design matrix X of shape [num_maps, num_features].

    For each map index i:
        - Read file wst_map_{i}.csv from `wst_dir`.
        - Filter to sample == 0.
        - Extract values for S0, S1, S2, sorted by index within each kind.
        - Concatenate in fixed order [S0, S1, S2] to a single 1D feature vector.

    Missing / NaN values are replaced by 0, and the feature dimension is kept
    identical for every map, defined from map 0.

    Args:
        num_maps:      number of maps / WST CSVs (e.g. 10_000).
        wst_dir:       directory containing WST files.
        file_template: pattern such that file_template.format(i=i)
                       gives the file name for map i.

    Returns:
        x: torch.FloatTensor of shape [num_maps, num_features].
    """
    # Determine global feature layout from the first file (map 0).
    example_path = os.path.join(wst_dir, file_template.format(i=0))
    if not os.path.exists(example_path):
        raise FileNotFoundError(f"Example WST file not found: {example_path}")

    base_kind_order, base_indices_per_kind = determine_wst_structure(example_path)

    # Number of features = sum_kinds (#indices for that kind)
    num_features = int(
        sum(len(base_indices_per_kind[kind]) for kind in base_kind_order)
    )
    print(f"Detected WST structure from {example_path}")
    for kind in base_kind_order:
        print(f"  kind={kind}, num_coeffs={len(base_indices_per_kind[kind])}")
    print(f"Total WST features per map: {num_features}")

    # Allocate array for all maps.
    x_all = np.zeros((num_maps, num_features), dtype=np.float32)

    # Loop over all maps and fill x_all[i, :]
    for i in range(num_maps):
        file_path = os.path.join(wst_dir, file_template.format(i=i))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"WST file for map {i} not found: {file_path}")

        features = _read_single_wst_file(
            file_path=file_path,
            base_kind_order=base_kind_order,
            base_indices_per_kind=base_indices_per_kind,
        )

        if features.shape[0] != num_features:
            raise ValueError(
                f"Feature length mismatch for map {i}: "
                f"expected {num_features}, got {features.shape[0]}"
            )

        x_all[i, :] = features

    # Convert to torch tensor.
    x = torch.as_tensor(x_all, dtype=torch.float32)
    return x


def standardize_features(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize features to zero mean and unit variance across the dataset.

    Args:
        x: [num_maps, num_features] tensor.

    Returns:
        x_std: standardized features.
        mean:  feature-wise mean.
        std:   feature-wise std (with zeros replaced by 1.0 to avoid division by zero).
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)

    # Avoid division by zero: if std is 0, set it to 1.
    std_replaced = std.clone()
    std_replaced[std_replaced == 0.0] = 1.0

    x_std = (x - mean) / std_replaced
    return x_std, mean, std_replaced


# ---------------------------------------------------------------------------
# 3. Prior definition
# ---------------------------------------------------------------------------

def build_box_prior_from_df(df: pd.DataFrame, param_names: List[str]) -> BoxUniform:
    """
    Define a BoxUniform prior using min/max of each parameter in the DataFrame.

    Args:
        df:           DataFrame with columns param_names.
        param_names:  list of parameter names in desired order.

    Returns:
        prior: sbi.utils.BoxUniform distribution over R^{len(param_names)}.
    """
    # Compute lower/upper bounds from data.
    lower_np = df[param_names].min().values
    upper_np = df[param_names].max().values

    lower = torch.as_tensor(lower_np, dtype=torch.float32)
    upper = torch.as_tensor(upper_np, dtype=torch.float32)

    # IMPORTANT:
    # You can replace these with physically motivated bounds, e.g.:
    #   lower = torch.tensor([logA_min, Ob0h2_min, Oc0h2_min, h_min, n_s_min, B_min])
    #   upper = torch.tensor([logA_max, Ob0h2_max, Oc0h2_max, h_max, n_s_max, B_max])
    #
    # Using min/max from df ensures that the prior covers your training data.
    # For true cosmological inference you might want a slightly wider or more
    # physically informed prior.

    prior = BoxUniform(low=lower, high=upper)
    return prior


# ---------------------------------------------------------------------------
# 4. NPE training and posterior construction
# ---------------------------------------------------------------------------

def train_npe(
    theta: torch.Tensor,
    x: torch.Tensor,
    prior: BoxUniform,
) -> NPE:
    """
    Set up and train an NPE model with sbi.

    Args:
        theta: [num_maps, num_params] tensor of parameters.
        x:     [num_maps, num_features] tensor of WST features.
        prior: sbi.utils.BoxUniform prior over parameters.

    Returns:
        inference: trained NPE inference object from sbi.
    """
    print("Setting up NPE with density_estimator='nsf' (neural spline flow).")
    inference = NPE(prior=prior, density_estimator="nsf")

    # Append simulations (theta, x).
    # Shape checks:
    print(f"theta.shape = {theta.shape}  (num_maps x num_params)")
    print(f"x.shape     = {x.shape}  (num_maps x num_features)")

    inference = inference.append_simulations(theta, x)

    # Train the density estimator.
    # You can pass keyword args to .train(), for example:
    #   density_estimator = inference.train(
    #       max_num_epochs=200,
    #       stop_after_epochs=20,
    #       show_train_summary=True,
    #   )
    density_estimator = inference.train()
    print("Training finished.")

    return inference


def build_posterior_from_inference(inference: NPE):
    """
    Build a DirectPosterior from a trained NPE inference object.

    Returns:
        posterior: DirectPosterior object.
    """
    posterior = inference.build_posterior()
    print("Built posterior:")
    print(posterior)
    return posterior


# ---------------------------------------------------------------------------
# 5. Posterior sampling and visualization
# ---------------------------------------------------------------------------

def sample_posterior_for_observation(
    posterior,
    x: torch.Tensor,
    obs_index: int,
    num_samples: int,
    param_names: List[str],
) -> torch.Tensor:
    """
    Sample from the posterior for a given observation index.

    Args:
        posterior:    sbi DirectPosterior object.
        x:            [num_maps, num_features] tensor of (standardized) WST features.
        obs_index:    integer index of the observation (e.g. 13).
        num_samples:  number of posterior samples to draw.
        param_names:  list of parameter names (for pretty printing).

    Returns:
        samples: [num_samples, num_params] tensor of posterior samples.
    """
    # Extract single observation, keep batch dimension as [1, num_features].
    x_obs = x[obs_index].unsqueeze(0)

    print(f"Sampling {num_samples} posterior samples for map index {obs_index}...")
    samples = posterior.sample((num_samples,), x=x_obs)

    # Compute posterior mean and std per parameter.
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=False)

    print("\nPosterior summary (mean ± std) for each parameter:")
    for name, m, s in zip(param_names, mean.tolist(), std.tolist()):
        print(f"  {name:>6}: {m:.4f} ± {s:.4f}")

    return samples


def plot_posterior_pairplot(samples: torch.Tensor, param_names: List[str]):
    """
    Visualize 1D and 2D marginals using sbi.analysis.pairplot.

    Args:
        samples:     [num_samples, num_params] tensor of posterior samples.
        param_names: list of parameter names (used as labels).
    """
    _ = pairplot(samples, labels=param_names, figsize=(6, 6))


# ---------------------------------------------------------------------------
# 6. Main script
# ---------------------------------------------------------------------------

def main():
    # ----------------------------
    # 6.1 Load parameter DataFrame
    # ----------------------------
    # Load df from the .pt file (first 6 columns only: logA, Ob0h2, Oc0h2, h, n_s, B)
    df = load_df_from_pt(THETA_PT_PATH)

    num_maps = len(df)
    print(f"Loaded parameter DataFrame with {num_maps} rows and columns {df.columns.tolist()}")

    # Extract theta.
    theta = build_theta_from_df(df, PARAM_NAMES)
    print(f"Theta tensor shape: {theta.shape}")

    # ---------------------------------------
    # 6.2 Load and preprocess WST CSVs into x
    # ---------------------------------------
    print("Building WST design matrix X from CSV files...")
    x = build_wst_design_matrix(
        num_maps=num_maps,
        wst_dir=WST_BASE_DIR,
        file_template=WST_FILE_TEMPLATE,
    )
    print(f"WST tensor shape (before standardization): {x.shape}")

    # Optionally standardize WST features.
    if STANDARDIZE_WST:
        x, x_mean, x_std = standardize_features(x)
        print("Standardized WST features (zero mean, unit variance).")
        print(f"x_mean shape: {x_mean.shape}, x_std shape: {x_std.shape}")
    else:
        x_mean = None
        x_std = None

    # --------------------------------
    # 6.3 Define prior over parameters
    # --------------------------------
    prior = build_box_prior_from_df(df, PARAM_NAMES)
    print("Defined BoxUniform prior using df min/max:")
    print(f"  low  = {prior.low}")
    print(f"  high = {prior.high}")
    print("You can replace these bounds with physically motivated ones if desired.")

    # ------------------------------
    # 6.4 Train NPE and build posterior
    # ------------------------------
    inference = train_npe(theta, x, prior)
    posterior = build_posterior_from_inference(inference)

    # ------------------------------------------------------
    # 6.5 Example: posterior for a single WST observation
    # ------------------------------------------------------
    obs_index = min(OBS_INDEX, num_maps - 1)  # ensure in range
    samples = sample_posterior_for_observation(
        posterior=posterior,
        x=x,
        obs_index=obs_index,
        num_samples=NUM_POSTERIOR_SAMPLES,
        param_names=PARAM_NAMES,
    )

    #print("\nSaving posterior samples to GetDist chain...")

    #import numpy as np
    #from getdist import MCSamples
    
    #samples_np = samples.cpu().numpy()
    #names = ["logA","Ob0h2","Oc0h2","h","n_s","B"]
    
    #gd = MCSamples(samples=samples_np, names=names)
    
    #outname = f"chain_tsz_wst_map{obs_index}"
    #gd.saveAsText(outname)
    
    #print(f"GetDist chain saved as: {outname}.txt\n")
    # ------------------------------------------------------
    # 6.6 Optional: visualize posterior with pairplot
    # ------------------------------------------------------
    # This will open a Matplotlib figure if you run the script in an environment
    # that can display plots (Jupyter, local Python, etc.).
    try:
        import matplotlib.pyplot as plt
        _ = pairplot(samples, labels=PARAM_NAMES, figsize=(6, 6))
        plt.show()
    except Exception as e:
        print(f"Could not generate pairplot (maybe no display?): {e}")


if __name__ == "__main__":
    main()
