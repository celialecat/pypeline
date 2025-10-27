#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to generate a 2D Latin Hypercube (LHS) over (sigma8, Omega_m).

This script generates N samples from an LHS spanning sigma8 and Omega_m.
For each sample, it keeps other cosmological parameters fixed but converts
the (sigma8, Omega_m) pair into the corresponding (omega_cdm, ln10^{10}A_s)
values.

The conversion ensures that the resulting cosmology, when computed by the
'classy_sz' library, exactly reproduces the target sigma8 value from the LHS.
This is achieved by:
1. Converting Omega_m to the physical density omega_cdm.
2. Using the linear scaling sigma8 ∝ sqrt(A_s) to find the exact
   ln10^{10}A_s required to match the target sigma8.

The main function `make_lhs_and_convert` handles this entire process and
returns a pandas DataFrame with the computed parameters.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # Attempt to import the QMC module from SciPy for LHS
    from scipy.stats import qmc as _qmc
    _HAS_SCIPY_QMC = True
except Exception:
    _HAS_SCIPY_QMC = False

# Import the specific CLASS wrapper
from classy_sz import Class as Class_sz

# Type hints for clarity
Number = float
Interval = Tuple[Number, Number]


@dataclass
class FixedCosmo:
    """Container for fixed cosmological parameters.

    Parameters are **physical densities** for baryons and CDM (omega_x == Ω_x h^2).
    """
    omega_b: Number = 0.02242
    omega_cdm: Number = 0.11933  # will be overridden per-sample
    H0: Number = 67.66
    tau_reio: Number = 0.054
    ln10_10_As: Number = 3.043    # baseline; will be adjusted per-sample
    n_s: Number = 0.9665
    omega_ncdm: Number = 0.0      # set if you include massive neutrinos (physical)

    def to_classy(self) -> dict:
        """Return parameters formatted for classy_sz."""
        return {
            "omega_b": float(self.omega_b),
            "omega_cdm": float(self.omega_cdm),
            "H0": float(self.H0),
            "tau_reio": float(self.tau_reio),
            "ln10^{10}A_s": float(self.ln10_10_As),
            "n_s": float(self.n_s),
        }


def _lhs_2d(N: int, intervals: Sequence[Interval], seed: Optional[int] = None) -> np.ndarray:
    """Latin Hypercube in 2D over given intervals; uniform margins.

    Falls back to simple uniform sampling if SciPy QMC is unavailable.
    Returns shape (N, 2).
    """
    lows = np.array([a for a, _ in intervals], dtype=float)
    highs = np.array([b for _, b in intervals], dtype=float)

    if _HAS_SCIPY_QMC:
        # Use SciPy's robust LHS implementation
        sampler = _qmc.LatinHypercube(d=2, seed=seed)
        U = sampler.random(n=N) # U is in [0, 1]^d
    else:
        # Fallback: Simple stratified sampling per dimension
        # This is less optimal than LHS but avoids the dependency.
        rng = np.random.default_rng(seed)
        U = (rng.permutation(N) + rng.random((N, 1))) / N
        V = (rng.permutation(N) + rng.random((N, 1))) / N
        U = np.hstack([U, V])
    print("Using SciPy LHS" if _HAS_SCIPY_QMC else "Using fallback sampling")

    # Scale the samples from [0, 1]^d to the target intervals
    if _HAS_SCIPY_QMC:
        return _qmc.scale(U, lows, highs)
    else:
        return lows + U * (highs - lows)


def _compute_sigma8(cpars: dict) -> float:
    """Compute sigma8 using classy_sz for a given set of parameters."""
    cls = Class_sz()
    cls.set(cpars)
    cls.set({"output": "mPk"})  # Linear P(k) is sufficient for sigma8()
    
    # Run the computation
    cls.compute_class_szfast()
    
    # Retrieve the sigma8 value
    val = float(cls.sigma8())
    
    # Cleanup to free memory
    try:
        cls.struct_cleanup(); cls.empty()
    except Exception:
        pass
    return val


def _omega_cdm_from_omegam(omegam: float, H0: float, omega_b: float, omega_ncdm: float = 0.0) -> float:
    """Convert total matter density Ω_m to physical CDM density ω_cdm.

    Uses the relation: Ω_m = (ω_b + ω_cdm + ω_ncdm) / h^2
    Therefore: ω_cdm = (Ω_m * h^2) - ω_b - ω_ncdm
    """
    h = H0 / 100.0
    return (h * h * float(omegam)) - float(omega_b) - float(omega_ncdm)


def _adjust_lnAs_for_sigma8(target_sigma8: float, fixed: FixedCosmo) -> float:
    """Return ln10^{10}A_s that yields the target sigma8 for the given cosmology.

    Uses the exact linear-theory scaling: sigma8 ∝ sqrt(A_s).
    """
    # 1) Evaluate sigma8 at the baseline amplitude (A_s_0)
    base_pars = fixed.to_classy()
    sigma8_0 = _compute_sigma8(base_pars)

    # 2) Apply linear scaling to find the new amplitude (A_s_new)
    # We know:
    # (sigma8_new / sigma8_0) = sqrt(A_s_new / A_s_0)
    # (target_sigma8 / sigma8_0)^2 = A_s_new / A_s_0
    # A_s_new = A_s_0 * (target_sigma8 / sigma8_0)^2
    
    # We work in log-space for ln(10^{10} A_s), which we call 'lnAs'
    # lnAs = ln(10^{10} A_s) => A_s = 10^{-10} * exp(lnAs)
    #
    # A_s_new / A_s_0 = (10^{-10} * exp(lnAs_new)) / (10^{-10} * exp(lnAs_0))
    # A_s_new / A_s_0 = exp(lnAs_new - lnAs_0)
    #
    # Equating the two:
    # exp(lnAs_new - lnAs_0) = (target_sigma8 / sigma8_0)^2
    # lnAs_new - lnAs_0 = ln[ (target_sigma8 / sigma8_0)^2 ]
    # lnAs_new = lnAs_0 + 2.0 * ln(target_sigma8 / sigma8_0)
    
    lnAs0 = float(fixed.ln10_10_As)
    lnAs_new = lnAs0 + 2.0 * math.log(float(target_sigma8) / sigma8_0)
    
    return lnAs_new


def make_lhs_and_convert(
    N: int,
    sigma8_interval: Interval,
    omegam_interval: Interval,
    seed: Optional[int] = 12345,
    base: FixedCosmo = FixedCosmo(),
    omega_ncdm: float = 0.0,
    verify: bool = True,
    extra_params: Optional[dict] = None,
    returned_params: Optional[Sequence[str]] = None,  # <-- NEW ARGUMENT
) -> pd.DataFrame:
    """
    Create an LHS over (sigma8, Omega_m) and convert to physical parameters,
    returning a DataFrame with all cosmological parameters and verification checks.

    Args:
        N: Number of samples to generate.
        sigma8_interval: (min, max) for sigma8.
        omegam_interval: (min, max) for Omega_m.
        seed: Random seed for reproducibility.
        base: A FixedCosmo object with baseline parameters.
        omega_ncdm: Physical density of massive neutrinos (if any).
        verify: If True, runs CLASS a second time on each sample to
                check if the target sigma8 was accurately reproduced.
        extra_params: A dictionary of extra nuisance parameters to add.
        returned_params: A list of column names to return.
                         If None (default), returns the basic cosmological
                         parameter grid: [h, logA, n_s, Ob0h2, Oc0h2].
    
    Returns:
        A pandas DataFrame containing the parameter grid.
    """
    start_time = time.time()  # <-- Start timing

    # Default nuisance parameters
    default_extra = {
        "B": 1.35,
        "A_cib": 4.7,
        "A_ir": 3.2,
        "A_rs": 0.94,
    }
    if extra_params:
        default_extra.update(extra_params)

    # Note: The original code defined 'rng' here but did not use it.
    # The seed is passed directly to _lhs_2d.
    # rng = np.random.default_rng(seed)

    # Generate the 2D LHS grid in the (sigma8, Omega_m) plane
    lhs = _lhs_2d(N, intervals=[sigma8_interval, omegam_interval], seed=seed)
    sigma8_targets = lhs[:, 0]
    omegam_targets = lhs[:, 1]

    rows = []
    print(f"Generating {N} samples...")
    for i, (s8_t, Om_t) in enumerate(zip(sigma8_targets, omegam_targets)):
        
        # 1) Compute physical CDM density from Omega_m
        omega_cdm = _omega_cdm_from_omegam(Om_t, base.H0, base.omega_b, omega_ncdm)
        if omega_cdm <= 0:
            raise ValueError(f"Derived omega_cdm <= 0 for Omega_m={Om_t:.4f}. Check intervals.")

        # 2) Adjust A_s to reproduce the target sigma8
        #    Create a temporary FixedCosmo instance with the *new* omega_cdm
        fixed_for_this_sample = FixedCosmo(
            omega_b=base.omega_b,
            omega_cdm=omega_cdm,       # Use the derived value
            H0=base.H0,
            tau_reio=base.tau_reio,
            ln10_10_As=base.ln10_10_As, # Use the baseline A_s for scaling
            n_s=base.n_s,
        )
        
        # This function runs CLASS once to get the scaling factor
        lnAs = _adjust_lnAs_for_sigma8(s8_t, fixed_for_this_sample)

        # 3) Verify consistency (optional)
        sigma8_chk = np.nan
        Omegam_chk = np.nan
        if verify:
            # Create the final parameter set for this sample
            test_pars = {
                **fixed_for_this_sample.to_classy(), # Contains correct omega_cdm
                "ln10^{10}A_s": float(lnAs),         # Contains correct lnAs
            }
            # Add omega_ncdm if it is non-zero
            if omega_ncdm > 0:
                test_pars["omega_ncdm"] = float(omega_ncdm)

            # Run CLASS a second time with final params to check sigma8
            sigma8_chk = _compute_sigma8(test_pars)
            
            # Also verify the Omega_m conversion
            h2 = (base.H0 / 100.0) ** 2
            Omegam_chk = (base.omega_b + omega_cdm + omega_ncdm) / h2

        # 4) Record one full sample, including all parameters
        row = {
            "h": base.H0 / 100.0,
            "logA": float(lnAs), # Use 'logA' as the column name
            "n_s": base.n_s,
            "Ob0h2": base.omega_b,
            "Oc0h2": omega_cdm,
            # Nuisance parameters
            "B": default_extra["B"],
            "A_cib": default_extra["A_cib"],
            "A_ir": default_extra["A_ir"],
            "A_rs": default_extra["A_rs"],
            # Target and verification values
            "Omega_m_target": float(Om_t),
            "Omega_m_check": float(Omegam_chk),
            "sigma8_target": float(s8_t),
            "sigma8_check": float(sigma8_chk),
        }
        rows.append(row)

    # Create the full DataFrame with all possible columns
    all_columns_order = [
        "h", "logA", "n_s", "Ob0h2", "Oc0h2", "B", "A_cib", "A_ir", "A_rs",
        "Omega_m_target", "Omega_m_check", "sigma8_target", "sigma8_check"
    ]
    df_full = pd.DataFrame(rows, columns=all_columns_order)

    # --- NEW LOGIC TO FILTER COLUMNS ---
    if returned_params is None:
        # Default behavior:
        # Return the basic cosmological parameters.
        # In this grid, only logA and Oc0h2 vary; the others are fixed.
        final_columns = ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]
    else:
        # User provided a specific list of columns
        available_cols = df_full.columns
        # Filter the list, keeping the user's specified order
        final_columns = [col for col in returned_params if col in available_cols]
        
        # Warn if some requested columns do not exist
        missing_cols = set(returned_params) - set(final_columns)
        if missing_cols:
            print(f"\nWarning: Requested parameters not available and ignored: {missing_cols}")

    # Select the final columns for the output DataFrame
    df_out = df_full[final_columns]
    # --- END OF NEW LOGIC ---

    # Compute total elapsed time
    elapsed = time.time() - start_time
    print(f"\nExecution completed in {elapsed:.2f} seconds.")

    return df_out


if __name__ == "__main__":
    # This block runs only when the script is executed directly
    
    # --- Example 1: Default behavior (basic cosmological parameters) ---
    print("--- Example 1: Default parameters (base cosmology) ---")
    df_default = make_lhs_and_convert(
        N=16, # Reduced N for a quick test
        sigma8_interval=(0.7, 0.9),
        omegam_interval=(0.25, 0.35),
        seed=20250831,
        verify=True, # Keep verification enabled
        returned_params=None # Explicitly show default
    )
    df_default.to_csv("lhs_default_params.csv", index=False)
    print(df_default.head())


    # --- Example 2: Custom parameters (cosmo + verification checks) ---
    print("\n--- Example 2: Custom parameters (logA, Oc0h2, and checks) ---")
    custom_cols = [
        "logA", 
        "Oc0h2", 
        "sigma8_target", 
        "sigma8_check", 
        "Omega_m_target", 
        "Omega_m_check",
        "B" # Let's also add a nuisance parameter
    ]
    df_custom = make_lhs_and_convert(
        N=16, # Reduced N for a quick test
        sigma8_interval=(0.7, 0.9),
        omegam_interval=(0.25, 0.35),
        seed=20250831,
        verify=True,
        returned_params=custom_cols
    )
    df_custom.to_csv("lhs_custom_params.csv", index=False)
    print(df_custom.head())
    
    # --- Example 3: Requesting all parameters (by listing them) ---
    print("\n--- Example 3: Requesting all parameters ---")
    all_cols = [
        "h", "logA", "n_s", "Ob0h2", "Oc0h2", "B", "A_cib", "A_ir", "A_rs",
        "Omega_m_target", "Omega_m_check", "sigma8_target", "sigma8_check"
    ]
    df_all = make_lhs_and_convert(
        N=16, # Reduced N for a quick test
        sigma8_interval=(0.7, 0.9),
        omegam_interval=(0.25, 0.35),
        seed=20250831,
        verify=True,
        returned_params=all_cols
    )
    df_all.to_csv("lhs_all_params.csv", index=False)
    print(df_all.head())