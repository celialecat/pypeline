import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

def validate_ps(
    theory: Tuple[np.ndarray, np.ndarray],
    empirical: Tuple[np.ndarray, np.ndarray],
    *,
    ell_range: Optional[Tuple[float, float]] = None,
    interp_kind: str = "linear",
    eps: float = 1e-30,
    yerr_emp: Optional[np.ndarray] = None,
    yerr_th: Optional[np.ndarray] = None,
    xscale: str = "linear",
    yscale: str = 'linear',
    return_interp: bool = False,
    plot: bool = False,
    title: str = "D_ℓ Validation: Empirical vs Theoretical",
    legend_emp: str = "Empirical",
    legend_th: str = "Theoretical"
) -> Dict[str, np.ndarray | float]:
    """
    Compares an empirical power spectrum (D_ell) with a theoretical one.
    
    Calculates relative differences, statistics (mean, median, RMSE), 
    and an optional reduced chi-squared. Can also generate plots.
    """

    def _as_1d(a, name):
        """Helper to ensure input is a 1D float array."""
        a = np.asarray(a)
        if a.ndim == 1: 
            return a.astype(float)
        a = a.squeeze()
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1D after squeeze; received shape: {a.shape}")
        return a.astype(float)

    ell_th, D_th = theory
    ell_emp, D_emp = empirical
    ell_th  = _as_1d(ell_th,  "ell_th")
    D_th    = _as_1d(D_th,    "D_th")
    ell_emp = _as_1d(ell_emp, "ell_emp")
    D_emp   = _as_1d(D_emp,   "D_emp")

    # -- Fix inconsistent theory pair (len mismatch) by building a grid for D_th
    if len(D_th) != len(ell_th):
        if np.any(~np.isfinite(ell_th)) or len(ell_th) < 2:
            raise ValueError("ell_th invalid for reconstructing a grid.")
        ell_low = float(np.nanmin(ell_th))
        ell_high = float(np.nanmax(ell_th))
        if not (np.isfinite(ell_low) and np.isfinite(ell_high)) or ell_high <= ell_low:
            raise ValueError("Invalid ell_th bounds for reconstructing a grid.")
        # Recreate ell_th as a geometric space matching D_th's length
        ell_th = np.geomspace(max(1.0, ell_low), ell_high, num=len(D_th))
        # yerr_th will be processed similarly below

    # -- Optional yerr -> 1D and consistent length
    if yerr_emp is not None:
        yerr_emp = _as_1d(yerr_emp, "yerr_emp")
        if len(yerr_emp) != len(ell_emp):
            raise ValueError("yerr_emp must have the same length as ell_emp.")
    if yerr_th is not None:
        yerr_th = _as_1d(yerr_th, "yerr_th")
        if len(yerr_th) != len(D_th):
            # If yerr_th has a length mismatch (e.g., from the original, 
            # mismatched ell_th), we resample it to match the new D_th length.
            old_len = len(yerr_th)
            if old_len < 2:
                yerr_th = None  # Not usable
            else:
                # Remap yerr_th by uniform resampling (index -> new size)
                idx_old = np.linspace(0, 1, old_len)
                idx_new = np.linspace(0, 1, len(D_th))
                yerr_th = np.interp(idx_new, idx_old, yerr_th)

    # -- Sort + deduplicate (interpolation requires strictly increasing x)
    def _sort_unique(x, y):
        """Sorts x and y by x, and handles duplicates in x by averaging y."""
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # Deduplicate x; aggregate y by mean if needed
        dx = np.diff(x)
        if np.any(dx == 0):
            unique_x, inverse_indices, counts = np.unique(x, return_inverse=True, return_counts=True)
            accumulator = np.zeros_like(unique_x, dtype=float)
            for i, yy in enumerate(y):
                accumulator[inverse_indices[i]] += yy
            y = accumulator / counts
            x = unique_x
        return x, y

    ell_th, D_th = _sort_unique(ell_th, D_th)
    ell_emp, D_emp = _sort_unique(ell_emp, D_emp)
    if yerr_emp is not None:
        # Sort yerr_emp according to the same ell_emp sorting
        ell_emp_sorted, yerr_emp = _sort_unique(ell_emp, yerr_emp)
        assert np.all(ell_emp_sorted == ell_emp), "ell_emp mismatch after sorting yerr_emp"
    if yerr_th is not None:
        # Sort yerr_th according to the same ell_th sorting
        ell_th_sorted, yerr_th = _sort_unique(ell_th, yerr_th)
        assert np.all(ell_th_sorted == ell_th), "ell_th mismatch after sorting yerr_th"


    # -- Determine ℓ overlap (avoids extrapolation)
    ell_min_overlap = max(ell_th.min(), ell_emp.min())
    ell_max_overlap = min(ell_th.max(), ell_emp.max())
    mask_overlap = (ell_emp >= ell_min_overlap) & (ell_emp <= ell_max_overlap)

    # -- User cut
    if ell_range is not None:
        lo, hi = ell_range
        mask_user = (ell_emp >= lo) & (ell_emp <= hi)
        mask = mask_overlap & mask_user
    else:
        mask = mask_overlap

    if not np.any(mask):
        raise ValueError("No ℓ overlap between theory and empirical (after cuts).")

    x = ell_emp[mask]
    y_emp = D_emp[mask]
    err_emp = yerr_emp[mask] if yerr_emp is not None else None

    # -- Interpolate theory -> empirical grid
    if interp_kind not in ("linear", "nearest"):
        raise ValueError("interp_kind must be 'linear' or 'nearest'.")

    if interp_kind == "linear":
        y_th_full = np.interp(ell_emp, ell_th, D_th)
        err_th_full = np.interp(ell_emp, ell_th, yerr_th) if yerr_th is not None else None
    else:  # nearest
        # Find nearest indices
        indices = np.searchsorted(ell_th, ell_emp, side="left")
        indices = np.clip(indices, 0, len(ell_th)-1)
        left = np.maximum(indices-1, 0)
        # Check which is closer: the one at 'left' or the one at 'indices'
        choose_left = (indices == 0) | ((ell_emp - ell_th[left]) <= (ell_th[indices] - ell_emp))
        nearest_indices = np.where(choose_left, left, indices)
        
        y_th_full = D_th[nearest_indices]
        err_th_full = yerr_th[nearest_indices] if yerr_th is not None else None

    # Apply the final mask
    y_th = y_th_full[mask]
    err_th = err_th_full[mask] if err_th_full is not None else None

    # -- Relative difference and propagated errors
    # Avoid division by zero
    denominator = np.where(np.abs(y_th) > eps, y_th, np.sign(y_th)*eps + (y_th==0)*eps)
    rel_diff = (y_emp - y_th) / denominator

    rel_error = None
    if (err_emp is not None) or (err_th is not None):
        # Propagate errors for (A-B)/B -> sqrt[ (err_A/B)^2 + (A*err_B / B^2)^2 ]
        rel_variance = np.zeros_like(rel_diff, dtype=float)
        if err_emp is not None:
            rel_variance += (err_emp / denominator)**2
        if err_th is not None:
            rel_variance += ((y_emp * err_th) / (denominator**2))**2
        rel_error = np.sqrt(rel_variance)

    # -- Statistics
    output = {
        "ell": x,
        "D_emp": y_emp,
        "rel_diff": rel_diff,
        "rel_mean": float(np.mean(rel_diff)),
        "rel_median": float(np.median(rel_diff)),
        "rel_rmse": float(np.sqrt(np.mean(rel_diff**2))),
        "rel_max_abs": float(np.max(np.abs(rel_diff))),
    }
    if return_interp:
        output["D_th_interp"] = y_th
        if err_th is not None:
            output["yerr_th_interp"] = err_th
    if rel_error is not None:
        output["rel_err"] = rel_error

    # -- Reduced χ² if possible
    if (err_emp is not None) or (err_th is not None):
        total_variance = np.zeros_like(y_emp)
        if err_emp is not None:
            total_variance += err_emp**2
        if err_th is not None:
            total_variance += err_th**2
            
        # Avoid division by zero if variance is tiny
        total_variance = np.clip(total_variance, a_min=eps**2, a_max=None)
        
        chi2 = np.sum((y_emp - y_th)**2 / total_variance)
        dof = max(1, len(x)-1) # Degrees of freedom (N_points - N_params)
        output["chi2_reduced"] = float(chi2/dof)

    # -- Plots
    if plot:
        plt.figure(figsize=(6.6, 4.6))
        # Plot empirical data
        if err_emp is not None:
            plt.errorbar(x, y_emp, yerr=err_emp, fmt='o', ms=3, lw=1, capsize=2, label=legend_emp)
        else:
            plt.plot(x, y_emp, 'o', ms=3, label=legend_emp)
            
        # Plot theoretical data
        plt.plot(x, y_th, '-', lw=1.5, label=legend_th)
        if err_th is not None:
            # Show theoretical uncertainty as a band
            ymin = np.maximum(y_th - err_th, eps) # Avoid log-scale issues
            ymax = np.maximum(y_th + err_th, eps)
            plt.fill_between(x, ymin, ymax, alpha=0.2, linewidth=0, color=plt.gca().lines[-1].get_color())
            
        plt.xscale(xscale); plt.yscale(yscale)
        plot_title = title + (f"  (χ²_r={output['chi2_reduced']:.2f})" if 'chi2_reduced' in output else "")
        plt.title(plot_title); plt.xlabel(r'$\ell$'); plt.ylabel(r'$D_\ell$'); plt.legend(); plt.tight_layout(); plt.show()

        # Plot relative difference
        plt.figure(figsize=(6.6, 3.8))
        if rel_error is not None:
            plt.errorbar(x, rel_diff, yerr=rel_error, fmt='.', ms=3, lw=1, capsize=2)
        else:
            plt.plot(x, rel_diff, '.', ms=3)
        plt.axhline(0, lw=1, color='black', linestyle='--')
        plt.xscale(xscale) # Use same x-scale
        plt.xlabel(r'$\ell$'); plt.ylabel(r'$(D^{\rm emp}-D^{\rm th})/D^{\rm th}$')
        plt.title("Relative Difference" + (" (±σ propagated)" if rel_error is not None else ""))
        plt.tight_layout(); plt.show()

    return output