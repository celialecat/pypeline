import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

def validation_ps(
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
    title: str = "Validation D_ℓ: empirical vs theoretical",
    legend_emp: str = "Empirical",
    legend_th: str = "Theoretical"
) -> Dict[str, np.ndarray | float]:

    def _as_1d(a, name):
        a = np.asarray(a)
        if a.ndim == 1: 
            return a.astype(float)
        a = a.squeeze()
        if a.ndim != 1:
            raise ValueError(f"{name} doit être 1D après squeeze; shape reçu: {a.shape}")
        return a.astype(float)

    ell_th, D_th = theory
    ell_emp, D_emp = empirical
    ell_th  = _as_1d(ell_th,  "ell_th")
    D_th    = _as_1d(D_th,    "D_th")
    ell_emp = _as_1d(ell_emp, "ell_emp")
    D_emp   = _as_1d(D_emp,   "D_emp")

    # -- Répare un couple théorie incohérent (len mismatch) en construisant une grille pour D_th
    if len(D_th) != len(ell_th):
        if np.any(~np.isfinite(ell_th)) or len(ell_th) < 2:
            raise ValueError("ell_th invalide pour reconstruire une grille.")
        ell_lo = float(np.nanmin(ell_th))
        ell_hi = float(np.nanmax(ell_th))
        if not (np.isfinite(ell_lo) and np.isfinite(ell_hi)) or ell_hi <= ell_lo:
            raise ValueError("Bornes de ell_th invalides pour reconstruire une grille.")
        ell_th = np.geomspace(max(1.0, ell_lo), ell_hi, num=len(D_th))
        # yerr_th sera traité plus bas de la même manière

    # -- yerr optionnels → 1D et longueur cohérente
    if yerr_emp is not None:
        yerr_emp = _as_1d(yerr_emp, "yerr_emp")
        if len(yerr_emp) != len(ell_emp):
            raise ValueError("yerr_emp doit avoir la même longueur que ell_emp.")
    if yerr_th is not None:
        yerr_th = _as_1d(yerr_th, "yerr_th")
        if len(yerr_th) != len(D_th):
            # Si yerr_th vient d’une autre grille, on reconstruit une grille cohérente pour elle aussi
            # en l’interpolant d’abord sur l’ancienne ell_th si possible (fallback simple)
            # Ici: on la remet simplement à la bonne taille par interpolation géométrique uniforme
            # (plus sûr: on la régénère après qu’on a fixé ell_th)
            # On suppose qu’avant mismatch, yerr_th correspondait à une grille fine monotone.
            # Si ce n’est pas le cas, on ravalera ci-dessous lors de l’interp finale.
            old_n = len(yerr_th)
            if old_n < 2:
                yerr_th = None  # pas exploitable
            else:
                # Remappe yerr_th par rééchantillonnage uniforme (index → nouvelle taille)
                idx_old = np.linspace(0, 1, old_n)
                idx_new = np.linspace(0, 1, len(D_th))
                yerr_th = np.interp(idx_new, idx_old, yerr_th)

    # -- Tri + dédoublonnage (interpolation exige xp strictement croissant)
    def _sort_unique(x, y):
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # dédoublonne x ; agrège y par moyenne si besoin
        dx = np.diff(x)
        if np.any(dx == 0):
            uniq_x, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
            accum = np.zeros_like(uniq_x, dtype=float)
            for i, yy in enumerate(y):
                accum[inv[i]] += yy
            y = accum / counts
            x = uniq_x
        return x, y

    ell_th, D_th = _sort_unique(ell_th, D_th)
    ell_emp, D_emp = _sort_unique(ell_emp, D_emp)
    if yerr_emp is not None:
        ell_emp, yerr_emp = _sort_unique(ell_emp, yerr_emp)
    if yerr_th is not None:
        ell_th, yerr_th = _sort_unique(ell_th, yerr_th)

    # -- Détermine l’overlap en ℓ (évite toute extrapolation)
    ell_min_overlap = max(ell_th.min(), ell_emp.min())
    ell_max_overlap = min(ell_th.max(), ell_emp.max())
    mask_overlap = (ell_emp >= ell_min_overlap) & (ell_emp <= ell_max_overlap)

    # Coupe utilisateur
    if ell_range is not None:
        lo, hi = ell_range
        mask_user = (ell_emp >= lo) & (ell_emp <= hi)
        m = mask_overlap & mask_user
    else:
        m = mask_overlap

    if not np.any(m):
        raise ValueError("Aucun recouvrement en ℓ entre théorie et empirique (après coupes).")

    x = ell_emp[m]
    y_emp = D_emp[m]
    err_emp = yerr_emp[m] if yerr_emp is not None else None

    # -- Interpolation théorie → grille empirique
    if interp_kind not in ("linear", "nearest"):
        raise ValueError("interp_kind doit être 'linear' ou 'nearest'.")

    if interp_kind == "linear":
        y_th_full = np.interp(ell_emp, ell_th, D_th)
        err_th_full = np.interp(ell_emp, ell_th, yerr_th) if yerr_th is not None else None
    else:  # nearest
        idx = np.searchsorted(ell_th, ell_emp, side="left")
        idx = np.clip(idx, 0, len(ell_th)-1)
        left = np.maximum(idx-1, 0)
        choose_left = (idx == 0) | ((ell_emp - ell_th[left]) <= (ell_th[idx] - ell_emp))
        nearest_idx = np.where(choose_left, left, idx)
        y_th_full = D_th[nearest_idx]
        err_th_full = yerr_th[nearest_idx] if yerr_th is not None else None

    y_th = y_th_full[m]
    err_th = err_th_full[m] if err_th_full is not None else None

    # -- Différence relative et erreurs propagées
    denom = np.where(np.abs(y_th) > eps, y_th, np.sign(y_th)*eps + (y_th==0)*eps)
    rel = (y_emp - y_th) / denom

    rel_err = None
    if (err_emp is not None) or (err_th is not None):
        var_rel = np.zeros_like(rel, dtype=float)
        if err_emp is not None:
            var_rel += (err_emp / denom)**2
        if err_th is not None:
            var_rel += ((y_emp * err_th) / (denom**2))**2
        rel_err = np.sqrt(var_rel)

    # -- Statistiques
    out = {
        "ell": x,
        "D_emp": y_emp,
        "rel_diff": rel,
        "rel_mean": float(np.mean(rel)),
        "rel_median": float(np.median(rel)),
        "rel_rmse": float(np.sqrt(np.mean(rel**2))),
        "rel_max_abs": float(np.max(np.abs(rel))),
    }
    if return_interp:
        out["D_th_interp"] = y_th
        if err_th is not None:
            out["yerr_th_interp"] = err_th
    if rel_err is not None:
        out["rel_err"] = rel_err

    # -- χ² réduit si possible
    if (err_emp is not None) or (err_th is not None):
        var_tot = np.zeros_like(y_emp)
        if err_emp is not None:
            var_tot += err_emp**2
        if err_th is not None:
            var_tot += err_th**2
        var_tot = np.clip(var_tot, a_min=eps**2, a_max=None)
        chi2 = np.sum((y_emp - y_th)**2 / var_tot)
        dof = max(1, len(x)-1)
        out["chi2_reduced"] = float(chi2/dof)

    # -- Plots
    if plot:
        plt.figure(figsize=(6.6, 4.6))
        if err_emp is not None:
            plt.errorbar(x, y_emp, yerr=err_emp, fmt='o', ms=3, lw=1, capsize=2, label=legend_emp)
        else:
            plt.plot(x, y_emp, 'o', ms=3, label=legend_emp)
        plt.plot(x, y_th, '-', lw=1.5, label=legend_th)
        if err_th is not None:
            ymin = np.maximum(y_th - err_th, eps)
            ymax = np.maximum(y_th + err_th, eps)
            plt.fill_between(x, ymin, ymax, alpha=0.2, linewidth=0)
        plt.xscale(xscale); plt.yscale(yscale) #or "log"
        ttl = title + (f"  (χ²_r={out['chi2_reduced']:.2f})" if 'chi2_reduced' in out else "")
        plt.title(ttl); plt.xlabel(r'$\ell$'); plt.ylabel(r'$D_\ell$'); plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(6.6, 3.8))
        if rel_err is not None:
            plt.errorbar(x, rel, yerr=rel_err, fmt='.', ms=3, lw=1, capsize=2)
        else:
            plt.plot(x, rel, '.', ms=3)
        plt.axhline(0, lw=1)
        plt.xlabel(r'$\ell$'); plt.ylabel(r'$(D^{\rm emp}-D^{\rm th})/D^{\rm th}$')
        plt.title("Différence relative" + (" (±σ propagé)" if rel_err is not None else ""))
        plt.tight_layout(); plt.show()

    return out
