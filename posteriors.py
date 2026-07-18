# -*- coding: utf-8 -*-
"""
Cosmology posterior pipeline on (Omega_c, sigma8) using GetDist.

This module implements:
- Gaussian log-likelihood from a fiducial covariance and a data/model summary vector
- Posterior evaluation on a parameter grid (with optional interpolation)
- Priors (uniform or Gaussian) and posterior normalization
- Contours (68% and 95%), areas, and comparison metrics (vs. power spectrum)
- Marginal statistics (means, 68% intervals), and the optimized combination Σ8
- Fisher matrix forecast (optional ∂C/∂θ), and Fisher ellipse area
- Utilities for observable combination, sensitivity by slices, instrumental scenarios,
  convergence checks, and f_sky rescaling
- Integration with GetDist (MCSamples & plotting helpers)

Author: (your name)
Requirements: numpy, scipy, matplotlib, getdist
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from scipy import linalg
from scipy.special import logsumexp
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

from getdist import mcsamples, plots
from getdist.densities import Density2D, getContourLevels


# ---------- Data classes -----------------------------------------------------
from scipy.interpolate import CloughTocher2DInterpolator

def interpolate_loglike_ct(
    grid: ParamGrid,
    lnL: np.ndarray,
    *,
    fill_value: float = -np.inf,
    rescale: bool = True
) -> Callable[[ArrayLike], np.ndarray]:
    """
    Construis un interpolateur Clough–Tocher (C¹) pour ln L(Ω_c, σ8).

    Notes
    -----
    - Extrapolation hors de l'enveloppe convexe des points n'est pas définie : on renvoie fill_value.
    - `rescale=True` normalise les coordonnées (utile si les échelles diffèrent beaucoup).

    Renvoie
    -------
    f : callable
        f([[oc1, s81], [oc2, s82], ...]) -> ln L interpolé (np.ndarray)
    """
    oc_grid, s8_grid = grid.mesh                 # shapes (Ny, Nx)
    pts = np.column_stack([oc_grid.ravel(), s8_grid.ravel()])  # (Ny*Nx, 2)
    vals = lnL.ravel()                            # (Ny*Nx,)

    ct = CloughTocher2DInterpolator(pts, vals, rescale=rescale)

    def _call(points: ArrayLike) -> np.ndarray:
        ptsq = np.atleast_2d(points)
        vals = ct(ptsq)
        # Clough-Tocher renvoie NaN en dehors de l’enveloppe convexe -> remplace par fill_value
        vals = np.where(np.isfinite(vals), vals, fill_value)
        return np.squeeze(vals)

    return _call


def resample_loglike_on_grid(
    interp_fn: Callable[[ArrayLike], np.ndarray],
    oc_new: np.ndarray,
    s8_new: np.ndarray
) -> np.ndarray:
    """
    Ré-échantillonne l'interpolateur sur une nouvelle grille (oc_new, s8_new).
    Renvoie un tableau 2D de shape (len(s8_new), len(oc_new)).
    """
    ocm, s8m = np.meshgrid(oc_new, s8_new, indexing="xy")
    pts = np.column_stack([ocm.ravel(), s8m.ravel()])
    lnL_new = interp_fn(pts).reshape(s8m.shape)
    return lnL_new


@dataclass
class ParamGrid:
    """Hold a 2D parameter grid and helpers for indexing and spacing."""
    oc: np.ndarray      # 1D grid for Omega_c
    s8: np.ndarray      # 1D grid for sigma8

    @property
    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return 2D meshgrid (OC, S8)."""
        return np.meshgrid(self.oc, self.s8, indexing="xy")

    @property
    def dA(self) -> float:
        """Return the cell area (assumes uniform spacing in each dimension)."""
        if len(self.oc) < 2 or len(self.s8) < 2:
            raise ValueError("Grid must have at least 2 points per dimension")
        return (self.oc[1] - self.oc[0]) * (self.s8[1] - self.s8[0])

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((oc_min, oc_max), (s8_min, s8_max))."""
        return (self.oc.min(), self.oc.max()), (self.s8.min(), self.s8.max())


@dataclass
class PosteriorGrid:
    """Container for log-posterior and normalized posterior on a 2D grid."""
    grid: ParamGrid
    logpost: np.ndarray  # shape (Ny, Nx) matching grid.mesh (S8 rows, OC cols)
    post: np.ndarray     # same shape, normalized so that sum(P)*dA = 1
    contours: Tuple[float, float] = (0.68, 0.95)  # default confidence levels

    def density2d(self) -> Density2D:
        """Convert to GetDist Density2D (normalized by integral)."""
        oc, s8 = self.grid.mesh
        # Create Density2D with (x=Omega_c axis, y=sigma8 axis)
        dens = Density2D(self.grid.oc, self.grid.s8, self.post.copy())
        # Ensure exact normalization by integral:
        dens = dens.normalize(by="integral", in_place=False)
        return dens

    def contour_levels(self) -> List[float]:
        """Return posterior levels corresponding to the target 'contours'."""
        # getContourLevels expects a binned, normalized density (sum to ~1 with bin area)
        # We pass the *probability density* on the grid and let the routine compute levels:
        return getContourLevels(self.post, contours=self.contours)

    def area_for_level(self, level: float) -> float:
        """Approximate area enclosed by the HPD iso-probability level."""
        mask = self.post >= level
        return mask.sum() * self.grid.dA

    def A68(self) -> float:
        """Area of the 68% HPD region."""
        lvl68, *_ = self.contour_levels()
        return self.area_for_level(lvl68)

    def A95(self) -> float:
        """Area of the 95% HPD region."""
        lvls = self.contour_levels()
        lvl95 = lvls[1] if len(lvls) > 1 else lvls[0]
        return self.area_for_level(lvl95)


# ---------- Core likelihood & prior ------------------------------------------


def gaussian_loglike(
    data_vec: ArrayLike,
    model_vec: ArrayLike,
    cov: ArrayLike,
    *, use_log_norm: bool = True,
    inv_cov: Optional[np.ndarray] = None,
    hartlap_nsim: Optional[int] = None
) -> float:
    """
    Compute the Gaussian log-likelihood: ln L = -0.5[(d-m)^T C^{-1} (d-m) + ln|2πC|].

    Parameters
    ----------
    data_vec : array-like
        Observed summary statistics (flattened vector).
    model_vec : array-like
        Model-predicted summary statistics at given parameters (same shape as data_vec).
    cov : array-like
        Covariance matrix of the summary statistics (assumed parameter-independent here).
    use_log_norm : bool
        If True include the normalization term (-0.5*logdet(2πC)).
    inv_cov : np.ndarray or None
        Optional precomputed inverse covariance. If provided, overrides `cov` inversion.
    hartlap_nsim : int or None
        If provided (and > p+2), apply Hartlap correction to C^{-1}:
        C^{-1}_corr = ((N - p - 2)/(N - 1)) * C^{-1}, where p = len(data_vec).

    Returns
    -------
    float
        ln L value.
    """
    d = np.atleast_1d(data_vec).astype(float)
    m = np.atleast_1d(model_vec).astype(float)
    if d.shape != m.shape:
        raise ValueError("data_vec and model_vec must have the same shape")

    p = d.size
    if inv_cov is None:
        C = np.array(cov, dtype=float)
        invC = linalg.inv(C)
    else:
        invC = np.array(inv_cov, dtype=float)

    # Hartlap factor for unbiased inverse covariance from finite simulations
    if hartlap_nsim is not None:
        N = int(hartlap_nsim)
        if N > p + 2:
            f = (N - p - 2) / (N - 1)
            invC = f * invC
        # If N <= p+2, we skip correction (unsafe regime).

    r = d - m
    chi2 = float(r @ invC @ r)
    lnL = -0.5 * chi2
    if use_log_norm:
        if inv_cov is None:
            sign, logdet = np.linalg.slogdet(2.0 * np.pi * np.array(cov, dtype=float))
            # logdet should be positive-definite; if sign<0 something is off with covariance
        else:
            # If only inv_cov was provided, we cannot get log|C| cheaply.
            # For most comparisons the additive constant is irrelevant; drop it.
            logdet = 0.0
        lnL += -0.5 * logdet
    return lnL


# ---------- Grid construction, evaluation, interpolation ---------------------


def make_param_grid(
    fiducial: Dict[str, float],
    *,
    npts: Tuple[int, int] = (121, 121),
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    frac: float = 0.20
) -> ParamGrid:
    """
    Build a 2D parameter grid in (Omega_c, sigma8).

    By default, bounds span ±20% around fiducial values.

    Parameters
    ----------
    fiducial : dict
        {'Omega_c': value, 'sigma8': value}
    npts : (int, int)
        Number of grid points along (Omega_c, sigma8).
    bounds : dict or None
        Optionally override bounds: {'Omega_c': (min, max), 'sigma8': (min, max)}
    frac : float
        If bounds is None, use [p0*(1-frac), p0*(1+frac)] around fiducial p0.

    Returns
    -------
    ParamGrid
    """
    oc0 = float(fiducial["Omega_c"])
    s80 = float(fiducial["sigma8"])

    if bounds is None:
        oc_min, oc_max = oc0 * (1 - frac), oc0 * (1 + frac)
        s8_min, s8_max = s80 * (1 - frac), s80 * (1 + frac)
    else:
        oc_min, oc_max = bounds["Omega_c"]
        s8_min, s8_max = bounds["sigma8"]

    oc = np.linspace(oc_min, oc_max, int(npts[0]))
    s8 = np.linspace(s8_min, s8_max, int(npts[1]))
    return ParamGrid(oc=oc, s8=s8)


def evaluate_loglike_grid(
    grid: ParamGrid,
    *,
    data_vec: ArrayLike,
    cov: ArrayLike,
    model_fn: Callable[[float, float], ArrayLike],
    include_norm: bool = True,
    inv_cov: Optional[np.ndarray] = None,
    hartlap_nsim: Optional[int] = None
) -> np.ndarray:
    """
    Evaluate ln L on the (Omega_c, sigma8) grid.

    Parameters
    ----------
    grid : ParamGrid
        Parameter grid.
    data_vec : array-like
        Observed summary vector.
    cov : array-like
        Covariance of the summary stats (fiducial, assumed constant here).
    model_fn : function
        model_fn(oc, s8) -> model summary vector matching data_vec shape.
        This can be a power spectrum model or wavelet scattering S0/S1/S2, etc.
    include_norm : bool
        Include Gaussian normalization term in ln L.
    inv_cov : np.ndarray or None
        Optionally provide C^{-1} to avoid repeated inversions.
    hartlap_nsim : int or None
        Optional Hartlap correction for inverse covariance.

    Returns
    -------
    lnL : np.ndarray
        2D array with shape (Ny, Nx): sigma8 rows, Omega_c columns.
    """
    oc_grid, s8_grid = grid.mesh
    lnL = np.empty_like(oc_grid, dtype=float)

    for j in range(s8_grid.shape[0]):
        for i in range(oc_grid.shape[1]):
            model_vec = np.asarray(model_fn(oc_grid[j, i], s8_grid[j, i]), dtype=float)
            lnL[j, i] = gaussian_loglike(
                data_vec, model_vec, cov,
                use_log_norm=include_norm,
                inv_cov=inv_cov,
                hartlap_nsim=hartlap_nsim
            )
    return lnL


def interpolate_loglike(
    grid: ParamGrid,
    lnL: np.ndarray
) -> Callable[[ArrayLike], np.ndarray]:
    """
    Build a continuous interpolator for ln L on the (Omega_c, sigma8) grid.

    Returns
    -------
    f : callable
        f([[oc1, s81], [oc2, s82], ...]) -> ln L array
        (uses RegularGridInterpolator with bounds_error=False, fill_value=-inf)
    """
    f = RegularGridInterpolator(
        (grid.oc, grid.s8), lnL.T,  # RegularGridInterpolator expects axes in order (x,y); our lnL is [Ny, Nx]
        bounds_error=False, fill_value=-np.inf
    )

    def _call(points: ArrayLike) -> np.ndarray:
        pts = np.atleast_2d(points)
        vals = f(pts)
        return np.squeeze(vals)

    return _call


# ---------- Priors & posterior normalization ---------------------------------


def add_prior_to_logpost(
    grid: ParamGrid,
    lnL: np.ndarray,
    *,
    prior_kind: str = "uniform",
    # Uniform prior settings (default ±20% around fiducial grid bounds)
    uniform_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    # Gaussian prior settings
    gaussian_mean: Optional[Tuple[float, float]] = None,
    gaussian_cov: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Add a prior to ln L to get ln posterior = ln L + ln prior.

    Parameters
    ----------
    grid : ParamGrid
        Parameter grid.
    lnL : np.ndarray
        Grid of ln L values, shape (Ny, Nx).
    prior_kind : {'uniform', 'gaussian'}
        Type of prior to add.
    uniform_bounds : dict or None
        If uniform, hard bounds {'Omega_c':(min,max), 'sigma8':(min,max)}; flat inside, -inf outside.
        If None, defaults to full grid bounds (i.e. effectively flat over the grid).
    gaussian_mean : (float, float) or None
        Mean for Gaussian prior in (Omega_c, sigma8).
    gaussian_cov : np.ndarray or None
        2x2 covariance for Gaussian prior.

    Returns
    -------
    ln_post : np.ndarray
        ln posterior over the grid.
    """
    oc, s8 = grid.mesh
    lnP = np.zeros_like(lnL)

    if prior_kind.lower() == "uniform":
        if uniform_bounds is None:
            # Flat on the grid range (no extra penalty)
            mask = np.isfinite(lnP)
        else:
            oclo, ochi = uniform_bounds["Omega_c"]
            s8lo, s8hi = uniform_bounds["sigma8"]
            mask = (oc >= oclo) & (oc <= ochi) & (s8 >= s8lo) & (s8 <= s8hi)
            lnP[~mask] = -np.inf

    elif prior_kind.lower() == "gaussian":
        if gaussian_mean is None or gaussian_cov is None:
            raise ValueError("Provide gaussian_mean and gaussian_cov for Gaussian prior")
        mean = np.array(gaussian_mean, dtype=float)
        cov = np.array(gaussian_cov, dtype=float)
        invC = linalg.inv(cov)
        # Normalization adds an additive constant; include it so you can compare models fairly
        sign, logdet = np.linalg.slogdet(2.0 * np.pi * cov)
        # Evaluate quadratic form at each grid point
        dx = np.stack([oc - mean[0], s8 - mean[1]], axis=-1)  # shape (...,2)
        # (x-μ)^T C^{-1} (x-μ)
        quad = dx[..., 0] * (invC[0, 0] * dx[..., 0] + invC[0, 1] * dx[..., 1]) + \
               dx[..., 1] * (invC[1, 0] * dx[..., 0] + invC[1, 1] * dx[..., 1])
        lnP = -0.5 * (quad + logdet)
    else:
        raise ValueError("prior_kind must be 'uniform' or 'gaussian'")

    ln_post = lnL + lnP
    return ln_post


def normalize_logposterior(
    grid: ParamGrid,
    lnpost: np.ndarray
) -> PosteriorGrid:
    """
    Convert ln posterior to a normalized posterior density on the grid.

    We normalize such that sum(post) * dA = 1.

    Returns
    -------
    PosteriorGrid
    """
    # Stabilize with logsumexp in continuous measure:
    # We want Z = ∫ exp(lnpost) dθ ≈ sum exp(lnpost) * dA
    # => ln Z ≈ logsumexp(lnpost) + log(dA)
    lZ = logsumexp(lnpost) + np.log(grid.dA)
    post = np.exp(lnpost - lZ)
    return PosteriorGrid(grid=grid, logpost=lnpost, post=post)


# ---------- Contours, areas, and comparisons ---------------------------------


def hpd_levels_and_areas(
    post_grid: PosteriorGrid,
    levels: Tuple[float, float] = (0.68, 0.95)
) -> Dict[str, float]:
    """
    Compute HPD levels and areas for given probabilities.

    Returns
    -------
    dict
        {'lvl68': level_value, 'A68': area, 'lvl95': level_value, 'A95': area}
    """
    dens = post_grid.density2d()  # normalized density
    lvls = getContourLevels(dens.P, contours=levels)
    out = {"lvl68": lvls[0], "A68": post_grid.area_for_level(lvls[0])}
    if len(lvls) > 1:
        out["lvl95"] = lvls[1]
        out["A95"] = post_grid.area_for_level(lvls[1])
    return out


def compare_stat(A68_ps: float, A68_other: float) -> float:
    """
    Compute improvement factor vs power spectrum:
        ratio = A68_ps / A68_other  (larger is better than PS).
    """
    if A68_other <= 0:
        return np.inf
    return float(A68_ps / A68_other)


# ---------- Marginals & Σ8 combination --------------------------------------


def _weighted_hist_axis(
    post: PosteriorGrid, axis: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marginalize posterior along one axis (0: Omega_c, 1: sigma8).

    Returns
    -------
    x : grid points for the chosen parameter
    p : marginalized pdf evaluated at x (normalized to unit integral)
    """
    P = post.post
    dA = post.grid.dA
    if axis == 0:
        # marginal over s8 (sum over rows), keep oc
        p = P.sum(axis=0) * (post.grid.s8[1] - post.grid.s8[0])
        x = post.grid.oc
    elif axis == 1:
        # marginal over oc (sum over columns), keep s8
        p = P.sum(axis=1) * (post.grid.oc[1] - post.grid.oc[0])
        x = post.grid.s8
    else:
        raise ValueError("axis must be 0 (Omega_c) or 1 (sigma8)")

    # Normalize to 1
    Z = np.trapz(p, x)
    if Z <= 0:
        raise RuntimeError("Zero/negative normalization in marginal")
    return x, p / Z


def marginalized_mean_and_ci_68(
    post: PosteriorGrid, param: str
) -> Dict[str, float]:
    """
    Compute marginalized mean and 68% equal-tail interval for a parameter.

    Parameters
    ----------
    post : PosteriorGrid
    param : {'Omega_c', 'sigma8'}

    Returns
    -------
    dict
        {'mean': ..., 'low68': ..., 'high68': ...}
    """
    axis = 0 if param == "Omega_c" else 1
    x, p = _weighted_hist_axis(post, axis)
    mean = np.trapz(x * p, x)

    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    # 16% and 84% for 68% equal-tail (can also implement HPD if preferred)
    lo = np.interp(0.16, cdf, x)
    hi = np.interp(0.84, cdf, x)

    return {"mean": float(mean), "low68": float(lo), "high68": float(hi)}


def grid_to_mcsamples(
    post: PosteriorGrid,
    *,
    label: str = "Posterior (grid)",
    ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
) -> mcsamples.MCSamples:
    """
    Convert a PosteriorGrid to a GetDist MCSamples object.

    Notes
    -----
    - We pass grid points with weights = posterior density * dA so that sum(weights) ~ 1.
    - We do NOT set loglikes here (optional); GetDist mainly needs weights and samples.
    - Remember: loglikes in GetDist are -log(posterior) if you want to set them.
    """
    oc, s8 = post.grid.mesh
    samples = np.column_stack([oc.ravel(), s8.ravel()])
    weights = (post.post * post.grid.dA).ravel()

    # Filter zero-weight points for cleaner KDE
    mask = weights > 0
    samples = samples[mask]
    weights = weights[mask]

    names = ["Omega_c", "sigma8"]
    labels = [r"\Omega_c", r"\sigma_8"]
    samps = mcsamples.MCSamples(
        samples=samples, weights=weights, names=names, labels=labels, label=label
    )

    if ranges is not None:
        samps.setRanges(ranges)
    return samps


def optimal_alpha_S8(
    samples: mcsamples.MCSamples,
    oc_ref: float = 0.264,
    *,
    return_scatter: bool = True
) -> Dict[str, float]:
    """
    Compute the optimal α for Σ8 = σ8 * (Ω_c / oc_ref)^α minimizing Var[ln Σ8].

    Analytic result in log-space:
    Let x = ln(Ω_c/oc_ref), y = ln(σ8). Then ln Σ8 = y + α x.
    The α minimizing Var[y + α x] is α* = -Cov(x,y)/Var(x).

    Returns
    -------
    dict
        {'alpha': α*, 'sigma_S8': marginalized 68% error on Σ8 (if return_scatter), 'mean_S8': ...}
    """
    p = samples.getParams()
    oc = np.asarray(p.Omega_c)
    s8 = np.asarray(p.sigma8)
    w = np.asarray(samples.weights)

    x = np.log(oc / oc_ref)
    y = np.log(s8)

    # Weighted covariance in log-space
    W = w / np.sum(w)
    xm, ym = np.sum(W * x), np.sum(W * y)
    varx = np.sum(W * (x - xm) ** 2)
    covxy = np.sum(W * (x - xm) * (y - ym))

    if varx <= 0:
        raise RuntimeError("Var(ln Omega_c) is non-positive; cannot define α.")

    alpha = -covxy / varx

    out = {"alpha": float(alpha)}

    if return_scatter:
        lnS8 = y + alpha * x
        S8 = np.exp(lnS8)
        mean_S8 = np.sum(W * S8)
        # 68% equal-tails from weighted CDF
        ix = np.argsort(S8)
        cdf = np.cumsum(W[ix])
        cdf /= cdf[-1]
        lo = np.interp(0.16, cdf, S8[ix])
        hi = np.interp(0.84, cdf, S8[ix])
        out["mean_S8"] = float(mean_S8)
        out["sigma_S8"] = 0.5 * float(hi - lo)
        out["low68_S8"] = float(lo)
        out["high68_S8"] = float(hi)
    return out


# ---------- Fisher matrix & ellipse ------------------------------------------


def fisher_matrix(
    theta0: Tuple[float, float],
    model_fn: Callable[[float, float], ArrayLike],
    cov: ArrayLike,
    *,
    step_frac: Tuple[float, float] = (0.01, 0.01),
    include_cov_deriv: bool = False,
    cov_derivs: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    Compute the (2x2) Fisher matrix for parameters (Omega_c, sigma8).

    Assumes Gaussian data with mean μ(θ) and covariance C(θ):
        F_ij = (∂μ/∂θ_i)^T C^{-1} (∂μ/∂θ_j)  +  0.5 Tr[C^{-1} ∂C/∂θ_i C^{-1} ∂C/∂θ_j]

    If include_cov_deriv=False, the second term is dropped (common approximation).

    Parameters
    ----------
    theta0 : (float, float)
        Fiducial (Omega_c, sigma8).
    model_fn : function
        model_fn(oc, s8) -> mean summary vector μ(θ).
    cov : array-like
        Fiducial covariance C(θ0).
    step_frac : (float, float)
        Relative steps for finite differences (e.g. 1%).
    include_cov_deriv : bool
        Include the ∂C/∂θ term if True. Requires cov_derivs or user to supply a function.
    cov_derivs : (dC/dOmega_c, dC/dsigma8) or None
        Optional derivatives of covariance. If None and include_cov_deriv=True, the second
        term is not added (user must provide derivatives if wanted).

    Returns
    -------
    F : np.ndarray
        2x2 Fisher matrix
    """
    oc0, s80 = map(float, theta0)
    mu0 = np.asarray(model_fn(oc0, s80), dtype=float)

    # Finite difference steps
    h_oc = step_frac[0] * max(abs(oc0), 1e-8)
    h_s8 = step_frac[1] * max(abs(s80), 1e-8)

    mu_oc_plus = np.asarray(model_fn(oc0 + h_oc, s80), dtype=float)
    mu_oc_minus = np.asarray(model_fn(oc0 - h_oc, s80), dtype=float)
    dmu_doc = (mu_oc_plus - mu_oc_minus) / (2 * h_oc)

    mu_s8_plus = np.asarray(model_fn(oc0, s80 + h_s8), dtype=float)
    mu_s8_minus = np.asarray(model_fn(oc0, s80 - h_s8), dtype=float)
    dmu_ds8 = (mu_s8_plus - mu_s8_minus) / (2 * h_s8)

    invC = linalg.inv(np.asarray(cov, dtype=float))

    # Mean-derivative term
    F = np.zeros((2, 2), dtype=float)
    J = np.vstack([dmu_doc, dmu_ds8])  # shape (2, p)
    # F_ij = dmu_i^T C^{-1} dmu_j
    F = J @ invC @ J.T

    if include_cov_deriv and cov_derivs is not None:
        dC_doc, dC_ds8 = cov_derivs
        term = np.zeros_like(F)
        term[0, 0] = 0.5 * np.trace(invC @ dC_doc @ invC @ dC_doc)
        term[1, 1] = 0.5 * np.trace(invC @ dC_ds8 @ invC @ dC_ds8)
        term[0, 1] = term[1, 0] = 0.5 * np.trace(invC @ dC_doc @ invC @ dC_ds8)
        F += term

    return F


def fisher_covariance(F: np.ndarray) -> np.ndarray:
    """Return the covariance matrix Σ = F^{-1}, with basic stability checks."""
    return linalg.inv(F)


def fisher_area_ellipse_68(cov2x2: np.ndarray) -> float:
    """
    Area enclosed by the 68% ellipse for a 2D Gaussian:
        A68 = π * Δχ²_68 * sqrt(det(Σ))
    where Δχ²_68 ≈ 2.30 for 2 dof.
    """
    delta_chi2_68 = 2.2957  # more precise than 2.30
    det = np.linalg.det(cov2x2)
    if det <= 0:
        raise RuntimeError("Non-positive determinant for covariance.")
    return np.pi * delta_chi2_68 * np.sqrt(det)


# ---------- Combinations, slices, scenarios, robustness, f_sky ---------------


def combine_posteriors(
    post_a: PosteriorGrid, post_b: PosteriorGrid
) -> PosteriorGrid:
    """
    Combine two posteriors defined on the *same grid* by multiplication (assuming independence).
    """
    if not np.allclose(post_a.grid.oc, post_b.grid.oc) or not np.allclose(post_a.grid.s8, post_b.grid.s8):
        raise ValueError("Grids for post_a and post_b must match to combine.")
    lnP = np.log(post_a.post + 1e-300) + np.log(post_b.post + 1e-300)
    return normalize_logposterior(post_a.grid, lnP)


def slice_indices(total_bins: int, start: int, stop: int) -> np.ndarray:
    """Utility: boolean mask selecting bins [start:stop] out of total_bins."""
    mask = np.zeros(total_bins, dtype=bool)
    mask[start:stop] = True
    return mask


def sensitivity_by_slices(
    data_vec: np.ndarray,
    model_fn: Callable[[float, float], np.ndarray],
    cov: np.ndarray,
    grid: ParamGrid,
    slices: List[np.ndarray],
    *,
    include_norm: bool = True,
    inv_cov: Optional[np.ndarray] = None,
    hartlap_nsim: Optional[int] = None
) -> List[Tuple[PosteriorGrid, Dict[str, float]]]:
    """
    Recompute the likelihood posterior for a set of index masks (slices) of the summary vector.

    Parameters
    ----------
    slices : list of boolean masks or index arrays
        Each defines which entries of the summary vector to keep.

    Returns
    -------
    list of (PosteriorGrid, stats)
        stats includes {'A68', 'A95'} for each slice
    """
    outputs = []
    for sel in slices:
        sel = np.asarray(sel)
        d_sel = data_vec[sel]
        C_sel = cov[np.ix_(sel, sel)]

        def model_sel(oc, s8):
            return np.asarray(model_fn(oc, s8))[sel]

        lnL = evaluate_loglike_grid(
            grid,
            data_vec=d_sel,
            cov=C_sel,
            model_fn=model_sel,
            include_norm=include_norm,
            inv_cov=inv_cov,
            hartlap_nsim=hartlap_nsim
        )
        lnpost = add_prior_to_logpost(grid, lnL, prior_kind="uniform")  # flat on grid
        post = normalize_logposterior(grid, lnpost)
        stats = hpd_levels_and_areas(post)
        outputs.append((post, stats))
    return outputs


def scenario_compare(
    scenarios: Dict[str, Dict[str, Union[np.ndarray, int]]],
    data_vec: np.ndarray,
    model_fn: Callable[[float, float], np.ndarray],
    grid: ParamGrid
) -> Dict[str, Dict[str, float]]:
    """
    Compare A68 (and A95) across different instrumental scenarios (different covariances, Nsims, etc.)

    Parameters
    ----------
    scenarios : dict
        Mapping: name -> {'cov': C, 'hartlap_nsim': N, ...}
    Returns
    -------
    dict
        name -> {'A68': ..., 'A95': ...}
    """
    out = {}
    for name, s in scenarios.items():
        C = np.asarray(s["cov"])
        Ns = s.get("hartlap_nsim", None)
        lnL = evaluate_loglike_grid(grid, data_vec=data_vec, cov=C, model_fn=model_fn, hartlap_nsim=Ns)
        lnpost = add_prior_to_logpost(grid, lnL, prior_kind="uniform")
        post = normalize_logposterior(grid, lnpost)
        stats = hpd_levels_and_areas(post)
        out[name] = {"A68": stats["A68"], "A95": stats.get("A95", np.nan)}
    return out


def convergence_curve_A68(
    cov_estimates: List[np.ndarray],
    data_vec: np.ndarray,
    model_fn: Callable[[float, float], np.ndarray],
    grid: ParamGrid,
    *,
    hartlap_nsims: Optional[List[int]] = None
) -> List[float]:
    """
    Compute A68 as a function of the number of realizations used to build the covariance.

    Parameters
    ----------
    cov_estimates : list of covariance estimates
        Typically built from increasing Nsims (e.g. 200, 500, 1000, ...)
    hartlap_nsims : optional list of Nsims corresponding to each covariance
        Used to apply Hartlap correction (if provided).

    Returns
    -------
    list of A68 values
    """
    A68s = []
    for i, C in enumerate(cov_estimates):
        Ns = None if hartlap_nsims is None else hartlap_nsims[i]
        lnL = evaluate_loglike_grid(grid, data_vec=data_vec, cov=C, model_fn=model_fn, hartlap_nsim=Ns)
        post = normalize_logposterior(grid, add_prior_to_logpost(grid, lnL, prior_kind="uniform"))
        A68s.append(post.A68())
    return A68s


def rescale_error_fsky(
    sigma: float,
    fsky_ref: float,
    fsky_new: float,
    *,
    exponent: float = 0.5
) -> float:
    """
    Rescale an error bar with sky fraction: sigma ∝ f_sky^{-exponent}.

    By default exponent=0.5 (common scaling ~ 1/sqrt(f_sky)); set exponent=1.0 if you want 1/f_sky.
    """
    if fsky_new <= 0 or fsky_ref <= 0:
        raise ValueError("fsky must be positive")
    return sigma * (fsky_ref / fsky_new) ** exponent


# ---------- Plotting helpers with GetDist ------------------------------------


def plot_contours_getdist(
    posts: List[PosteriorGrid],
    labels: Optional[List[str]] = None,
    *,
    filled: bool = True,
    legend_loc: str = "upper right",
    width_inch: float = 4.0
):
    """
    Plot 2D contours with GetDist from one or several PosteriorGrid(s).
    """
    g = plots.get_single_plotter(width_inch=width_inch, ratio=1.0)
    roots = [grid_to_mcsamples(P, label=lab or f"post{i+1}") for i, (P, lab) in enumerate(zip(posts, labels or []))]
    if labels is None:
        labels = [s.getLabel() for s in roots]
    g.plot_2d(roots, "Omega_c", "sigma8", filled=filled)
    g.add_legend(labels, legend_loc=legend_loc)
    plt.show()
    return g


# ---------- Example usage skeleton -------------------------------------------

if __name__ == "__main__":
    # Minimal runnable skeleton (replace model_fn and data_vec with your own)
    # 1) Define fiducial parameters and grid
    fid = {"Omega_c": 0.264, "sigma8": 0.8}
    grid = make_param_grid(fid, npts=(121, 121), frac=0.20)

    # 2) Define a toy model function for the summary vector (replace with your power spectrum or scattering)
    def model_fn(oc, s8):
        # Example: 5-bin toy summary = [oc, s8, oc*s8, oc^2, s8^2] just for demonstration
        return np.array([oc, s8, oc * s8, oc ** 2, s8 ** 2], dtype=float)

    # 3) Build synthetic "observed" data around fiducial model
    rng = np.random.default_rng(42)
    data_vec = model_fn(fid["Omega_c"], fid["sigma8"])
    p = data_vec.size

    # Fiducial covariance (toy)
    A = rng.standard_normal((p, p))
    cov = A @ A.T * 1e-4

    # 4) Evaluate ln L on the grid and add a flat prior
    # 4) Évalue ln L sur la grille brute
    lnL = evaluate_loglike_grid(
    grid, data_vec=data_vec, cov=cov, model_fn=model_fn, hartlap_nsim=3000)

# 4b) (Optionnel) Interpolation Clough–Tocher + remaillage fin
    use_ct = True  # mets False pour désactiver
    if use_ct:
        f_ct = interpolate_loglike_ct(grid, lnL, fill_value=-np.inf, rescale=True)
        oc_fine = np.linspace(grid.oc.min(), grid.oc.max(), 241)
        s8_fine = np.linspace(grid.s8.min(), grid.s8.max(), 241)
        lnL = resample_loglike_on_grid(f_ct, oc_fine, s8_fine)
        grid = ParamGrid(oc=oc_fine, s8=s8_fine)  # IMPORTANT: mettre à jour le grid

    # 4c) Prior plat et normalisation sur la (nouvelle) grille
    lnpost = add_prior_to_logpost(grid, lnL, prior_kind="uniform")
    post = normalize_logposterior(grid, lnpost)


    # 5) Areas and comparison with a "power spectrum" reference (here re-using same post for demo)
    stats = hpd_levels_and_areas(post)
    A68 = stats["A68"]
    # Suppose A68 for power spectrum alone was A68_ps (use your PS posterior):
    A68_ps = A68 * 1.5  # dummy
    ratio = compare_stat(A68_ps, A68)
    print(f"A68={A68:.4e}, improvement vs PS={ratio:.2f}x")

    # 6) Marginal means & 68% intervals
    marg_oc = marginalized_mean_and_ci_68(post, "Omega_c")
    marg_s8 = marginalized_mean_and_ci_68(post, "sigma8")
    print("Omega_c:", marg_oc)
    print("sigma8:", marg_s8)

    # 7) Σ8 optimization and error
    samps = grid_to_mcsamples(post, ranges={"Omega_c": (None, None), "sigma8": (None, None)})
    s8_opt = optimal_alpha_S8(samps, oc_ref=0.264)
    print("Sigma8 combo:", s8_opt)

    # 8) Fisher forecast (on the toy model)
    F = fisher_matrix((fid["Omega_c"], fid["sigma8"]), model_fn, cov, step_frac=(0.01, 0.01))
    cov_par = fisher_covariance(F)
    Af68 = fisher_area_ellipse_68(cov_par)
    print(f"Fisher A68 (ellipse) ~ {Af68:.4e}")

    # 9) (Optional) Plot with GetDist
    # g = plot_contours_getdist([post], labels=["Toy posterior"], filled=True)
