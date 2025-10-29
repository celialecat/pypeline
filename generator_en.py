import os
import numpy as np
import pandas as pd
import math
import logging
import time
from typing import Optional, Tuple, Dict, Any, Literal
import scattering  # already imported in your code
setattr(scattering, "np", np)  # hotfix: provide 'np' to the module


# Try to import functions from the repo; if they exist, we'll use them
try:
    import ST  # repo utility (get_random_data, power spectrum helpers)
except Exception:
    ST = None

try:
    import scattering  # scattering package from the repo (synthesis, ...)
except Exception:
    scattering = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------
# Timing helpers (decorator/context)
# ---------------------------------

def timed(func):
    """Decorator that logs the execution time of the wrapped function."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            logger.info(f"{func.__name__} executed in {t1 - t0:.3f}s")
    return wrapper


# ------------------------------
# CSV -> type & parsing helpers
# ------------------------------


@timed
def read_params_csv(csv_path: str) -> pd.DataFrame:
    """
    Read a table of cosmological parameters from either:
      - a CSV (tries common encodings), or
      - a NumPy file (.npy/.npz) containing a structured array or a dict.
    """
    import os
    ext = os.path.splitext(csv_path)[1].lower()

    if ext in (".npy", ".npz"):
        arr = np.load(csv_path, allow_pickle=True)
        # .npz -> mapping key->array
        if hasattr(arr, "files"):
            # Heuristic: if there's a single key holding a record array
            if len(arr.files) == 1 and arr[arr.files[0]].dtype.names:
                rec = arr[arr.files[0]]
                return pd.DataFrame.from_records(rec)
            # Otherwise, stack as columns
            data = {k: arr[k] for k in arr.files}
            return pd.DataFrame(data)
        else:
            # .npy
            if arr.dtype.names:  # structured array
                return pd.DataFrame.from_records(arr)
            elif isinstance(arr, np.ndarray):
                # If it's a 2D array, fabricate generic column names
                if arr.ndim == 2:
                    cols = [f"col{i}" for i in range(arr.shape[1])]
                    return pd.DataFrame(arr, columns=cols)
                # If it's an array of objects/dicts
                if arr.dtype == object:
                    try:
                        return pd.DataFrame(list(arr))
                    except Exception:
                        pass
            raise ValueError("Unrecognized .npy/.npz format to build a DataFrame.")
    else:
        # CSV: try several common encodings
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(csv_path, encoding=enc, engine="python")
            except Exception as e:
                last_err = e
        raise last_err


@timed
def _read_csv_detect_type(csv_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Read the CSV ignoring commented lines (#) and detect whether it is
    a 'wst' file (scattering coefficients) or a 'spectrum' file (ell, D_ell, ...).

    Returns: (type_str, dataframe) where type_str ∈ {'wst','spectrum'}.
    """
    # Let pandas detect the separator; keep header if present.
    df = pd.read_csv(csv_path, comment='#', engine='python')

    # Normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    # 'spectrum' heuristic: presence of 'ell' column OR any column containing 'D_ell'/'Cl'/'power'
    low = [c.lower() for c in cols]
    if any(c == 'ell' for c in low):
        return 'spectrum', df
    if any(('d_ell' in c or 'dell' in c or 'power' in c or c == 'cl') for c in low):
        return 'spectrum', df

    # 'wst' heuristic: presence of a column (or values) like 'S0','S1','S2', ...
    # Typical provided case: columns [sample, kind, index, value]
    if 'kind' in low:
        if df['kind' if 'kind' in cols else cols[low.index('kind')]].astype(str).str.match(r'^\s*S\d+').any():
            return 'wst', df
    # Otherwise, search any column where values look like 'S\d'
    for c in cols:
        if df[c].astype(str).str.match(r'^\s*S\d+').any():
            return 'wst', df

    # Fallback: if the first column is numeric and there are at least two columns -> spectrum
    if df.shape[1] >= 2:
        try:
            _ = pd.to_numeric(df.iloc[:, 0])
            return 'spectrum', df
        except Exception:
            pass

    raise ValueError("Could not detect CSV format: check headers or content.")


@timed
def _parse_spectrum_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the (ells, power) pair from a 'spectrum' dataframe.
    We look for 'ell' (or the 1st column) and 'D_ell_mean' / 'power' / 'Cl' (or the 2nd column).
    """
    cols = list(df.columns)
    low = [c.lower() for c in cols]

    # ell column
    ell_col = None
    if 'ell' in low:
        ell_col = cols[low.index('ell')]
    else:
        ell_col = cols[0]

    # power column
    power_candidates = []
    for c in cols[1:]:
        lc = c.lower()
        if ('d_ell' in lc) or ('dell' in lc) or ('power' in lc) or (lc == 'cl'):
            power_candidates.append(c)
    power_col = power_candidates[0] if power_candidates else (cols[1] if len(cols) >= 2 else None)
    if power_col is None:
        raise ValueError("Could not find a power column in the CSV.")

    ells = pd.to_numeric(df[ell_col], errors='coerce').to_numpy(dtype=float)
    power = pd.to_numeric(df[power_col], errors='coerce').to_numpy(dtype=float)

    if np.any(~np.isfinite(ells)) or np.any(~np.isfinite(power)):
        raise ValueError("ell/power columns contain non-finite values.")

    return ells, power


@timed
def _parse_wst_df(
    df: pd.DataFrame,
    sample: Optional[int] = 0,
    aggregate: Literal['none','mean'] = 'none',
    kinds: Optional[Tuple[str, ...]] = ("S0","S1"),
) -> np.ndarray:
    """
    Transform a WST CSV of shape [sample, kind, index, value] into a 1D vector.
    - sample: which sample index to select (default 0). If None and aggregate='mean', average over all samples.
    - aggregate: 'none' (select a single sample) or 'mean' (average on the sample dimension).
    - kinds: tuple of 'kinds' to keep and their order (default ('S0','S1')).

    Returns a 1D numpy vector ordered as [S0(idx=0), S1(idx=0..), ...] for a sample or the average.
    """
    # Expected columns (with tolerance on names)
    cols = list(df.columns)
    low = [c.lower() for c in cols]

    # locate columns
    def _find(colname, default=None):
        return cols[low.index(colname)] if colname in low else default

    col_sample = _find('sample', None)
    col_kind   = _find('kind', None)
    col_index  = _find('index', None)

    # value column: last numeric column by default
    value_col = None
    for c in reversed(cols):
        if pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break
    value_col = value_col or cols[-1]

    # Filter desired kinds
    if col_kind is None:
        raise ValueError("WST CSV: 'kind' column is missing.")

    work = df.copy()
    work[col_kind] = work[col_kind].astype(str).str.strip()

    if kinds is not None:
        work = work[work[col_kind].isin(kinds)]
        if work.empty:
            raise ValueError(f"WST CSV: none of the kinds {kinds} were found.")

    # Sample handling / averaging
    if aggregate == 'mean':
        # average by (kind, index)
        if col_index is None:
            # S0 only (missing index) → add index 0
            work['_idx'] = 0
            g = work.groupby([col_kind, '_idx'])[value_col].mean().reset_index()
            g = g.rename(columns={'_idx':'index'})
        else:
            g = work.groupby([col_kind, col_index])[value_col].mean().reset_index()
            g = g.rename(columns={col_index:'index'})
    else:
        # select a specific sample (default: 0)
        if col_sample is None:
            # no sample column → treat the whole table as a single sample
            sub = work
        else:
            s = 0 if sample is None else int(sample)
            sub = work[work[col_sample] == s]
            if sub.empty:
                raise ValueError(f"WST CSV: sample={s} not found.")
        if col_index is None:
            sub = sub.assign(index=0)
        else:
            sub = sub.rename(columns={col_index:'index'})
        g = sub[[col_kind, 'index', value_col]].copy()

    # Order by requested kinds then by increasing index
    order_kind = list(kinds) if kinds is not None else sorted(g[col_kind].unique())
    g['_korder'] = g[col_kind].apply(lambda x: order_kind.index(x) if x in order_kind else len(order_kind))
    g = g.sort_values(['_korder','index']).reset_index(drop=True)

    vec = pd.to_numeric(g[value_col], errors='coerce').to_numpy(dtype=np.float32)
    if np.any(~np.isfinite(vec)):
        raise ValueError("WST CSV: non-finite values found in the coefficient column.")

    return vec


# --------------------------------------------
# Spectrum -> Fourier amplitude & synthesis
# --------------------------------------------

@timed
def _build_fourier_amplitude_from_spectrum(
    ells: np.ndarray, power: np.ndarray, M: int, N: int, pixel_scale_arcmin: float
) -> np.ndarray:
    """
    Build an (M,N) Fourier-space amplitude grid from a vector (ell, power).
    - power: assumed ≥ 0; if it's D_ell, convert upstream if needed.
    - pixel_scale_arcmin : pixel size in arcmin → radians.
    amplitude = sqrt( interpolated_power(ell_grid) ).
    """
    pix_rad = pixel_scale_arcmin / 60.0 * (math.pi / 180.0)  # rad/pixel

    kx = np.fft.fftfreq(M, d=pix_rad)  # cycles / radian
    ky = np.fft.fftfreq(N, d=pix_rad)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell_grid = 2.0 * math.pi * np.sqrt(KX ** 2 + KY ** 2)

    # 1D linear interpolation on ell, clamping at boundaries
    sort_idx = np.argsort(ells)
    ells_s = np.maximum(ells[sort_idx], 0.0)
    power_s = np.maximum(power[sort_idx], 0.0)

    interp = np.interp(ell_grid.ravel(), ells_s, power_s, left=power_s[0], right=power_s[-1])
    interp = interp.reshape(ell_grid.shape)

    amplitude = np.sqrt(interp, dtype=np.float64)
    return amplitude


@timed
def _hermitian_symmetric_complex_field(amplitude: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate a Hermitian complex field F (M,N): F[-i,-j] = conj(F[i,j]) from a given amplitude (M,N).
    """
    M, N = amplitude.shape
    F = np.zeros((M, N), dtype=np.complex128)

    def sym(i, j):
        return (-i) % M, (-j) % N

    for i in range(M):
        for j in range(N):
            si, sj = sym(i, j)
            if (si < i) or (si == i and sj < j):
                continue
            amp = float(amplitude[i, j])
            if amp < 0:
                amp = 0.0
            if (si == i) and (sj == j):
                # self-conjugate points (DC/Nyquist) -> non-negative real
                F[i, j] = amp
            else:
                phi = rng.uniform(0.0, 2.0 * math.pi)
                val = amp * np.exp(1j * phi)
                F[i, j] = val
                F[si, sj] = np.conj(val)
    return F


# ------------------------------
# Public API
# ------------------------------

@timed
def generate_patch_from_csv(
    method: str,
    patch_size: Tuple[int, int],
    pixel_scale_arcmin: float,
    csv_path: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = 'cpu',
    synthesis_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Generate a 2D patch from a CSV containing either a spectrum (ell, D_ell_mean, ...)
    or WST coefficients (moments S0/S1, etc.).

    Args:
      - method: 'wst'/'Dell' for scattering, or 'spectrum' for a Gaussian field from a spectrum.
      - patch_size: (M, N) output patch size.
      - pixel_scale_arcmin: pixel size in arcminutes.
      - csv_path: path to the CSV.
      - output_path: if provided, saves the patch as .npy at this path.
      - seed: RNG seed.
      - device: 'cpu' or 'gpu'.
      - synthesis_kwargs: kwargs forwarded to scattering.synthesis (J, L, steps, learning_rate, estimator_name, ...).

    Returns: float32 numpy patch of shape (M, N).
    """
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 2 ** 31 - 1))
    method_norm = (method or '').strip().lower()
    if method_norm == 'dell':
        method_norm = 'wst'

    kind, df = _read_csv_detect_type(csv_path)
    M, N = patch_size
    synthesis_kwargs = dict(synthesis_kwargs or {})

    # ---------------- Spectrum path ----------------
    if method_norm in ('spectrum', 'power', 'wst_spectrum') or (kind == 'spectrum' and method_norm != 'wst'):
        logger.info("Generation via power spectrum (mode 'spectrum').")
        ells, power = _parse_spectrum_df(df)
        amplitude = _build_fourier_amplitude_from_spectrum(ells, power, M, N, pixel_scale_arcmin)
        F = _hermitian_symmetric_complex_field(amplitude, rng)
        field = np.fft.ifft2(F).real.astype(np.float32)
        # Optional normalization: zero-mean and unit-variance
        field -= field.mean()
        std = float(field.std())
        if std > 0:
            field /= std
        if output_path:
            np.save(output_path, field)
        return field

    # ---------------- WST path ----------------
    if method_norm == 'wst' or (kind == 'wst' and method_norm == 'wst'):
        if scattering is None:
            raise ImportError("Module 'scattering' not found: add the repo package to PYTHONPATH or install it (pip install -e).")
        logger.info("Generation via Wavelet Scattering Transform (synthesis).")

        # --- Select a sample or average over all (avoid the 80 vs 8 trap) ---
        sample_id = synthesis_kwargs.pop('sample', 0)      # int or None
        aggregate = synthesis_kwargs.pop('aggregate', 'none')  # 'none' | 'mean'
        kinds = synthesis_kwargs.pop('kinds', ("S0","S1"))   # keep S0/S1 by default

        coef_vec = _parse_wst_df(df, sample=sample_id, aggregate=aggregate, kinds=kinds)

        # --- Adjustment according to the estimator type ---
        J = synthesis_kwargs.get("J", 7)
        L = synthesis_kwargs.get("L", 4)
        estimator = synthesis_kwargs.get("estimator_name", "wst")

        # If isotropic mode (wst_iso), expect a vector of length J
        if estimator == "wst_iso":
            if coef_vec.size != J:
                raise ValueError(f"wst_iso expects J={J} S0 values; got {coef_vec.size}.")
            coef_vec = np.concatenate([[0.0], coef_vec])  # → size 1+J

        # If full WST (non-isotropic), expect J*L values
        elif estimator == "wst":
            expected = J * L
            if coef_vec.size != expected:
                raise ValueError(f"wst expects J*L={expected} S1(j,l) values; got {coef_vec.size}.")
            coef_vec = np.concatenate([[0.0], coef_vec])  # → size 1+J*L

        # --- Prepare target & synthesis arguments ---
        import torch
        target_t = torch.as_tensor(coef_vec, dtype=torch.float32).view(1, -1)  # (1, K)

        # Default estimator name for S0/S1 (adapt to the lib if needed)
        estimator_name = synthesis_kwargs.pop('estimator_name', 'moments')

        synth_args = dict(
            estimator_name=estimator_name,
            target=target_t,
            mode='estimator',     # we directly provide the target vector
            M=M, N=N,
            device=device,
        )
        # Merge user kwargs (J, L, steps, learning_rate, etc.)
        synth_args.update(synthesis_kwargs)

        logger.info(f"Calling scattering.synthesis with args: {list(synth_args.keys())}")

        # --- Synthesis call ---
        image_syn = scattering.synthesis(**synth_args)

        # --- Convert to numpy ---
        if isinstance(image_syn, (list, tuple)):
            image_syn = image_syn[0]
        if hasattr(image_syn, 'detach'):
            image_syn = image_syn.detach()
        if hasattr(image_syn, 'cpu'):
            image_syn = image_syn.cpu().numpy()

        # Expected shapes: (1,M,N) or (M,N)
        if image_syn.ndim == 3 and image_syn.shape[0] == 1:
            patch = image_syn[0].astype(np.float32)
        elif image_syn.ndim == 2:
            patch = image_syn.astype(np.float32)
        else:
            patch = np.squeeze(image_syn).astype(np.float32)

        # Simple normalization
        patch -= patch.mean()
        std = float(patch.std())
        if std > 0:
            patch /= std

        if output_path:
            np.save(output_path, patch)
        return patch

    # -------------------------------------------------
    raise ValueError(f"method '{method}' not supported or CSV type='{kind}' not compatible.")


# Small helper example for notebooks
@timed
def example_notebook_usage():
    """
    Quick example (adapt as needed) for a notebook:
    >>> from generators import generate_patch_from_csv
    >>> patch = generate_patch_from_csv(
    ...   'wst', patch_size=(256,256), pixel_scale_arcmin=0.5,
    ...   csv_path='coefs_wst.csv', seed=0, device='cpu',
    ...   synthesis_kwargs={'J':6,'L':4,'steps':400,'learning_rate':0.4,
    ...                     'sample':0, 'aggregate':'none', 'kinds':('S0','S1'),
    ...                     'estimator_name':'moments'}
    ... )
    >>> import matplotlib.pyplot as plt; plt.imshow(patch); plt.colorbar()
    """
    pass
