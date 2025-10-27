# wst_utils_sihao_grouped.py
from __future__ import annotations
import os
import glob
import time
import numpy as np
from typing import Dict, Optional, Union, Tuple

from astropy.io import fits


# -----------------------------
# I/O utilities for planar FITS maps
# -----------------------------
def read_fits_image(path: Union[str, os.PathLike], hdu: int = 0) -> np.ndarray:
    """
    Read a planar FITS image and return a float32 ndarray with shape [H,W] or [N,H,W].
    - If data.ndim == 3 and it's a pseudo single-channel stack, take the first plane.
    """
    data = fits.getdata(str(path), ext=hdu)
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2:
        return data  # [H,W]

    if data.ndim == 3:
        # Common cases: [C,H,W] or [N,H,W]. If there's an extra channel, take the first one.
        # If it's a real batch [N,H,W], keep as is.
        if data.shape[0] == 1:
            return data[0].astype(np.float32, copy=False)  # [H,W]
        return data  # [N,H,W]

    raise ValueError(
        f"FITS data has unsupported shape {data.shape} (need [H,W] or [N,H,W])."
    )


def load_images_from_path(
    path: Union[str, os.PathLike],
    patterns: Tuple[str, ...] = ("*.fits", "*.fit", "*.fits.gz", "*.FITS"),
) -> np.ndarray:
    """
    Accepts: ndarray / path to a FITS file / directory path.
    Returns a float32 ndarray as [N,H,W] (batch) or [H,W] if a single file.
    """
    # Already an array -> return as is
    if isinstance(path, np.ndarray):
        return path

    path = str(path)
    if os.path.isfile(path):
        arr = read_fits_image(path)
        return arr

    if os.path.isdir(path):
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(path, pat)))
        files = sorted(files)
        if len(files) == 0:
            raise FileNotFoundError(f"No FITS file found in directory: {path}")

        # Load + size checking
        imgs = []
        ref_hw = None
        for f in files:
            arr = read_fits_image(f)
            # Normalize to [H,W]
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    arr = arr[0]
                else:
                    # If we encounter a real batch stored in one file, expand it into the list
                    imgs.extend([arr_i.astype(np.float32, copy=False) for arr_i in arr])
                    continue

            if arr.ndim != 2:
                raise ValueError(f"File {f} has an unsupported shape: {arr.shape}")

            if ref_hw is None:
                ref_hw = arr.shape
            elif arr.shape != ref_hw:
                raise ValueError(
                    "All images in a directory must have the same size. "
                    f"Found {arr.shape} differing from {ref_hw} (file {f})."
                )

            imgs.append(arr.astype(np.float32, copy=False))

        if len(imgs) == 1:
            return imgs[0]  # [H,W]
        return np.stack(imgs, axis=0)  # [N,H,W]

    raise FileNotFoundError(f"Path not found: {path}")


# -----------------------------
# Pre-processing / conversions
# -----------------------------
def _as_batch(images: np.ndarray) -> np.ndarray:
    """Ensure a [N,H,W] float32 array with no NaN/Inf."""
    x = np.asarray(images)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"images must be [N,H,W] or [H,W], got shape {x.shape}")
    x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("images contain NaN or Inf.")
    return x


def _to_numpy(a):
    """Torch tensor -> numpy (CPU); return object unchanged if already numpy/list/etc."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return a


def _pick_from_keys(d: dict, keys: Tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


# -----------------------------
# Helpers to derive file paths in grouped mode
# -----------------------------
def _derive_plot_path(base, cosmo):
    if base is None:
        return None
    base = os.fspath(base)
    root, ext = os.path.splitext(base)
    img_exts = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}
    if ext.lower() in img_exts:
        return f"{root}_{cosmo}{ext}"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{cosmo}.png")


def _derive_csv_path(base, cosmo):
    if base is None:
        return None
    base = os.fspath(base)
    root, ext = os.path.splitext(base)
    if ext.lower() == '.csv':
        return f"{root}_{cosmo}.csv"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{cosmo}.csv")


def _derive_samples_csv_path(base, cosmo):
    """Derive a path for exporting (S0/S1/S2) samples in grouped mode."""
    if base is None:
        return None
    base = os.fspath(base)
    root, ext = os.path.splitext(base)
    if ext.lower() == '.csv':
        # Suffix with cosmology name without forcing "S1" in the file name
        return f"{root}_{cosmo}.csv"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{cosmo}_samples.csv")


# -----------------------------
# WST computation (Sihao Cheng)
# -----------------------------
def compute_wst_S012(
    images_or_path: Union[np.ndarray, str, os.PathLike],
    J: int = 7,
    L: int = 4,
    device: str = "auto",   # "cpu" | "gpu" | "auto"
    whiten: bool = False,
    return_means_for_dir: bool = True,
    # --- New ---
    plot: bool = False,
    which: str = "S1",                 # "S1" or "S2" for plotting
    save_plot: Optional[Union[str, os.PathLike]] = None,
    save_csv: Optional[Union[str, os.PathLike]] = None,
    strict_iso: bool = True,
    quiet: bool = False,
    # === NEW ===
    save_samples_csv: Optional[Union[str, os.PathLike]] = None,  # file or directory
    samples_format: str = "wide",  # "wide" (recommended) or "long"
    samples_which: Tuple[str, ...] = ("S1", "S2", "S0"),
    save_indices_csv: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Compute S0, S1, S2 (WST) for a batch [N,H,W] using 'scattering_transform' (Sihao Cheng).
    Accepts an ndarray, a .fits file path, a directory of .fits, **or**
    a parent directory **containing subdirectories** (grouped mode).

    - S0 = spatial mean -> [N,1]
    - S1 = s_mean[...] (isotropic if available, fallback otherwise) -> [N,K1]
    - S2 = s_cov [...] (isotropic if available, fallback otherwise) -> [N,K2]

    If `images_or_path` is a directory and `return_means_for_dir=True`:
      - adds "S0_mean" (shape [1,1]) and "S1_mean" (shape [1,K1]) + their std:
        "S1_std", "S2_mean", "S2_std".

    If `images_or_path` is a parent directory with subdirectories (grouped mode):
      - process each subdirectory that contains FITS files as a cosmology
      - returns a dict { 'mode': 'group', 'per_cosmo': {cosmo: results, ...} }
      - produces a plot per subdirectory if `plot=True` (with mean ± std)
        and saves it if `save_plot` is provided (file or output directory).

    Execution time:
      - In simple mode, the returned dict includes 'elapsed_sec'.
      - In grouped mode, the group-level dict includes 'elapsed_sec_total'.
    """
    _t0_total = time.perf_counter()

    # ---------- Grouped mode detection ----------
    if isinstance(images_or_path, (str, os.PathLike)) and os.path.isdir(str(images_or_path)):
        path = str(images_or_path)
        # Are there FITS directly in this directory?
        direct_fits = []
        for pat in ("*.fits", "*.fit", "*.fits.gz", "*.FITS"):
            direct_fits.extend(glob.glob(os.path.join(path, pat)))
        if len(direct_fits) == 0:
            # Look for subdirectories with FITS files
            subdirs = sorted([d.path for d in os.scandir(path) if d.is_dir()])
            cosmo_dirs = []
            for d in subdirs:
                ff = []
                for pat in ("*.fits", "*.fit", "*.fits.gz", "*.FITS"):
                    ff.extend(glob.glob(os.path.join(d, pat)))
                if ff:
                    cosmo_dirs.append(d)
            if cosmo_dirs:
                if not quiet:
                    print(f"Grouped mode detected: {len(cosmo_dirs)} subdirectories found.")

                per_cosmo = []
                per_cosmo_dict = {}
                for sd in cosmo_dirs:
                    cosmo = os.path.basename(sd.rstrip(os.sep))
                    if not quiet:
                        print(f"\n=== Cosmology: {cosmo} ===")
                    res = compute_wst_S012(
                        sd,
                        J=J, L=L, device=device, whiten=whiten,
                        return_means_for_dir=return_means_for_dir,
                        plot=plot, which=which,
                        save_plot=_derive_plot_path(save_plot, cosmo),
                        save_csv=_derive_csv_path(save_csv, cosmo),
                        strict_iso=strict_iso,
                        quiet=quiet,
                        # export raw samples per subdirectory/cosmology
                        save_samples_csv=_derive_samples_csv_path(save_samples_csv, cosmo),
                        samples_format=samples_format,
                        samples_which=samples_which,
                        save_indices_csv=_derive_csv_path(save_indices_csv, cosmo) if save_indices_csv else None,
                    )
                    per_cosmo.append(cosmo)
                    per_cosmo_dict[cosmo] = res
                elapsed_total = time.perf_counter() - _t0_total
                if not quiet:
                    print(f"[timing] grouped total elapsed = {elapsed_total:.3f} s")
                return {
                    'mode': 'group',
                    'per_cosmo': per_cosmo_dict,
                    'cosmologies': per_cosmo,
                    'n_cosmologies': len(per_cosmo),
                    'elapsed_sec_total': float(elapsed_total),
                }

    # ---------- Simple mode: single FITS file or directory of FITS ----------
    # 1) Load (if path) and normalize to batch
    arr = load_images_from_path(images_or_path)
    x = _as_batch(arr)

    # 2) Optional per-image whitening
    if whiten:
        mu = x.mean(axis=(-2, -1), keepdims=True)
        sd = x.std(axis=(-2, -1), keepdims=True)
        x = (x - mu) / (sd + 1e-12)

    # 3) Import scattering (Sihao Cheng)
    try:
        import scattering  # pip install git+https://github.com/SihaoCheng/scattering_transform.git
    except Exception as e:
        raise ImportError(
            "The package 'scattering_transform' (Sihao Cheng) is missing.\n"
            "Install via:\n"
            "  pip install git+https://github.com/SihaoCheng/scattering_transform.git\n"
            "or clone then:\n"
            "  git clone https://github.com/SihaoCheng/scattering_transform.git\n"
            "  pip install -e scattering_transform"
        ) from e

    # 4) Device
    M, N = x.shape[-2], x.shape[-1]
    dev = (device or "auto").strip().lower()
    if dev not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be 'cpu', 'gpu' or 'auto'")
    st_kwargs = {}
    if dev in {"cpu", "gpu"}:
        st_kwargs["device"] = dev

    # 5) Compute
    st_calc = scattering.Scattering2d(M, N, J, L, **st_kwargs)
    s_mean = st_calc.scattering_coef(x)   # dict
    s_cov  = st_calc.scattering_cov (x)   # dict

    # 6) Retrieve S1/S2 (isotropic only if strict_iso=True)
    # S1 ISO mandatory
    if "S1_iso" in s_mean:
        S1_t = s_mean["S1_iso"]
        idx_S1 = s_mean.get("index_S1_iso", None)
    else:
        if strict_iso:
            raise KeyError("S1_iso not found in s_mean: strict_iso=True requires pure isotropic part.")
        # Non-strict mode (conscious fallback): raw S1
        S1_t = s_mean.get("S1", None)
        idx_S1 = s_mean.get("index_S1", None)
        if S1_t is None:
            raise KeyError("S1 not found in s_mean.")

    # S2 ISO (optional but strict if requested)
    if "S2_iso" in s_cov:
        S2_t = s_cov["S2_iso"]
        idx_S2 = s_cov.get("index_S2_iso", None)
    else:
        # In strict mode, require S2_iso only if we PLOT S2 (which == "S2")
        if strict_iso and which.upper() == "S2":
            raise KeyError("S2_iso not found in s_cov: strict_iso=True.")
        S2_t = None
        idx_S2 = None

    # 7) Convert -> 2D numpy
    S1 = _to_numpy(S1_t)
    S1 = S1.reshape(x.shape[0], -1)
    S2 = None
    if S2_t is not None:
        S2 = _to_numpy(S2_t)
        S2 = S2.reshape(x.shape[0], -1)

    # 8) S0: spatial mean (numpy)
    S0 = x.mean(axis=(-2, -1), keepdims=False)[:, None]  # [N,1]

    # 9) Indices -> numpy if torch
    idx_S1 = None if idx_S1 is None else _to_numpy(idx_S1)
    idx_S2 = None if idx_S2 is None else _to_numpy(idx_S2)

    out: Dict[str, Optional[np.ndarray]] = {
        "S0": S0,
        "S1": S1,
        "S2": S2,
        "index_S1": idx_S1,
        "index_S2": idx_S2,
    }

    # === Export raw S0/S1/S2 samples (for covariance/Hartlap) ===
    if save_samples_csv is not None:
        import pandas as pd

        # Resolve final path (file vs directory)
        outpath = os.fspath(save_samples_csv)
        # If an explicit directory is provided (or ends with /), use a default file name
        if (os.path.isdir(outpath)) or (outpath.endswith(os.sep)):
            os.makedirs(outpath, exist_ok=True)
            outpath = os.path.join(outpath, "samples.csv")
        outdir = os.path.dirname(outpath)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        N_s = x.shape[0]  # number of samples (patches)
        rows_long = []
        wide_blocks = []

        # Helper to add a "wide" block
        def _add_wide_block(kind: str, mat: np.ndarray):
            cols = {f"{kind.lower()}_{i}": mat[:, i] for i in range(mat.shape[1])}
            df = pd.DataFrame(cols)
            wide_blocks.append(df)

        # S0
        if "S0" in samples_which and S0 is not None:
            if samples_format.lower() == "long":
                for i_sample in range(N_s):
                    rows_long.append({
                        "sample": int(i_sample),
                        "kind": "S0",
                        "index": 0,
                        "value": float(S0[i_sample, 0]),
                    })
            else:
                _add_wide_block("S0", S0)

        # S1
        if "S1" in samples_which and S1 is not None:
            if samples_format.lower() == "long":
                p1 = S1.shape[1]
                rows_long.extend({
                    "sample": int(i_sample),
                    "kind": "S1",
                    "index": int(j),
                    "value": float(S1[i_sample, j]),
                } for i_sample in range(N_s) for j in range(p1))
            else:
                _add_wide_block("S1", S1)

        # S2
        if "S2" in samples_which and S2 is not None:
            if samples_format.lower() == "long":
                p2 = S2.shape[1]
                rows_long.extend({
                    "sample": int(i_sample),
                    "kind": "S2",
                    "index": int(j),
                    "value": float(S2[i_sample, j]),
                } for i_sample in range(N_s) for j in range(p2))
            else:
                _add_wide_block("S2", S2)

        # Build final DataFrame
        if samples_format.lower() == "long":
            df_samples = pd.DataFrame(rows_long, columns=["sample", "kind", "index", "value"])
        elif samples_format.lower() == "wide":
            import pandas as pd  # ensure pd in scope
            df_samples = pd.concat(
                [pd.Series(np.arange(N_s, dtype=int), name="sample")] + wide_blocks,
                axis=1
            )
        else:
            raise ValueError("samples_format must be 'wide' or 'long'.")

        # CSV write + meta as first commented line
        meta = {
            'n_samples': int(N_s),
            'J': int(J), 'L': int(L), 'M': int(M), 'N': int(N),
            'whiten': bool(whiten), 'device': dev,
            'kinds': ','.join([k for k in samples_which if ((k=="S0" and S0 is not None) or
                                                            (k=="S1" and S1 is not None) or
                                                            (k=="S2" and S2 is not None))])
        }
        df_samples.to_csv(outpath, index=False)
        header = '# ' + ', '.join(f"{k}={v}" for k, v in meta.items())
        with open(outpath, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + '\n' + content)

        if not quiet:
            print(f"[export] samples -> {outpath} (format={samples_format}, kinds={meta['kinds']})")

        # --- Export S1/S2 index mapping if requested ---
        if save_indices_csv is not None:
            idx_out = os.fspath(save_indices_csv)
            idx_dir = os.path.dirname(idx_out)
            if idx_dir and not os.path.exists(idx_dir):
                os.makedirs(idx_dir, exist_ok=True)

            rows_idx = []

            def _flatten_index(kind: str, idx_arr):
                if idx_arr is None:
                    return
                arr = np.asarray(idx_arr)
                # Flatten carefully: one entry per "flattened" column
                # Goal: keep a stable mapping flat_index -> parameter tuple
                flat_size = arr.shape[0] if arr.ndim == 1 else int(np.prod(arr.shape[:-1]))
                # Assume the last dimension stores parameters (j, l, ...),
                # otherwise convert to raw list.
                for flat in range(flat_size):
                    try:
                        v = arr[flat] if arr.ndim == 1 else arr.reshape(flat_size, -1)[flat]
                        v = np.atleast_1d(v).tolist()
                    except Exception:
                        v = np.array(arr).ravel()[flat:flat+1].tolist()
                    rows_idx.append({"kind": kind, "flat_index": int(flat), "tuple": str(v)})

            _flatten_index("S1", idx_S1)
            _flatten_index("S2", idx_S2)

            if rows_idx:
                import pandas as pd
                df_idx = pd.DataFrame(rows_idx, columns=["kind", "flat_index", "tuple"])
                df_idx.to_csv(idx_out, index=False)
                if not quiet:
                    print(f"[export] indices -> {idx_out}")

    # 10) If input is a directory → add requested means/std + optional plot
    is_dir = isinstance(images_or_path, (str, os.PathLike)) and os.path.isdir(str(images_or_path))
    if is_dir and return_means_for_dir:
        # Mean/std over the batch
        S0_mean = S0.mean(axis=0, keepdims=True)     # [1,1]
        out["S0_mean"] = S0_mean

        S1_mean = S1.mean(axis=0, keepdims=True)     # [1,K1]
        S1_std  = S1.std(axis=0, ddof=1, keepdims=True) if S1.shape[0] > 1 else np.zeros_like(S1_mean)
        out["S1_mean"], out["S1_std"] = S1_mean, S1_std

        if S2 is not None:
            S2_mean = S2.mean(axis=0, keepdims=True)
            S2_std  = S2.std(axis=0, ddof=1, keepdims=True) if S2.shape[0] > 1 else np.zeros_like(S2_mean)
            out["S2_mean"], out["S2_std"] = S2_mean, S2_std

        # --- Save CSV of stats ---
        if save_csv is not None:
            import pandas as pd
            rows = []
            # S0
            rows.append({'kind': 'S0', 'index': 0,
                         'mean': float(S0_mean.ravel()[0]), 'std': np.nan})
            # S1
            m1 = S1_mean.ravel()
            s1 = S1_std.ravel() if S1_std is not None else np.zeros_like(m1)
            for i, (mi, si) in enumerate(zip(m1, s1)):
                rows.append({'kind': 'S1', 'index': int(i), 'mean': float(mi), 'std': float(si)})
            # S2
            if 'S2_mean' in out:
                m2 = out['S2_mean'].ravel()
                s2 = out['S2_std'].ravel()
                for i, (mi, si) in enumerate(zip(m2, s2)):
                    rows.append({'kind': 'S2', 'index': int(i), 'mean': float(mi), 'std': float(si)})
            df = pd.DataFrame(rows)
            meta = {
                'n_samples': int(S1.shape[0]),
                'J': int(J), 'L': int(L), 'M': int(M), 'N': int(N),
                'whiten': bool(whiten), 'device': dev
            }
            out_csv = os.fspath(save_csv)
            out_dir = os.path.dirname(out_csv)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            df.to_csv(out_csv, index=False)
            header = '# ' + ', '.join(f"{k}={v}" for k, v in meta.items())
            with open(out_csv, 'r+', encoding='utf-8') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header + '\n' + content)

        # --- Plot ---
        if plot:
            import matplotlib.pyplot as plt

            if which.upper() == "S1":
                y = S1_mean.ravel()
                yerr = S1_std.ravel() if S1_std is not None else None
                label_feat = "S1"
            elif which.upper() == "S2":
                if 'S2_mean' not in out:
                    if strict_iso:
                        raise ValueError("Requested to plot S2 but S2_iso is unavailable in strict mode.")
                    # Non-strict mode: nothing to plot
                    y = np.array([]); yerr = None; label_feat = "S2"
                else:
                    y = out['S2_mean'].ravel()
                    yerr = out['S2_std'].ravel()
                    label_feat = "S2"
            else:
                raise ValueError("which must be 'S1' or 'S2'.")

            x_idx = np.arange(y.size)

            plt.figure(figsize=(6.5, 4.2))
            plt.errorbar(x_idx, y, yerr=yerr,
                         fmt='o', ms=3, lw=1, capsize=2,
                         label=f"{label_feat} (N={S1.shape[0]})")
            plt.xlabel('Coefficient index')
            plt.ylabel(f"{label_feat} mean ± std")
            plt.title('Wavelet Scattering Statistics')
            plt.legend()
            plt.tight_layout()

            if save_plot is not None:
                out_dir = os.path.dirname(os.fspath(save_plot))
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.fspath(save_plot), dpi=200)
            plt.show()

    # Timing (simple mode)
    elapsed = time.perf_counter() - _t0_total
    out["elapsed_sec"] = float(elapsed)
    if not quiet:
        print(f"[timing] elapsed = {elapsed:.3f} s")

    return out
