"""Patch for covariance binning to match ell_eval = geomspace(400, 5000, 18).

This file provides *drop-in* replacements for two functions in your
`covariance.py`:
    - _load_dell_block(...)
    - compute_covariance_mixed(...)
plus a tiny CLI helper to allow `--dell_target_ell geom:400,5000,18`.

Usage (Python):
    from covariance_patch import compute_covariance_mixed
    import numpy as np
    target = np.geomspace(400, 5000, 18)
    res = compute_covariance_mixed(
        dell_inputs="/path/to/example_dell_...csv",
        align="truncate",
        dell_cut_head=0,
        dell_cut_tail=0,
        dell_target_ell=target,
        save_npz="mcmc_out/cov_all.npz",
        plot=True,
    )

This will compute the covariance *after rebinning* every D_ell patch to the
**target** ell grid (by 1D interpolation), and will save the grid into
`ell_after_cut` inside the NPZ, so that `gaussian_loglike` aligns automatically.
"""
from __future__ import annotations

import os, glob, pickle, time
from typing import Union, Sequence, Dict, Any, Optional, Tuple, List
import numpy as np


def _align_blocks(blocks, *, mode: str = "truncate"):
    """
    Aligne plusieurs blocs [N_i, p_i] pour former X=[N_min, sum p_i] (truncate)
    ou lève en 'strict' si les N diffèrent.
    """
    valid = [b for b in blocks if (isinstance(b, np.ndarray) and b.size > 0)]
    if not valid:
        return np.empty((0, 0), dtype=np.float64)
    Ns = [b.shape[0] for b in valid]
    if len(set(Ns)) != 1:
        if mode == "strict":
            raise ValueError(f"Incompatible sample counts (N): {Ns}")
        Nmin = min(Ns)
        valid = [b[:Nmin] for b in valid]
    return np.concatenate(valid, axis=1)


# ------------------------------
# Helpers copied from your file
# ------------------------------

def _is_csv(p: str) -> bool:
    return str(p).lower().endswith(".csv")

def _resolve_paths(maybe_paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None]) -> Sequence[str]:
    if maybe_paths is None:
        return []
    if isinstance(maybe_paths, (str, os.PathLike)):
        maybe_paths = [str(maybe_paths)]
    out: List[str] = []
    for item in maybe_paths:
        s = str(item)
        if os.path.isfile(s):
            out.append(s)
        elif os.path.isdir(s):
            out.extend(sorted(glob.glob(os.path.join(s, "*.csv"))))
        else:
            out.extend(sorted(glob.glob(s)))
    return out

def _load_emp_ps_from_csv(csv_path: str, *, quiet: bool = False) -> Dict[str, Any]:
    import pandas as pd
    df = pd.read_csv(csv_path, comment="#")
    if "ell" not in df.columns:
        raise ValueError("Invalid CSV: 'ell' column missing.")
    patch_cols = [c for c in df.columns if c.startswith("D_ell_patch")]
    if len(patch_cols) == 0:
        raise ValueError("Invalid CSV: no 'D_ell_patch{i}' columns found.")
    D_stack = df[patch_cols].to_numpy(dtype=float).T  # (N, d)
    ell = df["ell"].to_numpy(dtype=float)
    d = ell.shape[0]
    N = D_stack.shape[0]
    if "D_ell_mean" in df.columns:
        D_mean = df["D_ell_mean"].to_numpy(dtype=float)
    else:
        D_mean = D_stack.mean(axis=0)
    if "D_ell_std" in df.columns:
        D_std = df["D_ell_std"].to_numpy(dtype=float)
    else:
        D_std = D_stack.std(axis=0, ddof=1) if N > 1 else np.zeros(d, dtype=np.float64)
    if ell.shape[0] != D_stack.shape[1]:
        raise ValueError("Inconsistency: len(ell) != number of bins in D_ell_stack.")
    if not quiet:
        print(f"[info] D_ell CSV loaded: {os.path.basename(csv_path)} • d={d}, N={N}")
    return {"ell": ell, "D_ell_mean": D_mean, "D_ell_std": D_std, "D_ell_stack": D_stack, "n_maps": int(N)}

# ---------------------------------------------
# Patched: rebin to a target ell grid if given
# ---------------------------------------------

def _load_dell_block(
    paths: Sequence[str], *,
    cut_head: int = 1,
    cut_tail: int = 1,
    target_ell: Optional[np.ndarray] = None,
    rebin_method: str = "interp",  # future-proof hook
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a D_ell block [N, p_D] and *optionally* rebin to `target_ell`.

    If `target_ell` is provided, each patch is interpolated onto this grid
    (1D linear interpolation, monotonic ell assumed). The returned meta
    contains `ell_after_cut` = `target_ell` so downstream code aligns bins.
    """
    mats, meta = [], {}
    for f in paths:
        if not _is_csv(f):
            continue
        d = _load_emp_ps_from_csv(f, quiet=True)
        D = np.asarray(d["D_ell_stack"], dtype=np.float64)  # [N, d]
        ell = np.asarray(d["ell"], dtype=np.float64)
        h = int(max(0, cut_head)); t = int(max(0, cut_tail))
        if h + t >= D.shape[1]:
            raise ValueError(f"Cut (head={h}, tail={t}) too aggressive for d={D.shape[1]} in {os.path.basename(f)}")
        if t > 0:
            D = D[:, h:-t]; ell_cut = ell[h:-t]
        else:
            D = D[:, h:];    ell_cut = ell[h:]

        # --- Rebin to target grid if requested ---
        if target_ell is not None:
            tgt = np.asarray(target_ell, dtype=float)
            if (np.any(np.diff(ell_cut) <= 0)):
                raise ValueError("ell grid must be strictly increasing for interpolation")
            Dr = np.vstack([np.interp(tgt, ell_cut, row) for row in D])  # (N, p_target)
            D = Dr
            ell_used = tgt
        else:
            ell_used = ell_cut

        mats.append(D)
        meta = {"ell_after_cut": np.asarray(ell_used, dtype=np.float64)}

    if not mats:
        return np.empty((0, 0), dtype=np.float64), meta

    pset = {m.shape[1] for m in mats}
    if len(pset) != 1:
        raise ValueError(f"D_ell: inconsistent feature dimensions after cut/rebin: {sorted(pset)}")

    return np.concatenate(mats, axis=0), meta


# ------------------------------------------------
# Patched: expose `dell_target_ell` in main API
# ------------------------------------------------

def compute_covariance_mixed(
    *,
    wst_inputs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None] = None,
    dell_inputs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None] = None,
    prefer_s0s1: bool = True,
    align: str = "truncate",
    dell_cut_head: int = 1,
    dell_cut_tail: int = 1,
    dell_target_ell: Optional[np.ndarray] = None,
    save_csv: Optional[str] = None,
    save_npy: Optional[str] = None,
    save_npz: Optional[str] = None,
    save_pkl: Optional[str] = None,
    plot: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Compute empirical covariance with optional D_ell rebinning to `dell_target_ell`.

    When `dell_target_ell` is provided, *all* D_ell inputs are interpolated
    onto this common target grid before building the covariance. The saved
    NPZ will include this grid under the key `ell_after_cut`.
    """
    t_start = time.time()

    wst_paths  = _resolve_paths(wst_inputs)
    dell_paths = _resolve_paths(dell_inputs)

    # WST block (unchanged): you can paste your existing implementation here
    W = np.empty((0, 0), dtype=np.float64)
    if wst_paths:
        from pandas import read_csv  # lazy import for speed
        def _try_load_one(f):
            df = read_csv(f, comment="#")
            # Prefer S0+S1 if available, else fallback to numeric columns
            cols = [c.lower() for c in df.columns]
            if prefer_s0s1 and ("kind" in cols and "index" in cols):
                # reuse your original loader if you want full behaviour
                from covariance import _infer_s0s1_matrix_from_df as _s0s1
                return _s0s1(df).astype(np.float64, copy=False)
            else:
                from covariance import _infer_s1_matrix_from_df as _s1
                return _s1(df).astype(np.float64, copy=False)
        mats = []
        for f in wst_paths:
            try:
                mats.append(_try_load_one(f))
            except Exception:
                if not quiet:
                    print(f"[skip] Incompatible WST: {f}")
        if mats:
            p_set = {m.shape[1] for m in mats}
            if len(p_set) != 1:
                raise ValueError(f"WST: inconsistent feature dimensions: {sorted(p_set)}")
            W = np.concatenate(mats, axis=0)

    # D_ell block with *rebin*
    D = np.empty((0, 0), dtype=np.float64)
    meta_d: Dict[str, Any] = {}
    if dell_paths:
        D, meta_d = _load_dell_block(
            dell_paths,
            cut_head=dell_cut_head,
            cut_tail=dell_cut_tail,
            target_ell=dell_target_ell,
        )

    # Align and covariance
    X = _align_blocks([W, D], mode=align)
    if X.size == 0:
        raise ValueError("No readable samples/observables: WST and D_ell are empty.")
    N, p = X.shape
    if N < 2 or p < 1:
        raise ValueError(f"Insufficient samples for covariance: N={N}, p={p} (requires N>=2 and p>=1).")

    cov = np.cov(X, rowvar=False, ddof=1)

    meta = {
        "N": int(N),
        "p": int(p),
        "wst_block_shape": tuple(W.shape),
        "dell_block_shape": tuple(D.shape),
        "dell_cut_head": int(dell_cut_head),
        "dell_cut_tail": int(dell_cut_tail),
        "ell_after_cut": meta_d.get("ell_after_cut", None),
        "definition": "C = (1/(N-1)) * sum_i (x_i - mean)(x_i - mean)^T [no Hartlap]",
    }

    # Saves (same as your original; shortened here)
    if save_npz is not None:
        out_npz = os.path.splitext(os.fspath(save_npz))[0] + ".npz"
        out_dir = os.path.dirname(out_npz)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            out_npz,
            cov=cov,
            N=N, p=p,
            wst_block_shape=np.array(W.shape, dtype=int),
            dell_block_shape=np.array(D.shape, dtype=int),
            dell_cut_head=int(dell_cut_head),
            dell_cut_tail=int(dell_cut_tail),
            ell_after_cut=(meta["ell_after_cut"] if meta["ell_after_cut"] is not None else np.array([])),
        )
        if not quiet:
            print(f"[ok] covariance -> {out_npz}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 5.4))
        plt.imshow(cov, origin="lower", aspect="auto")
        plt.colorbar(label="Covariance")
        ttl = "Covariance (Dℓ)" if W.size == 0 else ("Covariance (WST + Dℓ)" if D.size > 0 else "Covariance (WST)")
        plt.title(f"{ttl} • N={N}, p={p}")
        plt.tight_layout(); plt.show()

    if not quiet:
        print(f"[ok] C built • N={N}, p={p} • WST={W.shape} • D_ell={D.shape}")
        print(f"[info] ell grid saved: {meta['ell_after_cut'].shape if meta['ell_after_cut'] is not None else 'None'}")

    return {"cov": cov, "N": N, "p": p, "X_shape": (N, p),
            "blocks": {"wst": W.shape, "dell": D.shape},
            "ell_after_cut": meta.get("ell_after_cut")}


# ------------------------------
# Optional: CLI helper for target
# ------------------------------

def _parse_target_ell(arg: Optional[str]) -> Optional[np.ndarray]:
    if arg is None:
        return None
    s = str(arg)
    if s.startswith("geom:"):
        # format: geom:start,stop,num
        _, spec = s.split(":", 1)
        start, stop, num = spec.split(",")
        return np.geomspace(float(start), float(stop), int(num))
    # else: try to load from .npy
    if os.path.isfile(s):
        arr = np.load(s)
        return np.asarray(arr, dtype=float)
    raise ValueError("--dell_target_ell expects 'geom:a,b,n' or a .npy file path")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dell", dest="dell", nargs="*", default=None)
    p.add_argument("--dell_cut_head", type=int, default=0)
    p.add_argument("--dell_cut_tail", type=int, default=0)
    p.add_argument("--dell_target_ell", type=str, default=None,
                   help="geom:400,5000,18 or path/to/ell.npy")
    p.add_argument("--save_npz", type=str, default="mcmc_out/cov_all.npz")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    target = _parse_target_ell(args.dell_target_ell)
    res = compute_covariance_mixed(
        dell_inputs=args.dell,
        dell_cut_head=args.dell_cut_head,
        dell_cut_tail=args.dell_cut_tail,
        dell_target_ell=target,
        save_npz=args.save_npz,
        plot=args.plot,
    )
    print("Done.")
