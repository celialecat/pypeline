# covariance.py
from __future__ import annotations
import os, glob, pickle
from typing import Union, Sequence, Dict, Any, Optional, Tuple, List
import numpy as np

# =========================
# File / path utils
# =========================
def _is_csv(p: str) -> bool:
    return str(p).lower().endswith(".csv")

def _resolve_paths(maybe_paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None]) -> Sequence[str]:
    """
    Accepts: file, directory, glob, list -> returns a list of found CSV files.
    """
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
            out.extend(sorted(glob.glob(s)))  # glob pattern
    return out

# ======================================
# WST loaders (S1 only, or S0+S1)
# ======================================
def _infer_s1_matrix_from_df(df):
    """
    Extracts an [N,p] matrix of S1 (or S1_iso) from a DataFrame ('wide' or 'long' formats).
    - 'long' : columns ['kind','index','value'/'mean', 'sample'(opt)] ; filter S1/S1_ISO then pivot.
    - 'wide' : numeric columns (s1_0, s1_1, ... or 0,1,2,...) ; numeric 'sample' is ignored.
    """
    import pandas as pd

    df = df.copy()

    # --- LONG format ---
    if {'kind', 'index'}.issubset(set(df.columns)):
        df['kind'] = df['kind'].astype(str).str.upper()
        df_s1 = df[df['kind'].isin(['S1', 'S1_ISO'])].copy()
        if df_s1.empty:
            raise ValueError("Format long détecté, mais aucune ligne 'S1'/'S1_ISO' trouvée.")
        value_col = 'value' if 'value' in df_s1.columns else ('mean' if 'mean' in df_s1.columns else None)
        if value_col is None:
            raise ValueError("Format long: aucune colonne 'value' ni 'mean' pour les S1.")
        sample_col = 'sample' if 'sample' in df_s1.columns else None
        if sample_col is None:
            # No sample column -> assume a single S1 mean -> N=1
            pivot = df_s1.pivot_table(index=None, columns='index', values=value_col, aggfunc='first')
            return pivot.to_numpy(dtype=np.float32)[None, ...]  # [1,p]
        pivot = df_s1.pivot_table(index=sample_col, columns='index', values=value_col, aggfunc='first').sort_index()
        if pivot.isna().any().any():
            raise ValueError("S1: valeurs manquantes après pivot (échantillons incomplets).")
        return pivot.to_numpy(dtype=np.float32)

    # --- WIDE format ---
    num_df = df.select_dtypes(include=['number']).copy()
    if 'sample' in num_df.columns:
        num_df = num_df.drop(columns=['sample'])
    if num_df.shape[1] == 0:
        raise ValueError("Aucune colonne numérique détectée pour inférer S1.")
    return num_df.to_numpy(dtype=np.float32)

def load_s1_samples_from_path(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Loads/concatenates S1 samples from:
      - a single CSV file;
      - a directory (concatenates all plausible CSVs);
      - a glob pattern ('/path/*.csv').
    Returns [N, p].
    """
    import pandas as pd

    path = os.fspath(path)
    def _read_one_csv(csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path, comment='#')
        return _infer_s1_matrix_from_df(df)

    if os.path.isfile(path):
        return _read_one_csv(path)

    if any(ch in path for ch in ['*', '?', '[']):
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"Aucun CSV ne matche le motif : {path}")
        mats = [_read_one_csv(f) for f in files]
        _p = {m.shape[1] for m in mats}
        if len(_p) != 1:
            raise ValueError(f"CSV S1 avec dimensions p incohérentes : {sorted(_p)}")
        return np.concatenate(mats, axis=0)

    if os.path.isdir(path):
        # Heuristic: read all and filter those that pass S1
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"Aucun CSV dans le dossier : {path}")
        mats = []
        for f in files:
            try:
                mats.append(_read_one_csv(f))
            except Exception:
                continue
        if not mats:
            raise ValueError("Aucun CSV compatible S1 n'a pu être lu dans le dossier.")
        _p = {m.shape[1] for m in mats}
        if len(_p) != 1:
            raise ValueError(f"CSV S1 avec dimensions p incohérentes : {sorted(_p)}")
        return np.concatenate(mats, axis=0)

    raise FileNotFoundError(f"Chemin introuvable : {path}")

def _infer_s0s1_matrix_from_df(df):
    """
    Extracts an [N,1+p] matrix (S0 + S1) from a DataFrame ('long' or 'wide' formats).
    """
    import pandas as pd

    df = df.copy()

    # --- LONG format ---
    if {'kind', 'index'}.issubset(set(df.columns)):
        df['kind'] = df['kind'].astype(str).str.upper()
        value_col = 'value' if 'value' in df.columns else ('mean' if 'mean' in df.columns else None)
        if value_col is None:
            raise ValueError("Format long détecté, mais aucune colonne 'value' ni 'mean'.")

        if 'sample' not in df.columns:
            # infer sample via S0 occurrences
            s0_rows = df[df['kind'] == 'S0']
            if s0_rows.empty:
                raise ValueError("Impossible d'inférer 'sample' (aucun S0 en format long).")
            s0_idx = s0_rows.index
            sample_map = {}
            cur = 0
            for i in s0_idx:
                sample_map[i] = cur; cur += 1
            sample_col = []
            current_sample = None
            for i in df.index:
                if i in sample_map:
                    current_sample = sample_map[i]
                if current_sample is None:
                    raise ValueError("S1 rencontré avant tout S0: 'sample' non inféré.")
                sample_col.append(current_sample)
            df['sample'] = sample_col

        # S0
        df_s0 = df[df['kind'] == 'S0'].copy()
        s0_pivot = df_s0.pivot_table(index='sample', columns='index', values=value_col, aggfunc='first').sort_index()
        if s0_pivot.shape[1] < 1 or 0 not in s0_pivot.columns:
            raise ValueError("S0 absent/mal formé (index 0 manquant).")
        S0 = s0_pivot[0].to_numpy(dtype=np.float32)[:, None]  # [N,1]

        # S1 (priority 'S1' else 'S1_ISO')
        if (df['kind'] == 'S1').any():
            df_s1 = df[df['kind'] == 'S1'].copy()
        else:
            df_s1 = df[df['kind'] == 'S1_ISO'].copy()
        if df_s1.empty:
            raise ValueError("Aucun S1/S1_ISO trouvé dans le CSV (format long).")

        s1_pivot = df_s1.pivot_table(index='sample', columns='index', values=value_col, aggfunc='first').sort_index()
        if s1_pivot.isna().any().any():
            raise ValueError("Des valeurs S1 manquent pour certains échantillons (format long).")
        s1_pivot = s1_pivot.reindex(sorted(s1_pivot.columns), axis=1)
        S1 = s1_pivot.to_numpy(dtype=np.float32)

        if S1.shape[0] != S0.shape[0]:
            raise ValueError(f"Incohérence N entre S0 (N={S0.shape[0]}) et S1 (N={S1.shape[0]}).")
        return np.concatenate([S0, S1], axis=1)

    # --- WIDE format ---
    num_df = df.select_dtypes(include=['number']).copy()
    if 'sample' in num_df.columns:
        num_df = num_df.drop(columns=['sample'])
    if num_df.shape[1] == 0:
        raise ValueError("Aucune colonne numérique détectée (S0+S1).")

    cols = [str(c).lower() for c in num_df.columns]
    s0_idx = [i for i, c in enumerate(cols) if c.startswith('s0_')]
    if not s0_idx:
        s0_idx = [0]  # fallback: 1st column = S0
    if len(s0_idx) != 1:
        raise ValueError(f"Format wide ambigu: attendu exactement 1 colonne S0, trouvé {len(s0_idx)}.")

    s1_idx = [i for i, c in enumerate(cols) if c.startswith('s1_')]
    if not s1_idx:
        s1_idx = [i for i in range(num_df.shape[1]) if i not in s0_idx]

    def _suffix_num(name):
        try: return int(name.split('_', 1)[1])
        except Exception: return name

    s1_cols = [num_df.columns[i] for i in s1_idx]
    s1_cols_sorted = sorted(s1_cols, key=lambda c: _suffix_num(str(c).lower()))

    S0 = num_df.iloc[:, s0_idx].to_numpy(dtype=np.float32)  # [N,1]
    S1 = num_df.loc[:, s1_cols_sorted].to_numpy(dtype=np.float32)  # [N,p]
    return np.concatenate([S0, S1], axis=1)

def load_s0s1_samples_from_csv(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Loads a single CSV (format 'long' or 'wide') and returns X=[N,1+p] with S0 in the first column.
    """
    import pandas as pd
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, comment='#')
    return _infer_s0s1_matrix_from_df(df)

# ======================================
# D_ell loader (CSV from compute_dell_empiriques)
# ======================================
def _load_emp_ps_from_csv(csv_path: str, *, quiet: bool = False) -> Dict[str, Any]:
    """
    Reads a CSV produced by compute_dell_empiriques(...) and prepares a dict:
      - 'ell'
      - 'D_ell_mean' (optional)
      - 'D_ell_std'  (optional)
      - 'D_ell_stack' (N,d) with 'D_ell_patch{i}'
    """
    import pandas as pd
    df = pd.read_csv(csv_path, comment="#")
    if "ell" not in df.columns:
        raise ValueError("CSV invalide: colonne 'ell' absente.")
    patch_cols = [c for c in df.columns if c.startswith("D_ell_patch")]
    if len(patch_cols) == 0:
        raise ValueError("CSV invalide: aucune colonne 'D_ell_patch{i}' trouvée.")
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
        raise ValueError("Incohérence: len(ell) != nb de bins dans D_ell_stack.")
    if not quiet:
        print(f"[info] CSV D_ell chargé: {os.path.basename(csv_path)} • d={d}, N={N}")
    return {"ell": ell, "D_ell_mean": D_mean, "D_ell_std": D_std, "D_ell_stack": D_stack, "n_maps": int(N)}

def _load_dell_block(paths: Sequence[str], *, cut_head: int = 1, cut_tail: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Builds a D_ell block [N, p_D] by reading one or more 'compute_dell_empiriques' CSV outputs.
    Cuts 'cut_head' and 'cut_tail' bins (default 1/1).
    Returns (X, meta).
    """
    mats, meta = [], {}
    for f in paths:
        if not _is_csv(f):
            continue
        d = _load_emp_ps_from_csv(f, quiet=True)  # {'D_ell_stack': (N,d), 'ell': (d,), ...}
        D = np.asarray(d["D_ell_stack"], dtype=np.float64)  # [N, d]
        h = int(max(0, cut_head)); t = int(max(0, cut_tail))
        if h + t >= D.shape[1]:
            raise ValueError(f"Coupe (head={h}, tail={t}) trop agressive pour d={D.shape[1]} dans {os.path.basename(f)}")
        if t > 0:
            D = D[:, h:-t]
            ell = d["ell"][h:-t]
        else:
            D = D[:, h:]
            ell = d["ell"][h:]
        mats.append(D)
        meta = {"ell_after_cut": np.asarray(ell, dtype=np.float64)}
    if not mats:
        return np.empty((0, 0), dtype=np.float64), meta
    pset = {m.shape[1] for m in mats}
    if len(pset) != 1:
        raise ValueError(f"D_ell: dimensions de features incohérentes après coupe: {sorted(pset)}")
    return np.concatenate(mats, axis=0), meta

# ======================================
# Alignment / X construction
# ======================================
def _align_blocks(blocks: Sequence[np.ndarray], *, mode: str = "truncate") -> np.ndarray:
    """
    Aligns multiple blocks [N_i, p_i] to form X=[N_min, sum p_i] (truncate)
    or raises in 'strict' mode if N_i differ.
    """
    valid = [b for b in blocks if (isinstance(b, np.ndarray) and b.size > 0)]
    if not valid:
        return np.empty((0, 0), dtype=np.float64)
    Ns = [b.shape[0] for b in valid]
    if len(set(Ns)) != 1:
        if mode == "strict":
            raise ValueError(f"N d'échantillons incompatibles: {Ns}")
        Nmin = min(Ns)
        valid = [b[:Nmin] for b in valid]
    return np.concatenate(valid, axis=1)

# ======================================
# Main function (API)
# ======================================
def compute_covariance_mixed(
    *,
    wst_inputs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None] = None,
    dell_inputs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None] = None,
    prefer_s0s1: bool = True,           # True: takes S0+S1 if available, else S1
    align: str = "truncate",            # 'truncate' (truncates to common N) or 'strict' (requires identical N)
    dell_cut_head: int = 1,             # default: remove the 1st bin (apodization / edges)
    dell_cut_tail: int = 1,             # default: remove the last bin
    save_csv: Optional[str] = None,
    save_npy: Optional[str] = None,
    save_npz: Optional[str] = None,
    save_pkl: Optional[str] = None,
    plot: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Builds a feature matrix X from WST and/or D_ell, then computes the empirical covariance (ddof=1).
    - WST: accepts 'wide' or 'long' CSVs (S1 only, or S0+S1 if prefer_s0s1=True).
    - D_ell: accepts 'compute_dell_empiriques' CSVs; cuts head/tail bins.
    - Mix: concatenates WST and D_ell by columns (cross-covariances included).
    - No inversion or Hartlap correction here.
    Saves in CSV, NPY, NPZ, PKL format depending on options. Plots heatmap if plot=True.
    """
    # -- Resolve file lists --
    wst_paths  = _resolve_paths(wst_inputs)
    dell_paths = _resolve_paths(dell_inputs)

    # -- Load WST --
    W = np.empty((0, 0), dtype=np.float64)
    if wst_paths:
        mats = []
        for f in wst_paths:
            try:
                if prefer_s0s1:
                    Xw = load_s0s1_samples_from_csv(f)  # [N, 1+p]
                else:
                    Xw = load_s1_samples_from_path(f)   # [N, p]
            except Exception:
                # fallback if prefer_s0s1 but S1-only file
                if prefer_s0s1:
                    try:
                        Xw = load_s1_samples_from_path(f)
                    except Exception:
                        if not quiet:
                            print(f"[skip] WST non compatible: {f}")
                        continue
                else:
                    if not quiet:
                        print(f"[skip] WST non compatible: {f}")
                    continue
            mats.append(Xw.astype(np.float64, copy=False))
        if mats:
            p_set = {m.shape[1] for m in mats}
            if len(p_set) != 1:
                raise ValueError(f"WST: dimensions de features incohérentes: {sorted(p_set)}")
            W = np.concatenate(mats, axis=0)

    # -- Load D_ell --
    D = np.empty((0, 0), dtype=np.float64)
    meta_d: Dict[str, Any] = {}
    if dell_paths:
        D, meta_d = _load_dell_block(dell_paths, cut_head=dell_cut_head, cut_tail=dell_cut_tail)

    # -- Merge / alignment --
    X = _align_blocks([W, D], mode=align)
    if X.size == 0:
        raise ValueError("Aucun échantillon/observable lisible: WST et D_ell vides.")
    N, p = X.shape
    if N < 2 or p < 1:
        raise ValueError(f"Samples insuffisants pour une covariance: N={N}, p={p} (besoin N>=2 et p>=1).")

    # -- Covariance (sample estimator ddof=1) --
    cov = np.cov(X, rowvar=False, ddof=1)

    # -- Meta export --
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

    # -- Saves --
    if save_csv is not None:
        import pandas as pd
        out_csv = os.fspath(save_csv)
        out_dir = os.path.dirname(out_csv)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(cov).to_csv(out_csv, index=False)
        header = '# ' + ', '.join(f"{k}={v if v is not None else 'NA'}" for k, v in meta.items())
        with open(out_csv, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + '\n' + content)
        if not quiet:
            print(f"[ok] covariance -> {out_csv}")

    if save_npy is not None:
        base = os.path.splitext(os.fspath(save_npy))[0]
        out_dir = os.path.dirname(base)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        np.save(f"{base}.npy", cov)
        if meta.get("ell_after_cut") is not None and len(np.asarray(meta["ell_after_cut"])) > 0:
            np.save(f"{base}_ell.npy", np.asarray(meta["ell_after_cut"]))
        if not quiet:
            print(f"[ok] covariance -> {base}.npy")

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

    if save_pkl is not None:
        out_pkl = os.path.splitext(os.fspath(save_pkl))[0] + ".pkl"
        out_dir = os.path.dirname(out_pkl)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        with open(out_pkl, "wb") as f:
            pickle.dump({"cov": cov, **meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
        if not quiet:
            print(f"[ok] covariance -> {out_pkl}")

    # -- Plot (heatmap) --
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 5.4))
        plt.imshow(cov, origin="lower", aspect="auto")
        plt.colorbar(label="Covariance")
        ttl = "Covariance"
        if W.size > 0 and D.size > 0:
            ttl += " (WST + Dℓ)"
        elif W.size > 0:
            ttl += " (WST)"
        else:
            ttl += " (Dℓ)"
        plt.title(f"{ttl} • N={N}, p={p}")
        plt.tight_layout()
        plt.show()

    if not quiet:
        print(f"[ok] C construit • N={N}, p={p} • WST={W.shape} • D_ell={D.shape}")
    return {"cov": cov, "N": N, "p": p, "X_shape": (N, p),
            "blocks": {"wst": W.shape, "dell": D.shape},
            "ell_after_cut": meta.get("ell_after_cut")}

# ======================================
# CLI
# ======================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Builds an empirical covariance (ddof=1) from WST (S1 or S0+S1) and/or D_ell.\n"
            "Inputs = CSV files, directories, or globs (can be mixed)."
        )
    )
    p.add_argument("--wst", dest="wst", nargs="*", default=None,
                   help="WST paths (CSV, directory, or glob). S0+S1 if available, else S1.")
    p.add_argument("--dell", dest="dell", nargs="*", default=None,
                   help="D_ell paths (compute_dell_empiriques CSV, directory, or glob).")
    p.add_argument("--no_s0", action="store_true",
                   help="Do not attempt S0+S1 (forces S1 only).")
    p.add_E('`ArgumentParser` object has no attribute `E`', '`add_argument`')
    p.add_argument("--align", type=str, default="truncate", choices=["truncate", "strict"],
                   help="Align blocks by N: truncate to common N or require equality.")
    p.add_argument("--dell_cut_head", type=int, default=1, help="Nb of bins to cut at the beginning (D_ell).")
    p.add_argument("--dell_cut_tail", type=int, default=1, help="Nb of bins to cut at the end (D_ell).")
    p.add_argument("--save_csv", type=str, default=None, help="Output CSV file (raw covariance).")
    p.add_argument("--save_npy", type=str, default=None, help="Base .npy path (writes base.npy [+ base_ell.npy]).")
    p.add_argument("--save_npz", type=str, default=None, help="Compressed .npz file (cov, meta).")
    p.add_argument("--save_pkl", type=str, default=None, help="Pickle file .pkl (cov + meta via pickle).")
    p.add_argument("--plot", action="store_true", help="Displays a heatmap of the covariance.")
    p.add_argument("--quiet", action="store_true", help="Less verbose.")

    args = p.parse_args()

    res = compute_covariance_mixed(
        wst_inputs=args.wst,
        dell_inputs=args.dell,
        prefer_s0s1=(not args.no_s0),
        align=args.align,
        dell_cut_head=args.dell_cut_head,
        dell_cut_tail=args.dell_cut_tail,
        save_csv=args.save_csv,
        save_npy=args.save_npy,
        save_npz=args.save_npz,
        save_pkl=args.save_pkl,
        plot=args.plot,
        quiet=args.quiet,
    )