#!/usr/bin/env python3
import os, sys, glob, re
import numpy as np
import pandas as pd

# --------------------------
# Headless + thread sanity (before any heavy libs)
# --------------------------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.pop("DISPLAY", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --------------------------
# MPI
# --------------------------
from mpi4py import MPI
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

# --------------------------
# Your WST code on PYTHONPATH
# --------------------------
sys.path.insert(0, "/rds/rds-clecat/pipeline_alina_full/alina_paper/scattering_transform")
from wst_en import compute_wst_S012, read_fits_image

# --------------------------
# Config
# --------------------------
INPUT_DIR   = "/rds/rds-clecat/pipeline_alina_full/alina_paper/pipeline_outputs/paint_cov_big_sim"
PATTERN     = "map_sbi_10t10patch_1.4beam_1024_*.fits"  # ..._0.fits to ..._9999.fits
J, L        = 7, 4
PREF_DEVICE = "cpu"          # preferred; will auto-fallback to CPU if GPU unsupported
WHITEN      = False
SAMPLES_FMT = "long"
BATCH_SIZE  = 1             # tune for memory

# Writable outputs under your project area
BASE_OUT        = "/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs"
OUT_SAMPLES_DIR = os.path.join(BASE_OUT, "wst_full_coefs")
OUT_MEAN_DIR    = os.path.join(BASE_OUT, "wst_mean_coefs")

# --------------------------
# Helpers
# --------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def split_list_round_robin(items, r, n):
    """
    Distribute 'items' across 'n' ranks in a round-robin way
    and return the slice corresponding to rank 'r'.
    """
    return items[r::n]

def load_batch(paths):
    imgs = []
    for p in paths:
        arr = read_fits_image(p)  # [H,W] float32
        if arr.ndim != 2:
            raise ValueError(f"Unexpected shape {arr.shape} for {p}")
        imgs.append(arr)
    return np.stack(imgs, axis=0).astype(np.float32, copy=False)

def run_wst_with_fallback(x, **kw):
    """
    Try GPU first if requested, else CPU. If GPU is incompatible (e.g., Blackwell sm_120 with older PyTorch),
    automatically retry on CPU. Other errors are raised.
    """
    device = kw.pop("device", PREF_DEVICE)

    if device == "gpu":
        try_order = ["gpu", "cpu"]
    elif device == "cpu":
        try_order = ["cpu"]
    else:
        try_order = ["gpu", "cpu"]

    last_exc = None
    for dev in try_order:
        try:
            return compute_wst_S012(x, device=dev, **kw)
        except Exception as e:
            msg = str(e)
            arch_err = ("no kernel image is available" in msg
                        or "not compatible with the current PyTorch" in msg)
            if dev == "gpu" and arch_err:
                if rank == 0:
                    print("[WST] GPU not compatible with this PyTorch build; falling back to CPU.")
                continue  # try CPU
            last_exc = e
            break      # don't mask other errors
    if last_exc:
        raise last_exc
    raise RuntimeError("WST fallback: no device succeeded and no exception captured")

def get_map_label_from_path(path):
    """
    Extract the integer map index from a FITS filename of the form
    'map_sbi_10t10patch_1.4beam_1024_XXXX.fits' -> XXXX (int).
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)  # e.g. 'map_sbi_10t10patch_1.4beam_1024_1234'
    idx_str = name.split('_')[-1]
    return int(idx_str)

# --------------------------
# Discover files (rank 0) and share
# --------------------------
if rank == 0:
    files_all = sorted(glob.glob(os.path.join(INPUT_DIR, PATTERN)))
    if len(files_all) == 0:
        raise FileNotFoundError(f"No FITS found under {INPUT_DIR} with pattern {PATTERN}")
else:
    files_all = None

files_all = comm.bcast(files_all if rank == 0 else None, root=0)
n_total   = len(files_all)

my_files = split_list_round_robin(files_all, rank, size)
n_local  = len(my_files)

if rank == 0:
    print(f"[MPI] size={size}  total_files={n_total}")

# Only rank 0 creates output dirs, then barrier
if rank == 0:
    ensure_dir(OUT_SAMPLES_DIR)
    ensure_dir(OUT_MEAN_DIR)
comm.Barrier()

# --------------------------
# Per-rank processing in batches
# --------------------------
rank_batch_paths = []

has_s2  = None
sum_s0  = None; sumsq_s0  = None; n_seen = 0
sum_s1  = None; sumsq_s1  = None
sum_s2  = None; sumsq_s2  = None

for start in range(0, n_local, BATCH_SIZE):
    batch_paths = my_files[start:start+BATCH_SIZE]
    if not batch_paths:
        continue

    x = load_batch(batch_paths)   # [B,H,W]

    # We want one CSV per map: BATCH_SIZE must be 1
    if BATCH_SIZE != 1:
        raise RuntimeError("This script assumes BATCH_SIZE=1 to produce one CSV per map.")

    # batch_paths has a single FITS file; extract its index
    map_label = get_map_label_from_path(batch_paths[0])  # 0..9999
    batch_samples_file = os.path.join(OUT_SAMPLES_DIR, f"wst_map_{map_label}.csv")


    res = run_wst_with_fallback(
        x,
        J=J, L=L,
        whiten=WHITEN,
        return_means_for_dir=False,
        plot=False,
        which="S1",
        save_samples_csv=batch_samples_file,
        samples_format=SAMPLES_FMT,
        quiet=True,
    )
    rank_batch_paths.append(batch_samples_file)
    B = x.shape[0]
    n_seen += B

    S0 = res["S0"]               # [B,1]
    S1 = res["S1"]               # [B,K1]
    S2 = res["S2"]               # [B,K2] or None

    if sum_s0 is None:
        sum_s0   = np.zeros((1,), dtype=np.float64)
        sumsq_s0 = np.zeros((1,), dtype=np.float64)
    if sum_s1 is None:
        sum_s1   = np.zeros((S1.shape[1],), dtype=np.float64)
        sumsq_s1 = np.zeros((S1.shape[1],), dtype=np.float64)
    if S2 is not None and sum_s2 is None:
        sum_s2   = np.zeros((S2.shape[1],), dtype=np.float64)
        sumsq_s2 = np.zeros((S2.shape[1],), dtype=np.float64)

    # accumulate
    s0v = S0[:, 0].astype(np.float64)
    sum_s0   += np.sum(s0v)
    sumsq_s0 += np.sum(s0v * s0v)

    s1v = S1.astype(np.float64)
    sum_s1   += np.sum(s1v, axis=0)
    sumsq_s1 += np.sum(s1v * s1v, axis=0)

    if S2 is not None:
        has_s2 = True if has_s2 is None else has_s2
        s2v = S2.astype(np.float64)
        if sum_s2 is None:
            sum_s2   = np.zeros((S2.shape[1],), dtype=np.float64)
            sumsq_s2 = np.zeros((S2.shape[1],), dtype=np.float64)
        sum_s2   += np.sum(s2v, axis=0)
        sumsq_s2 += np.sum(s2v * s2v, axis=0)
    else:
        has_s2 = False if has_s2 is None else has_s2

comm.Barrier()

# --------------------------
# Reduce accumulators to rank 0
# --------------------------
def reduce_array(a, op=MPI.SUM):
    if a is None:
        return None
    out = np.zeros_like(a)
    comm.Reduce(a, out, op=op, root=0)
    return out

total_n     = comm.reduce(n_seen, op=MPI.SUM, root=0)
glob_sum0   = reduce_array(sum_s0)
glob_sumsq0 = reduce_array(sumsq_s0)
glob_sum1   = reduce_array(sum_s1)
glob_sumsq1 = reduce_array(sumsq_s1)
glob_sum2   = reduce_array(sum_s2) if has_s2 else None
glob_sumsq2 = reduce_array(sumsq_s2) if has_s2 else None

all_rank_batch_paths = comm.gather(rank_batch_paths, root=0)

# --------------------------
# Rank 0: write global mean/std
# --------------------------
if rank == 0:
    print(f"[MPI] Processed {total_n} maps across {size} ranks.")

    def mean_and_std(glob_sum, glob_sumsq, n):
        mean = glob_sum / n
        var  = (glob_sumsq - (glob_sum * glob_sum) / n) / max(n - 1, 1)
        var  = np.maximum(var, 0.0)
        std  = np.sqrt(var)
        return mean, std

    m0, _  = mean_and_std(glob_sum0, glob_sumsq0, total_n)
    m1, s1 = mean_and_std(glob_sum1, glob_sumsq1, total_n)
    m2 = s2 = None
    if (glob_sum2 is not None) and (glob_sumsq2 is not None):
        m2, s2 = mean_and_std(glob_sum2, glob_sumsq2, total_n)

    out_mean_dir  = ensure_dir(os.path.join(OUT_MEAN_DIR, "wst_mean"))
    out_mean_csv  = os.path.join(out_mean_dir, "mean_stats.csv")

    rows = []
    rows.append({"kind": "S0", "index": 0, "mean": float(m0[0]), "std": np.nan})
    for i, (mi, si) in enumerate(zip(m1.tolist(), s1.tolist())):
        rows.append({"kind": "S1", "index": i, "mean": mi, "std": si})
    if m2 is not None:
        for i, (mi, si) in enumerate(zip(m2.tolist(), s2.tolist())):
            rows.append({"kind": "S2", "index": i, "mean": mi, "std": si})

    df_mean = pd.DataFrame(rows, columns=["kind", "index", "mean", "std"])
    tmp = out_mean_csv + ".tmp"
    df_mean.to_csv(tmp, index=False)
    header = f"# n_samples={total_n}, J={J}, L={L}, whiten={WHITEN}, device={PREF_DEVICE}"
    with open(tmp, "r", encoding="utf-8") as f:
        content = f.read()
    with open(out_mean_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n" + content)
    os.replace(tmp, out_mean_csv)
    print(f"[export] mean/std -> {out_mean_csv}")


