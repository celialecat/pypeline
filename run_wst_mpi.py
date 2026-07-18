# Save as run_wst_mpi.py
# Run with:
#   mpirun -np 32 python run_wst_mpi.py --start 0 --end 9999
# Example test run:
#   mpirun -np 4 python run_wst_mpi.py --start 0 --end 31 --which S1
import sys
sys.path.insert(0, "/rds/rds-clecat/pipeline_alina_full/alina_paper/scattering_transform")
import os
import argparse
from mpi4py import MPI
from wst_sv_1 import compute_wst_S012
# from yourmodule import compute_wst_S012   # ← Uncomment and fix import path

def parse_args():
    p = argparse.ArgumentParser(description="MPI runner for compute_wst_S012 over many FITS maps.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=9999)
    p.add_argument("--pattern", type=str,
                   default="/rds/maps_sbi_10by10patch_final/map_sbi_10t10patch_1.4beam_1024_{i}.fits")
    p.add_argument("--J", type=int, default=5)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--which", type=str, default="S1",
                   choices=["S0","S1","S2","S01","S012","S12"])
    p.add_argument("--save_samples_csv", type=str,
                   default="/rds/rds-clecat/pipeline_alina_full/alina_paper/pipeline_outputs/paints_100_test_wst/")
    p.add_argument("--save_plot", type=str,
                   default="/rds/rds-clecat/pipeline_alina_full/alina_paper/pipeline_outputs/paints_100_wst_plot/")
    p.add_argument("--whiten", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()

def ensure_dirs(*paths):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)

def process_index(i, args):
    path = args.pattern.format(i=i)
    if not os.path.exists(path):
        return f"MISS {i}: {path} (not found)"
    attempt, last_err = 0, None
    while attempt <= args.retries:
        try:
            res = compute_wst_S012(
                path,
                J=args.J, L=args.L,
                device="auto",
                whiten=args.whiten,
                samples_format="long",
                plot=(not args.no_plot),
                which=args.which,
                save_samples_csv=args.save_samples_csv,
                save_plot=args.save_plot,
                quiet=args.quiet
            )
            return f"DONE {i}"
        except Exception as e:
            last_err = e
            attempt += 1
    return f"FAIL {i}: {type(last_err).__name__}: {last_err}"

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if rank == 0:
        ensure_dirs(args.save_samples_csv, args.save_plot)
    comm.Barrier()
    indices = range(args.start, args.end + 1)
    my_indices = [i for j, i in enumerate(indices) if j % size == rank]
    results = []
    for i in my_indices:
        msg = process_index(i, args)
        results.append(msg)
        print(f"[rank {rank}/{size}] {msg}", flush=True)
    all_results = comm.gather(results, root=0)
    if rank == 0:
        done = sum(m.startswith("DONE") for r in all_results for m in r)
        miss = sum(m.startswith("MISS") for r in all_results for m in r)
        fail = sum(m.startswith("FAIL") for r in all_results for m in r)
        print(f"\nSummary: DONE={done}  MISS={miss}  FAIL={fail}")

if __name__ == "__main__":
    main()
