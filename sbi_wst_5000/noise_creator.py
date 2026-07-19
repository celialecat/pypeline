#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate N Simons Observatory LAT ILC tSZ noise maps (Compton-y),
projected onto 10° × 10° CAR patches using pixell, in parallel with MPI.

Usage (12 MPI ranks):
    mpirun -np 12 python generate_so_tsz_noise_patches_mpi.py
or:
    srun -n 12 python generate_so_tsz_noise_patches_mpi.py
"""

import os
import numpy as np
import healpy as hp
from pixell import enmap, reproject, utils
from mpi4py import MPI

# =========================
# GLOBAL CONFIGURATION
# =========================

# Path to the SO N_ell file (baseline tSZ ILC)
NELL_FILE = "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"

# Total number of noise maps to generate
N_MAPS = 5000

# HEALPix settings
NSIDE = 2048
LMAX_CLIP = None   # None → use full ell range from the file

# Output patches (10°×10°)
OUTDIR_PATCH = "noise_patches_car"
PATCH_SIZE_DEG = 10.0    # patch side length in degrees
PATCH_NPIX     = 512     # number of pixels per side

# Patch center in degrees (RA, Dec)
PATCH_RA0_DEG  = 0.0
PATCH_DEC0_DEG = 0.0

# Base RNG seed (a different seed is used for each global map index)
BASE_SEED = 12345


# =========================
# MPI INITIALIZATION
# =========================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# =========================
# UTILITY FUNCTIONS
# =========================

def load_nell_tsz(nell_file):
    """Load SO LAT ILC tSZ N_ell file: returns (ell, N_ell)."""
    ell, Nell = np.loadtxt(nell_file, unpack=True)
    return ell, Nell


def build_cl_noise_from_nell(ell, Nell, lmax_clip=None):
    """
    Build C_ell^noise = N_ell.

    Returns:
        Cl_noise : array (float64)
        lmax     : int
    """
    ell_int = ell.astype(int)
    lmax_file = int(ell_int.max())

    if lmax_clip is not None:
        lmax = min(lmax_file, int(lmax_clip))
        mask = ell_int <= lmax
        ell_int = ell_int[mask]
        Nell = Nell[mask]
    else:
        lmax = lmax_file

    Cl = np.zeros(lmax + 1, dtype=float)
    Cl[ell_int] = Nell

    # Fill ℓ < ℓ_min with the first value (not important for small patches)
    ell_min = ell_int[0]
    if ell_min > 0:
        Cl[:ell_min] = Nell[0]

    return Cl, lmax


def make_patch_geometry(size_deg, npix, ra0_deg, dec0_deg):
    """
    Create a CAR patch geometry (shape, wcs) for pixell.enmap.

    Returns:
        shape, wcs
    """
    res_rad = (size_deg * utils.degree) / npix

    ra0  = ra0_deg  * utils.degree
    dec0 = dec0_deg * utils.degree

    dec_min = dec0 - (size_deg / 2.0) * utils.degree
    dec_max = dec0 + (size_deg / 2.0) * utils.degree
    ra_min  = ra0  - (size_deg / 2.0) * utils.degree
    ra_max  = ra0  + (size_deg / 2.0) * utils.degree

    shape, wcs = enmap.geometry(
        pos=[[dec_min, ra_min],
             [dec_max, ra_max]],
        res=res_rad,
        proj="car"
    )

    return shape, wcs


def generate_noise_healpix(Cl_noise, nside, lmax, seed=None):
    """
    Generate one full-sky Gaussian HEALPix noise map with power spectrum Cl_noise.

    Returns:
        noise_map : float32 array (HEALPix)
    """
    if seed is not None:
        np.random.seed(seed)

    if lmax > 3 * nside - 1:
        raise ValueError(
            f"lmax={lmax} is too large for NSIDE={nside}. "
            f"Max allowed is {3*nside-1}."
        )

    noise_map = hp.synfast(
        Cl_noise,
        nside=nside,
        lmax=lmax,
        new=True,
        verbose=False
    )

    return noise_map.astype(np.float32)


def reproject_healpix_to_patch(hp_map, shape, wcs):
    """
    Project a HEALPix map onto a CAR patch using pixell.

    Returns:
        patch : float32 enmap array
    """
    patch = enmap.empty(shape, wcs, dtype=np.float32)
    patch = reproject.healpix2map(hp_map, patch, unit=1.0)
    return patch.astype(np.float32)


def ensure_dir(path):
    """Create directory if needed (only rank 0 calls this)."""
    if rank == 0 and not os.path.isdir(path):
        os.makedirs(path)
    # Ensure all ranks see the directory
    comm.Barrier()


def distribute_indices(n_total, size, rank):
    """
    Distribute [0, n_total) across MPI ranks.

    Returns:
        start_idx, end_idx  (end_idx is exclusive)
    """
    # Base number per rank
    n_base = n_total // size
    # Remainder: the first 'remainder' ranks get one extra
    remainder = n_total % size

    if rank < remainder:
        start = rank * (n_base + 1)
        end   = start + n_base + 1
    else:
        start = remainder * (n_base + 1) + (rank - remainder) * n_base
        end   = start + n_base

    return start, end


# =========================
# MAIN SCRIPT
# =========================

def main():

    if rank == 0:
        print(f"[MPI] Running with {size} ranks")

    # 1. Load N_ell and build Cl_noise (rank 0) then broadcast
    if rank == 0:
        print(f"[rank 0] Loading N_ell from {NELL_FILE} ...")
        ell, Nell = load_nell_tsz(NELL_FILE)
        Cl_noise, lmax = build_cl_noise_from_nell(ell, Nell, lmax_clip=LMAX_CLIP)
        meta = np.array([ell.min(), ell.max(), lmax], dtype=float)
    else:
        ell = Nell = Cl_noise = None
        meta = np.zeros(3, dtype=float)

    # Broadcast metadata and Cl_noise to all ranks
    meta = comm.bcast(meta, root=0)
    lmin_val, lmax_file_val, lmax = meta
    lmax = int(lmax)

    if rank != 0:
        # Allocate Cl_noise on non-root ranks
        Cl_noise = np.empty(lmax + 1, dtype=float)
    comm.Bcast(Cl_noise, root=0)

    if rank == 0:
        print(f"  -> ell range in file: {lmin_val:.0f} .. {lmax_file_val:.0f}")
        print(f"  -> using lmax = {lmax}")
        print(f"  -> NSIDE = {NSIDE}")

    # 2. Build patch geometry (all ranks do this locally)
    shape, wcs = make_patch_geometry(
        PATCH_SIZE_DEG,
        PATCH_NPIX,
        PATCH_RA0_DEG,
        PATCH_DEC0_DEG
    )
    if rank == 0:
        print(f"  -> patch shape = {shape} pixels ({PATCH_SIZE_DEG}° / {shape[-1]} px)")

    # 3. Ensure output directory exists
    ensure_dir(OUTDIR_PATCH)

    # 4. Distribute map indices among MPI ranks
    start_idx, end_idx = distribute_indices(N_MAPS, size, rank)
    n_local = end_idx - start_idx

    if rank == 0:
        print(f"Distributing {N_MAPS} maps across {size} ranks")
    print(f"[rank {rank}] handling indices {start_idx} .. {end_idx-1} (total {n_local})")

    # 5. Local loop for each rank
    for i_global in range(start_idx, end_idx):
        seed_i = BASE_SEED + i_global

        # 5.1 Full-sky noise
        noise_hp = generate_noise_healpix(
            Cl_noise,
            nside=NSIDE,
            lmax=lmax,
            seed=seed_i
        )

        # 5.2 Reproject to 10°×10° patch
        noise_patch = reproject_healpix_to_patch(noise_hp, shape, wcs)

        # 5.3 Save patch
        out_patch = os.path.join(OUTDIR_PATCH, f"noise_tsz_SO_patch_{i_global:04d}.npy")
        np.save(out_patch, np.asarray(noise_patch))

        # Optional: progress per rank
        if (i_global - start_idx + 1) % 50 == 0 or i_global == start_idx:
            print(f"[rank {rank}] -> {i_global+1}/{N_MAPS} total maps done")

    # Barrier for clean finish
    comm.Barrier()
    if rank == 0:
        print("All ranks finished. Done.")


if __name__ == "__main__":
    main()
