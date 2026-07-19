#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import healpy as hp
from pixell import enmap, reproject

# ============================================================
# 1) Lecture du fichier N_ell Simons Observatory (tSZ ILC)
# ============================================================

def load_nell_tsz(nell_file, deproj=0):
    """
    Charge le fichier SO N_ell.
    Par défaut, on prend la colonne 'Deproj-0' (standard ILC).

    Paramètres
    ----------
    nell_file : str
        Chemin vers SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt
    deproj : int
        0 → Deproj-0, 1 → Deproj-1, 2 → Deproj-2.

    Retour
    ------
    ell : array (float)
    Nell : array (float)
    """
    data = np.loadtxt(nell_file)
    ell  = data[:, 0]
    Nell = data[:, 1 + deproj]  # 1: Deproj-0, 2: Deproj-1, 3: Deproj-2
    return ell, Nell


def build_cl_noise_from_nell(ell, Nell, lmax_clip=None):
    """
    Construit C_ell^noise = N_ell (on suppose que le fichier donne bien C_ell).

    Paramètres
    ----------
    ell, Nell : arrays
        Colonnes du fichier N_ell.
    lmax_clip : int ou None
        Si non-None, coupe à ce lmax.

    Retour
    ------
    Cl_noise : array de taille (lmax+1,)
    lmax : int
    """
    ell_int   = ell.astype(int)
    lmax_file = int(ell_int.max())

    if lmax_clip is not None:
        lmax = min(lmax_file, int(lmax_clip))
        mask = ell_int <= lmax
        ell_int = ell_int[mask]
        Nell    = Nell[mask]
    else:
        lmax = lmax_file

    Cl = np.zeros(lmax + 1, dtype=float)
    Cl[ell_int] = Nell

    # Remplit les très bas multipôles (pas critique pour un patch 10°x10°)
    ell_min = ell_int[0]
    if ell_min > 0:
        Cl[:ell_min] = Nell[0]

    return Cl, lmax


# ============================================================
# 2) Génération d'un bruit full-sky HEALPix
# ============================================================

def generate_noise_healpix(Cl_noise, nside, lmax, seed=None):
    """
    Génère une carte HEALPix de bruit gaussien avec spectre Cl_noise.
    """
    if seed is not None:
        np.random.seed(seed)

    if lmax > 3 * nside - 1:
        raise ValueError(
            f"lmax={lmax} trop grand pour NSIDE={nside} (max={3*nside-1})."
        )

    hp_map = hp.synfast(
        Cl_noise,
        nside=nside,
        lmax=lmax,
        new=True,
        verbose=False
    )
    return hp_map.astype(np.float32)


# ============================================================
# 3) Reprojection du bruit sur la même géométrie que TA carte
# ============================================================

def noise_patch_from_nell(
    nell_file,
    template_map,
    nside=2048,
    lmax_clip=None,
    seed=None,
    deproj=0
):
    """
    Génère un patch de bruit tSZ SO sur la géométrie de `template_map`.

    Paramètres
    ----------
    nell_file : str
        Fichier N_ell.
    template_map : enmap.ndmap
        Carte pixell (shape, wcs) servant de modèle (10°x10° y-map).
    nside : int
        NSIDE pour hp.synfast.
    lmax_clip : int ou None
        lmax optionnel pour couper le spectre (sinon 3*nside-1).
    seed : int ou None
        Graine RNG.
    deproj : int
        Colonne de N_ell (0, 1 ou 2).

    Retour
    ------
    noise_patch : enmap.ndmap (même shape/wcs que template_map)
    """
    # 1) Charge N_ell
    ell, Nell = load_nell_tsz(nell_file, deproj=deproj)

    # 2) lmax_clip par défaut = 3*nside-1 (contrainte healpy)
    if lmax_clip is None:
        lmax_clip = 3 * nside - 1

    # 3) Construit C_ell^noise en respectant lmax_clip
    Cl_noise, lmax = build_cl_noise_from_nell(ell, Nell, lmax_clip=lmax_clip)

    # 4) Full-sky bruit HEALPix
    hp_noise = generate_noise_healpix(Cl_noise, nside=nside, lmax=lmax, seed=seed)

    # 5) Reprojection HEALPix -> patch CAR sur la géométrie de template_map
    shape = template_map.shape
    wcs   = template_map.wcs

    # Version pixell 0.20.4 : healpix2map(iheal, shape, wcs, ...)
    noise_patch = reproject.healpix2map(
        hp_noise,
        shape,
        wcs,
    )

    return noise_patch.astype(np.float32)


# ============================================================
# 4) Calcul du spectre D_ell pour une enmap (équivalent _cl_from_map)
# ============================================================

def cl_from_enmap(
    imap,
    bsize=300,
    max_ell=10_000,
    apod_width=100,
    unit_scale=1e12,
    normalize='phys',
    area_weighted=True
):
    """
    Calcule D_ell pour une carte pixell.enmap (patch) SANS passer par les fichiers .fits.

    C'est le même pipeline que dans compute_dell_empirical :
     - apodisation cosinus
     - FFT (normalize='phys')
     - ell binning
     - <w^2> correction
     - conversion C_ell -> D_ell = ell(ell+1)/(2π) C_ell * unit_scale

    Retour
    ------
    ell_b : array
    D_ell_b : array
    """
    from pixell import enmap as _enmap
    import numpy as _np

    # 1) Masque d'apodisation et carte apodisée
    taper_mask = _enmap.apod(_enmap.ones(imap.shape, imap.wcs), width=apod_width)
    imap_apod  = imap.apod(width=apod_width)

    # 2) FFT & power 2D
    kmap  = _enmap.fft(imap_apod, normalize=normalize)  # <-- 'phys' vient du param normalize
    cl_2d = _np.abs(kmap)**2

    # 3) Binning en ell
    Cl_b, ell_b = _enmap.lbin(cl_2d, bsize=bsize)

    # 4) <w^2> (area-weighted)
    pix_area = imap.pixsizemap(separable=False, broadcastable=False)
    if area_weighted:
        w2 = _np.sum(pix_area * taper_mask**2) / _np.sum(pix_area)
    else:
        w2 = _np.mean(taper_mask**2)
    if not _np.isfinite(w2) or w2 <= 0:
        raise ValueError(f"<w^2> invalide : {w2}")

    Cl_b = Cl_b / w2

    # 5) Coupe en ell
    m     = ell_b <= max_ell
    ell_b = ell_b[m]
    Cl_b  = Cl_b[m]

    # 6) D_ell
    D_b =  1e12*ell_b * (ell_b + 1) / (2.0 * _np.pi) * Cl_b * unit_scale

    if not (_np.all(_np.isfinite(ell_b)) and _np.all(_np.isfinite(D_b))):
        raise ValueError("NaN/Inf dans ell ou D_ell")

    return ell_b, D_b
