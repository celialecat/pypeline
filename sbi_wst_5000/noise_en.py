import numpy as np
import healpy as hp
from pixell import enmap, reproject


def load_nell_tsz(nell_file, deproj=0):
    """
    Load the SO N_ell file.
    By default, use the 'Deproj-0' column (standard ILC).

    Parameters
    ----------
    nell_file : str
        Path to SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt
    deproj : int
        0 → Deproj-0, 1 → Deproj-1, 2 → Deproj-2.

    Returns
    -------
    ell : array (float)
    Nell : array (float)
    """
    data = np.loadtxt(nell_file)
    ell  = data[:, 0]
    Nell = data[:, 1 + deproj]
    return ell, Nell


def build_cl_noise_from_nell(ell, Nell, lmax_clip=None):
    """
    Build C_ell^noise = N_ell (assuming the file already provides C_ell).

    Parameters
    ----------
    ell, Nell : arrays
        Columns from the N_ell file.
    lmax_clip : int or None
        If not None, truncate at this lmax.

    Returns
    -------
    Cl_noise : array of size (lmax+1,)
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

    ell_min = ell_int[0]
    if ell_min > 0:
        Cl[:ell_min] = Nell[0]

    return Cl, lmax


def generate_noise_healpix(Cl_noise, nside, lmax, seed=None):
    """
    Generate a Gaussian HEALPix noise map with power spectrum Cl_noise.
    """
    if seed is not None:
        np.random.seed(seed)

    if lmax > 3 * nside - 1:
        raise ValueError(
            f"lmax={lmax} too large for NSIDE={nside} (max={3*nside-1})."
        )

    hp_map = hp.synfast(
        Cl_noise,
        nside=nside,
        lmax=lmax,
        new=True,
        verbose=False
    )
    return hp_map.astype(np.float32)


def noise_patch_from_nell(
    nell_file,
    template_map,
    nside=2048,
    lmax_clip=None,
    seed=None,
    deproj=0
):
    """
    Generate a Simons Observatory tSZ noise patch
    on the geometry of `template_map`.

    Parameters
    ----------
    nell_file : str
        N_ell file.
    template_map : enmap.ndmap
        Pixell map (shape, wcs) used as a template (10°x10° y-map).
    nside : int
        NSIDE for hp.synfast.
    lmax_clip : int or None
        Optional lmax cut (otherwise 3*nside-1).
    seed : int or None
        RNG seed.
    deproj : int
        N_ell column (0, 1 or 2).

    Returns
    -------
    noise_patch : enmap.ndmap
        Same shape/wcs as template_map.
    """
    ell, Nell = load_nell_tsz(nell_file, deproj=deproj)

    if lmax_clip is None:
        lmax_clip = 3 * nside - 1

    Cl_noise, lmax = build_cl_noise_from_nell(ell, Nell, lmax_clip=lmax_clip)

    hp_noise = generate_noise_healpix(
        Cl_noise,
        nside=nside,
        lmax=lmax,
        seed=seed
    )

    noise_patch = reproject.healpix2map(
        hp_noise,
        template_map.shape,
        template_map.wcs,
    )

    return noise_patch.astype(np.float32)


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
    Compute D_ell for a pixell.enmap (patch) WITHOUT using .fits files.

    Pipeline:
     - cosine apodization
     - FFT (normalize='phys')
     - ell binning
     - <w^2> correction
     - conversion C_ell → D_ell = ell(ell+1)/(2π) C_ell * unit_scale

    Returns
    -------
    ell_b : array
    D_ell_b : array
    """
    from pixell import enmap as _enmap
    import numpy as _np

    taper_mask = _enmap.apod(
        _enmap.ones(imap.shape, imap.wcs),
        width=apod_width
    )
    imap_apod = imap.apod(width=apod_width)

    kmap  = _enmap.fft(imap_apod, normalize=normalize)
    cl_2d = _np.abs(kmap)**2

    Cl_b, ell_b = _enmap.lbin(cl_2d, bsize=bsize)

    pix_area = imap.pixsizemap(
        separable=False,
        broadcastable=False
    )
    if area_weighted:
        w2 = _np.sum(pix_area * taper_mask**2) / _np.sum(pix_area)
    else:
        w2 = _np.mean(taper_mask**2)

    if not _np.isfinite(w2) or w2 <= 0:
        raise ValueError(f"Invalid <w^2> : {w2}")

    Cl_b = Cl_b / w2

    m     = ell_b <= max_ell
    ell_b = ell_b[m]
    Cl_b  = Cl_b[m]

    D_b = (
        ell_b * (ell_b + 1)
        / (2.0 * _np.pi)
        * Cl_b
        * unit_scale
    )

    if not (
        _np.all(_np.isfinite(ell_b))
        and _np.all(_np.isfinite(D_b))
    ):
        raise ValueError("NaN/Inf in ell or D_ell")

    return ell_b, D_b
