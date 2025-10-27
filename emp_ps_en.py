import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap


def _fits_list(path_like):
    """Return the list of .fits paths from either a single file or a directory."""
    if os.path.isdir(path_like):
        files = sorted(glob.glob(os.path.join(path_like, "*.fits"))) + \
                sorted(glob.glob(os.path.join(path_like, "*.fits.gz")))
        if not files:
            raise FileNotFoundError(f"No .fits found in the folder: {path_like}")
        return files
    elif os.path.isfile(path_like):
        if not (path_like.endswith(".fits") or path_like.endswith(".fits.gz")):
            raise ValueError(f"files non-FITS given: {path_like}")
        return [path_like]
    else:
        raise FileNotFoundError(f"file not found at this path : {path_like}")


def _area_weighted_mean_w2(taper_mask, pix_area, area_weighted=True):
    """Compute <w^2>. By default, performs an area-weighted average over pixels."""
    w2 = (np.sum(pix_area * taper_mask**2) / np.sum(pix_area)) if area_weighted else np.mean(taper_mask**2)
    return float(w2)


def _cl_from_map(imap, bsize=300, apod_width=100, max_ell=10_000, normalize='phys', area_weighted=True):
    """
    Compute binned C_ell and ell_b for a patch:
      - apodization (cosine taper)
      - FFT (normalize='phys')
      - ell binning (enmap.lbin)
      - <w^2> correction
      - cut at ell <= max_ell

    Returns: ell_b_cut, C_ell_cut, fsky_full, fsky_eff
    """
    # Apodization mask (same WCS/shape as the map)
    taper_mask = enmap.apod(enmap.ones(imap.shape, imap.wcs), width=apod_width)
    # Apply apodization to the map
    imap_apod = imap.apod(width=apod_width)

    # FFT & 2D power
    kmap = enmap.fft(imap_apod, normalize=normalize)
    cl_2d = np.abs(kmap)**2

    # Bin in ell
    Cl_b, ell_b = enmap.lbin(cl_2d, bsize=bsize)

    # <w^2> correction (area-weighted by default)
    pix_area = imap.pixsizemap(separable=False, broadcastable=False)  # [sr], same shape as the map
    w2 = _area_weighted_mean_w2(taper_mask, pix_area, area_weighted=area_weighted)
    if not np.isfinite(w2) or w2 <= 0:
        raise ValueError(f"w^2 not valid: {w2}")
    Cl_b = Cl_b / w2

    # f_sky (useful for logs/diagnostics)
    area_full = np.sum(pix_area)                    # [sr]
    fsky_full = area_full / (4*np.pi)
    area_eff  = np.sum(pix_area * taper_mask**2)    # [sr]
    fsky_eff  = area_eff / (4*np.pi)

    # ell cut
    m = ell_b <= max_ell
    ell_b_cut = ell_b[m]
    Cl_b_cut  = Cl_b[m]

    if not (np.all(np.isfinite(ell_b_cut)) and np.all(np.isfinite(Cl_b_cut))):
        raise ValueError("NaN/Inf detected in ell or C_ell (after correction)")

    return ell_b_cut, Cl_b_cut, float(fsky_full), float(fsky_eff)


def compute_dell_empiriques(
    path_like,
    bsize=300,
    max_ell=10_000,
    apod_width=100,
    unit_scale=1e12,
    normalize='phys',
    area_weighted=True,
    plot=False,
    overlay_theory=None,   # tuple (ell_theory, D_ell_theory) or None
    label='Patches',
    save_csv=None,         # CSV path OR directory (in grouped mode)
    save_plot=None,        # image path OR directory (in grouped mode)
    quiet=False
):
    """
    Compute empirical D_ell from a patch .fits, a directory of patch .fits
    **or a directory containing subdirectories (one per cosmology)**.

    Pipeline (per patch):
      - apodization + normalized FFT ('phys')
      - ell binning
      - <w^2> correction
      - conversion C_ell -> D_ell = ell(ell+1)/(2π) C_ell * unit_scale
      - cross-patch aggregation: mean and standard deviation (ddof=1)

    Grouped mode:
      - If `path_like` is a directory WITHOUT direct FITS but WITH subdirectories,
        each subdirectory containing .fits is treated as a cosmology.
      - We write a **detailed CSV per cosmology** if `save_csv` is provided (file with suffix
        or to a directory), and a **summary CSV (mean/std) per cosmology**.  # [NEW]

    Detailed CSV (compatibility & covariance):
      - Columns: `ell`, `D_ell_mean`, `D_ell_std`, then `D_ell_patch{i}` for i=0..N-1.
      - A commented header includes metadata and the index↔file mapping.

    Returns also an execution time:
      - In simple mode: results['runtime_sec']
      - In grouped mode: top-level dict includes 'runtime_sec_total'
    """
    t0_group = time.perf_counter()

    # Internal helpers for output paths in grouped mode
    def _derive_csv_path(base, cosmo):
        if base is None:
            return None
        base = os.fspath(base)
        root, ext = os.path.splitext(base)
        if ext.lower() == '.csv':
            return f"{root}_{cosmo}.csv"
        # otherwise treat as a directory
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{cosmo}.csv")

    def _derive_plot_path(base, cosmo):
        if base is None:
            return None
        base = os.fspath(base)
        root, ext = os.path.splitext(base)
        img_exts = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}
        if ext.lower() in img_exts:
            return f"{root}_{cosmo}{ext}"
        # otherwise treat as a directory
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{cosmo}.png")

    # [NEW] path for “summary” CSV (mean/std) per cosmology
    def _derive_csv_meanstd_path(base, cosmo):  # [NEW]
        if base is None:
            return None
        base = os.fspath(base)
        root, ext = os.path.splitext(base)
        if ext.lower() == '.csv':
            return f"{root}_{cosmo}_meanstd.csv"
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{cosmo}_meanstd.csv")

    # Detect grouped mode: directory without direct FITS but with subdirectories containing FITS
    if os.path.isdir(path_like):
        direct_fits = sorted(glob.glob(os.path.join(path_like, "*.fits"))) + \
                      sorted(glob.glob(os.path.join(path_like, "*.fits.gz")))
        if len(direct_fits) == 0:
            subdirs = sorted([d.path for d in os.scandir(path_like) if d.is_dir()])
            cosmo_dirs = []
            for d in subdirs:
                ff = glob.glob(os.path.join(d, "*.fits")) + glob.glob(os.path.join(d, "*.fits.gz"))
                if ff:
                    cosmo_dirs.append(d)
            if cosmo_dirs:
                if not quiet:
                    print(f"Grouped mode detected: {len(cosmo_dirs)} subfolders containing cosmologies found")
                results_by_cosmo = {}
                for sd in cosmo_dirs:
                    cosmo = os.path.basename(sd.rstrip(os.sep))
                    if not quiet:
                        print(f"\n=== Cosmology: {cosmo} ===")
                    res = compute_dell_empiriques(
                        path_like=sd,
                        bsize=bsize,
                        max_ell=max_ell,
                        apod_width=apod_width,
                        unit_scale=unit_scale,
                        normalize=normalize,
                        area_weighted=area_weighted,
                        plot=plot,
                        overlay_theory=overlay_theory,
                        label=cosmo,
                        save_csv=_derive_csv_path(save_csv, cosmo),
                        save_plot=_derive_plot_path(save_plot, cosmo),
                        quiet=quiet,
                    )
                    results_by_cosmo[cosmo] = res

                    # [NEW] Write “mean/std” CSV per cosmology
                    meanstd_csv = _derive_csv_meanstd_path(save_csv, cosmo)
                    if meanstd_csv is not None:
                        try:
                            import pandas as pd
                            df_summary = pd.DataFrame({
                                'ell': res['ell'],
                                'D_ell_mean': res['D_ell_mean'],
                                'D_ell_std': res['D_ell_std'],
                            })
                            out_dir = os.path.dirname(meanstd_csv)
                            if out_dir and not os.path.exists(out_dir):
                                os.makedirs(out_dir, exist_ok=True)

                            header = (
                                f"# SUMMARY (mean/std) for cosmology={cosmo}\n"
                                f"# n_maps={res['n_maps']}, bsize={bsize}, max_ell={max_ell}, apod_width={apod_width}, "
                                f"unit_scale={unit_scale}, normalize={normalize}, area_weighted={area_weighted}\n"
                                f"# fsky_eff_mean={res['fsky_eff_mean']:.6e}, fsky_eff_std={res['fsky_eff_std']:.6e}\n"
                                f"# fsky_full_mean={res['fsky_full_mean']:.6e}, fsky_full_std={res['fsky_full_std']:.6e}"
                            )
                            # Write with commented header
                            df_summary.to_csv(meanstd_csv, index=False)
                            with open(meanstd_csv, "r+", encoding='utf-8') as f:
                                content = f.read()
                                f.seek(0, 0)
                                f.write(header + "\n" + content)
                            if not quiet:
                                print(f"Summary mean/std written: {meanstd_csv}")
                        except Exception as e:
                            print(f"[WARN] Impossible to write the csv summary for {cosmo}: {e}")

                total_elapsed = time.perf_counter() - t0_group
                if not quiet:
                    print(f"\nRuntime (grouped mode): {total_elapsed:.3f} s")
                return {
                    'mode': 'group',
                    'per_cosmo': results_by_cosmo,
                    'cosmologies': list(results_by_cosmo.keys()),
                    'n_cosmologies': len(results_by_cosmo),
                    'runtime_sec_total': float(total_elapsed),
                }
            # else: no direct FITS and no valid subfolders → fall through to error later

    # ------ “Simple” mode unchanged (single file or directory of FITS) ------
    t0_simple = time.perf_counter()
    files = _fits_list(path_like)
    n_maps = len(files)

    ell_ref = None
    D_all = []
    fsky_full_list = []
    fsky_eff_list  = []

    for i, f in enumerate(files):
        imap = enmap.read_map(f)
        ell_b, Cl_b, fsky_full, fsky_eff = _cl_from_map(
            imap, bsize=bsize, apod_width=apod_width,
            max_ell=max_ell, normalize=normalize, area_weighted=area_weighted
        )
        # Convert to D_ell
        D_b = ell_b*(ell_b+1)/(2*np.pi) * Cl_b * unit_scale

        if ell_ref is None:
            ell_ref = ell_b
        else:
            # Enforce same ell grid to stack/aggregate
            if not (len(ell_b) == len(ell_ref) and np.allclose(ell_b, ell_ref, rtol=0, atol=1e-8)):
                raise ValueError(
                    "The ell grids are not the same for every patch "
                    "Check that the patches have the same shape/WCS and that bsize is identical."
                )

        if not np.all(np.isfinite(D_b)):
            raise ValueError(f"NaN/Inf in D_ell for the file: {f}")

        D_all.append(D_b)
        fsky_full_list.append(fsky_full)
        fsky_eff_list.append(fsky_eff)

        if not quiet:
            print(f"[{i+1}/{n_maps}] {os.path.basename(f)}  f_sky(full)={fsky_full:.4e}  f_sky(eff)={fsky_eff:.4e}")

    D_all = np.asarray(D_all)  # shape: (n_maps, n_ell)

    # Patch statistics
    D_mean = np.mean(D_all, axis=0)
    # ddof=1 for sample std if n_maps>1; else zeros
    D_std  = np.std(D_all, axis=0, ddof=1) if n_maps > 1 else np.zeros_like(D_mean)

    fsky_full_arr = np.array(fsky_full_list)
    fsky_eff_arr  = np.array(fsky_eff_list)

    results = {
        'ell': ell_ref,
        'D_ell_mean': D_mean,
        'D_ell_std': D_std,
        'D_ell_stack': D_all,  # per-patch stack (N, d)
        'n_maps': n_maps,
        'fsky_eff_mean': float(np.mean(fsky_eff_arr)),
        'fsky_eff_std':  float(np.std(fsky_eff_arr, ddof=1) if n_maps > 1 else 0.0),
        'fsky_full_mean': float(np.mean(fsky_full_arr)),
        'fsky_full_std':  float(np.std(fsky_full_arr, ddof=1) if n_maps > 1 else 0.0),
        'files': files,
        'label': label,
    }

    # Detailed CSV save — with columns per patch
    if save_csv is not None:
        import pandas as pd

        # Patch columns: one column per patch (N_maps)
        patch_cols = {f"D_ell_patch{i}": results["D_ell_stack"][i] for i in range(n_maps)}

        df = pd.DataFrame({
            'ell': results['ell'],
            'D_ell_mean': results['D_ell_mean'],
            'D_ell_std': results['D_ell_std'],
            **patch_cols
        })

        # Header with metadata + mapping patch index -> filename
        file_map_lines = [f"# patch_index={i} file={os.path.basename(fn)}" for i, fn in enumerate(files)]
        header = (
            f"# n_maps={n_maps}, bsize={bsize}, max_ell={max_ell}, apod_width={apod_width}, "
            f"unit_scale={unit_scale}, normalize={normalize}, area_weighted={area_weighted}\n"
            f"# fsky_eff_mean={results['fsky_eff_mean']:.6e}, fsky_eff_std={results['fsky_eff_std']:.6e}\n"
            f"# fsky_full_mean={results['fsky_full_mean']:.6e}, fsky_full_std={results['fsky_full_std']:.6e}\n" +
            "\n".join(file_map_lines)
        )

        # Create directory if a directory path was given
        out_csv = save_csv
        out_dir = os.path.dirname(out_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Write: first without header, then prepend commented header
        df.to_csv(out_csv, index=False)
        with open(out_csv, "r+", encoding='utf-8') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + "\n" + content)

    # Plotting
    if plot:
        plt.figure(figsize=(6.5, 4.5))
        yerr = results['D_ell_std'] if n_maps > 1 else None
        plt.errorbar(results['ell'], results['D_ell_mean'], yerr=yerr, fmt='o', ms=3, lw=1, capsize=2, label=f"{label} (N={n_maps})")

        if overlay_theory is not None:
            ell_th, D_th = overlay_theory
            plt.plot(ell_th, D_th, linestyle='dashed', label='Théorie')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$')
        plt.title('Angular power spectrum with error bars')
        plt.legend()
        plt.tight_layout()
        if save_plot is not None:
            out_dir = os.path.dirname(save_plot)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            plt.savefig(save_plot, dpi=200)
        plt.show()

    elapsed = time.perf_counter() - t0_simple
    results['runtime_sec'] = float(elapsed)

    if not quiet:
        print(f"Executed. Patches: {n_maps} | mean f_sky(eff) = {results['fsky_eff_mean']:.4e}")
        print(f"Runtime: {elapsed:.3f} s")

    return results
