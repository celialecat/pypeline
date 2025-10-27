import os
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import time

TWOPI = 2*np.pi

def wrap_pm_pi(x):
    """Wrap angle x (rad) to [-π, π]."""
    return (x + np.pi) % (2*np.pi) - np.pi

def sample_lonlat_patch(n, ra0_deg=0.0, dec0_deg=0.0, w_deg=10.0, h_deg=10.0, rng=None):
    """
    Sample n uniform-in-area points in a w×h (degrees) patch centered at (ra0_deg, dec0_deg).
    Returns:
      - lon (rad) in [-π, π]
      - theta (colatitude, rad) in [0, π]
    """
    if rng is None:
        rng = np.random.default_rng()
    w = np.deg2rad(w_deg)
    h = np.deg2rad(h_deg)

    lon0   = np.deg2rad(ra0_deg)              # RA center (rad)
    theta0 = np.pi/2.0 - np.deg2rad(dec0_deg) # colatitude center (rad)

    # Uniform longitude then wrap to [-π, π]
    lon = lon0 - w/2.0 + rng.random(n) * w
    lon = wrap_pm_pi(lon)

    # Colatitude uniform in area (uniform in cos(theta))
    theta1 = np.clip(theta0 - h/2.0, 0.0, np.pi)
    theta2 = np.clip(theta0 + h/2.0, 0.0, np.pi)
    tmin, tmax = (theta1, theta2) if theta1 <= theta2 else (theta2, theta1)

    u = rng.random(n)
    cos_t = np.cos(tmax) + u * (np.cos(tmin) - np.cos(tmax))
    theta = np.arccos(np.clip(cos_t, -1.0, 1.0))

    return lon, theta

def _seed_from_path(path: Path, base_seed: int | None):
    """Generate a per-file stable seed from file path + base_seed (or None → random)."""
    if base_seed is None:
        return None
    h = hashlib.sha256((str(path.resolve()) + f"|{base_seed}").encode()).digest()
    return int.from_bytes(h[:8], "little", signed=False)  # 64-bit seed

def process_catalogues(
    inputs,
    out_dir: str | Path | None = None,
    ra0_deg: float = 0.0,
    dec0_deg: float = 0.0,
    w_deg: float = 10.0,
    h_deg: float = 10.0,
    seed: int | None = 42,
    pattern: str = "*.csv",
    recursive: bool = True,
    overwrite: bool = False,
    lon_wrap: str = "pm_pi",     # "pm_pi" => [-π,π], "0_2pi" => [0,2π)
    out_suffix: str = "_with_coords",
    lon_col: str = "lon",
    lat_col: str = "lat",
):
    start_time = time.time()

    """
    Process one or several paths (CSV files or folders) and generate output CSVs
    containing two additional columns:
      - lon : longitude (rad)
      - lat : colatitude (rad)  (NB: this is the colatitude, not declination)

    Parameters
    ----------
    inputs : str | Path | list[str|Path]
        Path(s) to individual CSVs or directories containing CSVs.
    out_dir : str | Path | None
        Output root directory. If None, writes next to source files.
    ra0_deg, dec0_deg : float
        Center of the patch (degrees).
    w_deg, h_deg : float
        Width / height of the patch (degrees).
    seed : int | None
        Base seed. A per-file derived seed is used for reproducibility.
        If None, random sampling.
    pattern : str
        Search pattern for CSVs in folders (default *.csv).
    recursive : bool
        Whether to search recursively in subdirectories.
    overwrite : bool
        If False and output file already exists, skip it.
    lon_wrap : {"pm_pi","0_2pi"}
        Longitude format to write.
    out_suffix : str
        Suffix inserted before the extension (e.g., foo.csv -> foo_with_coords.csv).
    lon_col, lat_col : str
        Names of the columns to write.

    Returns
    -------
    summary : list[dict]
        List of per-file summaries: {"in":..., "out":..., "n":..., "skipped": bool}
    """
    # Normalize the list of inputs
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    inputs = [Path(p) for p in inputs]

    # Collect final list of CSVs to process
    csv_files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            if recursive:
                csv_files.extend(sorted(p.rglob(pattern)))
            else:
                csv_files.extend(sorted(p.glob(pattern)))
        elif p.is_file() and p.suffix.lower() == ".csv":
            csv_files.append(p)
        else:
            # Silently ignore everything else
            pass

    if not csv_files:
        print("No CSV found.")
        return []

    out_root = Path(out_dir).resolve() if out_dir is not None else None
    summary = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            summary.append({"in": str(f), "out": None, "n": 0, "skipped": True, "reason": f"read_error: {e}"})
            continue

        n = len(df)
        if n == 0:
            summary.append({"in": str(f), "out": None, "n": 0, "skipped": True, "reason": "empty_csv"})
            continue

        # Build output path
        if out_root is None:
            out_path = f.with_name(f.stem + out_suffix + f.suffix)
        else:
            rel = f.name if f.is_file() else f.as_posix()
            # Keeps the file name; to reproduce the directory tree, replace with:
            # rel = f.relative_to(common_root).as_posix()
            out_path = out_root / (Path(f.stem + out_suffix + f.suffix).name)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            summary.append({"in": str(f), "out": str(out_path), "n": n, "skipped": True, "reason": "exists"})
            continue

        # Per-file (stable) seed
        file_seed = _seed_from_path(f, seed)
        rng = np.random.default_rng(file_seed) if file_seed is not None else np.random.default_rng()

        # Sample and write
        lon, theta = sample_lonlat_patch(n, ra0_deg=ra0_deg, dec0_deg=dec0_deg,
                                         w_deg=w_deg, h_deg=h_deg, rng=rng)

        if lon_wrap == "0_2pi":
            lon = lon % TWOPI
        elif lon_wrap == "pm_pi":
            lon = wrap_pm_pi(lon)
        else:
            raise ValueError("lon_wrap must be 'pm_pi' or '0_2pi'.")

        df[lon_col] = lon
        df[lat_col] = theta

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
        except Exception as e:
            summary.append({"in": str(f), "out": str(out_path), "n": n, "skipped": True, "reason": f"write_error: {e}"})
            continue

        summary.append({"in": str(f), "out": str(out_path), "n": n, "skipped": False, "reason": None})

    # Small readable recap
    done = sum(1 for s in summary if not s["skipped"])
    skipped = len(summary) - done
    print(f"Done: {done} written, {skipped} skipped.")

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds.")
    return summary
