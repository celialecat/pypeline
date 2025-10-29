import os
import time
import json
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import sys
import pandas as pd
from numpy.lib import recfunctions as rfn
from importlib import invalidate_caches, reload
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import importlib

# -----------------------------------------------------------------------------
# Monitoring via psutil (+ threading) — self-contained "resources" helpers
# -----------------------------------------------------------------------------
import psutil
import threading
from collections import deque
from dataclasses import dataclass, asdict

BYTES_IN_MB = 1024 ** 2

# ---- Quick SLURM helpers -----------------------------------------------------
def _slurm_cpu_guess() -> Optional[int]:
    for k in ("SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS"):
        v = os.environ.get(k)
        if v:
            try:
                # Handle values like "64(x2)" by extracting the first integer
                digits = ''.join(ch for ch in v if ch.isdigit())
                return int(digits) if digits else None
            except Exception:
                pass
    return None

_proc = psutil.Process()
_cpu_logical = psutil.cpu_count() or 1
_cpu_physical = psutil.cpu_count(logical=False)
try:
    _ = _proc.cpu_percent(None)  # prime CPU percentage counters
except Exception:
    pass

@dataclass
class ProcSnapshot:
    ts: float
    cpu_percent: float
    cpu_count_logical: int
    cpu_count_physical: Optional[int]
    cpu_affinity: Optional[List[int]]
    loadavg_1m: Optional[float]
    rss_mb: float
    uss_mb: Optional[float]
    vms_mb: float
    mem_percent: float
    num_threads: int
    open_files: int
    read_bytes_mb: float
    write_bytes_mb: float
    read_count: int
    write_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def advertised_cpu_capacity() -> int:
    """Number of CPUs the process is expected to use (affinity > SLURM > logical)."""
    try:
        aff = _proc.cpu_affinity()
        if aff:
            return max(1, len(aff))
    except Exception:
        pass
    return _slurm_cpu_guess() or (_cpu_logical or 1)


def take_snapshot() -> ProcSnapshot:
    p = _proc
    now = time.time()
    try:
        mem = p.memory_full_info()  # includes USS on major OSes
        uss_mb = getattr(mem, "uss", None)
        if uss_mb is not None:
            uss_mb = float(uss_mb) / BYTES_IN_MB
    except Exception:
        mem = p.memory_info()
        uss_mb = None

    try:
        io = p.io_counters()
        rb, wb = io.read_bytes, io.write_bytes
        rc, wc = io.read_count, io.write_count
    except Exception:
        rb = wb = rc = wc = 0

    try:
        load1, _, _ = psutil.getloadavg()
    except Exception:
        load1 = None

    try:
        aff = p.cpu_affinity()
    except Exception:
        aff = None

    return ProcSnapshot(
        ts=now,
        cpu_percent=p.cpu_percent(interval=None),
        cpu_count_logical=_cpu_logical,
        cpu_count_physical=_cpu_physical,
        cpu_affinity=aff,
        loadavg_1m=load1,
        rss_mb=float(mem.rss) / BYTES_IN_MB,
        uss_mb=uss_mb,
        vms_mb=float(mem.vms) / BYTES_IN_MB,
        mem_percent=p.memory_percent(),
        num_threads=p.num_threads(),
        open_files=len(p.open_files() or []),
        read_bytes_mb=float(rb) / BYTES_IN_MB,
        write_bytes_mb=float(wb) / BYTES_IN_MB,
        read_count=int(rc),
        write_count=int(wc),
    )


class ResourceLimitExceeded(RuntimeError):
    pass


class ResourceGuard:
    """
    Background sampler using psutil. Raises ResourceLimitExceeded if any limit is exceeded.
    """
    def __init__(
        self,
        interval_s: float = 1.0,
        *,
        max_rss_mb: Optional[float] = None,
        max_uss_mb: Optional[float] = None,
        max_wall_s: Optional[float] = None,
        max_cpu_percent: Optional[float] = None,  # applies to summed CPU percent across cores
        cpu_window: int = 5,
        collect: bool = True,
    ):
        self.interval_s = max(0.2, float(interval_s))
        self.max_rss_mb = max_rss_mb
        self.max_uss_mb = max_uss_mb
        self.max_wall_s = max_wall_s
        self.max_cpu_percent = max_cpu_percent
        self.cpu_window = max(1, int(cpu_window))
        self.collect = collect

        self._snapshots: List[Dict[str, Any]] = []
        self._cpu_ring = deque(maxlen=self.cpu_window)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self._exc: Optional[BaseException] = None

    @property
    def snapshots(self) -> List[Dict[str, Any]]:
        return self._snapshots

    def _tick(self):
        snap = take_snapshot().to_dict()
        self._cpu_ring.append(snap["cpu_percent"])
        if self.collect:
            self._snapshots.append(snap)

        # Limits checks
        if self.max_rss_mb is not None and snap["rss_mb"] > self.max_rss_mb:
            raise ResourceLimitExceeded(f"RSS {snap['rss_mb']:.1f} MB > {self.max_rss_mb} MB")

        if self.max_uss_mb is not None and snap.get("uss_mb") and snap["uss_mb"] > self.max_uss_mb:
            raise ResourceLimitExceeded(f"USS {snap['uss_mb']:.1f} MB > {self.max_uss_mb} MB")

        if self.max_cpu_percent is not None and len(self._cpu_ring) == self._cpu_ring.maxlen:
            cpu_avg = sum(self._cpu_ring) / len(self._cpu_ring)
            if cpu_avg > self.max_cpu_percent:
                secs = len(self._cpu_ring) * self.interval_s
                raise ResourceLimitExceeded(f"CPU avg {cpu_avg:.1f}% > {self.max_cpu_percent}% over ~{secs:.1f}s")

        if self.max_wall_s is not None and self._t0 is not None:
            elapsed = time.time() - self._t0
            if elapsed > self.max_wall_s:
                raise ResourceLimitExceeded(f"Wall time {elapsed:.1f}s > {self.max_wall_s}s")

    def _run(self):
        self._t0 = time.time()
        try:
            while not self._stop.wait(self.interval_s):
                self._tick()
        except BaseException as e:
            self._exc = e
            self._stop.set()

    def __enter__(self):
        self._exc = None
        try:
            _proc.cpu_percent(None)
        except Exception:
            pass
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=5)
        if self._exc is not None and exc is None:
            raise self._exc
        return False


# ---- Global monitoring config for this script --------------------------------
ENABLE_MONITORING = True
MONITOR_INTERVAL_S = 1.0
LIMITS = {
    "max_rss_mb": None,       # e.g., 32000 for 32 GB
    "max_uss_mb": None,       # optional strict bound on Unique Set Size
    "max_wall_s": None,       # e.g., 7200 for 2 hours
    "max_cpu_percent": None,  # by default allow ~98% per core below
    "cpu_window": 10,
}

# -----------------------------------------------------------------------------
# 0) Environment to avoid warnings (must be set before package import)
# -----------------------------------------------------------------------------
os.environ.setdefault("PATH_TO_COSMOPOWER_ORGANIZATION", "/rds-d4/user/iz221/hpc-work/cosmopower/")
os.environ.setdefault("PATH_TO_COSMOCNC", "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc")

# -----------------------------------------------------------------------------
# 1) Force-import cosmocnc from the ROOT REPO
# -----------------------------------------------------------------------------
PKG_ROOT = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc"  # <- repository root
MOD_NAME = "cosmocnc"

# Purge previous imports/paths of "cosmocnc"
for n in list(sys.modules):
    if n == MOD_NAME or n.startswith(MOD_NAME + "."):
        sys.modules.pop(n, None)

def _endswith_basename(p, name):
    try:
        return Path(p).name.lower() == name
    except Exception:
        return False

# Remove any sys.path entries that end with "cosmocnc"
sys.path = [p for p in sys.path if not _endswith_basename(p, MOD_NAME)]
# Add only the repo root
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

invalidate_caches()
cnc = importlib.import_module(MOD_NAME)

print("cosmocnc loaded from:", getattr(cnc, "__file__", None))
print("attrs scaling:", [n for n in dir(cnc) if "scal" in n.lower()])

# -----------------------------------------------------------------------------
# 2) Patch the generator module that consumes `cnc`
# -----------------------------------------------------------------------------
import cata_generator_cop as cg
importlib.reload(cg)
cg.cnc = cnc

# Safety net if the name changed depending on the version
if not hasattr(cg.cnc, "scaling_relation_params_default"):
    try:
        from cosmocnc import sr as _sr
        cg.cnc.scaling_relation_params_default = _sr.scaling_relation_params_default
    except Exception:
        pass


def load_local_cosmocnc(pkg_dir: str):
    """
    Manually load a local cosmocnc package from `pkg_dir` pointing to .../cosmocnc (the folder with __init__.py).
    This utility resets sys.modules entries and patches a 'cata_generator' module to use the loaded cnc.
    """
    pkg = Path(pkg_dir)
    for n in list(sys.modules):
        if n == "cosmocnc" or n.startswith("cosmocnc."):
            sys.modules.pop(n, None)
    sys.path[:] = [p for p in sys.path if Path(p).name.lower() != "cosmocnc"]
    invalidate_caches()

    spec = spec_from_file_location("cosmocnc", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    cnc = module_from_spec(spec)
    sys.modules["cosmocnc"] = cnc
    spec.loader.exec_module(cnc)  # type: ignore

    parent = str(pkg.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    import cata_generator
    reload(cata_generator)
    cata_generator.cnc = cnc
    try:
        if not hasattr(cata_generator.cnc, "scaling_relation_params_default"):
            from cosmocnc import sr as _sr
            cata_generator.cnc.scaling_relation_params_default = _sr.scaling_relation_params_default
    except Exception:
        pass
    return cnc, cata_generator


def _to_float32(obj):
    """
    Recursively convert all float arrays to float32 (keeps structure).
    """
    import numpy as np
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
        return obj.astype(np.float32)
    if isinstance(obj, dict):
        return {k: _to_float32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_float32(v) for v in obj]
    return obj


# 1) Use a pre-loaded 'cosmocnc' if already bootstrapped upstream
cnc = sys.modules.get("cosmocnc")

# 2) Otherwise, attempt a direct import
if cnc is None:
    try:
        import cosmocnc as cnc  # type: ignore
    except Exception:
        # 3) Fallback via environment variable: MUST point to the internal package (.../cosmocnc/cosmocnc)
        p = os.environ.get("PATH_TO_COSMOCNC")
        if not p:
            raise RuntimeError(
                "PATH_TO_COSMOCNC is not set. Set it to …/cosmocnc/cosmocnc (the folder with __init__.py)."
            )
        # Clean sys.path of '…/cosmocnc' folders (external repo) that create a namespace
        def _endswith_cosmocnc(x): return x.rstrip("/").split("/")[-1].lower() == "cosmocnc"
        sys.path = [x for x in sys.path if not _endswith_cosmocnc(x)]
        if p not in sys.path:
            sys.path.insert(0, p)
        import cosmocnc as cnc  # type: ignore

# Compact logs
print("cosmocnc file:", getattr(cnc, "__file__", None))
print("scaling names:", [n for n in dir(cnc) if "scal" in n.lower()])


# ----------------------------- Geometry helpers -----------------------------
TWOPI = 2.0 * np.pi

def wrap_pm_pi(phi: np.ndarray) -> np.ndarray:
    """Wrap longitudes into [-pi, pi]."""
    out = (phi + np.pi) % (2.0 * np.pi) - np.pi
    return out


def sample_lonlat_patch(
    n: int,
    *,
    ra0_deg: float = 0.0,
    dec0_deg: float = 0.0,
    w_deg: float = 10.0,
    h_deg: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample 'n' random directions uniformly within a rectangular patch on the sphere,
    approximated by longitude/latitude bands centered at (ra0_deg, dec0_deg) with size w x h (degrees).

    Returns:
      lon (rad), theta (colatitude in rad).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Uniform in the rectangle in (RA, Dec) degrees
    ra = ra0_deg + (rng.random(n) - 0.5) * w_deg
    dec = dec0_deg + (rng.random(n) - 0.5) * h_deg

    # Convert to radians
    lon = np.deg2rad(ra)
    lat = np.deg2rad(dec)  # latitude in [-pi/2, pi/2]
    theta = np.pi / 2.0 - lat  # colatitude in [0, pi]
    return lon, theta


def _seed_from_path(path: Union[str, Path], base_seed: Optional[int]) -> Optional[int]:
    """Derive a stable per-file seed from a path and an optional base seed."""
    if base_seed is None:
        return None
    p = str(path)
    h = 0
    for ch in p:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h ^ base_seed) & 0x7FFFFFFF


# ----------------------------- Core helpers ---------------------------------

def _compute_f_sky_from_patch(patch_deg_width: float, patch_deg_height: float) -> float:
    """Convert a rectangular sky patch size (in degrees) to f_sky fraction."""
    sky_area_deg2 = float(patch_deg_width) * float(patch_deg_height)
    full_sky_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
    f_sky = sky_area_deg2 / full_sky_deg2
    f_sky = max(0.0, min(1.0, f_sky))
    return f_sky


def _load_params_from_csv_first_row(path: Union[str, os.PathLike]) -> Dict[str, float]:
    """
    Load the first row from a CSV with header containing at least:
    h,logA,n_s,Ob0h2,Oc0h2
    Returns a dict for the cosmology fields we need.
    Extra columns are ignored by the pipeline (but later recorded in manifests).
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.shape == ():  # single row -> np.void
        row = data
    else:
        if data.shape[0] < 1:
            raise ValueError(f"No rows in parameter file: {path}")
        row = data[0]

    needed = {}
    for key in ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]:
        if key not in row.dtype.names:
            raise KeyError(f"CSV {path} missing required column '{key}'")
        needed[key] = float(row[key])
    return needed


def _default_cnc_params(
    survey_sr: Optional[str],
    survey_cat: Optional[str],
    observables: Optional[List[List[str]]] = None,
    obs_select: str = "q_so_sim",
) -> Dict[str, object]:
    """Build baseline cnc_params. Paths can be provided to local survey modules."""
    if observables is None:
        observables = [["q_so_sim"]]

    params = {
        "number_cores_hmf": 1,
        # resolution/precision (tune as needed)
        "n_points": 5000,
        "n_z": 5000,
        "z_min": 0.005,
        "z_max": 3.0,
        "M_min": 1e14,
        "M_max": 1e16,
        "cosmo_model": "lcdm",
        "hmf_type": "Tinker08",
        "mass_definition": "500c",
        "cosmology_tool": "classy_sz",
        "hmf_calc": "cnc",
        "hmf_type_deriv": "numerical",
        "power_spectrum_type": "cosmopower",
        "cosmo_amplitude_parameter": "logA",
        "Hubble_parameter": "h",
        "cosmo_param_density": "physical",
        "interp_tinker": "log",
        "class_sz_cosmo_model": "lcdm",
        "cosmocnc_verbose": "none",
        "load_catalogue": False,
        "class_sz_ndim_masses": 100,
        "class_sz_ndim_redshifts": 500,
        "class_sz_concentration_parameter": "B13",
        "class_sz_output": "mPk,m500c_to_m200c,m200c_to_m500c",
        "class_sz_hmf": "T08M500c",
        "class_sz_use_m500c_in_ym_relation": 1,
        "class_sz_use_m200c_in_ym_relation": 0,
        "observables": observables,
        "obs_select": obs_select,
        "stacked_likelihood": False,
        "obs_select_min": 0.0,
        "obs_select_max": 0.0,
        # Avoid KeyError in catalogue_generator if used internally
        "M_min_extended": None,
    }
    if survey_sr is not None:
        params["survey_sr"] = survey_sr
    if survey_cat is not None:
        params["survey_cat"] = survey_cat
    return params


def _format_float_for_filename(x: float, decimals: int = 6) -> str:
    """Format a float for filenames, fixed decimals, and strip trailing zeros."""
    s = f"{x:.{decimals}f}"
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s


# ---------- CSV writer helper ----------

def _write_catalogue_csv(catalogue: Dict[str, object], out_path: str) -> None:
    """
    Write a catalogue (dict columns -> arrays/lists) to CSV.
    - Keep only 1D columns with the same length (per-cluster columns).
    - Ignore scalars and non-1D arrays to avoid ambiguities.
    """
    import csv

    cols = {}
    N = None
    for k, v in (catalogue or {}).items():
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim != 1:
            continue
        n = arr.shape[0]
        if N is None:
            N = n
        elif n != N:
            # ignore mismatching columns
            continue
        cols[k] = np.asarray(arr, dtype=np.float32).tolist()

    # Valid empty file if no usable column
    if N is None:
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerow([])
        return

    headers = list(cols.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(N):
            writer.writerow([cols[h][i] for h in headers])


# -----------------------------------------------------------------------------
# process_npy_catalogues — with monitoring annotations
# -----------------------------------------------------------------------------

def process_npy_catalogues(
    inputs,
    ra0_deg: float = 0.0,
    dec0_deg: float = 0.0,
    w_deg: float = 10.0,
    h_deg: float = 10.0,
    seed: Optional[int] = 42,
    pattern: str = "*.npy",
    recursive: bool = True,
    lon_wrap: str = "pm_pi",      # "pm_pi" -> [-π,π], "0_2pi" -> [0,2π)
    lon_col: str = "lon",
    lat_col: str = "lat",
    make_backup: bool = False,    # True -> write <file>.bak.npy before overwriting
):
    """
    Process one or multiple paths (NPY files or folders) and rewrite *the same .npy file*
    by adding two extra coordinates: lon (rad), lat (colatitude rad).

    The summary now includes per-file proc snapshots and global monitoring can raise if limits exceeded.
    """
    # Monitoring guard (optional)
    guard = None
    if ENABLE_MONITORING:
        guard_limits = dict(LIMITS)
        if guard_limits.get("max_cpu_percent") is None:
            guard_limits["max_cpu_percent"] = 98.0 * advertised_cpu_capacity()
        guard = ResourceGuard(interval_s=MONITOR_INTERVAL_S, collect=True, **guard_limits)

    try:
        if guard is not None:
            guard.__enter__()

        # Normalize inputs
        if isinstance(inputs, (str, Path)):
            inputs = [inputs]
        inputs = [Path(p) for p in inputs]

        # Build final list of NPY files to process
        npy_files: list[Path] = []
        for p in inputs:
            if p.is_dir():
                if recursive:
                    npy_files.extend(sorted(p.rglob(pattern)))
                else:
                    npy_files.extend(sorted(p.glob(pattern)))
            elif p.is_file() and p.suffix.lower() == ".npy":
                npy_files.append(p)
            else:
                pass

        if not npy_files:
            print("No NPY files found.")
            return []

        summary = []

        for f in npy_files:
            # Stable per-file seed
            file_seed = _seed_from_path(f, seed)
            rng = np.random.default_rng(file_seed) if file_seed is not None else np.random.default_rng()

            # Read: allow pickle to handle dicts
            try:
                loaded = np.load(f, allow_pickle=True)
            except Exception as e:
                snapshot = take_snapshot().to_dict()
                summary.append({"in": str(f), "n": 0, "skipped": True, "reason": f"read_error: {e}",
                                "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                   "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                continue

            # Determine the logical “form” of the content
            is_scalar_object = isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object
            data_obj = None
            arr = None
            try:
                if is_scalar_object:
                    candidate = loaded.item()
                    if isinstance(candidate, dict):
                        data_obj = candidate
                    elif isinstance(candidate, np.ndarray):
                        arr = candidate
                    else:
                        # Unhandled type -> try keeping it in "data"
                        arr = candidate
                elif isinstance(loaded, np.ndarray):
                    arr = loaded
                else:
                    arr = np.asarray(loaded)
            except Exception:
                arr = np.asarray(loaded)

            # Case 2: dict
            if data_obj is not None:
                # Try to infer 'n' from any vector-like entry
                n = None
                for v in data_obj.values():
                    try:
                        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
                            n = v.shape[0]
                            break
                    except Exception:
                        pass
                if n is None:
                    snapshot = take_snapshot().to_dict()
                    summary.append({"in": str(f), "n": 0, "skipped": True, "reason": "cannot_infer_n_from_dict",
                                    "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                       "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                    continue

                # Sample coordinates
                lon, theta = sample_lonlat_patch(n, ra0_deg=ra0_deg, dec0_deg=dec0_deg, w_deg=w_deg, h_deg=h_deg, rng=rng)
                if lon_wrap == "0_2pi":
                    lon = lon % TWOPI
                elif lon_wrap == "pm_pi":
                    lon = wrap_pm_pi(lon)
                else:
                    snapshot = take_snapshot().to_dict()
                    summary.append({"in": str(f), "n": n, "skipped": True, "reason": "bad_lon_wrap",
                                    "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                       "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                    continue

                data_obj[lon_col] = lon
                data_obj[lat_col] = theta

                # Save (optional backup)
                try:
                    if make_backup:
                        f.with_suffix(".bak.npy").write_bytes(f.read_bytes())
                    np.save(f, data_obj, allow_pickle=True)
                    snapshot = take_snapshot().to_dict()
                    summary.append({"in": str(f), "n": n, "skipped": False, "reason": None,
                                    "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                       "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                except Exception as e:
                    snapshot = take_snapshot().to_dict()
                    summary.append({"in": str(f), "n": n, "skipped": True, "reason": f"write_error: {e}",
                                    "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                       "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                continue

            # Case 1 or 3: NumPy array
            if not isinstance(arr, np.ndarray):
                snapshot = take_snapshot().to_dict()
                summary.append({"in": str(f), "n": 0, "skipped": True, "reason": "unsupported_npy_content",
                                "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                   "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                continue

            if arr.ndim == 0:
                snapshot = take_snapshot().to_dict()
                summary.append({"in": str(f), "n": 0, "skipped": True, "reason": "scalar_array_unsupported",
                                "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                   "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                continue
            n = arr.shape[0]

            # Sample coordinates
            lon, theta = sample_lonlat_patch(n, ra0_deg=ra0_deg, dec0_deg=dec0_deg, w_deg=w_deg, h_deg=h_deg, rng=rng)
            if lon_wrap == "0_2pi":
                lon = lon % TWOPI
            elif lon_wrap == "pm_pi":
                lon = wrap_pm_pi(lon)
            else:
                snapshot = take_snapshot().to_dict()
                summary.append({"in": str(f), "n": n, "skipped": True, "reason": "bad_lon_wrap",
                                "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                   "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                continue

            # Case 1: structured array -> add two fields
            try:
                if arr.dtype.names is not None:
                    new_descr = list(arr.dtype.descr)
                    if lon_col in arr.dtype.names or lat_col in arr.dtype.names:
                        snapshot = take_snapshot().to_dict()
                        summary.append({"in": str(f), "n": n, "skipped": True, "reason": "field_name_conflict",
                                        "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                           "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                        continue
                    new_descr.append((lon_col, "<f8"))
                    new_descr.append((lat_col, "<f8"))
                    new_dtype = np.dtype(new_descr)

                    new_arr = np.empty(arr.shape, dtype=new_dtype)
                    for name in arr.dtype.names:
                        new_arr[name] = arr[name]
                    new_arr[lon_col] = lon
                    new_arr[lat_col] = theta

                    try:
                        if make_backup:
                            f.with_suffix(".bak.npy").write_bytes(f.read_bytes())
                        np.save(f, new_arr)
                        snapshot = take_snapshot().to_dict()
                        summary.append({"in": str(f), "n": n, "skipped": False, "reason": None,
                                        "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                           "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                    except Exception as e:
                        snapshot = take_snapshot().to_dict()
                        summary.append({"in": str(f), "n": n, "skipped": True, "reason": f"write_error: {e}",
                                        "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                           "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
                    continue
            except Exception:
                pass

            # Case 3: simple array -> rewrite as dict to preserve original array
            try:
                payload = {"data": arr, lon_col: lon, lat_col: theta}
                if make_backup:
                    f.with_suffix(".bak.npy").write_bytes(f.read_bytes())
                payload_f32 = _to_float32(payload)
                np.save(f, payload_f32, allow_pickle=True)
                snapshot = take_snapshot().to_dict()
                summary.append({"in": str(f), "n": n, "skipped": False, "reason": None,
                                "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                   "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})
            except Exception as e:
                snapshot = take_snapshot().to_dict()
                summary.append({"in": str(f), "n": n, "skipped": True, "reason": f"write_error: {e}",
                                "proc_snapshot": {"ts": snapshot["ts"], "rss_mb": snapshot["rss_mb"],
                                                   "cpu_percent": snapshot["cpu_percent"], "num_threads": snapshot["num_threads"]}})

        done = sum(1 for s in summary if not s["skipped"])
        skipped = len(summary) - done
        print(f"Completed: {done} file(s) rewritten, {skipped} skipped.")
        return summary

    finally:
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except ResourceLimitExceeded as e:
                last = take_snapshot().to_dict()
                print(f"[ResourceGuard] {e} | last rss={last['rss_mb']:.1f} MB, cpu={last['cpu_percent']:.1f}%")
                raise


# ---------- Input normalization ----------

def _read_csv_all_rows(path: Union[str, os.PathLike]) -> List[Dict[str, float]]:
    """
    Read all rows from a CSV file (strict mode).
    Requires header with at least the 5 required columns:
        h, logA, n_s, Ob0h2, Oc0h2
    Returns a list of dicts with only those columns.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if arr.shape == ():  # single row
        arr = np.array([arr], dtype=arr.dtype)

    required = ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]
    for key in required:
        if key not in arr.dtype.names:
            raise KeyError(f"CSV {path} missing required column '{key}'")

    rows: List[Dict[str, float]] = []
    for r in arr:
        rows.append({k: float(r[k]) for k in required})
    return rows


def _dataframe_to_param_rows(df: "pd.DataFrame") -> List[Dict[str, float]]:
    """
    Convert a pandas DataFrame (each row = one cosmology) to a list of param dicts.

    NEW: Accept either the full set ["h","logA","n_s","Ob0h2","Oc0h2"] or the minimal set ["Oc0h2","logA"].
    - If only ["Oc0h2","logA"] are present, missing parameters (h, n_s, Ob0h2) will be taken from
      `baseline_cosmo_params` within generate_cluster_catalogues() and left untouched here.
    """
    if pd is None:
        raise ImportError("pandas is required to pass a DataFrame as input, but it's not installed.")

    full_required = ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]
    minimal_required = ["Oc0h2", "logA"]

    has_full = all(c in df.columns for c in full_required)
    has_min = all(c in df.columns for c in minimal_required)

    if not (has_full or has_min):
        missing_full = [c for c in full_required if c not in df.columns]
        missing_min = [c for c in minimal_required if c not in df.columns]
        raise KeyError(
            "DataFrame must contain either all of {} or at least {}.\n"
            "Missing (full): {}\nMissing (minimal): {}".format(
                full_required, minimal_required, missing_full, missing_min
            )
        )

    keep_cols = full_required if has_full else minimal_required
    df_num = df[keep_cols].astype(float).copy()

    # Guard against NaN / Inf values
    vals = df_num.to_numpy()
    if not np.all(np.isfinite(vals)):
        raise ValueError("The DataFrame contains non-finite values (NaN or Inf) in required columns.")

    # Return dicts with only present keys; missing ones will be taken from defaults later
    return [dict(row) for _, row in df_num.iterrows()]


def _iter_cosmologies_from_input(
    cosmo_input: Union[str, os.PathLike, "pd.DataFrame"],
    n_cosmologies: int,
    csv_glob: str,
) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    """
    Normalize the three possible inputs into a list of cosmology parameter dicts.
    Returns:
      - param_rows: List[Dict[str, float]] of length <= n_cosmologies
      - provenance: List[Dict[str, object]] with metadata for manifests (e.g., source path, row index)
    """
    param_rows: List[Dict[str, float]] = []
    provenance: List[Dict[str, object]] = []

    # Case 1: DataFrame
    if pd is not None and isinstance(cosmo_input, pd.DataFrame):
        all_rows = _dataframe_to_param_rows(cosmo_input)
        for idx, r in enumerate(all_rows[: int(n_cosmologies)]):
            param_rows.append(r)
            provenance.append({"source_type": "dataframe", "row_index": idx})
        return param_rows, provenance

    # Otherwise treat as path-like
    if not isinstance(cosmo_input, (str, os.PathLike)):
        raise TypeError("cosmo_input must be a directory path, a CSV file path, or a pandas DataFrame.")
    path = str(cosmo_input)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path}")

    # Directory of CSV files (backward-compatible behavior: first row per file)
    if os.path.isdir(path):
        csv_paths = sorted(glob.glob(os.path.join(path, csv_glob)))
        if len(csv_paths) == 0:
            raise FileNotFoundError(f"No CSV files found in {path!r} matching pattern {csv_glob!r}")
        for c_idx, csv_path in enumerate(csv_paths[: int(n_cosmologies)]):
            params = _load_params_from_csv_first_row(csv_path)  # strict 5-col
            param_rows.append(params)
            provenance.append(
                {"source_type": "csv_dir", "csv_path": csv_path, "file_index": c_idx, "row_index": 0}
            )
        return param_rows, provenance

    # Single CSV file with potentially multiple rows (strict 5-col)
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        all_rows = _read_csv_all_rows(path)
        for r_idx, r in enumerate(all_rows[: int(n_cosmologies)]):
            param_rows.append(r)
            provenance.append(
                {"source_type": "csv_file", "csv_path": path, "row_index": r_idx}
            )
        return param_rows, provenance

    raise ValueError(
        "cosmo_input must be a directory containing CSV files, a single CSV file, or a pandas DataFrame."
    )


# -----------------------------------------------------------------------------
# generate_cluster_catalogues — with ResourceGuard + manifest samples
# -----------------------------------------------------------------------------

def generate_cluster_catalogues(
    cosmo_input: Union[str, os.PathLike, "pd.DataFrame"],
    n_cosmologies: int,
    n_catalogues_per_cosmo: int,
    patch_size_deg: Tuple[float, float],
    output_dir: Union[str, os.PathLike],
    *,
    csv_glob: str = "*.csv",
    # If you want full-sky, pass override_f_sky=1.0; otherwise it's computed from patch size.
    override_f_sky: Optional[float] = None,
    get_sky_coords: bool = False,
    # Survey module paths (pass absolute paths in your environment)
    survey_sr_path: Optional[str] = None,
    survey_cat_path: Optional[str] = None,
    # Random seeding base for reproducibility
    seed_offset: int = 20000401,
    # Allow the caller to tweak cnc/scaling/cosmology defaults
    cnc_params_overrides: Optional[Dict[str, object]] = None,
    baseline_cosmo_params: Optional[Dict[str, float]] = None,
    scaling_relation_overrides: Optional[Dict[str, float]] = None,
    # File naming (tpl kept but ignored for safety to satisfy requested naming)
    cosmology_folder_prefix: str = "cosmo",
    catalogue_file_tpl: str = "catalogue_{cosmo_idx:03d}_{cat_idx:04d}.npy",
    manifest_name: str = "manifest.json",
    output_format: str = "npy",  # "npy" | "csv" | "both"
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run the catalogue generation pipeline.

    NEW: integrates psutil-based monitoring. Each manifest contains resource_samples (tail),
    resource_snapshot_end; run_summary.json records a summary snapshot as well.
    """
    if cnc is None:
        raise ImportError("cosmocnc is not available in the current environment. Install and retry.")

    if output_format not in ("npy", "csv", "both"):
        raise ValueError(f"Invalid output_format='{output_format}'. Use 'npy', 'csv', or 'both'.")

    t_start = time.time()

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Normalize input to a list of param dicts + provenance metadata
    param_rows, provenance = _iter_cosmologies_from_input(
        cosmo_input=cosmo_input, n_cosmologies=n_cosmologies, csv_glob=csv_glob
    )
    if len(param_rows) == 0:
        raise ValueError("No cosmology rows found in the provided input.")

    # f_sky
    if override_f_sky is not None:
        f_sky = float(override_f_sky)
    else:
        w, h = patch_size_deg
        f_sky = _compute_f_sky_from_patch(w, h)
    f_sky = max(0.0, min(1.0, f_sky))

    # cnc params
    cnc_params = _default_cnc_params(survey_sr_path, survey_cat_path)
    if cnc_params_overrides:
        cnc_params.update(cnc_params_overrides)

    # Baseline cosmology (values used if missing in each row)
    if baseline_cosmo_params is None:
        cosmo_params = {
            "h": 0.6766,
            "Ob0h2": 0.02242,
            "Oc0h2": 0.1193,
            "logA": 3.047,
            "n_s": 0.9665,
            "m_nu": 0.06,
            "tau_reio": 0.0544,
        }
    else:
        cosmo_params = dict(baseline_cosmo_params)

    # Scaling relation parameters
    scal_rel_params = cnc.scaling_relation_params_default.copy()
    if scaling_relation_overrides:
        scal_rel_params.update(scaling_relation_overrides)

    # Initialise number counts once
    number_counts = cnc.cluster_number_counts(cnc_params=cnc_params)
    number_counts.cosmo_params = cosmo_params
    number_counts.scal_rel_params = scal_rel_params
    number_counts.initialise()

    run_started = time.time()
    written_files: List[str] = []
    cosmo_summaries = []

    # Monitoring setup
    guard = None
    res_samples_all: List[Dict[str, Any]] = []
    if ENABLE_MONITORING:
        limits = dict(LIMITS)
        if limits.get("max_cpu_percent") is None:
            limits["max_cpu_percent"] = 98.0 * advertised_cpu_capacity()
        guard = ResourceGuard(interval_s=MONITOR_INTERVAL_S, collect=True, **limits)

    try:
        if guard is not None:
            guard.__enter__()

        for c_idx, row_params in enumerate(param_rows):
            # Update cosmology from row (accepts either the full set or the minimal {Oc0h2, logA})
            for k in ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]:
                if k in row_params:
                    cosmo_params[k] = float(row_params[k])

            number_counts.update_params(cosmo_params, scal_rel_params)

            # Derived parameters for naming
            logA_val = float(cosmo_params["logA"])
            oc0h2_val = float(cosmo_params["Oc0h2"])
            omega_c_val = float(cosmo_params["Oc0h2"]) / (float(cosmo_params["h"]) ** 2)  # Omega_c = Oc0h2 / h^2

            logA_str = _format_float_for_filename(logA_val)
            oc0h2_str  = _format_float_for_filename(oc0h2_val)

            # Per-cosmology folder
            prov = provenance[c_idx]
            if prov["source_type"] == "csv_dir":
                stem = os.path.splitext(os.path.basename(prov["csv_path"]))[0]
                cosmo_folder = os.path.join(output_dir, f"{cosmology_folder_prefix}_{c_idx:03d}_{stem}")
            elif prov["source_type"] == "csv_file":
                base = os.path.splitext(os.path.basename(prov["csv_path"]))[0]
                cosmo_folder = os.path.join(output_dir, f"{cosmology_folder_prefix}_{c_idx:03d}_{base}_row{prov['row_index']:04d}")
            elif prov["source_type"] == "dataframe":
                cosmo_folder = os.path.join(output_dir, f"{cosmology_folder_prefix}_{c_idx:03d}_df_row{prov['row_index']:04d}")
            else:
                cosmo_folder = os.path.join(output_dir, f"{cosmology_folder_prefix}_{c_idx:03d}")
            os.makedirs(cosmo_folder, exist_ok=True)

            # Manifest skeleton
            manifest = {
                "provenance": prov,
                "cosmology_index": c_idx,
                "cosmo_params": {k: float(v) for k, v in cosmo_params.items()},
                "derived_params": {"omega_c": omega_c_val},
                "scaling_relation_params": {
                    k: float(v) for k, v in scal_rel_params.items()
                    if isinstance(v, (int, float, np.floating))
                },
                "cnc_params": cnc_params,
                "n_catalogues": int(n_catalogues_per_cosmo),
                "f_sky": f_sky,
                "get_sky_coords": bool(get_sky_coords),
                "seed_offset": int(seed_offset),
                "started_at": time.time(),
                "catalogues": [],
                "output_format": output_format,
                # Monitoring payloads
                "resource_samples": [],
                "resource_snapshot_end": None,
            }

            # Record extra columns when applicable
            try:
                extras: Dict[str, float] = {}
                if prov["source_type"] == "csv_file":
                    import csv as _csv
                    with open(prov["csv_path"], newline="") as fh:
                        reader = _csv.DictReader(fh)
                        for i, row in enumerate(reader):
                            if i == prov["row_index"]:
                                for k, v in row.items():
                                    if k in ("h", "logA", "n_s", "Ob0h2", "Oc0h2"):
                                        continue
                                    if v is None or v == "":
                                        continue
                                    try:
                                        extras[k] = float(v)
                                    except Exception:
                                        extras[k] = v
                                break
                elif prov["source_type"] == "dataframe" and pd is not None and isinstance(cosmo_input, pd.DataFrame):
                    row = cosmo_input.iloc[prov["row_index"]]
                    for k, v in row.items():
                        if k in ("h", "logA", "n_s", "Ob0h2", "Oc0h2"):
                            continue
                        try:
                            extras[k] = float(v)
                        except Exception:
                            extras[k] = v
                manifest["input_extra_columns"] = extras
            except Exception:
                pass

            for j in range(int(n_catalogues_per_cosmo)):
                unique_seed = int(seed_offset) + int(c_idx) * 10_000 + int(j)
                np.random.seed(unique_seed)

                cat_gen = cnc.catalogue_generator(
                    number_counts=number_counts,
                    n_catalogues=1,
                    seed=unique_seed,
                    get_sky_coords=get_sky_coords,
                    sky_frac=f_sky,
                )
                t0 = time.time()
                cat_gen.generate_catalogues_hmf()
                catalogue_list = cat_gen.catalogue_list
                catalogue = catalogue_list[0] if isinstance(catalogue_list, list) else catalogue_list

                # File naming stem
                out_stem = f"logA={logA_str}_Oc0h2={oc0h2_str}_{j:04d}"
                files_this_catalogue = []

                # Write .npy
                if output_format in ("npy", "both"):
                    npy_path = os.path.join(cosmo_folder, out_stem + ".npy")
                    catalogue_list_f32 = _to_float32(catalogue_list)
                    np.save(npy_path, catalogue_list_f32)
                    written_files.append(npy_path)
                    files_this_catalogue.append({"file": os.path.basename(npy_path), "format": "npy"})

                # Write .csv (tabular per-cluster)
                if output_format in ("csv", "both"):
                    csv_path = os.path.join(cosmo_folder, out_stem + ".csv")
                    _write_catalogue_csv(catalogue, csv_path)
                    written_files.append(csv_path)
                    files_this_catalogue.append({"file": os.path.basename(csv_path), "format": "csv"})

                elapsed = time.time() - t0
                manifest["catalogues"].append({
                    "files": files_this_catalogue,
                    "seed": unique_seed,
                    "n_clusters": int(len(catalogue.get("M", []))) if isinstance(catalogue, dict) else None,
                    "elapsed_s": elapsed,
                    "logA": logA_val,
                    "omega_c": omega_c_val,
                })
                if verbose:
                    human_files = ", ".join(f["file"] for f in files_this_catalogue)
                    print(f"[cosmo {c_idx:03d}] catalogue {j:04d} -> {human_files} (gen {elapsed:.3f}s)")

            # Monitoring attachments (tail only to limit size)
            snap_now = take_snapshot().to_dict()
            manifest["resource_snapshot_end"] = snap_now
            if guard is not None:
                tail = guard.snapshots[-min(len(guard.snapshots), 200):]
                manifest["resource_samples"] = tail
                res_samples_all.extend(tail)

            # finalize manifest
            manifest["finished_at"] = time.time()
            with open(os.path.join(cosmo_folder, manifest_name), "w") as f:
                json.dump(manifest, f, indent=2)

            cosmo_summaries.append({
                "cosmo_folder": cosmo_folder,
                "provenance": prov,
                "n_catalogues": int(n_catalogues_per_cosmo),
            })

        summary = {
            "output_dir": output_dir,
            "n_cosmologies": len(param_rows),
            "f_sky": f_sky,
            "patch_size_deg": list(map(float, patch_size_deg)),
            "files_written": written_files,
            "cosmo_summaries": cosmo_summaries,
            "output_format": output_format,
            "total_elapsed_s": time.time() - run_started,
            # Monitoring summary
            "resource_snapshot_end": take_snapshot().to_dict(),
            "resource_samples_count": len(res_samples_all),
        }
        # Write a top-level summary too
        with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if verbose:
            print(f"Pipeline total (inside function): {time.time() - t_start:.3f}s")

        return summary

    finally:
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except ResourceLimitExceeded as e:
                # Record an error file with last snapshot and re-raise
                try:
                    err_path = os.path.join(output_dir, "resource_limit_error.json")
                    with open(err_path, "w") as f:
                        json.dump({"error": str(e), "last_snapshot": take_snapshot().to_dict()}, f, indent=2)
                except Exception:
                    pass
                raise


if __name__ == "__main__":
    # --------------- Automatic runtime display ---------------
    _script_start = time.time()

    # EXAMPLES (edit paths for your environment):

    # 1) Directory containing CSV files (first row used per file) [strict 5-col]
    # summary = generate_cluster_catalogues(
    #     cosmo_input="../tszsbi/catalogue_demo",
    #     n_cosmologies=10,
    #     n_catalogues_per_cosmo=100,
    #     patch_size_deg=(10.0, 10.0),
    #     output_dir="../tszsbi/catalogue_generated",
    #     survey_sr_path="../cosmocnc/surveys/survey_sr_so_sim.py",
    #     survey_cat_path="../cosmocnc/surveys/survey_cat_so_sim.py",
    #     scaling_relation_overrides={"bias_sz": 0.8, "dof": 0.0},
    #     output_format="both",
    # )

    # 2) Single CSV file (each row = one cosmology) [strict 5-col]
    # summary = generate_cluster_catalogues(
    #     cosmo_input="../tszsbi/multirow_cosmologies.csv",
    #     n_cosmologies=5,
    #     n_catalogues_per_cosmo=50,
    #     patch_size_deg=(10.0, 10.0),
    #     output_dir="../tszsbi/catalogue_generated_from_file",
    #     output_format="csv",
    # )

    # 3) pandas DataFrame (each row = one cosmology) — now supports 2 columns: Oc0h2 & logA
    # df = pd.DataFrame({
    #     "Oc0h2": [0.116816, 0.121944, 0.092288, 0.108687, 0.129850],
    #     "logA":  [3.038456, 2.884068, 3.570701, 2.910129, 3.033971],
    # })
    # summary = generate_cluster_catalogues(
    #     cosmo_input=df,
    #     n_cosmologies=len(df),
    #     n_catalogues_per_cosmo=20,
    #     patch_size_deg=(10.0, 10.0),
    #     output_dir="../tszsbi/catalogue_generated_from_df",
    #     baseline_cosmo_params={
    #         "h": 0.6766,
    #         "Ob0h2": 0.02242,
    #         "Oc0h2": 0.1193,  # will be overwritten by df values
    #         "logA": 3.047,    # will be overwritten by df values
    #         "n_s": 0.9665,
    #         "m_nu": 0.06,
    #         "tau_reio": 0.0544,
    #     },
    #     output_format="npy",
    # )

    _script_end = time.time()
    print(f"[Script] Execution time: {_script_end - _script_start:.3f} seconds")
