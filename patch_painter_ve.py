# tsz_painter_fixed_grouped.py
from __future__ import annotations
import os
import re  # (ADD) for parsing parameters from filenames
from typing import Sequence, Optional, Union, List
from juliacall import Main as jl
from pathlib import Path
import json
import pandas as pd
import numpy as np
import time  # (ADD) global runtime measurement


def _match_row_in_lhs(lhs_csv: str, logA: float, Oc0h2: float, tol: float = 5e-6) -> dict:
    """
    Find the closest matching row in an LHS CSV given target (logA, Oc0h2).
    If nothing is within the absolute tolerance, choose the nearest by L1 distance.
    Returns a small dict with target/check values when present.
    """
    df = pd.read_csv(lhs_csv)
    logA = float(logA); Oc0h2 = float(Oc0h2)
    m = np.isclose(df["logA"].to_numpy(), logA, atol=tol) & \
        np.isclose(df["Oc0h2"].to_numpy(), Oc0h2, atol=tol)
    if not m.any():
        i = np.argmin((df["logA"] - logA).abs() + (df["Oc0h2"] - Oc0h2).abs())
        row = df.iloc[i]
    else:
        row = df[m].iloc[0]
    out = {
        "Omega_m_target": float(row.get("Omega_m_target", np.nan)),
        "sigma8_target":  float(row.get("sigma8_target",  np.nan)),
    }
    if "Omega_m_check" in row: out["Omega_m_check"] = float(row["Omega_m_check"])
    if "sigma8_check"  in row: out["sigma8_check"]  = float(row["sigma8_check"])
    return out


# ------------------ filename → params fallback (ADD: minimal) ------------------
# Parse logA & Oc0h2 from filenames like ..._logA=3.038_Oc0h2=0.116816.csv
_LOGA_RE  = re.compile(r"(?:^|[_-])logA=([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)")
_OC0H2_RE = re.compile(r"(?:^|[_-])Oc0h2=([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)")


def _default_extract_params_from_name(path: Union[str, os.PathLike]) -> dict:
    """
    Minimal parameter extractor from a filename when no explicit params CSV is provided.
    Expected filename segments: 'logA=<float>' and 'Oc0h2=<float>'.
    Provides defaults for h, Ob0h2, B to keep the pipeline running.
    """
    name = os.path.basename(str(path))
    m_logA = _LOGA_RE.search(name)
    m_oc   = _OC0H2_RE.search(name)
    if not (m_logA and m_oc):
        raise ValueError(f"Filename does not contain logA/Oc0h2: {name}")
    return {
        "h": 0.6766,
        "Ob0h2": 0.02242,
        "Oc0h2": float(m_oc.group(1)),
        "B": 1.35,
        "logA": float(m_logA.group(1)),  # used later for LHS matching
    }

# ------------------------- utils: expand dirs to csvs -------------------------

def _csvs_from_paths(paths: Sequence[Union[str, os.PathLike]],
                     recursive: bool = False,
                     require_params_in_title: bool = True) -> List[str]:
    """
    Expand user-specified files/directories to a sorted list of CSV paths.
    Optionally require that filenames contain 'logA=' and 'Oc0h2=' tokens.
    """
    out: List[str] = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            it = P.rglob("*.csv") if recursive else P.glob("*.csv")
            for f in it:
                name = f.name
                if (not require_params_in_title) or (_LOGA_RE.search(name) and _OC0H2_RE.search(name)):
                    out.append(str(f))
        else:
            if P.suffix.lower() == ".csv":
                name = P.name
                if (not require_params_in_title) or (_LOGA_RE.search(name) and _OC0H2_RE.search(name)):
                    out.append(str(P))
    out.sort()
    return out

# ----------------------------- Julia environment -----------------------------

def _setup_julia_env(
    envdir: str,
    xgpaint_url: str = "https://github.com/licongxu/XGPaint.jl",
    xgpaint_rev: Optional[str] = None,
    quiet: bool = False,
):
    """
    Create/activate a dedicated Julia environment, pin/prepare dependencies,
    and (re)develop XGPaint at the requested URL/revision.
    Also removes environment variables that can break Julia SSL on macOS.
    """
    for k in ("DYLD_FALLBACK_LIBRARY_PATH", "JULIA_OPENSSL_USE_SYSTEM_LIBS"):
        if k in os.environ:
            os.environ.pop(k)
    os.makedirs(envdir, exist_ok=True)
    os.environ["JULIA_PROJECT"] = envdir
    rev_line = f', rev="{xgpaint_rev}"' if xgpaint_rev else ""
    jl.seval(f"""
    import Pkg, Base
    Base.mkpath(raw"{envdir}")
    Pkg.activate(raw"{envdir}"; shared=false)
    try
        Pkg.Registry.add("General")
    catch
        Pkg.Registry.update()
    end
    Pkg.add(name="OpenSSL", version="1.4")
    Pkg.add(name="HTTP", version="1.10")
    Pkg.add(name="HDF5", version="0.17")
    try
        Pkg.rm("XGPaint"; force=true)
    catch
    end
    Pkg.develop(Pkg.PackageSpec(url="{xgpaint_url}"{rev_line}))
    Pkg.add(["CSV","DataFrames","Healpix","Pixell","Dates","Printf","ImageFiltering"])
    Pkg.instantiate()
    Pkg.precompile()
    """)


def _ensure_julia_symbols():
    jl.seval(r"""
    using XGPaint, CSV, DataFrames, Healpix, Pixell, Dates, Printf, ImageFiltering
    wall() = time()

    function paint_one_catalogue(cat_path::AbstractString,
                                 par_path::AbstractString,
                                 idx::Integer,
                                 patch_size_deg::Float64,
                                 pix_res_arcmin::Float64,
                                 out_dir::AbstractString;
                                 beam_fwhm_arcmin::Union{Nothing,Float64}=nothing,
                                 nx::Int=256)

        if !isfile(cat_path)
            @warn "Catalogue $(cat_path) not found"
            return nothing
        end
        if !isfile(par_path)
            @warn "Params $(par_path) not found"
            return nothing
        end

        # Read catalogue (z, M, lon, lat), build model, etc. (identical)
        df    = CSV.read(cat_path, DataFrame)
        z_arr = df.z
        m_arr = df.M .* 1e14
        ra_arr  = rem.(df.lon .+ π, 2π) .- π
        dec_arr = (π/2) .- df.lat

        p   = CSV.read(par_path, DataFrame)[1, :]
        h, Ωb, Ωc, B = p.h,  p.Ob0h2/p.h^2,  p.Oc0h2/p.h^2,  p.B

        t0 = wall()
        a10_base = Arnauld10ThermalSZProfile(Omega_c = Ωc, Omega_b = Ωb, h = h, B = B)
        y_model  = build_interpolator(a10_base; Nx = nx)
        @printf("   interpolator built in %.3f s\n", wall() - t0)

        half = patch_size_deg/2
        box        = [ half  -half; -half  half] * Pixell.degree
        shape, wcs = geometry(CarClenshawCurtis{Float64}, box, pix_res_arcmin * Pixell.arcminute)

        # Paint in Float64 to match XGPaint/Pixell internals
        sky_map64  = Enmap(zeros(shape), wcs)
        workspace  = profileworkspace(shape, wcs)

        t1 = wall()
        paint!(sky_map64, workspace, y_model, m_arr, z_arr, ra_arr, dec_arr)
        @printf("   painting finished in %.3f s\n", wall() - t1)

        if beam_fwhm_arcmin !== nothing
            σ_pix = (beam_fwhm_arcmin / (2*sqrt(2*log(2)))) / pix_res_arcmin
            k = ImageFiltering.KernelFactors.gaussian(σ_pix)
            sky_map64 .= ImageFiltering.imfilter(sky_map64, (k, k))
            println("   applied Gaussian beam of FWHM=$(beam_fwhm_arcmin) arcmin")
        end

        # Correct conversion to Float32 WITHOUT nesting Enmap:
        #    1) extract a plain Array from the Enmap,
        #    2) convert element type,
        #    3) build a new Enmap with that Matrix.
        buf32      = Float32.(Array(sky_map64))   # Matrix{Float32}
        sky_map32  = Enmap(buf32, wcs)            # Enmap{Float32, …, Matrix{Float32}, …}

        mkpath(out_dir)
        mapfile = joinpath(out_dir, @sprintf("y_map_%0.1fdeg_%0.3farcmin_%d.fits",
                                             patch_size_deg, pix_res_arcmin, idx))
        write_map(mapfile, sky_map32)
        println("   saved → $mapfile")
        return mapfile
    end
    """)


# ------------------------------ helpers (new) ------------------------------

def _read_params_csv_first_row(path: str) -> dict:
    """
    Read the first row of a parameter CSV.
    Expected columns: h, Ob0h2, Oc0h2, B, (optional) logA.
    """
    import csv
    with open(path, newline="") as f:
        r = list(csv.DictReader(f))
    if not r:
        raise ValueError(f"Empty params CSV: {path}")
    row = r[0]
    out = {
        "h": float(row["h"]),
        "Ob0h2": float(row["Ob0h2"]),
        "Oc0h2": float(row["Oc0h2"]),
        "B": float(row["B"]),
    }
    # logA is optional and may be missing or blank
    if "logA" in row and row["logA"] not in (None, "", "nan"):
        try: out["logA"] = float(row["logA"])
        except Exception: pass
    return out


def _slug_value(x: Union[str, float, int]) -> str:
    """
    Create a filename-safe slug from a value (float/int/str).
    Removes unsafe characters and collapses runs of separators.
    """
    if isinstance(x, (float, int)):
        s = f"{x:.8g}"  # short and stable formatting
    else:
        s = str(x)
    s = re.sub(r"[^A-Za-z0-9.+\-=]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _cosmo_subdir_name(p: dict) -> str:
    """
    Group output files by cosmology.
    Default grouping key: (logA, Oc0h2). If logA is missing, only use Oc0h2.
    Add more fields here if you want a finer granularity.
    """
    parts = []
    if "logA" in p and p["logA"] is not None and not (isinstance(p["logA"], float) and np.isnan(p["logA"])):
        parts.append(f"logA={_slug_value(p['logA'])}")
    parts.append(f"Oc0h2={_slug_value(p['Oc0h2'])}")
    return "_".join(parts)

# ------------------------------ main entrypoint ------------------------------

def paint_patches(
    patch_size_deg: float,
    pix_res_arcmin: float,
    output_dir: Union[str, os.PathLike],
    beam_fwhm_arcmin: Optional[float] = None,
    catalogs: Union[str, Sequence[str], None] = None,
    params: Union[str, Sequence[str], None] = None,
    nx: int = 256,
    *,
    envdir: Optional[str] = None,
    xgpaint_url: str = "https://github.com/licongxu/XGPaint.jl",
    xgpaint_rev: Optional[str] = None,
    recursive: Optional[bool] = False,
    lhs_csv: Optional[str] = None,              # Optional: join LHS metadata
    manifest_name: str = "paint_manifest.json", # Manifest filename
) -> List[str]:
    """
    Paint multiple y-map patches with XGPaint (Julia), grouped by cosmology.
    Key features:
      - Dedicated Julia project with pinned deps.
      - Float32 enmap storage and non-hardcoded pixel resolution.
      - Output directory structure: output_dir / <cosmo_subdir> / y_map_*.fits
      - Optional LHS matching (closest row in lhs_csv).
      - JSON manifest merged/appended on each run.
      - Prints total runtime for all patches (seconds).
    """
    # Start global timer (seconds)
    t_start = time.perf_counter()

    if envdir is None:
        envdir = os.path.join(os.path.expanduser("~"), ".julia", "environments", "xgpaint_env")
    _setup_julia_env(envdir, xgpaint_url=xgpaint_url, xgpaint_rev=xgpaint_rev, quiet=False)
    jl.seval(r"""
import Pkg, InteractiveUtils, Base
println("── Julia/Project info ─────────────────────────────")
InteractiveUtils.versioninfo()
println("active project: ", Base.active_project())
println("depot path    : ", Base.DEPOT_PATH)
println()
println("── Pkg.status ──────────────────────────────────────")
Pkg.status(["XGPaint","Healpix","Pixell","HDF5","HTTP","OpenSSL"])
println()
try
    @eval using XGPaint
    xgpath = pathof(XGPaint)
    println("XGPaint.path : ", xgpath)
    println("XGPaint.uuid : ", Base.PkgId(XGPaint).uuid)
    deps = Pkg.dependencies()
    xg = filter(d -> d.second.name == "XGPaint", deps) |> first |> last
    println("XGPaint.version     : ", get(xg, :version, "n/a"))
    println("XGPaint.repo        : ", get(xg, :repo, "n/a"))
    println("XGPaint.rev/branch  : ", get(xg, :rev, "n/a"))
    println("XGPaint.git-tree-sha: ", get(xg, :git_tree_sha1, "n/a"))
catch e
    @warn "XGPaint not loadable" exception=(e, catch_backtrace())
end
println("────────────────────────────────────────────────────")
""")

    _ensure_julia_symbols()

    # Validate and expand catalogue inputs
    if catalogs is None:
        raise ValueError("Provide at least one catalogue CSV or a directory via 'catalogs'.")

    if isinstance(catalogs, (str, os.PathLike)):
        cand = [str(catalogs)]
    else:
        cand = [str(p) for p in catalogs]

    # Enforce filename tokens unless require_params_in_title=False
    cat_list = _csvs_from_paths(cand, recursive=recursive, require_params_in_title=True)
    if not cat_list:
        raise ValueError("No CSV found in 'catalogs'. Expected filenames containing 'logA=' and 'Oc0h2='.")

    # Normalize params list to match cat_list arity
    if params is None:
        par_list = [None] * len(cat_list)
    elif isinstance(params, (str, os.PathLike)):
        par_list = [str(params)] * len(cat_list)
    else:
        par_list = [str(p) if p is not None else None for p in params]

    if any(p is not None for p in par_list) and len(par_list) != len(cat_list):
        raise ValueError("If 'params' are provided, they must match the number of expanded CSVs.")

    # Prepare manifest bookkeeping
    outs: List[str] = []
    records: List[dict] = []
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / manifest_name

    # Utility: create a temporary params CSV when none is provided
    import tempfile, csv
    def _write_tmp_param_row(params_dict, tmpdir, idx):
        """
        Write a one-row CSV with columns (h, Ob0h2, Oc0h2, B, logA).
        Missing values are left blank; logA may be blank.
        """
        needed = ("h","Ob0h2","Oc0h2","B","logA")
        out = os.path.join(tmpdir, f"params_{idx}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(needed))
            w.writeheader()
            row = {k: params_dict.get(k, "") for k in needed}
            w.writerow(row)
        return out

    with tempfile.TemporaryDirectory(prefix="tsz_params_") as tmpdir:
        for i, cfile in enumerate(cat_list):
            # (1) Build params dict 'p' (from user CSV, title_reader, or filename fallback)
            if par_list[i] is None:
                try:
                    from title_reader import extract_params  # type: ignore
                    p = extract_params(cfile)
                    for k in ("h","Ob0h2","Oc0h2","B"):
                        if k not in p:
                            raise KeyError(k)
                    if "logA" not in p:
                        # If logA missing, try to get it from filename
                        p["logA"] = _default_extract_params_from_name(cfile)["logA"]
                except Exception:
                    p = _default_extract_params_from_name(cfile)
                pfile = _write_tmp_param_row(p, tmpdir, i)
            else:
                pfile = par_list[i]
                # Read the real params CSV to populate manifest; fall back to extractors as needed
                try:
                    p = _read_params_csv_first_row(pfile)
                except Exception:
                    try:
                        from title_reader import extract_params  # type: ignore
                        p = extract_params(cfile)
                    except Exception:
                        p = _default_extract_params_from_name(cfile)

            # (2) Choose output subdirectory based on cosmology
            cosmo_subdir = _cosmo_subdir_name(p)
            target_dir = str(Path(output_dir) / cosmo_subdir)

            # (3) Call the Julia painter (non-hardcoded pixel resolution + float32 map)
            out = jl.paint_one_catalogue(
                str(cfile), str(pfile), int(i),
                float(patch_size_deg), float(pix_res_arcmin), str(target_dir),
                beam_fwhm_arcmin=(float(beam_fwhm_arcmin) if beam_fwhm_arcmin is not None else None),
                nx=int(nx),
            )

            # (4) Collect output and write a manifest record
            if out is not None:
                out_path = str(out)
                outs.append(out_path)

                h = float(p["h"]); Ob0h2 = float(p["Ob0h2"]); Oc0h2 = float(p["Oc0h2"])
                Omega_m = (Ob0h2 + Oc0h2) / (h*h)
                Omega_c = Oc0h2 / (h*h)
                rec = {
                    "paint_path": out_path,
                    "cosmo_subdir": cosmo_subdir,
                    "catalogue_csv": str(cfile),
                    "params": {
                        "h": h, "Ob0h2": Ob0h2, "Oc0h2": Oc0h2,
                        "B": float(p.get("B", np.nan)),
                        "logA": float(p.get("logA", np.nan)),
                    },
                    "derived": {"Omega_m": Omega_m, "Omega_c": Omega_c},
                    "pixel_resolution_arcmin": float(pix_res_arcmin),
                    "storage_dtype": "float32",
                }
                if lhs_csv:
                    try:
                        match = _match_row_in_lhs(lhs_csv, rec["params"]["logA"], rec["params"]["Oc0h2"])
                        rec["lhs_match"] = match
                    except Exception as e:
                        rec["lhs_match_error"] = str(e)
                records.append(rec)

    # (5) Merge/append manifest and print total runtime (in seconds)
    try:
        if manifest_path.exists():
            prev = json.loads(manifest_path.read_text())
            if isinstance(prev, list):
                records = prev + records
        manifest_path.write_text(json.dumps(records, indent=2))
        print(f"[paint_patches] manifest written → {manifest_path}")
    except Exception as e:
        print(f"[paint_patches] WARN: could not write manifest: {e}")

    t_total = time.perf_counter() - t_start
    print(f"[paint_patches] total_time_seconds={t_total:.3f}")

    return outs
