import os
import time
import json
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
import sys
import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
import os, sys

# utils_import.py
from importlib import invalidate_caches, reload
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

# --- 0) Env pour éviter les warnings (à faire avant tout import du paquet) ---
import os
os.environ.setdefault("PATH_TO_COSMOPOWER_ORGANIZATION", "/rds-d4/user/iz221/hpc-work/cosmopower/")
os.environ.setdefault("PATH_TO_COSMOCNC", "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc")

# --- 1) Forcer l'import de cosmocnc depuis le REPO RACINE ---
import sys, importlib
from pathlib import Path
from importlib import invalidate_caches

PKG_ROOT = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc"  # << le bon dossier
MOD_NAME = "cosmocnc"

# Purge d'anciens imports/chemins "cosmocnc"
for n in list(sys.modules):
    if n == MOD_NAME or n.startswith(MOD_NAME + "."):
        sys.modules.pop(n, None)

def _endswith_basename(p, name): 
    try: return Path(p).name.lower() == name
    except Exception: return False

sys.path = [p for p in sys.path if not _endswith_basename(p, MOD_NAME)]
# Ajouter UNIQUEMENT le repo racine
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

invalidate_caches()
cnc = importlib.import_module(MOD_NAME)

print("cosmocnc chargé depuis :", getattr(cnc, "__file__", None))
print("attrs scaling :", [n for n in dir(cnc) if "scal" in n.lower()])

# --- 2) Patcher le module générateur qui consomme `cnc` ---
import importlib
import cata_generator_cop as cg
importlib.reload(cg)
cg.cnc = cnc

# filet de sécurité si le nom a changé selon la version
if not hasattr(cg.cnc, "scaling_relation_params_default"):
    try:
        from cosmocnc import sr as _sr
        cg.cnc.scaling_relation_params_default = _sr.scaling_relation_params_default
    except Exception:
        pass

def load_local_cosmocnc(pkg_dir: str):
    pkg = Path(pkg_dir)
    for n in list(sys.modules):
        if n == "cosmocnc" or n.startswith("cosmocnc."): sys.modules.pop(n, None)
    sys.path[:] = [p for p in sys.path if Path(p).name.lower() != "cosmocnc"]; invalidate_caches()
    spec = spec_from_file_location("cosmocnc", pkg/"__init__.py", submodule_search_locations=[str(pkg)])
    cnc = module_from_spec(spec); sys.modules["cosmocnc"] = cnc; spec.loader.exec_module(cnc)
    parent = str(pkg.parent)
    if parent not in sys.path: sys.path.insert(0, parent)
    import cata_generator; reload(cata_generator); cata_generator.cnc = cnc
    try:
        if not hasattr(cata_generator.cnc, "scaling_relation_params_default"):
            from cosmocnc import sr as _sr
            cata_generator.cnc.scaling_relation_params_default = _sr.scaling_relation_params_default
    except Exception: pass
    return cnc, cata_generator

def _to_float32(obj):
    """Convertit récursivement tous les arrays float64 en float32."""
    import numpy as np
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
        return obj.astype(np.float32)
    if isinstance(obj, dict):
        return {k: _to_float32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_float32(v) for v in obj]
    return obj


# 1) Si un 'cosmocnc' est déjà chargé (bootstrap en amont), on le réutilise tel quel
cnc = sys.modules.get("cosmocnc")

# 2) Sinon, on essaie un import direct
if cnc is None:
    try:
        import cosmocnc as cnc  # type: ignore
    except Exception:
        # 3) Fallback via variable d’environnement: DOIT pointer vers le paquet interne (…/cosmocnc/cosmocnc)
        p = os.environ.get("PATH_TO_COSMOCNC")
        if not p:
            raise RuntimeError(
                "PATH_TO_COSMOCNC n'est pas défini. "
                "Définis-le vers …/cosmocnc/cosmocnc (le dossier avec __init__.py)."
            )
        # Nettoyer sys.path des dossiers '…/cosmocnc' (repo externe) qui créent un namespace
        def _endswith_cosmocnc(x): return x.rstrip("/").split("/")[-1].lower() == "cosmocnc"
        sys.path = [x for x in sys.path if not _endswith_cosmocnc(x)]
        if p not in sys.path:
            sys.path.insert(0, p)
        import cosmocnc as cnc  # type: ignore

# Logs compacts et fiables
print("cosmocnc file:", getattr(cnc, "__file__", None))
print("scaling names:", [n for n in dir(cnc) if "scal" in n.lower()])




def _compute_f_sky_from_patch(patch_deg_width: float, patch_deg_height: float) -> float:
    """
    Convert a rectangular sky patch size (in degrees) to f_sky fraction.
    """
    sky_area_deg2 = float(patch_deg_width) * float(patch_deg_height)
    full_sky_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
    f_sky = sky_area_deg2 / full_sky_deg2
    # Guard against pathological inputs
    f_sky = max(0.0, min(1.0, f_sky))
    #f_sky = 1.0
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
    # Ensure we can index like an array even if there's only one row
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
    """
    Build baseline cnc_params. Paths can be provided to local survey modules.
    """
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
    # Remove trailing zeros and possible trailing dot
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    # Keep minus signs and 'e' if present (though we avoid scientific notation by using fixed decimals)
    return s


# ---------- NEW: CSV writer helper ----------

def _write_catalogue_csv(catalogue: Dict[str, object], out_path: str) -> None:
    """
    Écrit un catalogue (dict colonnes -> arrays/listes) en CSV.
    - Conserve uniquement les colonnes 1D de même longueur (colonnes "par cluster").
    - Ignore les scalaires et tableaux non-1D pour éviter les ambiguïtés.
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
            # ignore colonnes de taille différente
            continue
        cols[k] = np.asarray(arr, dtype=np.float32).tolist()


    # Fichier vide mais valide si aucune colonne exploitable
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

def process_npy_catalogues(
    inputs,
    ra0_deg: float = 0.0,
    dec0_deg: float = 0.0,
    w_deg: float = 10.0,
    h_deg: float = 10.0,
    seed: int | None = 42,
    pattern: str = "*.npy",
    recursive: bool = True,
    lon_wrap: str = "pm_pi",      # "pm_pi" => [-π,π], "0_2pi" => [0,2π)
    lon_col: str = "lon",
    lat_col: str = "lat",
    make_backup: bool = False,    # True => écrit <file>.bak.npy avant d'écraser
):
    """
    Traite un ou plusieurs chemins (fichiers NPY ou dossiers) et réécrit *le même fichier .npy*
    en y ajoutant deux coordonnées supplémentaires :
      - lon : longitude (rad)
      - lat : colatitude (rad)

    Comportements selon le contenu NPY:
      1) Tableau structuré (arr.dtype.names != None) -> ajout de deux champs.
      2) Dictionnaire (np.load(..., allow_pickle=True).item() de type dict) -> ajout de deux clés.
      3) Tableau simple (ndarray 1D/2D) -> réécriture en dict {"data": arr, lon_col: ..., lat_col: ...}.

    Retour
    ------
    summary : list[dict]
        Liste de résumés par fichier : {"in":..., "n":..., "skipped": bool, "reason": str|None}
    """
    # Normalise la liste d'entrées
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    inputs = [Path(p) for p in inputs]

    # Récupère la liste finale de NPY à traiter
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
        print("Aucun NPY trouvé.")
        return []

    summary = []

    for f in npy_files:
        # Seed par fichier (stable)
        file_seed = _seed_from_path(f, seed)
        rng = np.random.default_rng(file_seed) if file_seed is not None else np.random.default_rng()

        # Lecture: on autorise le pickle pour gérer les dicts
        try:
            loaded = np.load(f, allow_pickle=True)
        except Exception as e:
            summary.append({"in": str(f), "n": 0, "skipped": True, "reason": f"read_error: {e}"})
            continue

        # Détermine la "forme logique" du contenu
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
                    # Type non géré -> on tente de le conserver dans "data"
                    arr = candidate
            elif isinstance(loaded, np.ndarray):
                arr = loaded
            else:
                # Cas exotique
                arr = np.asarray(loaded)
        except Exception:
            arr = np.asarray(loaded)

        # Cas 2: dict
        if data_obj is not None:
            # Détermine n
            # On essaie de deviner un "n" pertinent parmi les valeurs du dict
            n = None
            for v in data_obj.values():
                try:
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
                        n = v.shape[0]
                        break
                except Exception:
                    pass
            if n is None:
                summary.append({"in": str(f), "n": 0, "skipped": True, "reason": "cannot_infer_n_from_dict"})
                continue

            # Échantillonne
            lon, theta = sample_lonlat_patch(n, ra0_deg=ra0_deg, dec0_deg=dec0_deg, w_deg=w_deg, h_deg=h_deg, rng=rng)
            if lon_wrap == "0_2pi":
                lon = lon % TWOPI
            elif lon_wrap == "pm_pi":
                lon = wrap_pm_pi(lon)
            else:
                summary.append({"in": str(f), "n": n, "skipped": True, "reason": "bad_lon_wrap"})
                continue

            data_obj[lon_col] = lon
            data_obj[lat_col] = theta

            # sauvegarde (backup éventuel)
            try:
                if make_backup:
                    f.with_suffix(".bak.npy").write_bytes(f.read_bytes())
                np.save(f, data_obj, allow_pickle=True)
                summary.append({"in": str(f), "n": n, "skipped": False, "reason": None})
            except Exception as e:
                summary.append({"in": str(f), "n": n, "skipped": True, "reason": f"write_error: {e}"})
            continue

        # Cas 1 ou 3: tableau NumPy
        if not isinstance(arr, np.ndarray):
            summary.append({"in": str(f), "n": 0, "skipped": True, "reason": "unsupported_npy_content"})
            continue

        # Détermine n
        if arr.ndim == 0:
            summary.append({"in": str(f), "n": 0, "skipped": True, "reason": "scalar_array_unsupported"})
            continue
        n = arr.shape[0]

        # Échantillonne
        lon, theta = sample_lonlat_patch(n, ra0_deg=ra0_deg, dec0_deg=dec0_deg, w_deg=w_deg, h_deg=h_deg, rng=rng)
        if lon_wrap == "0_2pi":
            lon = lon % TWOPI
        elif lon_wrap == "pm_pi":
            lon = wrap_pm_pi(lon)
        else:
            summary.append({"in": str(f), "n": n, "skipped": True, "reason": "bad_lon_wrap"})
            continue

        # Cas 1: tableau structuré -> on ajoute deux champs
        try:
            if arr.dtype.names is not None:
                # construit un nouveau dtype
                new_descr = list(arr.dtype.descr)
                # évite les collisions de noms
                if lon_col in arr.dtype.names or lat_col in arr.dtype.names:
                    summary.append({"in": str(f), "n": n, "skipped": True, "reason": "field_name_conflict"})
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
                    summary.append({"in": str(f), "n": n, "skipped": False, "reason": None})
                except Exception as e:
                    summary.append({"in": str(f), "n": n, "skipped": True, "reason": f"write_error: {e}"})
                continue
        except Exception:
            # si l'accès à dtype.names plante, on retombe sur le cas 3
            pass

        # Cas 3: tableau simple -> on réécrit un dict pour préserver l'array original
        try:
            payload = {"data": arr, lon_col: lon, lat_col: theta}
            if make_backup:
                f.with_suffix(".bak.npy").write_bytes(f.read_bytes())
            payload_f32 = _to_float32(payload)
            np.save(f, payload_f32, allow_pickle=True)
            summary.append({"in": str(f), "n": n, "skipped": False, "reason": None})
        except Exception as e:
            summary.append({"in": str(f), "n": n, "skipped": True, "reason": f"write_error: {e}"})

    done = sum(1 for s in summary if not s["skipped"])
    skipped = len(summary) - done
    print(f"Terminé : {done} fichier(s) réécrit(s), {skipped} sauté(s).")
    return summary


# ---------- NEW: generic input normalization ----------

def _read_csv_all_rows(path: Union[str, os.PathLike]) -> List[Dict[str, float]]:
    """
    Read all rows from a CSV file.
    Requires header with at least the 5 required columns.
    Returns a list of dicts containing only the required columns; also returns
    (via side channel) the list of extra columns for manifest (handled by caller).
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
    Automatically extracts only required cosmology parameters, even if others are present.
    Ensures all required values are finite floats.
    """
    if pd is None:
        raise ImportError("pandas is required to pass a DataFrame as input, but it's not installed.")

    required = ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    df_num = df[required].astype(float).copy()

    # Guard against NaN / Inf values
    vals = df_num.to_numpy()
    if not np.all(np.isfinite(vals)):
        raise ValueError("The DataFrame contains non-finite values (NaN or Inf) in required columns.")

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
      - provenance: List[Dict[str, object]] with metadata for manifests
                    (e.g., source path, row index)
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
        raise TypeError(
            "cosmo_input must be a directory path, a CSV file path, or a pandas DataFrame."
        )
    path = str(cosmo_input)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path}")

    # Directory of CSV files (backward-compatible behavior: first row per file)
    if os.path.isdir(path):
        csv_paths = sorted(glob.glob(os.path.join(path, csv_glob)))
        if len(csv_paths) == 0:
            raise FileNotFoundError(f"No CSV files found in {path!r} matching pattern {csv_glob!r}")
        for c_idx, csv_path in enumerate(csv_paths[: int(n_cosmologies)]):
            params = _load_params_from_csv_first_row(csv_path)
            param_rows.append(params)
            provenance.append(
                {"source_type": "csv_dir", "csv_path": csv_path, "file_index": c_idx, "row_index": 0}
            )
        return param_rows, provenance

    # Single CSV file with potentially multiple rows
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
    output_format: str = "npy",  # NEW: "npy" | "csv" | "both"
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run the catalogue generation pipeline.

    Parameters
    ----------
    ...
    output_format : one of {"npy","csv","both"} to control file outputs.
    """
    if cnc is None:
        raise ImportError("cosmocnc is not available in the current environment. Install and retry.")

    if output_format not in ("npy", "csv", "both"):
        raise ValueError(f"Invalid output_format='{output_format}'. Use 'npy', 'csv', or 'both'.")

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

    # Baseline cosmology (will be overwritten per row)
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

    for c_idx, row_params in enumerate(param_rows):
        # update cosmology from row (required keys only)
        for k in ["h", "logA", "n_s", "Ob0h2", "Oc0h2"]:
            cosmo_params[k] = float(row_params[k])

        number_counts.update_params(cosmo_params, scal_rel_params)

        # Compute derived parameters for naming
        logA_val = float(cosmo_params["logA"])
        oc0h2_val = float(cosmo_params["Oc0h2"])

        # Omega_c = Oc0h2 / h^2
        omega_c_val = float(cosmo_params["Oc0h2"]) / (float(cosmo_params["h"]) ** 2)
        logA_str = _format_float_for_filename(logA_val)
        oc0h2_str  = _format_float_for_filename(oc0h2_val)
        omega_c_str = _format_float_for_filename(omega_c_val)

        # Per-cosmology folder (use index + source description for clarity when possible)
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

        # Manifest
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
            "output_format": output_format,  # NEW
        }

        # If input came from a CSV file or DataFrame and there were extra columns, record them
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
                n_catalogues=1,  # we save each catalogue separately
                seed=unique_seed,
                get_sky_coords=get_sky_coords,
                sky_frac=f_sky,
            )
            t0 = time.time()
            cat_gen.generate_catalogues_hmf()
            catalogue_list = cat_gen.catalogue_list
            # There is 1 catalogue because n_catalogues=1 above
            catalogue = catalogue_list[0] if isinstance(catalogue_list, list) else catalogue_list

            # === File naming stem (kept) ===
            out_stem = f"logA={logA_str}_Oc0h2={oc0h2_str}_{j:04d}"
            files_this_catalogue = []

            # --- write .npy (compat: keep the list) ---
            if output_format in ("npy", "both"):
                npy_path = os.path.join(cosmo_folder, out_stem + ".npy")
                catalogue_list_f32 = _to_float32(catalogue_list)
                np.save(npy_path, catalogue_list_f32)
                written_files.append(npy_path)
                files_this_catalogue.append({"file": os.path.basename(npy_path), "format": "npy"})

            # --- write .csv (tabular per-cluster) ---
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
                print(f"[cosmo {c_idx:03d}] catalogue {j:04d} -> {human_files}")

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
        "output_format": output_format,  # NEW
        "total_elapsed_s": time.time() - run_started,
    }
    # Write a top-level summary too
    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    # EXAMPLES (edit paths for your environment):

    # 1) Dossier contenant des CSV (première ligne utilisée pour chaque fichier)
    # summary = generate_cluster_catalogues(
    #     cosmo_input="../tszsbi/catalogue_demo",  # <- dossier
    #     n_cosmologies=10,
    #     n_catalogues_per_cosmo=100,
    #     patch_size_deg=(10.0, 10.0),
    #     output_dir="../tszsbi/catalogue_generated",
    #     survey_sr_path="../cosmocnc/surveys/survey_sr_so_sim.py",
    #     survey_cat_path="../cosmocnc/surveys/survey_cat_so_sim.py",
    #     scaling_relation_overrides={"bias_sz": 0.8, "dof": 0.0},
    #     output_format="both",
    # )

    # 2) Fichier CSV unique (chaque ligne = une cosmologie)
    # summary = generate_cluster_catalogues(
    #     cosmo_input="../tszsbi/multirow_cosmologies.csv",  # <- fichier CSV unique
    #     n_cosmologies=5,  # on peut tronquer si le fichier a plus de lignes
    #     n_catalogues_per_cosmo=50,
    #     patch_size_deg=(10.0, 10.0),
    #     output_dir="../tszsbi/catalogue_generated_from_file",
    #     output_format="csv",
    # )

    # 3) DataFrame pandas (chaque ligne = une cosmologie)
    # import pandas as pd
    # df = pd.read_csv("../tszsbi/multirow_cosmologies.csv")
    # summary = generate_cluster_catalogues(
    #     cosmo_input=df,  # <- DataFrame
    #     n_cosmologies=len(df),
    #     n_catalogues_per_cosmo=20,
    #     patch_size_deg=(10.0, 10.0),
    #     output_dir="../tszsbi/catalogue_generated_from_df",
    #     output_format="npy",
    # )

    pass
