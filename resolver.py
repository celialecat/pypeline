"""
Packaging-friendly import & path resolver for your catalogue pipeline.

Goal: Eliminate hard-coded absolute paths and sys.path hacks.

This file provides:
1) `resolver.py`: a small utility that
   - imports `cosmocnc` in a packaging-friendly way (installed dep or vendored copy)
   - exposes a survey registry so callers can pass logical keys instead of file paths
   - resolves resources via importlib.resources (PEP 302/451 compliant)
2) Patch snippets showing how to integrate resolver into your pipeline and public API.
3) A minimal `pyproject.toml` template.

Usage from user code after integration:

    from yourpkg.api import generate_cluster_catalogues

    summary = generate_cluster_catalogues(
        cosmo_input=df,
        n_cosmologies=1,
        n_catalogues_per_cosmo=2,
        patch_size_deg=(10.0, 10.0),
        output_dir="./outputs/test_56",
        survey="so_sim",              # <-- logical key instead of absolute paths
        cnc_params_overrides={"n_points": 10000, "n_z": 10000},
        verbose=True,
        output_formats=["csv"],
    )

"""

# ======================
# 1) NEW MODULE: resolver.py
# ======================

# Save this as yourpkg/resolver.py

from __future__ import annotations
import os
import types
import importlib
import importlib.util
import importlib.resources as ir
from typing import Optional, Tuple


class ImportErrorWithHint(ImportError):
    pass


# --- cosmocnc resolver -------------------------------------------------------

# resolver.py
import os, sys, types, importlib

class ImportErrorWithHint(ImportError):
    pass

def _looks_valid_cnc(mod: types.ModuleType) -> bool:
    # Indices légers d'une vraie API cosmocnc
    names = dir(mod)
    candidates = {
        "cluster_number_counts", "ClusterNumberCounts",
        "scaling_relation_params_default", "catalogue_generator"
    }
    return any(n in names for n in candidates)

def import_cnc() -> types.ModuleType:
    """
    Importe cosmocnc de manière robuste :
    - respecte PATH_TO_COSMOCNC s'il est défini (👉 doit pointer vers .../cosmocnc/cosmocnc)
    - nettoie les entrées sys.path '.../cosmocnc' qui créent un namespace vide
    - essaie plusieurs cibles et valide que l'API ressemble bien à cosmocnc
    """
    # 0) Priorité à PATH_TO_COSMOCNC s'il est présent
    p = os.environ.get("PATH_TO_COSMOCNC")
    if p:
        def _endswith_cosmocnc(x: str) -> bool:
            return str(x).rstrip("/").split("/")[-1].lower() == "cosmocnc"
        # Retire les namespaces 'root cosmocnc' qui masquent la vraie lib
        sys.path = [x for x in sys.path if not _endswith_cosmocnc(x)]
        if p not in sys.path:
            sys.path.insert(0, p)

    candidates = [
        "cosmocnc",                 # lib installée
        "cosmocnc.cosmocnc",        # parfois imbriquée
        "yourpkg.vendors.cosmocnc", # copie vendored éventuelle
    ]
    last_err = None
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if _looks_valid_cnc(mod):
                return mod
        except Exception as e:
            last_err = e
            continue

    raise ImportErrorWithHint(
        "Impossible d'importer 'cosmocnc'. Définis PATH_TO_COSMOCNC vers …/cosmocnc/cosmocnc "
        "(le dossier qui contient __init__.py), ou installe la librairie (pip install cosmocnc). "
        f"Dernière erreur : {last_err!r}"
    )


# --- survey registry ----------------------------------------------------------

"""
We provide a registry of survey modules that live *inside* the package,
so callers only pass a logical key (e.g. "so_sim").

Expected modules (examples) shipped in your package:
- yourpkg.surveys.survey_sr_so_sim
- yourpkg.surveys.survey_cat_so_sim

You can add more surveys later and just register them here.
"""

_SURVEYS = {
    # key: (survey_sr_module, survey_cat_module)
    "so_sim": (
        "yourpkg.surveys.survey_sr_so_sim",
        "yourpkg.surveys.survey_cat_so_sim",
    ),
}


def resolve_survey_modules(key: Optional[str]) -> Tuple[Optional[types.ModuleType], Optional[types.ModuleType]]:
    """Return (sr_module, cat_module) for a logical survey key.

    If key is None, returns (None, None) allowing cnc defaults.
    If key contains dots (".") we treat it as a fully-qualified module base:
      - f"{key}.survey_sr"
      - f"{key}.survey_cat"
    """
    if key is None:
        return None, None

    # treat "pkg.path" as a base and try two conventional names
    if "." in key:
        sr_mod_name = f"{key}.survey_sr"
        cat_mod_name = f"{key}.survey_cat"
        sr_mod = importlib.import_module(sr_mod_name)
        cat_mod = importlib.import_module(cat_mod_name)
        return sr_mod, cat_mod

    # registry lookup
    if key not in _SURVEYS:
        raise KeyError(f"Unknown survey key '{key}'. Known: {sorted(_SURVEYS.keys())}")
    sr_name, cat_name = _SURVEYS[key]
    sr_mod = importlib.import_module(sr_name)
    cat_mod = importlib.import_module(cat_name)
    return sr_mod, cat_mod


# --- resources helper ---------------------------------------------------------

def resource_path(package: str, name: str) -> str:
    """Return an absolute path to a packaged resource (for libs that need file paths).

    Uses importlib.resources.files to locate artefacts shipped inside wheels/sdists.
    """
    p = ir.files(package).joinpath(name)
    # Ensure it's present
    if not p.exists():
        raise FileNotFoundError(f"Resource '{name}' not found in package '{package}'.")
    return str(p)


# ============================
# 2) PATCHES / PUBLIC API LAYER
# ============================

# Save this as yourpkg/api.py (thin wrapper around your existing function)

API_SNIPPET = r"""
from __future__ import annotations
from typing import Optional, Iterable, Union, Tuple

from .resolver import import_cnc, resolve_survey_modules

# import the actual implementation from your current module
from .cata_generator import generate_cluster_catalogues as _impl


def generate_cluster_catalogues(
    cosmo_input,
    n_cosmologies: int,
    n_catalogues_per_cosmo: int,
    patch_size_deg: Tuple[float, float],
    output_dir,
    *,
    survey: Optional[str] = "so_sim",   # <-- logical key; defaults to packaged SO sim
    output_formats: Union[str, Iterable[str]] = ("npy",),
    cnc_params_overrides=None,
    baseline_cosmo_params=None,
    scaling_relation_overrides=None,
    seed_offset: int = 20000401,
    get_sky_coords: bool = False,
    override_f_sky: float | None = None,
    verbose: bool = True,
    csv_glob: str = "*.csv",
    cosmology_folder_prefix: str = "cosmo",
    catalogue_file_tpl: str = "catalogue_{cosmo_idx:03d}_{cat_idx:04d}.npy",
    manifest_name: str = "manifest.json",
):

    Public, packaging-friendly API: no absolute paths; no sys.path edits.
    - `survey` is a logical key or a dotted module base; we locate proper modules.
    - `cosmocnc` is imported as a dependency or vendored module.

    cnc = import_cnc()
    sr_mod, cat_mod = resolve_survey_modules(survey)

    # Call the underlying implementation; pass module paths if your impl expects strings,
    # or pass module objects if you've adapted it to accept them (recommended).
    survey_sr_path = sr_mod.__file__ if sr_mod is not None else None
    survey_cat_path = cat_mod.__file__ if cat_mod is not None else None

    return _impl(
        cosmo_input=cosmo_input,
        n_cosmologies=n_cosmologies,
        n_catalogues_per_cosmo=n_catalogues_per_cosmo,
        patch_size_deg=patch_size_deg,
        output_dir=output_dir,
        csv_glob=csv_glob,
        override_f_sky=override_f_sky,
        get_sky_coords=get_sky_coords,
        survey_sr_path=survey_sr_path,
        survey_cat_path=survey_cat_path,
        seed_offset=seed_offset,
        cnc_params_overrides=cnc_params_overrides,
        baseline_cosmo_params=baseline_cosmo_params,
        scaling_relation_overrides=scaling_relation_overrides,
        cosmology_folder_prefix=cosmology_folder_prefix,
        catalogue_file_tpl=catalogue_file_tpl,
        manifest_name=manifest_name,
        output_formats=output_formats,
        verbose=verbose,
    )
"""

# -----------------------------------------------------
# Optional patch in your implementation (recommended):
# allow passing *modules* directly for survey_sr / survey_cat.
# -----------------------------------------------------

ALLOW_MODULES_PATCH = r"""
@@
-def _default_cnc_params(
-    survey_sr: Optional[str],
-    survey_cat: Optional[str],
+def _default_cnc_params(
+    survey_sr: Optional[object],   # path str or module with __file__
+    survey_cat: Optional[object],
@@
-    if survey_sr is not None:
-        params["survey_sr"] = survey_sr
-    if survey_cat is not None:
-        params["survey_cat"] = survey_cat
+    def _to_path(x):
+        if x is None:
+            return None
+        if hasattr(x, "__file__"):
+            return x.__file__  # module object
+        return str(x)          # string path
+    if survey_sr is not None:
+        params["survey_sr"] = _to_path(survey_sr)
+    if survey_cat is not None:
+        params["survey_cat"] = _to_path(survey_cat)
    return params
"""

# ==========================
# 3) Minimal pyproject.toml
# ==========================

PYPROJECT = r"""
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "yourpkg"
version = "0.1.0"
description = "Cluster catalogue generator with packaging-friendly imports"
authors = [{name = "You"}]
readme = "README.md"
requires-python = ">=3.9"''
dependencies = [
  "numpy",
  "pandas; extra == 'full'",
  "cosmocnc>=0.1.0",  # if published; otherwise remove and vendor it
]

[tool.setuptools.packages.find]
where = ["."]
include = ["yourpkg*"]

[tool.setuptools.package-data]
# ship survey modules and any data files they need
"yourpkg.surveys" = ["*.py", "*.json", "*.yml", "*.txt"]

[project.optional-dependencies]
full = ["pandas", "pyarrow"]

[project.urls]
Homepage = "https://example.com/yourpkg"
"""

if __name__ == "__main__":
    print("This file contains resolver utilities, API snippet, optional patches, and a pyproject template.")
