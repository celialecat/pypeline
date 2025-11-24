# patch_generator_ju_fixed.py
import os
import sys
import json
import shutil
import time
import textwrap
import subprocess
from pathlib import Path

# -----------------------------------------------------------------------------
# Garde-fous threads + ENV Julia à définir AVANT tout import relié à Julia
# -----------------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JULIA_NUM_THREADS", "1")

os.environ.setdefault("JULIA_DEPOT_PATH", "/rds/rds-clecat/pipeline_alina_full/.julia_depot")
os.environ.setdefault("JULIA_PROJECT",    "/rds/rds-clecat/pipeline_alina_full/alina_paper/pipe_env/julia_env")
os.environ.setdefault("JULIA_PKG_PRECOMPILE_AUTO", "0")

import pandas as pd
import numpy as np

from cata_generator_en import load_local_cosmocnc, generate_cluster_catalogues
from coordinates_attributor_en import process_catalogues

BASE_WORK_DIR   = "/rds/rds-clecat/pipeline_alina_full/alina_paper/pipeline_outputs/mcmc_runs"
COSMOCNC_PATH   = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/cosmocnc"
SURVEY_SR_PATH  = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/surveys/survey_sr_so_sim.py"
SURVEY_CAT_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/surveys/survey_cat_so_sim.py"
XGPAINT_URL     = "/rds/rds-clecat/pipeline_alina_full/alina_paper/XGPaint.jl"  # repo local

print("--> Initialisation de cosmocnc...")
_, _cg = load_local_cosmocnc(COSMOCNC_PATH)

JSON_START = "<<JSON_START>>"
JSON_END   = "<<JSON_END>>"

# -----------------------------------------------------------------------------
# Lanceur enfant (processus Python séparé)
# -----------------------------------------------------------------------------
def _run_painter_subprocess(
    catalogs, params, output_dir, patch_size_deg, pix_res_arcmin, nx,
    envdir, xgpaint_url, recursive=False, beam_fwhm_arcmin=None,
    py_exec=sys.executable, log_path=None
):
    """
    Démarre un sous-processus Python qui :
      - active l'env Julia,
      - force XGPaint local via Pkg.develop(path=...),
      - garantit 'Arnauld10ThermalSZProfile' (alias 'Arnaud10...' si besoin),
      - appelle patch_painter_ve.paint_patches(...).

    Tous les logs peuvent aller sur stdout/stderr; on imprime à la FIN une ligne
    balisée contenant UNIQUEMENT le JSON pour le parent.
    """
    depot_json    = json.dumps(os.environ["JULIA_DEPOT_PATH"])
    envdir_json   = json.dumps(envdir)
    xgpaint_json  = json.dumps(xgpaint_url)
    outdir_json   = json.dumps(str(output_dir))
    catalogs_json = json.dumps([str(p) for p in catalogs])
    params_json   = json.dumps([str(p) for p in params])

    nx_int        = int(nx)
    pix_res       = float(pix_res_arcmin)
    patch_size    = float(patch_size_deg)
    beam_expr_py      = "None" if beam_fwhm_arcmin is None else repr(float(beam_fwhm_arcmin))
    recursive_expr_py = "True" if bool(recursive) else "False"

    child_code = textwrap.dedent(f"""
        import os, json, sys
        from juliacall import Main as jl

        JSON_START = {json.dumps(JSON_START)}
        JSON_END   = {json.dumps(JSON_END)}

        # --- ENV pour Julia ---
        os.environ["JULIA_DEPOT_PATH"] = {depot_json}
        os.environ["JULIA_PROJECT"]    = {envdir_json}
        os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("JULIA_NUM_THREADS", "1")

        # --- Variables Python (chemins) ---
        envdir_py   = {envdir_json}
        xgpaint_py  = {xgpaint_json}
        outdir_py   = {outdir_json}
        catalogs_py = {catalogs_json}
        params_py   = {params_json}

        # --- Pkg: activer et pinner le repo local via API PythonCall ---
        jl.seval("import Pkg")
        jl.Pkg.activate(envdir_py)
        jl.Pkg.develop(path=xgpaint_py)
        jl.Pkg.resolve()
        jl.Pkg.instantiate()

        # --- Charger XGPaint ---
        jl.seval("using XGPaint")
        if not jl.seval("isdefined(XGPaint, :Arnauld10ThermalSZProfile)"):
            if jl.seval("isdefined(XGPaint, :Arnaud10ThermalSZProfile)"):
                jl.seval("Base.eval(XGPaint, :(const Arnauld10ThermalSZProfile = Arnaud10ThermalSZProfile))")
            else:
                raise RuntimeError("XGPaint ne fournit ni Arnauld10ThermalSZProfile ni Arnaud10ThermalSZProfile.")

        # --- Appeler le wrapper Python existant ---
        from patch_painter_ve import paint_patches

        out = paint_patches(
            patch_size_deg={patch_size},
            pix_res_arcmin={pix_res},
            output_dir=outdir_py,
            catalogs=catalogs_py,
            params=params_py,
            nx={nx_int},
            envdir=envdir_py,
            xgpaint_url=xgpaint_py,
            recursive={recursive_expr_py},
            beam_fwhm_arcmin={beam_expr_py},
        )

        # IMPORTANT : une seule ligne JSON balisée à la fin
        sys.stdout.write(JSON_START + json.dumps(out or []) + JSON_END + "\\n")
        sys.stdout.flush()
    """)

    cmd = [py_exec, "-c", child_code]
    env = os.environ.copy()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    out, err = p.communicate()

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(out or "")
            f.write("\n\n=== STDERR ===\n")
            f.write(err or "")

    if p.returncode != 0:
        tail = "\n".join((err or "").splitlines()[-30:])
        raise RuntimeError(
            f"Le sous-processus de peinture a échoué (code={p.returncode}).\n"
            f"STDERR (extrait) :\n{tail}"
        )

    # ---- Extraction robuste de la section JSON balisée ----
    try:
        if out and (JSON_START in out) and (JSON_END in out):
            start = out.rfind(JSON_START)
            end   = out.rfind(JSON_END)
            if start != -1 and end != -1 and end > start:
                json_str = out[start+len(JSON_START):end]
                return json.loads(json_str)

        # Fallback: dernière ligne qui a l'air d'un tableau JSON
        if out:
            for line in reversed(out.strip().splitlines()):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    return json.loads(line)

        # Sinon : erreur explicite
        raise RuntimeError("Aucune section JSON balisée trouvée dans la sortie du sous-processus.")
    except Exception as e:
        raise RuntimeError(
            f"Impossible de parser la sortie JSON du sous-processus : {e}\n"
            f"STDOUT brut (fin) :\n" + "\n".join((out or "").splitlines()[-50:])
        )

# -----------------------------------------------------------------------------
# API principale
# -----------------------------------------------------------------------------
def get_simulation_patch(
    logA: float,
    Oc0h2: float,
    run_id: str = "run",
    cleanup: bool = True,
    beam_fwhm_arcmin=None,
    log_file: str | None = None,
):
    """
    Génère un patch tSZ simulé (profil Arnauld10...) pour (logA, Oc0h2).
    Renvoie le chemin du premier FITS généré.
    La peinture Julia se fait dans un sous-processus (stabilité du kernel).
    """
    iter_dir   = Path(BASE_WORK_DIR) / run_id
    dir_cat    = iter_dir / "1_catalogues"
    dir_coords = iter_dir / "2_coords"
    dir_maps   = iter_dir / "3_maps"
    dir_maps.mkdir(parents=True, exist_ok=True)

    # Cosmologie fixe
    h_fixed, Ob0h2_fixed, n_s = 0.6766, 0.02242, 0.9665

    # 1) Catalogues
    df_cosmo = pd.DataFrame({
        "logA":   [logA],
        "Oc0h2":  [Oc0h2],
        "h":      [h_fixed],
        "Ob0h2":  [Ob0h2_fixed],
        "n_s":    [n_s],
    })
    summary_gen = generate_cluster_catalogues(
        cosmo_input=df_cosmo,
        n_cosmologies=1,
        n_catalogues_per_cosmo=1,
        patch_size_deg=(10.0, 10.0),
        output_dir=str(dir_cat),
        survey_sr_path=SURVEY_SR_PATH,
        survey_cat_path=SURVEY_CAT_PATH,
        cnc_params_overrides={"n_points": 3000, "n_z": 3000},
        verbose=False,
        output_format="csv",
    )

    # 1.5) params pour le peintre
    params_csv_path = iter_dir / "params_for_painter.csv"
    pd.DataFrame({
        "h":     [h_fixed],
        "Ob0h2": [Ob0h2_fixed],
        "Oc0h2": [Oc0h2],
        "B":     [1.35],
        "logA":  [logA],
    }).to_csv(params_csv_path, index=False)

    # 2) Coordonnées
    cosmo_subdirs = [d["cosmo_folder"] for d in summary_gen["cosmo_summaries"]]
    if not cosmo_subdirs:
        raise RuntimeError("Aucun catalogue n'a été généré.")

    process_catalogues(
        cosmo_subdirs,
        out_dir=dir_coords,
        ra0_deg=0, dec0_deg=0,
        w_deg=10, h_deg=10,
        seed=None,
        recursive=True,
        lon_wrap="pm_pi",
    )

    coords_csvs = sorted(Path(dir_coords).glob("*.csv"))
    if not coords_csvs:
        raise RuntimeError("Aucun fichier de coordonnées trouvé.")

    # 3) Peinture (sous-processus)
    nx_calc = int(10.0 * 60.0 / 0.5)  # 10 deg / 0.5' ≈ 1200 px

    map_paths = _run_painter_subprocess(
        catalogs=[str(p) for p in coords_csvs],
        params=[str(params_csv_path)] * len(coords_csvs),
        output_dir=str(dir_maps),
        patch_size_deg=10.0,
        pix_res_arcmin=0.5,
        nx=nx_calc,
        envdir=os.environ["JULIA_PROJECT"],
        xgpaint_url=XGPAINT_URL,
        recursive=False,
        beam_fwhm_arcmin=beam_fwhm_arcmin,
        log_path=(log_file or str(iter_dir / "painter_subprocess.log")),
    )

    if not map_paths:
        raise RuntimeError("La génération de la map a échoué (liste vide).")

    final_map_path = map_paths[0]

    # 4) Nettoyage
    if cleanup:
        try:
            shutil.rmtree(dir_cat, ignore_errors=True)
            shutil.rmtree(dir_coords, ignore_errors=True)
            if params_csv_path.exists():
                params_csv_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Cleanup partiel non critique : {e}")

    return final_map_path


# --- Exécution directe (optionnelle) ---
if __name__ == "__main__":
    t0 = time.time()
    try:
        test_logA, test_Oc0h2 = 3.0, 0.12
        print(f"Lancement simulation patch pour logA={test_logA}, Oc0h2={test_Oc0h2}")
        out = get_simulation_patch(
            logA=test_logA,
            Oc0h2=test_Oc0h2,
            run_id="debug_run_subproc",
            cleanup=False,
            beam_fwhm_arcmin=None,
        )
        print("SUCCÈS :", out)
    except Exception as e:
        print("ERREUR:", e)
        raise
    finally:
        print(f"Temps total: {time.time()-t0:.2f}s")
