# -*- coding: utf-8 -*-
# Script pipeline avec instrumentation de debug et isolation de paint_patches en sous-processus.

import os, sys, platform, time, signal, tempfile, glob, gc, tracemalloc, resource, subprocess, json, pathlib
import faulthandler
import pandas as pd
import numpy as np
from astropy.io import fits

# juste avant Popen():
src_dir = os.path.dirname(os.path.abspath(__file__))  # dossier de ton script courant
env = {**os.environ,
       "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
       "NUMEXPR_NUM_THREADS":"1","JULIA_NUM_THREADS":"1",
       "JULIA_PKG_PRECOMPILE_AUTO":"0",
       "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH",""),
}


# ===================== Chemins statiques (à adapter si besoin) =====================
SURVEY_SR_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/surveys/survey_sr_so_sim.py"
SURVEY_CAT_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/surveys/survey_cat_so_sim.py"
XGPAINT_URL    = "/rds/rds-clecat/pipeline_alina_full/alina_paper/XGPaint.jl"
# ===================================================================================

# --------------------------------- LOG UTIL ---------------------------------------
def log_step(msg: str):
    print(f"[DIAGNOSTIC] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)
    sys.stdout.flush(); sys.stderr.flush()

def log_banner(title="Bannière"):
    print("\n" + "="*88)
    print(f"= {title}")
    print("="*88 + "\n", flush=True)

def log_env_snapshot():
    log_banner("Contexte d'exécution")
    print(f"PID              : {os.getpid()}")
    print(f"Python           : {sys.version.splitlines()[0]}")
    print(f"Platform         : {platform.platform()}")
    try:
        import numpy, astropy
        print(f"NumPy            : {numpy.__version__}")
        print(f"Astropy          : {astropy.__version__}")
    except Exception as e:
        print(f"(Version check)   EXCEPTION: {e}")
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"RLIMIT_AS        : soft={soft}, hard={hard}")
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        print(f"RLIMIT_DATA      : soft={soft}, hard={hard}")
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        print(f"RLIMIT_STACK     : soft={soft}, hard={hard}")
    except Exception as e:
        print(f"(resource)        EXCEPTION: {e}")
    env_keys = [
        "PATH_TO_COSMOCNC","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS","JULIA_NUM_THREADS","JULIA_DEPOT_PATH","JULIA_PROJECT",
        "PYTHONFAULTHANDLER","PYTHONMALLOC","MALLOC_ARENA_MAX"
    ]
    for k in env_keys:
        if k in os.environ:
            print(f"{k:>20s} = {os.environ.get(k)}")
    print("", flush=True)

def log_tree(label, root, limit=200):
    print(f"[DIAGNOSTIC] Tree {label} @ {root}")
    count = 0
    for p in glob.glob(os.path.join(root, "**/*"), recursive=True):
        print("  -", p)
        count += 1
        if count >= limit:
            print(f"  ... ({count}+ items, trunc.)"); break
    print("", flush=True)

def log_tracemalloc(topn=10, note=""):
    if not tracemalloc.is_tracing():
        return
    snap = tracemalloc.take_snapshot()
    stats = snap.statistics('lineno')
    print(f"[MEM] Top {topn} allocations {note}:")
    for s in stats[:topn]:
        print(f"  {s}")
    total = sum(s.size for s in stats)
    print(f"[MEM] Total tracked: {total/1024/1024:.2f} MiB\n", flush=True)

# --------------------------------- BREADCRUMBS ------------------------------------
_breadcrumb_path = f"/tmp/patch_gen_{os.getpid()}.crumb"
def crumb(msg: str):
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"{ts} | PID={os.getpid()} | {msg}\n"
        fd = os.open(_breadcrumb_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        os.write(fd, line.encode('utf-8'))
        os.fsync(fd); os.close(fd)
    except Exception:
        pass

# ----------------------------- FAULTHANDLER / SIGNALS -----------------------------
faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(120, repeat=True)
signal.signal(signal.SIGUSR1, lambda *a: faulthandler.dump_traceback(file=sys.__stderr__))
signal.signal(signal.SIGUSR2, lambda *a: faulthandler.dump_traceback(file=sys.__stderr__))

# -------------------------- Limiter l'agressivité des threads ----------------------
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("JULIA_NUM_THREADS",    "1")
os.environ.setdefault("PYTHONFAULTHANDLER",   "1")
os.environ.setdefault("PYTHONMALLOC",         "malloc")

# ----------------------------- GC & Tracemalloc -----------------------------------
# (Option) Si trop bruyant, commente la ligne suivante :
gc.set_debug(gc.DEBUG_STATS)
tracemalloc.start(25)

# -------------------------- Imports projet (Python pur) ----------------------------
from cata_generator_en import generate_cluster_catalogues
from coordinates_attributor_en import process_catalogues
# ⚠️ IMPORTANT: NE PAS IMPORTER patch_painter_ve ICI (risque d'init juliacall à l'import)
# (On l'importera uniquement dans le sous-processus.)

# ------------------------------- Aides Julia/XGPaint --------------------------------
def try_ping_xgpaint(repo_path):
    log_step("Ping XGPaint (ls -la) pour vérifier l'accès…")
    try:
        out = subprocess.run(["/bin/ls", "-la", repo_path], capture_output=True, text=True, timeout=15)
        print(out.stdout[:2000])
        if out.returncode != 0 and out.stderr:
            print(out.stderr[:2000])
    except Exception as e:
        log_step(f"Ping XGPaint EXCEPTION: {e}")

def julia_smoketest(repo_path):
    """Optionnel: tester Julia en amont (isolé) pour déclencher une éventuelle précompilation."""
    code = f'println("hello"); isdir("{repo_path}") || error("repo not found")'
    proc = subprocess.run(["julia", "--startup-file=no", "-e", code],
                          capture_output=True, text=True, env=os.environ)
    crumb(f"julia_smoketest rc={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"Julia smoketest failed:\n{proc.stderr[:2000]}")

def run_paint_patches_subproc(args_dict, timeout_sec=3600):
    """
    Exécute paint_patches(**args) dans un sous-processus Python pour isoler
    les segfaults/BLAS/Julia. Retourne la liste fits_paths.
    """
    code = r"""
import json, sys, os, tempfile
# Threads / dépôt Julia minimal
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("JULIA_NUM_THREADS","1")
os.environ.setdefault("JULIA_PKG_PRECOMPILE_AUTO","0")
os.environ.setdefault("JULIA_DEPOT_PATH", tempfile.mkdtemp(prefix="julia_depot_sub_"))

from tsz_painter import paint_patches  # <- import retardé, uniquement ici
try:
    params = json.loads(sys.stdin.read())
    out = paint_patches(**params)
    sys.stdout.write(json.dumps({"ok": True, "out": out}))
except Exception as e:
    sys.stdout.write(json.dumps({"ok": False, "err": str(e)}))
"""
    crumb("SPAWN paint_patches subprocess")
    env = {**os.environ,
           "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
           "NUMEXPR_NUM_THREADS":"1","JULIA_NUM_THREADS":"1",
           "JULIA_PKG_PRECOMPILE_AUTO":"0",
           # Important : DEPOT isolé aussi côté sous-processus si pas déjà défini
           #"JULIA_DEPOT_PATH": os.environ.get("JULIA_DEPOT_PATH", tempfile.mkdtemp(prefix="julia_depot_")),
          }
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env
    )
    try:
        stdout, stderr = proc.communicate(json.dumps(args_dict), timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        crumb("SUBPROC paint_patches TIMEOUT -> killed")
        raise TimeoutError("paint_patches subprocess timed out")

    crumb(f"SUBPROC paint_patches returncode={proc.returncode}")
    log_file = pathlib.Path(tempfile.gettempdir())/f"paint_subproc_{proc.pid}.stderr.log"
    try:
        log_file.write_text(stderr or "")
    except Exception:
        pass

    if proc.returncode != 0:
        raise RuntimeError(f"paint_patches crashed in subprocess (rc={proc.returncode}). See {log_file}")
    try:
        payload = json.loads(stdout or "{}")
    except Exception:
        raise RuntimeError(f"Invalid JSON from subprocess. See {log_file}")
    if not payload.get("ok"):
        raise RuntimeError(f"paint_patches Python exception: {payload.get('err')}")
    return payload.get("out", [])

# --------------------------------- Pipeline principal --------------------------------
def patch_generator(Oc0h2: float, logA: float, seed: int = None) -> np.ndarray:
    log_banner("patch_generator: entrée")
    print(f"PATH_TO_COSMOCNC = {os.environ.get('PATH_TO_COSMOCNC')}")
    print(f"cwd              = {os.getcwd()}\n", flush=True)
    log_env_snapshot()

    # Dépôt Julia isolé pour le parent (utile si le sous-processus hérite)
    #os.environ.setdefault("JULIA_DEPOT_PATH", tempfile.mkdtemp(prefix="julia_depot_"))
    os.environ.setdefault("JULIA_DEPOT_PATH", "/rds/rds-clecat/pipeline_alina_full/.julia_depot")

    crumb(f"Julia env (parent): DEPOT={os.environ['JULIA_DEPOT_PATH']} JTHREADS={os.environ['JULIA_NUM_THREADS']}")

    cosmo_dict = {
        "Oc0h2": float(Oc0h2),
        "logA": float(logA),
        "h": 0.6766,
        "Ob0h2": 0.02242,
        "n_s": 0.9665,
        "B": 1.35
    }
    log_step(f"Cosmologie = {cosmo_dict}")
    log_tracemalloc(note="(au démarrage)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_step(f"Tempdir = {tmp_dir}")
        dir_cat    = os.path.join(tmp_dir, "cat")
        dir_coords = os.path.join(tmp_dir, "coords")
        dir_paint  = os.path.join(tmp_dir, "paint")
        for d in [dir_cat, dir_coords, dir_paint]:
            os.makedirs(d, exist_ok=True)

        # --- PREP ---
        params_csv_path = os.path.join(tmp_dir, "params_paint.csv")
        df_input = pd.DataFrame([cosmo_dict])
        df_input.to_csv(params_csv_path, index=False)
        log_step(f"params_paint.csv écrit -> {params_csv_path}")
        log_tracemalloc(note="(après params csv)")

        # --- ÉTAPE 1 : generate_cluster_catalogues ---
        seed_offset = seed if seed is not None else np.random.randint(0, 100000)
        log_banner("Étape 1: generate_cluster_catalogues")
        crumb("ENTER generate_cluster_catalogues")
        try:
            generate_cluster_catalogues(
                cosmo_input=df_input,
                n_cosmologies=1,
                n_catalogues_per_cosmo=1,
                patch_size_deg=(10., 10.),
                output_dir=dir_cat,
                survey_sr_path=SURVEY_SR_PATH,
                survey_cat_path=SURVEY_CAT_PATH,
                seed_offset=seed_offset,
                cnc_params_overrides={"n_points": 3000, "n_z": 3000},
                output_format="csv",
                verbose=True
            )
        except Exception as e:
            log_step(f"[EXC] generate_cluster_catalogues: {e}")
            raise
        finally:
            crumb("EXIT  generate_cluster_catalogues")
            log_tracemalloc(note="(post generate_cluster_catalogues)")
            if os.path.exists(dir_cat): log_tree("dir_cat", dir_cat)

        try: del df_input
        except Exception: pass
        gc.collect()

        # --- ÉTAPE 2 : process_catalogues ---
        log_banner("Étape 2: process_catalogues")
        crumb("ENTER process_catalogues")
        try:
            process_catalogues(
                inputs=dir_cat,
                out_dir=dir_coords,
                ra0_deg=0, dec0_deg=0,
                w_deg=10, h_deg=10,
                seed=seed_offset,
                recursive=True,
                lon_wrap="pm_pi",
                overwrite=True
            )
        except Exception as e:
            log_step(f"[EXC] process_catalogues: {e}")
            raise
        finally:
            crumb("EXIT  process_catalogues")
            log_tracemalloc(note="(post process_catalogues)")
            if os.path.exists(dir_coords): log_tree("dir_coords", dir_coords)
        gc.collect()

        # --- ÉTAPE 3 : paint_patches (Julia/XGPaint) ---
        log_banner("Étape 3: paint_patches (Julia/XGPaint)")
        try_ping_xgpaint(XGPAINT_URL)

        generated_cats = glob.glob(os.path.join(dir_coords, "**/*.csv"), recursive=True)
        log_step(f"Nb catalogues trouvés : {len(generated_cats)}")
        if not generated_cats:
            raise RuntimeError("Aucun CSV après process_catalogues")

        param_files = [params_csv_path] * len(generated_cats)
        try:
            sz = os.path.getsize(generated_cats[0])
            log_step(f"Taille 1er catalogue: {sz/1024:.1f} KiB")
        except Exception as e:
            log_step(f"Impossible de lire la taille du CSV: {e}")

        log_tracemalloc(note="(avant paint_patches)")
        crumb("ENTER paint_patches (subprocess)")

        # *** Appel ISOLÉ en sous-processus (import juliacall uniquement là-bas) ***
        fits_paths = run_paint_patches_subproc(dict(
            patch_size_deg=10.0,
            pix_res_arcmin=0.5,
            output_dir=dir_paint,
            catalogs=generated_cats,
            params=param_files,
            nx=128,
            xgpaint_url=XGPAINT_URL,
        ))

        crumb("EXIT  paint_patches (subprocess)")
        log_tracemalloc(note="(post paint_patches)")
        if os.path.exists(dir_paint): log_tree("dir_paint", dir_paint)
        gc.collect()

        if not fits_paths:
            raise RuntimeError("paint_patches n'a retourné aucun fichier.")

        # --- ÉTAPE 4 : Chargement FITS ---
        log_banner("Étape 4: Chargement FITS")
        first_fits = fits_paths[0]
        log_step(f"FITS path = {first_fits} (exists={os.path.exists(first_fits)})")
        try:
            with fits.open(first_fits, memmap=False) as hdul:
                log_step(f"HDUs: {len(hdul)}; primary shape: {hdul[0].data.shape if hdul[0].data is not None else None}")
                map_data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        except Exception as e:
            log_step(f"[EXC] FITS open/load: {e}")
            raise
        finally:
            log_tracemalloc(note="(post FITS load)")
        gc.collect()

    log_banner("patch_generator: fin normale")
    return map_data

# --------------------------------- Entrée CLI --------------------------------------
if __name__ == "__main__":
    try:
        _ = patch_generator(0.12, 3.04, seed=1234)
        log_step("Tout est OK ✅")
    except Exception as e:
        log_step(f"Échec pipeline (Exception Python) : {e}")
        raise
