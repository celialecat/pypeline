import os
import sys
import shutil
import time
from pathlib import Path

# ==============================================================================
# 1. CONFIGURATION DE L'ENVIRONNEMENT (S'EXÉCUTE À L'IMPORT)
# ==============================================================================
print("--> [Auto-Setup] Configuration des variables d'environnement Julia...")

# Chemins critiques (Hardcoded selon votre infrastructure)
JULIA_DEPOT_PATH = "/rds/rds-clecat/pipeline_alina_full/.julia_depot"
JULIA_ENV_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/pipe_env/julia_env"

# Application des variables AVANT tout import de juliacall ou modules liés
os.environ["JULIA_DEPOT_PATH"] = JULIA_DEPOT_PATH
os.environ["JULIA_PROJECT"] = JULIA_ENV_PATH
os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

# ==============================================================================
# 2. INITIALISATION DU MOTEUR JULIA ET DES PAQUETS
# ==============================================================================
# Cette section remplace les cellules de votre notebook.
# Elle s'exécute une seule fois lors du premier import du module.

try:
    print(f"--> [Auto-Setup] Chargement de juliacall avec l'env : {JULIA_ENV_PATH}")
    from juliacall import Main as jl
    
    print("--> [Auto-Setup] Vérification et installation des paquets Julia...")
    # On force les opérations de maintenance de paquets ici
    jl.seval('import Pkg')
    
    # S'assure que XGPaint est bien tracké
    jl.seval('Pkg.add("XGPaint")') 
    jl.seval('Pkg.resolve()')
    
    print("--> [Auto-Setup] Instantiation de l'environnement...")
    jl.seval('Pkg.instantiate()')

    print("--> [Auto-Setup] Démarrage de la précompilation (Patientez)...")
    t_precomp = time.time()
    jl.seval('Pkg.precompile()')
    print(f"--> [Auto-Setup] Précompilation terminée en {time.time() - t_precomp:.1f}s")

    print("--> [Auto-Setup] Chargement et vérification de XGPaint...")
    jl.seval('using XGPaint')
    
    # Vérification de sécurité (comme dans votre notebook)
    exists = jl.seval("isdefined(XGPaint, :Arnauld10ThermalSZProfile)")
    if not exists:
        print("Warning: 'Arnauld10ThermalSZProfile' introuvable, tentative de listing...")
        all_names = jl.seval("names(XGPaint; all=true)")
        candidates = [str(n) for n in all_names if "ThermalSZ" in str(n)]
        print(f"Candidats trouvés : {candidates}")
    else:
        print("--> [Auto-Setup] SUCCÈS : XGPaint est totalement opérationnel.")

except Exception as e:
    print("\n!!! ERREUR CRITIQUE LORS DE L'INITIALISATION JULIA !!!")
    print(f"Erreur : {e}")
    print("Le pipeline risque d'échouer.")
    # On ne lève pas d'erreur bloquante ici pour laisser l'utilisateur débugger si besoin,
    # mais le script risque de planter plus loin.

# ==============================================================================
# 3. IMPORTS PYTHON CLASSIQUES (Post-Initialization)
# ==============================================================================
import pandas as pd
import numpy as np

# Imports locaux
from cata_generator_en import load_local_cosmocnc, generate_cluster_catalogues
from coordinates_attributor_en import process_catalogues

# ==============================================================================
# 4. CONSTANTES GLOBALES
# ==============================================================================
BASE_WORK_DIR = "/rds/rds-clecat/pipeline_alina_full/alina_paper/pipeline_outputs/mcmc_runs"
COSMOCNC_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/cosmocnc"
SURVEY_SR_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/surveys/survey_sr_so_sim.py"
SURVEY_CAT_PATH = "/rds/rds-clecat/pipeline_alina_full/alina_paper/cosmocnc/surveys/survey_cat_so_sim.py"
XGPAINT_URL = "/rds/rds-clecat/pipeline_alina_full/alina_paper/XGPaint.jl"

print("--> [Auto-Setup] Initialisation de cosmocnc...")
_, cg = load_local_cosmocnc(COSMOCNC_PATH)


# ==============================================================================
# 5. FONCTION PRINCIPALE
# ==============================================================================
def get_simulation_patch(
    logA: float, 
    Oc0h2: float, 
    run_id: str = "test", 
    cleanup: bool = True
):
    """
    Génère un patch tSZ simulé pour une paire de paramètres (logA, Oc0h2).
    L'environnement Julia est déjà initialisé au chargement du module.
    """
    # Importation locale pour la logique de peinture
    # L'environnement étant déjà configuré en haut du fichier, l'import ici est sûr.
    from patch_painter_ve import paint_patches

    # 1. Définition des dossiers
    iter_dir = Path(BASE_WORK_DIR) / run_id
    dir_cat = iter_dir / "1_catalogues"
    dir_coords = iter_dir / "2_coords"
    dir_maps = iter_dir / "3_maps"
    
    # Paramètres fixes
    h_fixed = 0.6766
    Ob0h2_fixed = 0.02242
    
    # ---------------------------------------------------------
    # ÉTAPE 1 : GÉNÉRATION DES CATALOGUES
    # ---------------------------------------------------------
    df_cosmo = pd.DataFrame({
        "logA": [logA],
        "Oc0h2": [Oc0h2],
        "h": [h_fixed],
        "Ob0h2": [Ob0h2_fixed],
        "n_s": [0.9665]
    })

    summary_gen = generate_cluster_catalogues(
        cosmo_input=df_cosmo,
        n_cosmologies=1,
        n_catalogues_per_cosmo=1,
        patch_size_deg=(10., 10.),
        output_dir=str(dir_cat),
        survey_sr_path=SURVEY_SR_PATH,
        survey_cat_path=SURVEY_CAT_PATH,
        cnc_params_overrides={"n_points": 3000, "n_z": 3000},
        verbose=False,
        output_format="csv",
    )
    
    # ---------------------------------------------------------
    # ÉTAPE 1.5 : PARAMÈTRES POUR LE PEINTRE
    # ---------------------------------------------------------
    params_csv_path = iter_dir / "params_for_painter.csv"
    pd.DataFrame({
        "h": [h_fixed],
        "Ob0h2": [Ob0h2_fixed],
        "Oc0h2": [Oc0h2],
        "B": [1.35], 
        "logA": [logA]
    }).to_csv(params_csv_path, index=False)

    # ---------------------------------------------------------
    # ÉTAPE 2 : ATTRIBUTION DES COORDONNÉES
    # ---------------------------------------------------------
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
        lon_wrap="pm_pi"
    )

    # ---------------------------------------------------------
    # ÉTAPE 3 : PEINTURE (XGPAINT.JL)
    # ---------------------------------------------------------
    coords_csvs = list(Path(dir_coords).glob("*.csv"))
    if not coords_csvs:
        raise RuntimeError("Aucun fichier de coordonnées trouvé.")
    
    # Calcul dynamique de la résolution pixel
    nx_calc = int(10.0 * 60.0 / 0.5)

    map_paths = paint_patches(
        patch_size_deg=10.0,
        pix_res_arcmin=0.5,
        output_dir=str(dir_maps),
        catalogs=[str(p) for p in coords_csvs],
        params=[str(params_csv_path)] * len(coords_csvs),
        nx=nx_calc, 
        envdir=os.environ["JULIA_PROJECT"], 
        xgpaint_url=XGPAINT_URL,
        recursive=False
    )

    if not map_paths:
        raise RuntimeError("La génération de la map a échoué (liste vide).")

    final_map_path = map_paths[0]

    # ---------------------------------------------------------
    # NETTOYAGE
    # ---------------------------------------------------------
    if cleanup:
        try:
            shutil.rmtree(dir_cat, ignore_errors=True)
            shutil.rmtree(dir_coords, ignore_errors=True)
            if os.path.exists(params_csv_path):
                os.remove(params_csv_path)
        except Exception as e:
            print(f"Warning: Cleanup failed for {run_id}: {e}")

    return final_map_path

# --- TEST LOCAL ---
if __name__ == "__main__":
    # Si on lance le script directement, on teste la fonction
    try:
        fits = get_simulation_patch(3.0, 0.12, "test_local_run", cleanup=False)
        print(f"\n[Main] Fichier généré : {fits}")
    except Exception as e:
        print(e)