import os
import shutil
import uuid
import pandas as pd
import numpy as np

# Import de tes modules existants
# Assure-toi que ces fichiers sont dans le PYTHONPATH ou dans le même dossier
from cata_generator_en import generate_cluster_catalogues
from coordinates_attributor_en import process_catalogues
from patch_painter_ve import paint_patches
from emp_ps_en import compute_dell_empirical
from wst_en import compute_wst_S012

class SimulationPipeline:
    def __init__(self, base_scratch_dir, xgpaint_path, survey_sr, survey_cat):
        self.base_scratch_dir = base_scratch_dir
        self.xgpaint_path = xgpaint_path
        self.survey_sr = survey_sr
        self.survey_cat = survey_cat
        
        # Définition du binning identique à celui utilisé pour la Covariance
        # IMPORTANT : Doit correspondre exactement à ell_eval utilisé pour la covariance
        self.ell_eval = np.geomspace(400, 5000, 18) 

    def run_simulation(self, Oc0h2, logA):
        """
        Exécute le pipeline complet pour un set de paramètres et retourne le vecteur concaténé.
        """
        # 1. Création d'un dossier unique pour ce step (pour éviter les conflits threads)
        unique_id = str(uuid.uuid4())
        step_dir = os.path.join(self.base_scratch_dir, f"step_{unique_id}")
        os.makedirs(step_dir, exist_ok=True)

        # Définition des sous-dossiers
        dir_cov = os.path.join(step_dir, "cov_pipe")
        dir_coords = os.path.join(step_dir, "cov_pipe_coords")
        dir_paint = os.path.join(step_dir, "paint")
        dir_stats = os.path.join(step_dir, "stats")
        
        try:
            # --- A. Génération Catalogues ---
            # On crée le dataframe d'entrée
            df_input = pd.DataFrame({'Oc0h2': [Oc0h2], 'logA': [logA]})
            
            # On appelle ta fonction existante
            generate_cluster_catalogues(
                cosmo_input=df_input,
                n_cosmologies=1,
                n_catalogues_per_cosmo=1, # Une seule map par step MCMC pour la théorie stochastique
                patch_size_deg=(10., 10.),
                output_dir=dir_cov,
                survey_sr_path=self.survey_sr,
                survey_cat_path=self.survey_cat,
                cnc_params_overrides={"n_points": 3000, "n_z": 3000}, # Ajuster pour vitesse vs précision
                verbose=False,
                output_format="csv"
            )

            # --- B. Attribution Coordonnées ---
            process_catalogues(
                dir_cov,
                out_dir=dir_coords,
                ra0_deg=0, dec0_deg=0, w_deg=10, h_deg=10,
                seed=42, # Ou None si tu veux de la variance cosmique à chaque step (attention à la convergence)
                recursive=True,
                lon_wrap="pm_pi"
            )

            # --- C. Peinture (XGPaint) ---
            paint_patches(
                patch_size_deg=10.0,
                pix_res_arcmin=0.5,
                output_dir=dir_paint,
                beam_fwhm_arcmin=None, # Ou la valeur de ton beam
                catalogs=[dir_coords],
                nx=128, # Doit matcher tes données
                xgpaint_url=self.xgpaint_path
            )

            # --- D. Calcul Statistiques ---
            
            # 1. Spectre de puissance (D_ell) avec le binning forcé
            res_dell = compute_dell_empirical(
                path_like=dir_paint,
                max_ell=5000,
                apod_width=100, # Doit matcher tes données
                unit_scale=1e12,
                area_weighted=False,
                plot=False,
                ell_eval=self.ell_eval, # <--- BINNING FORCÉ
                save_csv=os.path.join(dir_stats, "dell"),
                quiet=True
            )
            
            # Chargement du D_ell calculé
            # compute_dell_empirical sauvegarde un CSV, on le relit ou on utilise le retour 'res'
            # res["D_ell_mean"] contient le vecteur binné
            dell_vector = res_dell["D_ell_mean"]

            # 2. WST
            res_wst = compute_wst_S012(
                dir_paint,
                J=7, L=4, # Doit matcher tes données
                device="cpu",
                whiten=False,
                samples_format='long',
                plot=False,
                which="S1", # Ou S012 selon tes données
                save_samples_csv=os.path.join(dir_stats, "wst_samples"),
                save_csv=os.path.join(dir_stats, "wst_mean"),
                quiet=True
            )
            
            # Extraction vecteur WST (S1 ou S0+S1+S2)
            # Le format de retour de compute_wst dépend de ton implémentation exacte,
            # ici on suppose qu'on charge le fichier mean généré ou qu'on prend la moyenne
            # Si res_wst retourne un DataFrame :
            if isinstance(res_wst, pd.DataFrame):
                 # Attention : il faut s'assurer de l'ordre des colonnes pour qu'il matche la covariance
                 # Utilise la logique de ton covariance.py pour extraire le vecteur
                from covariance import _infer_s1_matrix_from_df # Ou s0s1 selon le cas
                wst_matrix = _infer_s1_matrix_from_df(res_wst)
                wst_vector = wst_matrix.mean(axis=0) # (N_features,)
            else:
                # Adapter selon ce que retourne ta fonction compute_wst_S012
                raise NotImplementedError("Adapter l'extraction WST selon le return de compute_wst")

            # --- E. Concaténation ---
            # On concatène WST puis Dell (comme dans compute_covariance_mixed)
            full_vector = np.concatenate([wst_vector, dell_vector])
            
            return full_vector

        except Exception as e:
            print(f"Simulation failed at {Oc0h2}, {logA}: {e}")
            return None
        
        finally:
            # --- F. Nettoyage ---
            # CRUCIAL : Supprimer le dossier temporaire sinon le disque va exploser
            if os.path.exists(step_dir):
                shutil.rmtree(step_dir)