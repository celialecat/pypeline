import numpy as np
import pandas as pd
import os

# Import des modules contenant vos fonctions
# Assurez-vous que ces fichiers sont dans le path ou ajustez les imports
from wst_en import compute_wst_S012
from emp_ps import compute_dell_empirical

def summary_stat(
    images_or_path, 
    stat_type="both", 
    ell_eval=None, 
    return_style="vector",
    # Arguments spécifiques WST (valeurs par défaut de votre exemple)
    wst_kwargs={'J': 7, 'L': 4, 'whiten': False, 'strict_iso': True},
    # Arguments spécifiques PS
    ps_kwargs={'max_ell': 5000, 'bsize': 300, 'unit_scale': 1e12, 'quiet': True}
):
    """
    Calcule les statistiques résumées (WST et/ou Power Spectrum) pour une carte ou un lot de cartes.
    
    Args:
        images_or_path (str): Chemin vers le fichier .fits ou dossier de fits.
        stat_type (str): "wst", "ps", ou "both".
        ell_eval (np.array): Grille des multipoles pour le binning du PS (obligatoire si type="ps" ou "both").
        return_style (str): 
            - "vector": Retourne un np.array 1D concaténé (idéal pour MCMC/Likelihood).
            - "pandas": Retourne un DataFrame avec noms des colonnes (idéal pour l'analyse).
        wst_kwargs (dict): Paramètres passés à compute_wst_S012.
        ps_kwargs (dict): Paramètres passés à compute_dell_empirical.
        
    Returns:
        np.array ou pd.DataFrame
    """
    
    results_vector = []
    col_names = []
    
    # ==========================================
    # 1. Calcul des Wavelet Scattering Transforms
    # ==========================================
    if stat_type.lower() in ["wst", "both"]:
        # On désactive le plot et l'écriture disque pour le MCMC (vitesse)
        wst_args = wst_kwargs.copy()
        wst_args.update({'plot': False, 'save_csv': None, 'save_plot': None, 'save_samples_csv': None, 'quiet': True})
        
        # Appel de votre fonction
        try:
            res_wst = compute_wst_S012(images_or_path, **wst_args)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du calcul WST : {e}")

        # Extraction et moyennage sur le batch (N, coeffs) -> (coeffs,)
        # S0
        s0 = np.mean(res_wst['S0'], axis=0).ravel() # [1]
        results_vector.append(s0)
        col_names.append("S0")
        
        # S1
        s1 = np.mean(res_wst['S1'], axis=0).ravel() # [K1]
        results_vector.append(s1)
        col_names.extend([f"S1_{i}" for i in range(len(s1))])
        
        # S2 (si présent)
        if res_wst['S2'] is not None:
            s2 = np.mean(res_wst['S2'], axis=0).ravel() # [K2]
            results_vector.append(s2)
            col_names.extend([f"S2_{i}" for i in range(len(s2))])

    # ==========================================
    # 2. Calcul du Power Spectrum Empirique
    # ==========================================
    if stat_type.lower() in ["ps", "both"]:
        if ell_eval is None:
            raise ValueError("Pour utiliser 'ps' ou 'both', vous devez fournir 'ell_eval' (la grille de binning).")
            
        # On désactive plots/sauvegardes
        ps_args = ps_kwargs.copy()
        ps_args.update({'ell_eval': ell_eval, 'plot': False, 'save_csv': None, 'save_plot': None, 'quiet': True})

        try:
            # Note: compute_dell_empirical attend un path_like strict
            res_ps = compute_dell_empirical(images_or_path, **ps_args)
        except Exception as e:
             raise RuntimeError(f"Erreur lors du calcul PS : {e}")

        # On récupère le D_ell moyen sur le batch
        d_ell = res_ps['D_ell_mean'] # Déjà 1D [n_bins]
        
        # Vérification d'intégrité (NaNs)
        if not np.all(np.isfinite(d_ell)):
            # En MCMC, on peut vouloir renvoyer -inf likelihood, mais ici on lève une erreur
            raise ValueError("NaN détecté dans le spectre de puissance.")

        results_vector.append(d_ell)
        # Nommage basé sur les centres de bins ell
        stats_ell = res_ps['ell']
        col_names.extend([f"Dell_{int(l)}" for l in stats_ell])

    # ==========================================
    # 3. Formatage de la sortie
    # ==========================================
    final_vec = np.concatenate(results_vector)
    
    if return_style == "pandas":
        return pd.DataFrame([final_vec], columns=col_names)
    else:
        return final_vec