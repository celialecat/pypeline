from typing import Dict, Optional, Tuple
from copy import deepcopy
import os
import csv
import matplotlib.pyplot as plt

import tszpower as tsz
from title_reader import extract_params  # doit savoir parser le "titre"

import numpy as np
print("tszpower from:", tsz.__file__)
ell = np.asarray(tsz.get_ell_range())
print("ell[-1] =", float(ell[-1]))


# ---- Valeurs par défaut (vous pouvez ajuster) ----
DEFAULT_ALLPARS: Dict[str, float | int] = {
    'omega_b': 0.02242,
    'omega_cdm': 0.1193,
    'H0': 67.66,
    'tau_reio': 0.0544,
    'ln10^{10}A_s': 2.972,
    'n_s': 0.9665,
    'M_min': 1.0e14 * 0.6766,
    'M_max': 1.0e16 * 0.6766,
    'z_min': 1e-2,
    'z_max': 3.0,
    'P0GNFW': 8.130,
    'c500': 1.156,
    'gammaGNFW': 0.3292,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,
    'B': 1.41,
    "cosmo_model": 0,   # 1: mnu-lcdm emulators; 0: lcdm neutrino mass fixed
    'jax': 1,
    #What we add : 
    #'m_nu' : 0.06,
    #'Ob0' : 0.04897,
    #'Om0' : 0.3096,
    #'Onu0' : 0.0014,
    #'sigma_8' : 0.8099,
    #'ln10^{10}A_s' : 3.04

}

# Mapping des clés "extract/CSV" -> clés tszpower
_KEY_MAP = {
    'Ob0h2': 'omega_b',
    'Oc0h2': 'omega_cdm',
    'n_s': 'n_s',
    'logA': 'ln10^{10}A_s',
    'B': 'B',
}

def _parse_params_from_csv(csv_path: str) -> Dict[str, float]:
    """
    Lit le premier enregistrement d'un CSV et renvoie un dict de paramètres.
    Attend des colonnes comme: h,n_s,Ob0h2,B,A_cib,A_ir,A_rs,logA,Oc0h2
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader)  # premier enregistrement
    # convertit tous les champs numériques en float si possible
    out = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            # On laisse le champ tel quel s'il n'est pas numérique
            out[k] = v
    return out

def _build_allpars_from_params(
    params: Dict[str, float],
    base: Optional[Dict[str, float | int]] = None,
    include_extras: bool = True
) -> Dict[str, float | int]:
    """
    Construit le dictionnaire allpars pour tszpower à partir d'un dict de paramètres.
    - h -> H0 (=100*h)
    - Rescale M_min/M_max par h si présent
    - Map Ob0h2/Oc0h2/n_s/logA/B vers les clés attendues
    - Ajoute A_cib, A_ir, A_rs s'ils existent
    """
    allpars = deepcopy(DEFAULT_ALLPARS if base is None else base)

    # 1) h -> H0 et masses
    h = params.get('h', None)
    if h is not None:
        try:
            h = float(h)
            allpars['H0'] = 100.0 * h
            if 'M_min' in allpars:
                allpars['M_min'] = 1.0e14 * h
            if 'M_max' in allpars:
                allpars['M_max'] = 1.0e16 * h
        except (TypeError, ValueError):
            pass

    # 2) mapping direct
    for src_key, dst_key in _KEY_MAP.items():
        if src_key in params:
            try:
                allpars[dst_key] = float(params[src_key])
            except (TypeError, ValueError):
                pass

    # 3) extras éventuels
    if include_extras:
        for extra in ('A_cib', 'A_ir', 'A_rs'):
            if extra in params:
                try:
                    allpars[extra] = float(params[extra])
                except (TypeError, ValueError):
                    pass

    return allpars

def compute_tsz_power_with_errors_from_title_or_csv(
    source: str,
    base_allpars: Optional[Dict[str, float | int]] = None,
    *,
    show_plot: bool = True,
    use_full_error: bool = True,
    f_sky: float = 1.0,
) -> Tuple:
    """
    Calcule et (optionnellement) trace le spectre tSZ théorique avec barres d'erreur,
    à partir d'un *titre de document* (parsable par `extract_params`) OU d'un *CSV*.

    Paramètres
    ----------
    source : str
        - Soit un chemin de fichier .csv contenant une ligne de cosmologie
          (colonnes typiques: h,n_s,Ob0h2,B,A_cib,A_ir,A_rs,logA,Oc0h2)
        - Soit un titre (string) que `title_reader.extract_params` sait parser.
    base_allpars : dict, optionnel
        Dictionnaire de base à partir duquel on met à jour les paramètres.
    show_plot : bool
        Si True, affiche un plot Matplotlib avec barres d'erreur.
    use_full_error : bool
        Si True, trace les barres d'erreur "complètes" (non-Gauss + Gauss).
        Sinon, trace les barres d'erreur gaussiennes.

    Renvoie
    -------
    (ell, D_ell_yy, sigma_full, sigma_gauss, allpars)
    """
    # 1) Récupération des paramètres depuis la source
    params = extract_params(source)
    
# --- PATCH : si extract_params a renvoyé {fichier: {...}}, on choisit une entrée ---
    if isinstance(params, dict) and params:
        first_val = next(iter(params.values()))
        if isinstance(first_val, dict):
        # Si 'source' est déjà un fichier, on tente de prendre son entrée ; sinon la première
            if os.path.isfile(source) and source in params:
                params = params[source]
            else:
                params = first_val  # prend la première entrée trouvée
# Nettoie les clés au cas où
    params = {str(k).strip().lstrip('\ufeff'): v for k, v in params.items()}
    print("[PARAMS] Oc0h2:", params.get("Oc0h2"), " | logA:", params.get("logA"))

# ----------------------------------------------------------------------

    print("params =", params)

    # 2) Construction de allpars pour tszpower
    allpars = _build_allpars_from_params(params, base=base_allpars, include_extras=True)
    print("allpars = " , allpars)
    # 3) Initialisation tszpower
    tsz.classy_sz.set(allpars)
    tsz.initialise()

    # 4) Calcul du spectre avec erreurs
    ell = tsz.get_ell_range()
    D_ell_yy, sigma_full, sigma_gauss = tsz.compute_Dell_yy_with_error(params_value_dict=allpars, f_sky=f_sky)
    f_sky = f_sky

    # 5) Plot si demandé
    if show_plot:
        plt.figure()
        if use_full_error:
            plt.errorbar(ell, D_ell_yy, yerr=sigma_full, fmt='-', label='Total error')
        else:
            plt.errorbar(ell, D_ell_yy, yerr=sigma_gauss, fmt='-', label='Gaussian error')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell^{yy}$')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ell, D_ell_yy, sigma_full, sigma_gauss, allpars


# --------- Exemple d'utilisation ----------
if __name__ == "__main__":
    # Cas 1: à partir d'un titre/nom parsable
    title = "logA=3.038456_Oc0h2=0.116816_h=0.6766_n_s=0.9665_B=1.35"
    ell, D, sig_full, sig_gauss, allpars = compute_tsz_power_with_errors_from_title_or_csv(
        title, show_plot=True
    )

    # Cas 2: à partir d'un CSV
    # csv_path = "/chemin/vers/cosmology.csv"
    # ell, D, sig_full, sig_gauss, allpars = compute_tsz_power_with_errors_from_title_or_csv(
    #     csv_path, show_plot=True
    # )
