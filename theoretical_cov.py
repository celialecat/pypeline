from typing import Tuple, Optional, Any, Dict
from copy import deepcopy
import os, csv
import numpy as np
import matplotlib.pyplot as plt

# --- Parsing via extract_params  ---
from title_reader import extract_params 

# --- Valeurs de base par défaut  ---
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
    "cosmo_model": 0,     # 1: mnu-lcdm emulators; 0: lcdm neutrino mass fixed
    'jax': 1,
}

# Mapping CSV/params -> clés tszpower
_KEY_MAP = {
    'Ob0h2': 'omega_b',
    'Oc0h2': 'omega_cdm',
    'n_s': 'n_s',
    'logA': 'ln10^{10}A_s',
    'B': 'B',
}

def _parse_params_from_csv(csv_path: str) -> Dict[str, float]:
    """Lit la première ligne d'un CSV et renvoie un dict de paramètres."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader)
    out = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out

def _build_allpars_from_params(
    params: Dict[str, float],
    base: Optional[Dict[str, float | int]] = None,
    include_extras: bool = True
) -> Dict[str, float | int]:
    """Construit allpars directement depuis un dict de paramètres (depuis CSV ou titre)."""
    allpars = deepcopy(DEFAULT_ALLPARS if base is None else base)

    # h -> H0 et rescale des masses
    h = params.get('h', None)
    if h is not None:
        try:
            h = float(h)
            allpars['H0'] = 100.0 * h
            # on force les masses cohérentes avec h
            allpars['M_min'] = 1.0e14 * h
            allpars['M_max'] = 1.0e16 * h
        except (TypeError, ValueError):
            pass

    # mapping direct des clés vers ce qu'attend tszpower
    for src_key, dst_key in _KEY_MAP.items():
        if src_key in params:
            try:
                allpars[dst_key] = float(params[src_key])
            except (TypeError, ValueError):
                pass

    # extras éventuels
    if include_extras:
        for extra in ('A_cib', 'A_ir', 'A_rs'):
            if extra in params:
                try:
                    allpars[extra] = float(params[extra])
                except (TypeError, ValueError):
                    pass
    return allpars

def _normalise_params_from_source(source: str) -> Dict[str, float]:
    """
    Renvoie un dict plat de paramètres à partir d'un CSV ou d'un *titre* parsé par extract_params.
    Gère le cas où extract_params renvoie {fichier: {...}}.
    """
    if os.path.isfile(source) and source.lower().endswith(".csv"):
        params = _parse_params_from_csv(source)
    else:
        params = extract_params(source)
        # PATCH: si extract_params renvoie {path: {...}} on choisit l'entrée ad hoc
        if isinstance(params, dict) and params:
            first_val = next(iter(params.values()))
            if isinstance(first_val, dict):
                if os.path.isfile(source) and source in params:
                    params = params[source]
                else:
                    params = first_val
    # Nettoyage des clés (BOM, espaces)
    params = {str(k).strip().lstrip('\ufeff'): v for k, v in params.items()}
    return params

def compute_theoretical_covariance_from_source(
    source: str,
    *,
    initialise_tsz: bool = True,
    use_scaled: bool = True,
    tsz_module: Optional[Any] = None,
    show: bool = True,
    return_fig: bool = True,
    imshow_kwargs: Optional[dict] = None,
    f_sky: float = None
) -> Tuple[np.ndarray, Optional[plt.Figure]]:
    """
    Calcule la covariance théorique (trispectrum) à partir d'un *titre* (via extract_params) ou d'un *CSV*.

    Paramètres
    ----------
    source : str
        - Titre/nom parsable (ex: "logA=..._Oc0h2=..._h=...") -> utilise extract_params
        - OU chemin d'un fichier CSV avec colonnes:
          h,n_s,Ob0h2,B,A_cib,A_ir,A_rs,logA,Oc0h2
    initialise_tsz : bool
        Si True, fait tsz.classy_sz.set(allpars) puis tsz.initialise().
    use_scaled : bool
        Si True, utilise compute_scaled_trispectrum, sinon compute_trispectrum.
    tsz_module : module
        Injection optionnelle du module tszpower (utile pour tests).
    show : bool
        Affiche la figure si True.
    return_fig : bool
        Retourne la Figure si True.
    imshow_kwargs : dict
        Arguments passés à imshow (ex. {'origin': 'lower'}).

    Retour
    ------
    (cov, fig)
      cov : np.ndarray
      fig : matplotlib.figure.Figure ou None
    """
    # 0) Import tszpower ici pour permettre l'injection tsz_module
    tsz = tsz_module
    if tsz is None:
        import tszpower as tsz

    # 1) Récupère un dict plat de paramètres puis construit allpars
    params = _normalise_params_from_source(source)
    allpars = _build_allpars_from_params(params)

    # 2) Initialisation (set + initialise)
    if initialise_tsz:
        tsz.classy_sz.set(allpars)
        tsz.initialise()
        # Optionnel : essaye de calculer sigma8 (suivant dispo)
        try:
            tsz.classy_sz.get_sigma8_and_der(params_values_dict=allpars)
        except Exception:
            try:
                tsz.classy_sz.get_sigma8_and_der(params_value_dict=allpars)
            except Exception:
                pass

    # 3) Calcul de la covariance via trispectrum
    def _call_with_fallback(fn_scaled: bool):
        if fn_scaled:
            try:
                return tsz.compute_scaled_trispectrum(params_value_dict=allpars)
            except TypeError:
                return tsz.compute_scaled_trispectrum(params_values_dict=allpars)
        else:
            try:
                return tsz.compute_trispectrum(params_value_dict=allpars)
            except TypeError:
                return tsz.compute_trispectrum(params_values_dict=allpars)

    T = _call_with_fallback(use_scaled)
    T = np.asarray(T, dtype=float)

    denom = 4.0 * np.pi * float(f_sky)
    M = T / denom
    cov = M

    # 4) Plot
    fig = None
    if return_fig or show:
        if imshow_kwargs is None:
            imshow_kwargs = {"origin": "lower"}
        else:
            imshow_kwargs = {"origin": "lower", **imshow_kwargs}

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cov, **imshow_kwargs)
        ax.set_title("Theoretical covariance (scaled trispectrum)" if use_scaled
                     else "Theoretical covariance (raw trispectrum)")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell'$")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Amplitude")
        fig.tight_layout()
        if show:
            plt.show()

    return (cov, fig) if return_fig else (cov, None)


# --- Compat: ancien nom (si tu veux garder la même API qu'avant) ---
def compute_theoretical_covariance_from_title(*args, **kwargs):
    """Wrapper rétrocompatible qui appelle la nouvelle fonction."""
    return compute_theoretical_covariance_from_source(*args, **kwargs)
