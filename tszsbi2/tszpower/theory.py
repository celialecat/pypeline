# tszpower/theory.py
import numpy as np
import jax.numpy as jnp
from .power_spectra import compute_Dell_yy
from .utils import get_ell_range
# import time

# You can either mimic the structure of your Cobaya theory classes
# or simply define functions. Here is one possible approach:

def compute_sz_power(params_value_dict=None):
    """
    Computes the tSZ power spectrum (1-halo term) for the given cosmological parameters.
    
    Parameters
    ----------
    cosmology_params : dict
        Dictionary with keys "omega_b", "omega_cdm", "H0", "tau_reio",
        "ln10_10A_s", "n_s", and "B".
    
    Returns
    -------
    ell : ndarray
        Multipole moments.
    cl_1h : ndarray
        tSZ power spectrum (1-halo term).
    """
    # Fixed astrophysical parameters
    # Build the parameter dictionary expected by tszpower (note the key for A_s)
    
    # Retrieve ell array and compute the power spectrum
    ell = get_ell_range()  # This function should already be in your package.
    dl_total = compute_Dell_yy(params_value_dict=params_value_dict)
    
    # For this example we assume only a 1-halo contribution:
    dl_1h = dl_total
    return ell, dl_1h


def compute_foreground_lkl(allparams, fg_template_path='/home/lx256/tsz_project/tszpower/data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt'):
    """
    Computes the foreground contribution given nuisance parameters.
    
    Parameters
    ----------
    nuisance_params : dict
        Dictionary with keys "A_cib", "A_ir", "A_rs".
    fg_template_path : str
        Path to the foreground template file.
    
    Returns
    -------
    cl_fg : ndarray
        Foreground power spectrum.
    """
    # Load the foreground template.
    # Expected columns: ell, CIB template, RS template, IR template, CN template.
    D_fg = np.loadtxt(fg_template_path)
    fg_ell = D_fg[:, 0]
    A_CIB_MODEL = D_fg[:, 1]
    A_RS_MODEL  = D_fg[:, 2]
    A_IR_MODEL  = D_fg[:, 3]
    A_CN_MODEL  = D_fg[:, 4]
    
    # Extract nuisance parameters
    A_cib = allparams["A_cib"]
    A_rs = allparams["A_rs"]
    A_ir = allparams["A_ir"]
    # Here A_CN is fixed by calibration
    A_cn = 0.9033
    
    dl_fg = A_cib * A_CIB_MODEL + A_rs * A_RS_MODEL + A_ir * A_IR_MODEL + A_cn * A_CN_MODEL
    return dl_fg