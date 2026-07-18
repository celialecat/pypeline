from .tsz import compute_integral, compute_trispectrum, compute_tsz_covariance
from . import classy_sz
import numpy as np

def initialise():
    """
    Pre-compile key JAX functions to avoid including compilation time in later measurements.
    This function must be called after setting the cosmological parameters.
    """
    classy_sz.compute_class_szfast()  # Ensure this is always called
    params = classy_sz.get_all_relevant_params()
    # Remove non-numeric values before passing to JAX
    params_filtered = {k: v for k, v in params.items() if isinstance(v, (int, float, np.ndarray))}

    if params is None:
        raise ValueError("classy_sz is not configured! Please call tszpower.classy_sz.set(...) first.")

    # Warm up key functions with filtered params
    _ = compute_integral(params_values_dict=params_filtered)
    _ = compute_trispectrum(params_values_dict=params_filtered)
    # _ = compute_tsz_covariance(params_values_dict=params_filtered, noise_ell=None, f_sky=1.0)