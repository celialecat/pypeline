import numpy as np
import jax
import jax.numpy as jnp
import functools
import jax.scipy as jscipy
from jax.scipy.interpolate import RegularGridInterpolator
import mcfit
from mcfit import TophatVar
from .utils import get_ell_range

# Precompute the x grid (make sure these limits match your needs)
_X_GRID = jnp.logspace(jnp.log10(1e-6), jnp.log10(1e5), num=1024)

# Construct the Hankel transform using the precomputed x grid.
# (This runs once at import time.)
_Hankel = mcfit.Hankel(_X_GRID, nu=0.5, lowring=True, backend='jax')
# Create a jitted version of its __call__ method.
_Hankel_jit = jax.jit(functools.partial(_Hankel, extrap=False))



# Import the shared classy_sz from the package
from . import classy_sz

def gnfw_pressure_profile(x, z, m, params_values_dict = None):
    """
    GNFW pressure profile
    """
    # Obtain cosmological parameters from classy_sz
    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    # rparams = classy_sz.pars
    conv_fac = 299792.458 # speed of light
    h = rparams['H0']/100
    H = classy_sz.get_hubble_at_z(z, params_values_dict = params_values_dict) * conv_fac  # Hubble parameter at given z
    H0 = rparams['H0']  # Hubble parameter at z=0

    # GNFW prefactor
    B = rparams['B']    
    m_delta_tilde = (m / B) # convert to M_sun 
    # C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * ((h / 0.7) * m_delta_tilde / (3e14))**(2 / 3 + 0.12) # eV cm^-3
    C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * (m_delta_tilde / (0.7 * 3e14))**(2 / 3 + 0.12) * (0.7/h)**1.5 # eV cm^-3
    c500 = rparams['c500']
    gamma = rparams['gammaGNFW']
    alpha = rparams['alphaGNFW']
    beta = rparams['betaGNFW']
    P0 = rparams['P0GNFW']

    # Calculate scaled radius and pressure profile
    scaled_x = c500 * x
    term1 = scaled_x**(-gamma)
    term2 = (1 + scaled_x**alpha)**((gamma - beta) / alpha)
    Pe = C * P0 * term1 * term2

    return Pe


def window_function(x, x_min, x_max):
    """
    Since the integral is between x_min and x_max
    """

    return jnp.where((x >= x_min) & (x <= x_max), 1.0, 0.0)

def hankel_integrand(x, z, m, x_min=1e-6, x_max=4, params_values_dict = None):
    """
    Compute x^0.5 * Pe(x) * W(x).
    Handles x and m as arrays using vmap for vectorization.
    """
    # Vectorize over m
    def single_m(m_val, params_values_dict = params_values_dict):
        # Compute GNFW pressure profile for given m
        Pe = gnfw_pressure_profile(x, z, m_val, params_values_dict = params_values_dict)
        
        # Compute the window function
        W_x = window_function(x, x_min, x_max)
        
        # Combine the result
        return x**0.5 * Pe * W_x
    
    # Apply vmap to vectorize over m
    result = jax.vmap(single_m)(m)
    
    return result  # Shape: (len(m), len(x))

def mpc_per_h_to_cm(mpc_per_h, h):
    """
    Converts a distance in Mpc/h to centimeters.
    """
    # 1 megaparsec (Mpc) in centimeters
    Mpc_to_cm = 3.085677581e24  # cm

    # Convert Mpc/h to Mpc by dividing by h
    mpc = mpc_per_h / h

    # Convert Mpc to cm
    cm = mpc * Mpc_to_cm

    return cm

def y_ell_prefactor(z, m, delta = 500, params_values_dict = None): 
    
    rparams = classy_sz.get_all_relevant_params(params_values_dict = params_values_dict)
    # rparams = classy_sz.pars
    h = rparams['H0']/100
    B = rparams['B']

    # print(rparams)
    me_in_eV = 510998.95 # electron mass in eV/c^2
    # me_in_eV = 511000 # electron mass in eV/c^2
    sigmat_cm = 6.6524587321e-25 # Thomson cross section in cm^2
    sigmat_over_mec2 = sigmat_cm / me_in_eV
    # sigmat_over_mec2 = 283.2980000259841 / 0.5176

    dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict)*h
    # dAz = classy_sz.get_angular_distance_at_z(z,params_values_dict = params_values_dict)/(1+z)*h # in Mpc/h
    r_delta = classy_sz.get_r_delta_of_m_delta_at_z(delta, m, z, params_values_dict = params_values_dict)/(B**(1/3))  # in Mpc/h
    ell_delta = dAz/r_delta
    # print(ell_delta)
    h = rparams['H0']/100

    r_delta_cm = mpc_per_h_to_cm(r_delta, h)  # r is in cm 
    # print(r_delta_cm/ell_delta**2)
    # print(sigmat_over_mec2)

    prefactor = sigmat_over_mec2 * 4 * jnp.pi * r_delta_cm/(ell_delta**2)

    return prefactor


def y_ell_complete(z, m, x_min=1e-6, x_max=4, params_values_dict=None):
    rparams = classy_sz.get_all_relevant_params(params_values_dict=params_values_dict)
    h = rparams['H0'] / 100
    B = rparams['B']
    
    prefactor = y_ell_prefactor(z, m, params_values_dict=params_values_dict)
    
    # Use the precomputed _X_GRID instead of recreating x
    integrand = hankel_integrand(_X_GRID, z, m, x_min=x_min, x_max=x_max, params_values_dict=params_values_dict)

    # Use the pre-compiled jitted Hankel transform.
    k, y_k = _Hankel_jit(integrand)  # k = ell/ell_delta

    dAz = classy_sz.get_angular_distance_at_z(z, params_values_dict=params_values_dict) * h  # in Mpc/h
    delta = 500
    r_delta = classy_sz.get_r_delta_of_m_delta_at_z(delta, m, z, params_values_dict=params_values_dict) / (B**(1/3))
    ell_delta = dAz / r_delta
    
    # Compute ell and combine to get y_ell
    ell = k[None, :] * ell_delta[:, None]
    y_ell = prefactor[:, None] * y_k * jnp.sqrt(jnp.pi / (2 * k[None, :]))

    return ell, y_ell


# @jax.jit
def y_ell_interpolate(z, m, params_values_dict = None):
    """
    Interpolate y_ell values onto a uniform ell grid for multiple m values.
    """
    # Get cosmological parameters
    # rparams = classy_sz.pars
    # l_min = rparams['ell_min']
    # l_max = rparams['ell_max']
    # dlogell = rparams['dlogell']


    # log10_l_min = jnp.log10(l_min)
    # log10_l_max = jnp.log10(l_max)
    # num = jnp.array(((log10_l_max - log10_l_min) / dlogell) + 1, int)
    # print(num)

    # Compute the complete y_ell values
    ell_nointer_list, y_ell_nointer_list = y_ell_complete(z, m, params_values_dict = params_values_dict)
    # Freeze these arrays so that JAX won't trace back through y_ell_complete.
    # ell_nointer_list = jax.lax.stop_gradient(y_ell_complete(z, m, params_values_dict = params_values_dict))[0]
    # y_ell_nointer_list = jax.lax.stop_gradient(y_ell_complete(z, m, params_values_dict = params_values_dict))[1]

    # Define evaluation ell values (uniform grid)
    # ell_eval = jnp.logspace(log10_l_min, log10_l_max, num=num)
    ell_eval = get_ell_range()

    # Interpolator function for a single m
    def interpolate_single(ell_nointer, y_ell_nointer):
        # interpolator = RegularGridInterpolator((ell_nointer,), y_ell_nointer, method='linear', bounds_error=False, fill_value=None)
        interpolator = RegularGridInterpolator((ell_nointer,), y_ell_nointer, method='linear', bounds_error=False, fill_value=None)
        return interpolator(ell_eval)

    # Vectorize the interpolation across all m
    interpolate_all = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)

    # Perform the interpolation
    y_ell_eval_list = interpolate_all(ell_nointer_list, y_ell_nointer_list)
    # print(ell_eval)

    return ell_eval, y_ell_eval_list


@jax.jit
def vectorized_y_ell_interpolate(z_array, m_array, params_values_dict):
    """
    Computes the y_ell profile for a one-to-one pair of (z, m) values.
    Each (z, m) pair is processed by converting the scalar m into a one-element 1D array.
    
    Parameters:
      z_array: 1D JAX array of redshifts.
      m_array: 1D JAX array of masses.
      params_values_dict: Dictionary containing the cosmological parameters.
    
    Returns:
      ell_eval: The evaluation grid for ℓ (assumed identical for all entries).
      y_ell_profiles: An array (shape [n, len(ell)]) of y_ell profiles.
    """
    if z_array.shape != m_array.shape:
        raise ValueError("z_array and m_array must have the same shape for 1-to-1 correspondence.")

    def compute_y_ell_single(z, m):
        m_wrapped = jnp.atleast_1d(m)
        ell_eval, y_ell_eval = y_ell_interpolate(z, m_wrapped, params_values_dict=params_values_dict)
        y_ell_eval = jnp.squeeze(y_ell_eval, axis=0)
        return ell_eval, y_ell_eval

    vectorized_fn = jax.vmap(compute_y_ell_single, in_axes=(0, 0))
    ell_array, y_ell_profiles = vectorized_fn(z_array, m_array)
    ell_eval = ell_array[0]
    return ell_eval, y_ell_profiles