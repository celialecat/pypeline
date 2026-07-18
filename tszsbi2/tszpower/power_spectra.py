from .tsz import compute_integral, compute_trispectrum, compute_tsz_covariance
from .utils import get_ell_range, simpson
from .tsz import dVdzdOmega, get_hmf_at_z_and_m
from .profiles import y_ell_interpolate
import jax.numpy as jnp
import jax
from functools import partial

def compute_Cell_yy(params_value_dict = None):
    
    ell = get_ell_range()
    C_ell_yy = compute_integral(params_values_dict=params_value_dict)

    return C_ell_yy


def compute_Dell_yy(params_value_dict = None):

    
    ell = get_ell_range()
    C_ell_yy = compute_integral(params_values_dict=params_value_dict)
    D_ell_yy = ell*(ell+1)*C_ell_yy/(2*jnp.pi)*1e12

    return D_ell_yy

def compute_Dell_yy_with_error(params_value_dict = None, f_sky=1):

    if not (0.0 < f_sky <= 1.0):
        raise ValueError(f"f_sky must be in (0,1], got {f_sky}")

    ell, M, M_G = compute_tsz_covariance(params_values_dict=params_value_dict, noise_ell=None, f_sky=f_sky)

    C_ell_yy = compute_integral(params_values_dict=params_value_dict)
    D_ell_yy = ell*(ell+1)*C_ell_yy/(2*jnp.pi)*1e12

    diag_M_full = jnp.diag(M)
    sigma_full = jnp.sqrt(diag_M_full)
    sigma_full = ell*(ell+1)*sigma_full/(2*jnp.pi)*1e12

    diag_M_gauss = jnp.diag(M_G)
    sigma_gauss = jnp.sqrt(diag_M_gauss)
    sigma_gauss = ell*(ell+1)*sigma_gauss/(2*jnp.pi)*1e12

    return D_ell_yy, sigma_full, sigma_gauss

def compute_scaled_trispectrum(params_value_dict = None):
    """
    Computes the trispectrum for the y-y correlation.
    """
    ell, T_llp = compute_trispectrum(params_values_dict=params_value_dict)

    ell_factor = ell * (ell + 1.0)   # shape (n_ell,)

    # Broadcast to 2D via outer product
    # shape: (n_ell, n_ell)
    ell2D_factor = jnp.outer(ell_factor, ell_factor)

    # The overall scaling factor
    prefactor = 1 / ((2.0 * jnp.pi)**2)
    # Finally, multiply element-wise
    scaled_T_ell_ellprime = T_llp * ell2D_factor * prefactor
    return scaled_T_ell_ellprime*1e24

@jax.jit
def compute_dlnCl_dlnz(params_values_dict=None, ell=None):
    """
    Compute d ln C_ell / d ln z for a given ell.
    Returns z_grid, dlnCl_dlnz (both 1D arrays, length n_z)
    """
    # Get parameter grids
    allparams = params_values_dict
    z_min = allparams.get('z_min')
    z_max = allparams.get('z_max')
    M_min = allparams.get('M_min')
    M_max = allparams.get('M_max')
    n_z = 150
    n_m = 150
    z_grid = jnp.geomspace(z_min, z_max, n_z)
    m_grid = jnp.geomspace(M_min, M_max, n_m)
    logm_grid = jnp.log(m_grid)
    dx_m = logm_grid[1] - logm_grid[0]
    
    # Get y_ell(m, z) for all z, m
    def get_yellm_for_z(zp):
        ell_arr, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        # Find index of ell
        idx = jnp.argmin(jnp.abs(ell_arr - ell))
        return y_ellm[:, idx]  # shape (n_m,)
    y_ell_mz = jax.vmap(get_yellm_for_z)(z_grid)  # shape (n_z, n_m)
    
    # Get dndlnm for all z, m
    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)
    dndlnm_grid = jax.vmap(get_hmf_for_z)(z_grid)  # shape (n_z, n_m)
    
    # dV/dz/dOmega for all z
    dVdz_grid = dVdzdOmega(z_grid, params_values_dict=params_values_dict)  # shape (n_z,)
    
    # Numerator: z * dV/dz * \int dM dndlnm * |y_ell|^2
    integrand = dndlnm_grid * (y_ell_mz ** 2)  # shape (n_z, n_m)
    integral_over_m = jax.vmap(lambda arr: simpson(arr, x=logm_grid))(integrand)  # shape (n_z,)
    numerator = z_grid * dVdz_grid * integral_over_m  # shape (n_z,)
    
    # Denominator: \int dz dV/dz \int dM dndlnm * |y_ell|^2
    denominator = simpson(dVdz_grid * integral_over_m, x=z_grid)  # scalar
    
    dlnCl_dlnz = numerator / denominator  # shape (n_z,)
    return z_grid, dlnCl_dlnz

@jax.jit
def compute_dlnCl_dlnM(params_values_dict=None, ell=None):
    """
    Compute d ln C_ell / d ln M for a given ell.
    Returns m_grid, dlnCl_dlnM (both 1D arrays, length n_m)
    """
    # Get parameter grids
    allparams = params_values_dict
    z_min = allparams.get('z_min')
    z_max = allparams.get('z_max')
    M_min = allparams.get('M_min')
    M_max = allparams.get('M_max')
    n_z = 150
    n_m = 150
    z_grid = jnp.geomspace(z_min, z_max, n_z)
    m_grid = jnp.geomspace(M_min, M_max, n_m)
    logm_grid = jnp.log(m_grid)
    
    # Get y_ell(m, z) for all z, m
    def get_yellm_for_z(zp):
        ell_arr, y_ellm = y_ell_interpolate(zp, m_grid, params_values_dict=params_values_dict)
        
        # Find the index in ell_arr that is closest to the given ell
        idx = jnp.argmin(jnp.abs(ell_arr - ell))
        return y_ellm[:, idx]  # shape (n_m,)
    y_ell_mz = jax.vmap(get_yellm_for_z)(z_grid)  # shape (n_z, n_m)
    
    # Get dndlnm for all z, m
    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)
    dndlnm_grid = jax.vmap(get_hmf_for_z)(z_grid)  # shape (n_z, n_m)
    
    # dV/dz/dOmega for all z
    dVdz_grid = dVdzdOmega(z_grid, params_values_dict=params_values_dict)  # shape (n_z,)
    
    # Numerator: M * \int dz dV/dz dndlnm * |y_ell|^2
    integrand = dndlnm_grid * (y_ell_mz ** 2)  # shape (n_z, n_m)
    # For each m, integrate over z with dV/dz included
    numerator = jax.vmap(lambda arr: simpson(dVdz_grid * arr, x=z_grid))(integrand.T)  # shape (n_m,)
    
    # Denominator: \int dz dV/dz \int dM dndlnm * |y_ell|^2
    # First, integrate over m for each z
    integral_over_m = jax.vmap(lambda arr: simpson(arr, x=logm_grid))(integrand)  # shape (n_z,)
    denominator = simpson(dVdz_grid * integral_over_m, x=z_grid)  # scalar
    
    dlnCl_dlnM = numerator / denominator  # shape (n_m,)
    return m_grid, dlnCl_dlnM


@partial(jax.jit, static_argnames=('N_z', 'N_m'))
def compute_N_clusters(N_z=512, N_m=512, params_values_dict=None):
    """
    Compute the number of clusters for a given ell.
    Returns m_grid, N_clusters (both 1D arrays, length n_m)
    """
    # Get parameter grids
    allparams = params_values_dict
    z_min = allparams.get('z_min')
    z_max = allparams.get('z_max')
    M_min = allparams.get('M_min')
    M_max = allparams.get('M_max')
    n_z = N_z
    n_m = N_m
    z_grid = jnp.geomspace(z_min, z_max, n_z)
    m_grid = jnp.geomspace(M_min, M_max, n_m)
    logm_grid = jnp.log(m_grid)
    
    # Get dndlnm for all z, m
    def get_hmf_for_z(zp):
        return get_hmf_at_z_and_m(z=zp, m=m_grid, params_values_dict=params_values_dict)
    dndlnm_grid = jax.vmap(get_hmf_for_z)(z_grid)
    comov_vol = dVdzdOmega(z_grid, params_values_dict=params_values_dict)

    comov_vol_expanded = comov_vol[:, None]  

    integrand = dndlnm_grid * comov_vol_expanded 

    # First integrate over m (using your Simpson routine along axis=1)
    partial_m = simpson(integrand, x=logm_grid, axis=1)  # shape (n_z,)
     # Then integrate the result over z (along axis=0)
    result = simpson(partial_m, x=z_grid, axis=0)  # scalar

    return result*4*jnp.pi 