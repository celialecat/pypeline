# tszpower/likelihood.py
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from .theory import compute_sz_power, compute_foreground_lkl
from .power_spectra import compute_Dell_yy
class TSZPowerLikelihood:
    def __init__(self, data_directory="data",
                 data_file="data_ps-ell-y2-erry2_total-planck-collab-15.txt",
                 cov_file=None):
        """
        Initialize by loading the observed data and constructing the covariance matrix.
        """
        data_path = os.path.join(data_directory, data_file)
        D = np.loadtxt(data_path)
        # Expected columns: ℓ, observed power spectrum, sigma.
        self.ell_data = D[:, 0]
        self.dl_obs = D[:, 1]
        self.sigma_obs = D[:, 2]

        if cov_file is None:
            self.covmat =jnp.diag(self.sigma_obs**2)
        else:
            cov_path = os.path.join(data_directory, cov_file)
            self.covmat = jnp.loadtxt(cov_path)
            
        self.inv_covmat = jnp.linalg.inv(self.covmat)
        self.det_covmat = jnp.linalg.det(self.covmat)
    
    # @partial(jax.jit, static_argnames=["self", "fg_template_path"])
    def compute(self, allparams, fg_template_path="data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"):
        """
        Computes the log-likelihood given cosmological and nuisance parameters.
        
        Parameters
        ----------
        cosmology_params : dict
            Dictionary with cosmological parameters (see `compute_sz_power`).
        nuisance_params : dict
            Dictionary with nuisance (foreground) parameters.
        fg_template_path : str
            Path to the foreground template file.
        
        Returns
        -------
        loglike : float
            The log-likelihood.
        """
        # Import our theory functions (local import to avoid circular dependencies)
        from .theory import compute_sz_power, compute_foreground_lkl
        
        # Compute the theoretical SZ power spectrum.
        ell_theory, dl_1h = compute_sz_power(allparams)
        # Compute the foreground contribution.
        dl_fg = compute_foreground_lkl(allparams, fg_template_path=fg_template_path)
        # Sum the contributions. (Adjust which components to sum as needed.)
        dl_model = dl_1h + dl_fg
        
        # (Optionally, you may want to check that the theory ell array matches self.ell_data.)
        
        # Compute the residual and chi^2.
        resid = self.dl_obs - dl_model
        chi2 = jnp.dot(resid, jnp.dot(self.inv_covmat, resid))
        loglike = -0.5 * chi2 - 0.5 * jnp.log(self.det_covmat)
        # loglike = -0.5 * chi2
        return loglike

@jax.jit    
def compute_tsz_likelihood(cosmo_param_dict=None, fg_param_dict=None, data_path="/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt", cov_path=None):
    D = np.loadtxt(data_path)
    # Expected columns: ℓ, observed power spectrum, sigma.
    ell_data = D[:, 0]
    dl_obs = D[:, 1]
    sigma_obs = D[:, 2]

    dl_signal = compute_Dell_yy(params_value_dict=cosmo_param_dict)
    dl_fg = compute_foreground_lkl(fg_param_dict)

    dl_model = dl_signal + dl_fg

    resid = dl_obs - dl_model
    covmat =jnp.diag(sigma_obs**2)
    inv_covmat = jnp.linalg.inv(covmat)
    chi2 = jnp.dot(resid, jnp.dot(inv_covmat, resid))

    loglike = -0.5 * chi2 - 0.5 
        # loglike = -0.5 * chi2
    return loglike
