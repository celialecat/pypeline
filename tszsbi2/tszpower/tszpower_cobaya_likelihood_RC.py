#!/usr/bin/env python3
"""
tSZ Power Spectrum Likelihood Module

This module implements a Gaussian likelihood for the tSZ power spectrum.
It reads the data from a text file and compares the theory prediction
(with multipole ℓ and power spectrum Cl) to the data.
"""

import os
import numpy as np
from cobaya.likelihood import Likelihood

class tSZ_PS_Likelihood(Likelihood):
    # Variables set via the YAML file.
    data_directory: str = "data"  
    data_file: str = "data_ps-ell-y2-erry2_total-planck-collab-15.txt"
    rc_file: str = "data_rc-ell-rc-errrc.txt"
    use_covariance: bool = False  
    cov_file: str = "tSZ_trispectrum_ref_total-planck-collab-15_step_1.txt" 

    def initialize(self):
        # Load the data file.
        data_path = os.path.join(self.data_directory, self.data_file)
        D = np.loadtxt(data_path)
        # Expected format: Column 0: ℓ; Column 1: observed power spectrum; Column 2: error (sigma).
        self.ell_data = D[:, 0][:-1]
        self.cl_obs = D[:, 1][:-1]
        self.sigma_obs = D[:, 2][:-1]
        # self.ell_data = D[:, 0]
        # self.cl_obs = D[:, 1]
        # self.sigma_obs = D[:, 2]

        self.cl_obs_testrc = D[:, 1]
        self.sigma_obs_testrc = D[:, 2]

        RC_path = os.path.join(self.data_directory, self.rc_file)
        R = np.loadtxt(RC_path)
        # Expected format: Column 0: ℓ; Column 1: resolved sources power spectrum; Column 2: error (sigma).
        self.ell_rc = R[:, 0]
        self.cl_rc = R[:, 1]
        self.sigma_rc = R[:, 2]

        # Build a diagonal covariance matrix if no external covariance is provided.
        if self.cov_file is None:
            self.covmat = np.diag(self.sigma_obs**2)
            print("Gaussian covariance matrix is used.")
        else:
            f_sky = 0.47
            trisp_path = os.path.join(self.data_directory, self.cov_file)
            T = np.loadtxt(trisp_path)
            self.covmat = np.diag(self.sigma_obs)**2 + T/(4.*np.pi*f_sky)
            print("Trispectrum is used.")

        # Pre-compute inverse and determinant.
        self.inv_covmat = np.linalg.inv(self.covmat)
        self.det_covmat = np.linalg.det(self.covmat)

        self.log.info("tSZ_PS_Likelihood: Data loaded and covariance matrix constructed.")
        super().initialize()

    def get_requirements(self):
        # Request the theory outputs from both SZ power and foreground components.
        return {"Cl_sz": {}, "Cl_sz_foreground": {}}

    def _get_data(self):
        return self.ell_data, self.cl_obs

    def _get_cov(self):
        return self.covmat

    def logp(self, **params_values):
        # Retrieve the SZ power spectrum from the theory.
        # Expecting Cl_sz to be a dictionary with keys "1h" and "2h"
        theory_dict = self.provider.get_Cl_sz()
        # Retrieve the foreground contribution.
        foreground = self.provider.get_Cl_sz_foreground()
        # Sum the SZ and foreground components.
        # cl_theory = theory_dict["1h"] + theory_dict["2h"] + foreground

        # Identify multipole bins where the prior applies.
        idx = (self.ell_rc >= 257.) & (self.ell_rc <= 1248.5)
        # print(idx)
        
        # Ensure that a resolved sources template (CˆRC) is available.
        # For example, load it in the initialize() method and store in self.cl_rc.
        # Here we assume that self.cl_rc is already defined; if not, use zeros (or raise an error).
        if not hasattr(self, "cl_rc"):
            # Ideally, load the resolved sources power from a file.
            self.log.warning("Resolved sources template not found; using zeros as a placeholder.")
            self.cl_rc = np.zeros_like(self.ell_rc)
        
        # Impose the physical prior:
        # The total foreground power (foreground) plus the resolved source contribution (cl_rc)
        # should not exceed the observed power (cl_obs) in these bins.
        if np.any(foreground[idx] + self.cl_rc[idx] > self.cl_obs_testrc[idx]):
            # print(self.ell_data[idx])
            self.log.info("MCMC step rejected: foreground power exceeds physical upper bound.")
            return -np.inf


        cl_theory = theory_dict["1h"] + foreground[:-1]
        # cl_theory = theory_dict["1h"] + foreground
        # Optionally, one could check that the ℓ array of the theory matches self.ell_data.
        resid = self.cl_obs - cl_theory
        chi2 = np.dot(resid, np.dot(self.inv_covmat, resid))
        # loglike = -0.5 * chi2 - 0.5 * np.log(self.det_covmat)
        loglike = -0.5 * chi2
        self.log.info("tSZ_PS_Likelihood: chi2 = {:.2f}, loglike = {:.2f}".format(chi2, loglike))
        return loglike