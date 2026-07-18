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
    data_directory: str = "benchmark"  
    data_file: str = "data_ell_yell_yerr_benchmark.txt"
    use_covariance: bool = False  
    cov_file: str = "trispectrum_matrix_benchmark.txt"  

    def initialize(self):
        # Load the data file.
        data_path = os.path.join(self.data_directory, self.data_file)
        D = np.loadtxt(data_path)
        # Expected format: Column 0: ℓ; Column 1: observed power spectrum; Column 2: error (sigma).
        self.ell_data = D[:, 0]
        self.cl_obs = D[:, 1]
        self.sigma_obs = D[:, 2]

        # Build a diagonal covariance matrix if no external covariance is provided.
        if self.cov_file is None:
            self.covmat = np.diag(self.sigma_obs**2)
            print("Gaussian covariance matrix is used.")
        else:
            f_sky = 1.
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
        cl_theory = theory_dict["1h"] + foreground
        # Optionally, one could check that the ℓ array of the theory matches self.ell_data.
        resid = self.cl_obs - cl_theory
        chi2 = np.dot(resid, np.dot(self.inv_covmat, resid))
        # loglike = -0.5 * chi2 - 0.5 * np.log(self.det_covmat)
        loglike = -0.5 * chi2
        self.log.info("tSZ_PS_Likelihood: chi2 = {:.2f}, loglike = {:.2f}".format(chi2, loglike))
        return loglike
