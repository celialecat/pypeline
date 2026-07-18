#!/usr/bin/env python3
"""
Cobaya theory modules to compute the tSZ power spectrum using the tszpower package,
with fixed astrophysical parameters and free cosmological parameters, plus additional
foreground residuals.
"""

from cobaya.theory import Theory
# import tszpower
from .config import classy_sz
from .power_spectra import compute_Dell_yy
from .initialise import initialise
from .utils import get_ell_range
import numpy as np
import time
import os

###########################################################################
# tSZ Power Spectrum Theory Module
###########################################################################

class tSZ_PS_Theory(Theory):
    # Declare the outputs that this module provides.
    output = ["Cl_sz"]

    # Free (sampled) cosmological parameters.
    params = {
        "omega_b": 0,
        "omega_cdm": 0,
        "H0": 0,
        "tau_reio": 0,
        "ln10_10A_s": 0,
        "n_s": 0,
        "B": 0,
    }

    def get_requirements(self):
        return {
            "omega_b": None,
            "omega_cdm": None,
            "H0": None,
            "tau_reio": None,
            "ln10_10A_s": None,
            "n_s": None,
            "B": None,
        }

    def initialize(self):
        self.log.info("tSZ_PS_Theory initialized (tSZ power spectrum part).")
        # Fixed astrophysical parameters.
        self.fixed_params = {
            "M_min": 1e14*0.55,
            "M_max": 1e16*0.9,
            "z_min": 5e-3,
            "z_max": 3.0,
            "P0GNFW": 8.130,
            "c500": 1.156,
            "gammaGNFW": 0.3292,
            "alphaGNFW": 1.0620,
            "betaGNFW": 5.4807,
            "jax": 1,
            "cosmo_model": 0,
        }
        # Set an initial parameter dictionary using default free parameters.
        initial_pars = self.fixed_params.copy()
        initial_pars.update({
            "omega_b": 0.022,
            "omega_cdm": 0.12,
            "H0": 67.0,
            "tau_reio": 0.06,
            # Use the key expected by tszpower.
            "ln10^{10}A_s": 3.0,
            "n_s": 0.965,
            "B": 1.0,
        })
        # Initialize tszpower with the complete set of parameters.
        classy_sz.set(initial_pars)
        # Perform the heavy computation only once.
        initialise()
        self.log.info("Initial tSZ fast computation completed.")

    def calculate(self, state, want_derived=True, **params_values):
        start_time = time.time()
        # Build the updated parameter dictionary with free parameters.
        updated_pars = {
            "omega_b": params_values["omega_b"],
            "omega_cdm": params_values["omega_cdm"],
            "H0": params_values["H0"],
            "tau_reio": params_values["tau_reio"],
            # Use the key notation expected by tszpower.
            "ln10^{10}A_s": params_values["ln10_10A_s"],
            "n_s": params_values["n_s"],
            "B": params_values["B"],
        }
        # Merge with fixed astrophysical parameters.
        updated_pars.update(self.fixed_params)
        
        # Retrieve the multipole array and compute the tSZ power spectrum.
        # ell = get_ell_range()
        dl_total = compute_Dell_yy(params_value_dict=updated_pars)

        # For this example, assume the entire signal is from the 1-halo term.
        cl_1h = dl_total
        cl_2h = np.zeros_like(dl_total)

        # Store the outputs.
        state["Cl_sz"] = {"1h": cl_1h, "2h": cl_2h}
        self._current_state = state

        self.log.info("SZ power spectrum computed in {:.4f} seconds".format(time.time() - start_time))

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)



class tSZ_FG_Theory(Theory):
    output = ["Cl_sz_foreground"]

    # Declare free nuisance parameters (foreground amplitudes)
    # --- Free nuisance (foreground) parameters ---
    # A_sz: float            # overall amplitude for the tSZ power spectrum
    # A_cib: float           # amplitude for the CIB foreground template
    # A_ir: float            # amplitude for the infrared foreground template
    # A_rs: float            # amplitude for the radio source foreground template

    # Explicitly register nuisance parameters with Cobaya:
    params = {"A_cib": 0, "A_ir": 0, "A_rs": 0}


    # Data file for the foreground template; using package resource filename.
    foreground_data_directory: str = "data"
    foreground_data_file: str = "data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"

    def initialize(self):
        self.log.info("SZForegroundTheory initialized (foreground part).")
        fg_path = os.path.join(self.foreground_data_directory, self.foreground_data_file)
        if not os.path.exists(fg_path):
            raise RuntimeError("Foreground template file not found: " + fg_path)
        D_fg = np.loadtxt(fg_path)
        # Expected columns: ell, CIB template, RS template, IR template, CN template.
        self.fg_ell = D_fg[:, 0]
        self.A_CIB_MODEL = D_fg[:, 1]
        self.A_RS_MODEL  = D_fg[:, 2]
        self.A_IR_MODEL  = D_fg[:, 3]
        self.A_CN_MODEL  = D_fg[:, 4]
        self._current_state = {}

    def calculate(self, state, want_derived=False, **params_values):
        # Retrieve free parameters
        A_cib = params_values["A_cib"]
        A_rs = params_values["A_rs"]
        A_ir = params_values["A_ir"]
        # A_CN is fixed (from external calibration)
        A_cn = 0.9033
        # Cl_fg = A_cib * self.A_CIB_MODEL + A_rs * self.A_RS_MODEL + A_ir * self.A_IR_MODEL + A_cn * self.A_CN_MODEL
        Cl_fg = A_cib * self.A_CIB_MODEL + A_rs * self.A_RS_MODEL + A_ir * self.A_IR_MODEL
        state["Cl_sz_foreground"] = Cl_fg
        self._current_state = state
        self.log.info("SZ foreground computed.")

    def get_Cl_sz_foreground(self):
        return self._current_state.get("Cl_sz_foreground", None)
