# run_mcmc_sigma8_omegam.py
import sys, os, numpy as np
from cobaya import run

sys.path.insert(0, ".")
sys.path.insert(0, "/rds/rds-clecat/pipeline_alina_full/alina_paper/")
sys.path.insert(0, "/rds/rds-clecat/pipeline_alina_full/alina_paper/tszsbi2")

INPDIR = "mcmc_inputs"
d_obs = np.load(os.path.join(INPDIR, "donnees_observees.npy"))
Cinv  = np.load(os.path.join(INPDIR, "inv_cov.npy"))
ell   = np.load(os.path.join(INPDIR, "ell_bins.npy"))

like_block = {
    "likelihood_cov_given.TszLikelihood": {
        "data_vector": d_obs,
        "inv_cov_matrix": Cinv,
        "ell_bins": ell,
        # "neutrino_kwargs": {"N_ncdm": 1, "m_ncdm": 0.06, "deg_ncdm": 3},
    }
}

params_block = {
    "sigma8":  {"prior": {"min": 0.6, "max": 1.0}, "ref": 0.81, "proposal": 0.01},
    "Omega_m": {"prior": {"min": 0.20, "max": 0.40}, "ref": 0.31, "proposal": 0.005},
}

sampler_block = {
    "mcmc": {
        "learn_proposal": True,
        "Rminus1_stop": 0.01,
        "max_samples": 5000
    }
}

info = {
    "likelihood": like_block,
    "params": params_block,
    "sampler": sampler_block,
    "output": "chains/run_sigma8_omegam"
}

print("Running Cobaya (sigma8, Omega_m)...")
updated_info, products = run(info)
print("Finished. Chains saved in:", info["output"])
