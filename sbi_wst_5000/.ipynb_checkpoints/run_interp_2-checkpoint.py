import numpy as np
import os
from getdist import plots, MCSamples
from inference_engine import FlexibleCosmoInterpolator, GaussianLikelihood

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
BASE_DIR = "/rds/rds-clecat/pipeline_alina_full/alina_paper"
DATA_DIR = os.path.join(BASE_DIR, "sbi_wst_5000")

# Fichiers .pt
WST_THETA = os.path.join(DATA_DIR, "theta_wst.pt")
WST_DATA  = os.path.join(DATA_DIR, "x_wst.pt")
DELL_DATA = os.path.join(DATA_DIR, "dell_dataset_maxell10000.pt")

# Covariance et paramètres
COV_FILE = os.path.join(BASE_DIR, "out/cov_all_0_fixed.csv")
N_SIMS_COV = 5000
CUTS = (1, 1)  # head, tail pour Dell

# Paramètres du scan
N_GRID = 80
LOGA_LIM = (2.7, 3.4)
OCH2_LIM = (0.10, 0.14)

# Fiducial
FID_LOGA = 3.05
FID_OCH2 = 0.12


def run_analysis(mode="combined"):
    """mode: 'wst', 'dell' ou 'combined' """

    if mode == "wst":
        emu = FlexibleCosmoInterpolator(
            wst_theta_pt=WST_THETA,
            wst_data_pt=WST_DATA,
            wst_keep=8,  # IMPORTANT
        )
    elif mode == "dell":
        emu = FlexibleCosmoInterpolator(
            dell_dataset_pt=DELL_DATA,
            dell_cut_head=CUTS[0],
            dell_cut_tail=CUTS[1],
            wst_keep=8,  # sans effet ici, mais ok
        )
    else:
        emu = FlexibleCosmoInterpolator(
            wst_theta_pt=WST_THETA,
            wst_data_pt=WST_DATA,
            dell_dataset_pt=DELL_DATA,
            dell_cut_head=CUTS[0],
            dell_cut_tail=CUTS[1],
            wst_keep=8,  # IMPORTANT
        )

    # Observation synthétique
    d_obs = emu.predict(FID_LOGA, FID_OCH2)

    # Likelihood
    lik = GaussianLikelihood(emu, COV_FILE, d_obs, n_sims_cov=N_SIMS_COV)

    # Scan grille
    print(f"--- Scan 2D ({mode}) ---")
    x_range = np.linspace(LOGA_LIM[0], LOGA_LIM[1], N_GRID)
    y_range = np.linspace(OCH2_LIM[0], OCH2_LIM[1], N_GRID)

    samples = []
    logLs = []

    for la in x_range:
        for oc in y_range:
            samples.append([la, oc])
            logLs.append(lik.compute(la, oc))

    samples = np.asarray(samples, dtype=float)
    logLs = np.asarray(logLs, dtype=float)

    # ===== weights stables =====
    logLs -= np.max(logLs)            # max = 0
    logLs = np.clip(logLs, -700, 0)   # évite exp underflow -> 0 partout
    weights = np.exp(logLs)

    # filtrage de sécurité (NaN/inf/weights=0)
    mask = np.isfinite(weights) & np.isfinite(samples).all(axis=1) & (weights > 0)
    samples = samples[mask]
    weights = weights[mask]

    if samples.shape[0] < 10:
        raise RuntimeError(f"[{mode}] Trop peu de points valides après filtrage: {samples.shape[0]}")

    weights /= np.sum(weights)

    return MCSamples(
        samples=samples,
        weights=weights,
        names=["logA", "Oc0h2"],
        labels=[r"\ln(10^{10}A_s)", r"\Omega_c h^2"],
        label=mode.upper(),
    )


if __name__ == "__main__":
    sample_dell = run_analysis(mode="dell")
    sample_combined = run_analysis(mode="combined")

    # plotter robuste
    g = plots.getSinglePlotter()
    g.triangle_plot(
        [sample_dell, sample_combined],
        filled=True,
        contour_colors=["red", "blue"],
        legend_labels=["Spectre de puissance seul", "WST + Dell"],
    )

    g.export("comparaison_wst_dell.png")
    print("Graphique généré : comparaison_wst_dell.png")
