import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import os

# Importe tes classes définies précédemment
from inference_utils import CosmoInterpolator, GaussianLikelihood

# --- CONFIGURATION ---
GRID_CSV = "lhs_sigma8_Omegam_full_cosmopars.csv"
WST_MEAN_DIR = "/chemin/vers/outputs_grid/wst/wst_mean"
DELL_MEAN_DIR = "/chemin/vers/outputs_grid/dell"
COV_PATH = "covariance_matrix.csv"

# Paramètres de la grille de scan (plus c'est fin, plus les contours sont lisses)
N_POINTS = 100  # 100x100 = 10,000 points de calcul (très rapide avec l'émulateur)
OM_RANGE = (0.25, 0.35)
S8_RANGE = (0.7, 0.9)

# Fiducial (pour l'exemple)
FIDUCIAL_PARAMS = [0.30, 0.80] 

def main():
    # 1. SETUP (Identique à avant)
    print("Initialisation...")
    emulator = CosmoInterpolator(
        grid_params_csv=GRID_CSV,
        wst_mean_dir=WST_MEAN_DIR,
        dell_mean_dir=DELL_MEAN_DIR,
        dell_cut_head=1,
        dell_cut_tail=1
    )
    
    # Création fausse donnée fiduciale
    d_obs = emulator.predict(*FIDUCIAL_PARAMS)
    
    likelihood = GaussianLikelihood(
        interpolator=emulator,
        cov_matrix_path=COV_PATH,
        fiducial_data_vector=d_obs,
        n_sims_cov=5000
    )

    # 2. CALCUL DE LA GRILLE (GRID SCAN)
    print(f"Calcul de la grille de Likelihood ({N_POINTS}x{N_POINTS} points)...")
    
    # On crée les axes 1D
    om_vals = np.linspace(OM_RANGE[0], OM_RANGE[1], N_POINTS)
    s8_vals = np.linspace(S8_RANGE[0], S8_RANGE[1], N_POINTS)
    
    # On prépare des listes pour stocker les résultats "aplatis"
    # GetDist veut une liste de points, pas une matrice 2D
    samples = []
    weights = []
    
    # Boucle sur la grille
    # (Note: on pourrait vectoriser si l'interpolateur le supporte, 
    # mais la boucle est assez rapide ici)
    for om in om_vals:
        for s8 in s8_vals:
            # Calcul du Log-Likelihood
            # On vérifie si on est dans les bornes de l'interpolateur par sécurité
            # (Bien que linspace assure d'être dedans si les ranges sont bons)
            logL = likelihood.log_likelihood(om, s8)
            
            samples.append([om, s8])
            weights.append(logL) # On stocke le logL d'abord

    samples = np.array(samples)
    log_weights = np.array(weights)
    
    # 3. CONVERSION LOG-LIKE -> PROBABILITÉ (POIDS)
    # Astuce numérique : on soustrait le max pour éviter les overflow d'exponentielle
    # P = exp(logL - max(logL))
    weights = np.exp(log_weights - np.max(log_weights))
    
    print("Génération des plots GetDist...")

    # 4. CRÉATION DE L'OBJET MCSamples
    # C'est l'étape clé : on transforme notre grille en objet GetDist
    mcsamples = MCSamples(
        samples=samples,   # Coordonnées [Om, s8]
        weights=weights,   # Probabilité en chaque point
        names=['Omega_m', 'sigma8'],
        labels=[r'\Omega_m', r'\sigma_8'],
        settings={'smooth_scale_2D': 0.5} # Lissage optionnel si la grille est grossière
    )

    # 5. TRACÉ DES CONTOURS
    g = plots.get_subplot_plotter()
    
    # Triangle plot (contours 2D + distributions 1D)
    g.triangle_plot(
        [mcsamples], 
        filled=True, 
        contour_colors=['darkblue'],
        title_limit=1 # Affiche les titres avec les valeurs +/- 1 sigma
    )
    
    # Ajouter la vraie valeur fiduciale pour comparer
    # (GetDist permet d'ajouter des lignes de vérité)
    g.subplots[1, 0].plot(FIDUCIAL_PARAMS[0], FIDUCIAL_PARAMS[1], 
                          color='red', marker='x', markersize=10, label='Vérité')

    filename = "contours_getdist_grid.png"
    g.export(filename)
    print(f"Figure sauvegardée : {filename}")

if __name__ == "__main__":
    main()