import numpy as np
import os
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

# On importe les classes depuis ton fichier inference_torch.py
from inference_torch import TensorCosmoInterpolator, GaussianLikelihood

# --- CONFIGURATION ---
DATA_DIR = "./data"
PT_WST_THETA = os.path.join(DATA_DIR, "theta_wst.pt")
PT_WST_DATA  = os.path.join(DATA_DIR, "x_wst.pt")
PT_DELL_DATA = os.path.join(DATA_DIR, "dell_dataset.pt")
COV_PATH     = os.path.join(DATA_DIR, "covariance_matrix.csv")

# Paramètres de coupe (DOIVENT être identiques à ceux de covariance.py)
CUT_HEAD = 1
CUT_TAIL = 1

# Définition de la grille de scan (Espace logA, Oc0h2)
# Ajuste min/max selon l'étendue de ta grille LHS d'origine
LOGA_RANGE  = (2.7, 3.3)   # Min, Max pour logA
OCH2_RANGE  = (0.10, 0.14) # Min, Max pour Oc0h2
N_POINTS    = 80           # 80x80 = 6400 points de calcul (rapide)

def main():
    # 1. INITIALISATION
    print(">>> Chargement de l'émulateur...")
    emulator = TensorCosmoInterpolator(
        wst_theta_pt=PT_WST_THETA,
        wst_data_pt=PT_WST_DATA,
        dell_dataset_pt=PT_DELL_DATA,
        dell_cut_head=CUT_HEAD,
        dell_cut_tail=CUT_TAIL
    )
    
    # 2. DONNÉES FIDUCIALES (OBSERVATION)
    # Pour l'exemple, on crée une fausse observation au centre de la grille
    # Dans la réalité, remplace ceci par le chargement de ton vrai vecteur de données
    fid_logA, fid_Oc0h2 = 3.05, 0.12
    d_obs = emulator.predict(fid_logA, fid_Oc0h2)
    print(f">>> Donnée fiduciale générée à logA={fid_logA}, Oc0h2={fid_Oc0h2}")

    # 3. INITIALISATION LIKELIHOOD
    if not os.path.exists(COV_PATH):
        raise FileNotFoundError(f"Matrice de covariance introuvable : {COV_PATH}")
        
    lik = GaussianLikelihood(
        interpolator=emulator,
        cov_matrix_path=COV_PATH,
        fiducial_data_vector=d_obs,
        n_sims_cov=5000
    )

    # 4. GRID SCAN (Calcul de la surface de probabilité)
    print(f">>> Scan de la grille {N_POINTS}x{N_POINTS}...")
    
    loga_vals = np.linspace(LOGA_RANGE[0], LOGA_RANGE[1], N_POINTS)
    och2_vals = np.linspace(OCH2_RANGE[0], OCH2_RANGE[1], N_POINTS)
    
    samples = []
    weights = []
    
    # Boucle double (Scan)
    for la in loga_vals:
        for oc in och2_vals:
            # Calcul Log-Likelihood
            logL = lik.log_likelihood(la, oc)
            
            # Stockage
            samples.append([la, oc])
            weights.append(logL)
            
    samples = np.array(samples)
    log_weights = np.array(weights)

    # 5. POST-PROCESSING (LogL -> Proba)
    # On soustrait le max pour stabilité numérique : P = exp(LogL - MaxLogL)
    # Les poids n'ont pas besoin d'être normalisés à 1, GetDist s'en charge.
    weights_proba = np.exp(log_weights - np.max(log_weights))

    # 6. GETDIST PLOTTING
    print(">>> Génération des contours GetDist...")
    
    # Création de l'objet MCSamples
    mcsamples = MCSamples(
        samples=samples, 
        weights=weights_proba,
        names=['logA', 'Oc0h2'],
        labels=[r'\ln(10^{10}A_s)', r'\Omega_c h^2'],
        settings={'smooth_scale_2D': 0.6} # Petit lissage pour rendre les contours jolis
    )

    # Configuration du plot
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        [mcsamples], 
        filled=True, 
        contour_colors=['#0044cc'], # Bleu foncé
        title_limit=1
    )
    
    # Ajout de la croix rouge (Vérité)
    g.subplots[1, 0].plot(fid_logA, fid_Oc0h2, color='red', marker='x', markersize=15, markeredgewidth=2, label='Fiducial')
    
    outfile = "contours_inference_wst_dell.png"
    g.export(outfile)
    print(f">>> Terminé ! Figure sauvegardée : {outfile}")

if __name__ == "__main__":
    main()