import torch
import numpy as np
from scipy.interpolate import RBFInterpolator

class TensorCosmoInterpolator:
    """
    Interpolateur basés sur des fichiers .pt (PyTorch).
    Entrée : Paramètres (logA, Oc0h2) -> Sortie : Vecteur concaténé [WST, Dell]
    """
    def __init__(self, 
                 wst_theta_pt: str, 
                 wst_data_pt: str, 
                 dell_dataset_pt: str,
                 dell_cut_head: int = 1,
                 dell_cut_tail: int = 1):
        """
        Args:
            wst_theta_pt: Chemin vers theta_wst.pt (N, 2)
            wst_data_pt: Chemin vers x_wst.pt (N, n_wst)
            dell_dataset_pt: Chemin vers dell_dataset.pt (N, 2 + n_dell)
            dell_cut_head/tail: Bins à couper pour Dell (doit matcher la covariance)
        """
        print("--- Chargement des Tenseurs d'entraînement ---")
        
        # 1. Chargement WST
        # On passe en CPU et numpy pour scipy
        t_wst = torch.load(wst_theta_pt, map_location="cpu").numpy()
        x_wst = torch.load(wst_data_pt, map_location="cpu").numpy()
        
        # 2. Chargement Dell
        # Format attendu : [logA, Oc0h2, D_ell_0, ...]
        dell_full = torch.load(dell_dataset_pt, map_location="cpu").numpy()
        t_dell = dell_full[:, :2]  # Les 2 premières colonnes sont les params
        x_dell = dell_full[:, 2:]  # Le reste est la data
        
        # 3. Vérification de l'alignement des cosmologies
        # On vérifie que les paramètres logA/Oc0h2 sont les mêmes ligne par ligne
        if not np.allclose(t_wst, t_dell, atol=1e-5):
            raise ValueError(
                "Désalignement détecté entre WST et Dell ! \n"
                "Les paramètres cosmologiques aux mêmes indices ne correspondent pas.\n"
                "Vérifiez que les fichiers .pt ont été générés avec le même ordre de fichiers (sorted glob)."
            )
        
        print(f"Alignement OK. Nombre de cosmologies : {t_wst.shape[0]}")
        
        # 4. Pré-traitement Dell (Coupures head/tail)
        # Gestion des indices pour couper le début et la fin
        n_bins = x_dell.shape[1]
        start = dell_cut_head
        end = n_bins - dell_cut_tail if dell_cut_tail > 0 else n_bins
        
        if start >= end:
            raise ValueError(f"Coupures Dell invalides : start={start}, end={end}, n_bins={n_bins}")
            
        x_dell_cut = x_dell[:, start:end]
        print(f"Dell coupé : {n_bins} bins -> {x_dell_cut.shape[1]} bins")
        
        # 5. Concaténation [WST, Dell] -> Vecteur Y
        self.X = t_wst # Inputs : (logA, Oc0h2)
        self.Y = np.concatenate([x_wst, x_dell_cut], axis=1) # Targets
        
        print(f"Dimension vecteur final (WST+Dell) : {self.Y.shape[1]}")
        
        # 6. Entraînement Interpolateur
        print("Entraînement RBFInterpolator (kernel='linear')...")
        self.interpolator = RBFInterpolator(self.X, self.Y, kernel='linear')
        print("Interpolateur prêt.")

    def predict(self, logA: float, Oc0h2: float) -> np.ndarray:
        """
        Prédiction pour un couple (logA, Oc0h2).
        Attention : L'input DOIT être (logA, Oc0h2), pas (Omega_m, sigma8).
        """
        # Formater pour (N, 2)
        coords = np.array([[logA, Oc0h2]])
        return self.interpolator(coords)[0]


class GaussianLikelihood:
    def __init__(self, 
                 interpolator: TensorCosmoInterpolator, 
                 cov_matrix_path: str, 
                 fiducial_data_vector: np.ndarray,
                 n_sims_cov: int = 5000):
        
        self.interpolator = interpolator
        self.d_obs = fiducial_data_vector
        
        # Chargement Covariance (support csv ou npy)
        if cov_matrix_path.endswith('.npy'):
            self.cov = np.load(cov_matrix_path)
        elif cov_matrix_path.endswith('.csv'):
            import pandas as pd
            self.cov = pd.read_csv(cov_matrix_path, comment='#').values
        else:
            raise ValueError("Format covariance inconnu (.csv ou .npy)")
            
        # Vérification dimension
        if self.cov.shape[0] != len(self.d_obs):
            raise ValueError(f"Erreur dimension : Covariance {self.cov.shape} vs Data {self.d_obs.shape}")

        # Hartlap correction
        p_bins = self.cov.shape[0]
        self.hartlap_factor = (n_sims_cov - p_bins - 2) / (n_sims_cov - 1)
        
        print(f"Inversion matrice de précision (Hartlap factor={self.hartlap_factor:.4f})...")
        self.precision_matrix = np.linalg.inv(self.cov) * self.hartlap_factor

    def log_likelihood(self, logA: float, Oc0h2: float) -> float:
        """
        Calcule la log-likelihood gaussienne pour les paramètres (logA, Oc0h2).
        """
        # 1. Prédiction modèle
        model = self.interpolator.predict(logA, Oc0h2)
        
        # 2. Résidu
        diff = model - self.d_obs
        
        # 3. Chi2
        chi2 = diff.T @ self.precision_matrix @ diff
        
        return -0.5 * chi2