import os
import shutil
import uuid
import numpy as np
from cobaya.likelihood import Likelihood

# Import de tes modules locaux
# Ils doivent être dans le même dossier que ce fichier
from my_likelihood import LikelihoodCalculator
from patch_generator_ju_fixed import get_simulation_patch

class FullPipelineLikelihood(Likelihood):
    """
    Classe wrapper pour Cobaya.
    Elle orchestre : Génération Patch -> Calcul Summary Stats -> LogLikelihood
    """
    
    # Ces variables seront automatiquement remplies par le fichier .yaml
    cov_path: str
    obs_data_path: str
    
    def initialize(self):
        """
        Appelé une seule fois au début de la MCMC.
        Charge la matrice de covariance et les données observées.
        """
        # Définition de la grille ell (doit être IDENTIQUE à celle utilisée pour ta covariance)
        self.ell_grid = np.geomspace(400, 5000, 18)
        
        print(f"[Cobaya Wrapper] Initialisation...")
        print(f"  - Covariance : {self.cov_path}")
        print(f"  - Observation: {self.obs_data_path}")
        
        # Instanciation de ton calculateur (lourd : inversion matrice, etc.)
        self.calculator = LikelihoodCalculator(
            cov_path=self.cov_path,
            obs_data_path=self.obs_data_path,
            ell_grid=self.ell_grid
        )
        print("[Cobaya Wrapper] Prêt.")

    def get_requirements(self):
        """Définit les paramètres requis par cette likelihood."""
        return {'logA': None, 'Oc0h2': None}

    def logp(self, **params_values):
        """
        Fonction appelée à CHAQUE pas de la MCMC.
        """
        # 1. Récupération des paramètres proposés par le sampler
        logA = params_values['logA']
        Oc0h2 = params_values['Oc0h2']
        
        # Vérification simple des bornes physiques (sécurité)
        # Tu peux ajuster ces bornes si nécessaire, mais le Prior dans le YAML fait déjà le travail.
        if not (1.0 < logA < 5.0) or not (0.0 < Oc0h2 < 0.5):
            return -np.inf

        # 2. Génération d'un ID unique pour éviter les conflits de fichiers
        unique_id = f"mcmc_{uuid.uuid4().hex[:8]}"
        
        fits_path = None
        try:
            # 3. Génération du Patch (Appel subprocess Julia/CosmoCNC)
            fits_path = get_simulation_patch(
                logA=logA,
                Oc0h2=Oc0h2,
                run_id=unique_id,
                cleanup=True, # Nettoie les catalogues intermédiaires (1_catalogues, 2_coords)
                beam_fwhm_arcmin=None 
            )
            
            # Si la génération a échoué (retour None)
            if fits_path is None or not os.path.exists(fits_path):
                return -np.inf

            # 4. Calcul de la Likelihood
            loglike = self.calculator.compute_loglike(fits_path)
            
            return loglike

        except Exception as e:
            print(f"[Error MCMC] {unique_id} : {e}")
            return -np.inf
            
        finally:
            # 5. Nettoyage FINAL (Critique pour ne pas saturer le disque)
            # get_simulation_patch(cleanup=True) garde le FITS final, on doit le supprimer ici
            if fits_path and os.path.exists(fits_path):
                try:
                    os.remove(fits_path)
                    # Suppression du dossier parent 'run_id' créé par le générateur
                    # fits_path est du style : .../mcmc_runs/unique_id/3_maps/.../map.fits
                    # On veut supprimer .../mcmc_runs/unique_id
                    # On remonte de 3 niveaux : map.fits -> dossier_map -> 3_maps -> unique_id
                    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(fits_path)))
                    
                    # Sécurité pour ne pas supprimer n'importe quoi
                    if unique_id in run_dir and "mcmc_runs" in run_dir: 
                        shutil.rmtree(run_dir, ignore_errors=True)
                except OSError:
                    pass