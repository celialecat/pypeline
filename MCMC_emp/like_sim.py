from cobaya.likelihood import Likelihood
import numpy as np
import os
from simulation_pipeline import SimulationPipeline

class SimBasedLikelihood(Likelihood):
    # Chemins définis dans le YAML
    data_wst_path: str
    data_dell_path: str
    cov_path: str
    
    # Config Pipeline
    scratch_dir: str
    xgpaint_path: str
    survey_sr_path: str
    survey_cat_path: str

    def initialize(self):
        print("Initialisation du Pipeline de Simulation...")
        self.pipeline = SimulationPipeline(
            base_scratch_dir=self.scratch_dir,
            xgpaint_path=self.xgpaint_path,
            survey_sr=self.survey_sr_path,
            survey_cat=self.survey_cat_path
        )
        
        # --- 1. Chargement Covariance ---
        print(f"Chargement Covariance: {self.cov_path}")
        cov_data = np.load(self.cov_path)
        self.cov = cov_data['cov']
        self.inv_cov = np.linalg.inv(self.cov)
        
        # --- 2. Chargement Données Observées ---
        # Il faut charger les données et les concaténer exactement comme dans la simulation
        # Ceci est une version simplifiée, adapte-la à tes fichiers exacts
        
        # A. Chargement WST Data
        # Utilise ta fonction de chargement standard
        from covariance import _infer_s1_matrix_from_df # Adapter import
        import pandas as pd
        
        df_wst = pd.read_csv(self.data_wst_path, comment="#")
        wst_obs = _infer_s1_matrix_from_df(df_wst).mean(axis=0) # Si plusieurs réal, prendre mean
        
        # B. Chargement Dell Data
        df_dell = pd.read_csv(self.data_dell_path, comment="#")
        # IMPORTANT : Si tes données observées ne sont pas déjà binnées sur ell_eval,
        # il faut faire l'interpolation ici aussi !
        # Supposons qu'elles le soient déjà (car sorties de compute_dell_empirical avec ell_eval):
        dell_obs = df_dell["D_ell_mean"].values 
        
        # C. Concaténation Vecteur Data
        self.data_vector = np.concatenate([wst_obs, dell_obs])
        
        # Vérification dimensions
        if len(self.data_vector) != self.cov.shape[0]:
            raise ValueError(f"Dimension Mismatch: Data ({len(self.data_vector)}) vs Cov ({self.cov.shape[0]})")

    def get_requirements(self):
        return {"Oc0h2": None, "logA": None}

    def logp(self, **params):
        Oc0h2 = params["Oc0h2"]
        logA = params["logA"]
        
        # 1. Run Simulation
        theory_vector = self.pipeline.run_simulation(Oc0h2, logA)
        
        if theory_vector is None:
            return -np.inf
            
        # Vérif dimension (au cas où le binning foire)
        if len(theory_vector) != len(self.data_vector):
            print(f"Shape mismatch in theory: got {len(theory_vector)}, expected {len(self.data_vector)}")
            return -np.inf

        # 2. Calcul Chi2
        delta = self.data_vector - theory_vector
        chi2 = delta @ self.inv_cov @ delta
        
        return -0.5 * chi2