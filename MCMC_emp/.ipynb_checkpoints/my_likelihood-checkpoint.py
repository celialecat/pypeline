import numpy as np
import pandas as pd
import os
import ast
from summary_stat import summary_stat

class LikelihoodCalculator:
    def __init__(self, 
                 cov_path: str, 
                 obs_data_path: str, 
                 ell_grid: np.ndarray,
                 wst_kwargs: dict = None,
                 ps_kwargs: dict = None):
        """
        Initialise le calculateur de Likelihood :
        1. Charge la covariance et ses métadonnées.
        2. Calcule la matrice de précision (inverse) avec correction de Hartlap.
        3. Calcule et stocke le vecteur de statistiques des données observées.

        Args:
            cov_path: Chemin vers le fichier de covariance (.csv ou .pkl/.npz).
            obs_data_path: Chemin vers les données observées (FITS).
            ell_grid: Grille des multipoles utilisée pour summary_stat.
            wst_kwargs: Arguments pour compute_wst_S012.
            ps_kwargs: Arguments pour compute_dell_empirical.
        """
        self.ell_grid = ell_grid
        self.wst_kwargs = wst_kwargs if wst_kwargs else {'J': 7, 'L': 4, 'whiten': False, 'strict_iso': True}
        self.ps_kwargs = ps_kwargs if ps_kwargs else {'max_ell': 5000, 'unit_scale': 1e12, 'quiet': True}
        
        # --- 1. Chargement de la Covariance et Métadonnées ---
        self.cov, self.meta = self._load_covariance_and_meta(cov_path)
        
        # Extraction des paramètres critiques depuis les métadonnées
        self.N_sims = int(self.meta['N'])
        self.p_dim = int(self.meta['p'])
        
        # Gestion des formes pour séparer WST et Dell
        # Le format tuple est parfois lu comme string depuis le CSV, on le convertit
        wst_shape = self._parse_tuple(self.meta.get('wst_block_shape', '(0,0)'))
        dell_shape = self._parse_tuple(self.meta.get('dell_block_shape', '(0,0)'))
        
        self.n_wst_features = wst_shape[1] if wst_shape[0] > 0 else 0
        self.n_dell_raw = dell_shape[1] if dell_shape[0] > 0 else 0
        
        self.cut_head = int(self.meta.get('dell_cut_head', 0))
        self.cut_tail = int(self.meta.get('dell_cut_tail', 0))

        print(f"[Likelihood] Covariance chargée. N_sims={self.N_sims}, p={self.p_dim}")
        print(f"[Likelihood] Structure détectée : WST={self.n_wst_features} coeffs, Dell (brut)={self.n_dell_raw} bins")
        print(f"[Likelihood] Cuts Dell : Head={self.cut_head}, Tail={self.cut_tail}")

        # --- 2. Correction de Hartlap & Inversion ---
        # Facteur de Hartlap : alpha = (N - p - 2) / (N - 1)
        if self.N_sims <= self.p_dim + 2:
            raise ValueError(f"Pas assez de simulations (N={self.N_sims}) pour inverser la covariance de taille p={self.p_dim}. Hartlap impossible.")
            
        self.hartlap_factor = (self.N_sims - self.p_dim - 2) / (self.N_sims - 1)
        
        # Inversion (Pseudo-inverse par sécurité si mal conditionnée, sinon inv classique)
        try:
            self.inv_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            print("[Warning] Inversion singulière, utilisation de pinv.")
            self.inv_cov = np.linalg.pinv(self.cov)
            
        self.precision_matrix = self.hartlap_factor * self.inv_cov
        print(f"[Likelihood] Facteur Hartlap calculé : {self.hartlap_factor:.4f}")

        # --- 3. Calcul du vecteur observé (cible) ---
        print(f"[Likelihood] Calcul des stats sur l'observation : {obs_data_path}")
        raw_obs_vec = summary_stat(
            obs_data_path, 
            stat_type="both", 
            ell_eval=self.ell_grid, 
            return_style="vector",
            wst_kwargs=self.wst_kwargs,
            ps_kwargs=self.ps_kwargs
        )
        self.obs_vec = self._process_vector(raw_obs_vec)
        
        # Vérification de dimension
        if len(self.obs_vec) != self.p_dim:
            raise ValueError(f"Dimension mismatch ! Covariance p={self.p_dim}, mais Obs vector p={len(self.obs_vec)}.")

    def _load_covariance_and_meta(self, path):
        """Charge CSV ou NPZ/PKL et extrait les métadonnées."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.csv':
            # Lecture des métadonnées dans le header (lignes commençant par #)
            meta = {}
            with open(path, 'r') as f:
                first_line = f.readline()
                if first_line.startswith('#'):
                    # Format attendu : # N=100, p=50, ...
                    content = first_line.strip().lstrip('#').strip()
                    pairs = content.split(',')
                    for pair in pairs:
                        if '=' in pair:
                            key, val = pair.split('=', 1)
                            meta[key.strip()] = val.strip()
            
            df = pd.read_csv(path, comment='#')
            cov = df.values
            return cov, meta
            
        elif ext == '.npz':
            data = np.load(path, allow_pickle=True)
            cov = data['cov']
            # Reconstruction du dict meta à partir des clés npz
            meta = {k: (v.item() if v.ndim==0 else v) for k, v in data.items() if k != 'cov'}
            return cov, meta
            
        else:
            raise NotImplementedError("Seuls CSV et NPZ sont supportés pour l'instant.")

    def _parse_tuple(self, val):
        """Convertit string '(100, 20)' en tuple (100, 20) de manière sûre."""
        if isinstance(val, (tuple, list, np.ndarray)):
            return val
        try:
            return ast.literal_eval(val)
        except:
            return (0, 0)

    def _process_vector(self, raw_vec):
        """
        Transforme le vecteur brut sorti de summary_stat pour qu'il colle à la covariance.
        Applique la séparation WST/PS et les cuts sur le PS.
        """
        # Le vecteur brut est [WST (S0+S1+S2), Dell]
        
        # 1. Partie WST
        if self.n_wst_features > 0:
            wst_part = raw_vec[:self.n_wst_features]
            remainder = raw_vec[self.n_wst_features:]
        else:
            wst_part = np.array([])
            remainder = raw_vec

        # 2. Partie Dell
        # remainder correspond maintenant au Dell brut
        if self.n_dell_raw > 0:
            # On applique les cuts head/tail
            # head
            start_idx = self.cut_head
            # tail (attention si cut_tail=0)
            end_idx = -self.cut_tail if self.cut_tail > 0 else None
            
            dell_cut = remainder[start_idx:end_idx]
        else:
            dell_cut = np.array([])

        # 3. Re-assemblage
        final_vec = np.concatenate([wst_part, dell_cut])
        return final_vec

    def compute_loglike(self, sim_patch_path: str) -> float:
        """
        Calcule la log-vraisemblance gaussienne pour un patch simulé donné.
        log L = -0.5 * (x_sim - x_obs)^T * Psi * (x_sim - x_obs)
        """
        # 1. Calcul des stats sur la simulation
        try:
            raw_sim_vec = summary_stat(
                sim_patch_path,
                stat_type="both",
                ell_eval=self.ell_grid,
                return_style="vector",
                wst_kwargs=self.wst_kwargs,
                ps_kwargs=self.ps_kwargs
            )
        except Exception as e:
            print(f"[Error] Echec summary_stat sur {sim_patch_path}: {e}")
            return -np.inf # Rejet du point MCMC
        
        # 2. Alignement (Cuts)
        sim_vec = self._process_vector(raw_sim_vec)
        
        # 3. Résidu
        diff = sim_vec - self.obs_vec
        
        # 4. Chi2
        chi2 = diff.T @ self.precision_matrix @ diff
        
        # 5. Log Likelihood
        return -0.5 * chi2