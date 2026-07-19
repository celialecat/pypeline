import torch
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator


class FlexibleCosmoInterpolator:
    """
    Émulateur par interpolation RBF.
    Gère WST, Dell, ou combiné, avec alignement robuste des cosmologies.
    """

    def __init__(
        self,
        wst_theta_pt=None,
        wst_data_pt=None,
        dell_dataset_pt=None,
        dell_cut_head=1,
        dell_cut_tail=1,
        align_round_decimals=8,  # arrondi pour matcher logA/Oc0h2 entre fichiers
    ):
        self.X, self.Y = None, None
        self.mode = ""  # 'wst', 'dell', 'combined'
        self.p_wst = 0
        self.p_dell = 0

        # --- CHARGEMENT WST ---
        x_wst, t_wst = None, None
        if wst_data_pt and wst_theta_pt:
                t_wst = torch.load(wst_theta_pt, map_location="cpu").numpy()
                x_wst = torch.load(wst_data_pt, map_location="cpu").numpy()
            
            # === TRONQUAGE WST POUR MATCHER LA COVARIANCE ===
                x_wst = x_wst[:, :8]
                self.p_wst = x_wst.shape[1]
            
                print(f">>> WST chargés : {self.p_wst} coefficients (tronqués pour matcher la covariance)")


        # --- CHARGEMENT DELL ---
        x_dell, t_dell = None, None
        if dell_dataset_pt:
            dell_full = torch.load(dell_dataset_pt, map_location="cpu").numpy()
            t_dell = dell_full[:, :2]   # (logA, Oc0h2)
            raw_dell = dell_full[:, 2:]

            n_bins = raw_dell.shape[1]
            end = n_bins - dell_cut_tail if dell_cut_tail > 0 else n_bins
            x_dell = raw_dell[:, dell_cut_head:end]
            self.p_dell = x_dell.shape[1]
            print(f">>> Dell chargés : {self.p_dell} bins (après cuts {dell_cut_head}:{end})")

        # --- ASSEMBLAGE / MODE ---
        if x_wst is not None and x_dell is not None:
            # Alignement robuste par clés (logA, Oc0h2) arrondies
            tw = np.round(t_wst.astype(np.float64), align_round_decimals)
            td = np.round(t_dell.astype(np.float64), align_round_decimals)

            w_map = {}
            for i, k in enumerate(tw):
                w_map[tuple(k)] = i

            d_map = {}
            for i, k in enumerate(td):
                d_map[tuple(k)] = i

            common = sorted(set(w_map.keys()) & set(d_map.keys()))
            if len(common) == 0:
                raise ValueError("Aucune cosmologie commune entre WST et Dell (après arrondi).")

            iw = np.array([w_map[k] for k in common], dtype=int)
            id = np.array([d_map[k] for k in common], dtype=int)

            self.X = t_wst[iw]
            self.Y = np.concatenate([x_wst[iw], x_dell[id]], axis=1)
            self.mode = "combined"
            print(
                f">>> Mode : Combiné (WST + Dell) | aligné sur {len(common)} cosmologies communes "
                f"(WST={t_wst.shape[0]}, Dell={t_dell.shape[0]})"
            )

        elif x_wst is not None:
            self.X = t_wst
            self.Y = x_wst
            self.mode = "wst"
            print(">>> Mode : WST uniquement")

        elif x_dell is not None:
            self.X = t_dell
            self.Y = x_dell
            self.mode = "dell"
            print(">>> Mode : Dell uniquement")

        else:
            raise ValueError("Aucune donnée (WST ou Dell) fournie à l'interpolateur.")

        print(f">>> Entraînement RBF sur {self.X.shape[0]} cosmologies.")
        self.interpolator = RBFInterpolator(self.X, self.Y, kernel="linear")

    def predict(self, logA, Oc0h2):
        coords = np.array([[logA, Oc0h2]], dtype=float)
        return self.interpolator(coords)[0]


def _load_cov_fixed_csv(path: str) -> np.ndarray:
    """
    Lecture robuste d'une covariance CSV.
    Hypothèses compatibles avec ton fichier:
      - lignes '#' commentées
      - header index '0,1,2,...' possible
      - ensuite: matrice carrée numérique
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if s.startswith("0,1,2,3,4,"):
                continue

            parts = s.split(",")
            if len(parts) < 10:
                continue
            try:
                vals = [float(x) for x in parts]
            except Exception:
                continue
            rows.append(vals)

    cov = np.array(rows, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance non carrée après parsing: shape={cov.shape}")
    return cov


class GaussianLikelihood:
    """
    Log-likelihood Gaussienne avec :
      - lecture robuste covariance
      - extraction auto du bloc (wst / dell) si cov_full est 'combined'
      - Hartlap sur l'inverse
    """

    def __init__(self, interpolator, cov_matrix_path, d_obs, n_sims_cov=5000):
        self.interpolator = interpolator
        self.d_obs = np.asarray(d_obs, dtype=float)
        p_target = self.d_obs.size

        print(f">>> Chargement de la covariance : {cov_matrix_path}")
        cov_full = _load_cov_fixed_csv(cov_matrix_path)

        p_full = cov_full.shape[0]
        print(f">>> Matrice complète détectée : {p_full}x{p_full}")

        if p_full != p_target:
            print(f">>> Dimensions mismatch : Cov ({p_full}) vs Data ({p_target})")
            print(f">>> Tentative d'extraction du bloc statistique '{interpolator.mode}'...")

            if interpolator.mode == "wst":
                cov = cov_full[:p_target, :p_target]         # top-left
            elif interpolator.mode == "dell":
                cov = cov_full[-p_target:, -p_target:]       # bottom-right
            else:
                raise ValueError("Impossible de faire correspondre la matrice de covariance aux données.")
        else:
            cov = cov_full

        p = cov.shape[0]
        hartlap = (n_sims_cov - p - 2) / (n_sims_cov - 1)
        print(f">>> Matrice finale {p}x{p} préparée. Facteur Hartlap : {hartlap:.4f}")

        self.inv_cov = np.linalg.inv(cov) * hartlap

    def compute(self, logA, Oc0h2):
        d_model = self.interpolator.predict(logA, Oc0h2)
        diff = d_model - self.d_obs
        chi2 = diff.T @ self.inv_cov @ diff
        return -0.5 * chi2
