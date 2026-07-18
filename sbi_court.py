import os
import numpy as np
import pandas as pd
import torch

from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.analysis import pairplot

# -------------------------------------------------------------------
# 0. Charger les paramètres cosmologiques (theta)
# -------------------------------------------------------------------

# Chargement du tensor initial
theta_tensor = torch.load(
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/theta_samples_paint_10000.pt",
    map_location="cpu",
)

columns = [
    "logA",
    "Ob0h2",
    "Oc0h2",
    "h",
    "n_s",
    "B",
    "param7",
    "param8",
    "param9",
]

df = pd.DataFrame(theta_tensor.numpy(), columns=columns)

# On garde uniquement les 6 premiers paramètres (logA, Ob0h2, Oc0h2, h, n_s, B)
df = df.iloc[:, :-3]
print("df shape:", df.shape)
print(df.head())

# Conversion en Tensor float32 pour sbi
theta = torch.as_tensor(df.values, dtype=torch.float32)  # [N, 6]
num_simulations, num_dim = theta.shape
print(f"theta: {theta.shape}  (N={num_simulations}, dim={num_dim})")


# -------------------------------------------------------------------
# 1. Charger les WST à partir des CSV : X (features) 
#    -> une ligne par carte i, colonnes = [S0, S1..., S2...]
# -------------------------------------------------------------------

BASE_WST_DIR = "/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/wst_full_coefs"

def load_wst_features_for_map(i: int) -> np.ndarray:
    """
    Lit wst_map_i.csv et renvoie un vecteur 1D numpy contenant
    [S0, tous les S1 ordonnés par index, tous les S2 ordonnés par index].
    """
    filename = os.path.join(BASE_WST_DIR, f"wst_map_{i}.csv")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"WST CSV not found: {filename}")

    # Le premier header est commenté avec '#', donc on l’ignore avec comment='#'
    df_wst = pd.read_csv(filename, comment="#")

    # On s’assure de l’ordre : d’abord S0, puis S1 par index, puis S2 par index.
    # S0
    s0 = df_wst[df_wst["kind"] == "S0"].sort_values("index")["value"].to_numpy()

    # S1
    s1 = df_wst[df_wst["kind"] == "S1"].sort_values("index")["value"].to_numpy()

    # S2 (il y en a beaucoup plus, mais c’est un simple vecteur)
    s2 = df_wst[df_wst["kind"] == "S2"].sort_values("index")["value"].to_numpy()

    # On concatène tout
    feat = np.concatenate([s0, s1, s2]).astype(np.float32)  # shape (D,)
    return feat


# Boucle sur toutes les cartes pour construire la matrice X
X_list = []
N = num_simulations  # normalement 10000
for i in range(N):
    feat_i = load_wst_features_for_map(i)
    X_list.append(feat_i)

X = np.stack(X_list, axis=0)  # shape [N, D]
print("X shape (before normalisation):", X.shape)

X = torch.as_tensor(X, dtype=torch.float32)


# -------------------------------------------------------------------
# 2. Normalisation des features WST
#    (centrage-réduction par coefficient)
# -------------------------------------------------------------------

# Moyenne et std par dimension
x_mean = X.mean(dim=0, keepdim=True)   # [1, D]
x_std = X.std(dim=0, keepdim=True)     # [1, D]

# Éviter les divisions par zéro pour les features constantes
x_std[x_std == 0.0] = 1.0

X_norm = (X - x_mean) / x_std
print("X_norm shape:", X_norm.shape)

# (Optionnel : on peut aussi normaliser theta si on veut, mais ce n’est pas obligatoire
#  pour sbi, car le prior s’en charge. On laisse theta dans ses unités physiques.)


# -------------------------------------------------------------------
# 3. Définir le prior sur les paramètres cosmologiques
#    (BoxUniform entre min et max de chaque paramètre)
# -------------------------------------------------------------------

low = theta.min(dim=0).values
high = theta.max(dim=0).values

prior = BoxUniform(low=low, high=high)
print("prior low:", low)
print("prior high:", high)


# -------------------------------------------------------------------
# 4. Entraîner NPE avec sbi
# -------------------------------------------------------------------

inference = NPE(prior=prior)

# On donne les paires (theta, x_norm) à sbi
inference = inference.append_simulations(theta, X_norm)

density_estimator = inference.train()   # tu verras le log de l'entraînement
posterior = inference.build_posterior()

print(posterior)


# -------------------------------------------------------------------
# 5. Utiliser le posterior pour une carte observée
# -------------------------------------------------------------------
# Supposons que tu as une carte "observée" pour laquelle tu as déjà
# calculé les WST dans un CSV de la même forme (1 sample, S0/S1/S2).
# Exemple : OBS_MAP_INDEX ou bien un fichier séparé.
# -------------------------------------------------------------------

def load_and_normalize_obs_wst(csv_path: str) -> torch.Tensor:
    """
    Charge un CSV WST (même format que wst_map_i.csv),
    construit le vecteur [S0, S1..., S2...],
    puis le normalise avec x_mean et x_std calculés sur le training.
    """
    df_wst = pd.read_csv(csv_path, comment="#")
    s0 = df_wst[df_wst["kind"] == "S0"].sort_values("index")["value"].to_numpy()
    s1 = df_wst[df_wst["kind"] == "S1"].sort_values("index")["value"].to_numpy()
    s2 = df_wst[df_wst["kind"] == "S2"].sort_values("index")["value"].to_numpy()

    feat = np.concatenate([s0, s1, s2]).astype(np.float32)  # shape (D,)
    x_obs = torch.as_tensor(feat, dtype=torch.float32)

    # Normalisation avec les mêmes stats que le training
    x_obs_norm = (x_obs - x_mean.squeeze(0)) / x_std.squeeze(0)
    return x_obs_norm


# Exemple : si tu as une carte observée sauvée en WST dans ce CSV
# (remplace le chemin par le tien)
OBS_WST_CSV = "/rds/rds-clecat/pipeline_alina_full/alina_paper/wst_outputs/wst_full_coefs/wst_map_13.csv"

x_o = load_and_normalize_obs_wst(OBS_WST_CSV)  # shape (D,)

# Tirer des échantillons postérieurs p(theta | x_o)
posterior_samples = posterior.sample((10_000,), x=x_o)
print("posterior_samples shape:", posterior_samples.shape)  # [10000, 6]

# Petit corner plot
labels = [r"$\log A$", r"$\Omega_b h^2$", r"$\Omega_c h^2$", r"$h$", r"$n_s$", r"$B$"]
_ = pairplot(posterior_samples, figsize=(6, 6), labels=labels)
