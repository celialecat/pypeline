#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import torch

# ==========================
# CONFIGURATION À ADAPTER
# ==========================
WST_DIR = "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/wst_full_coefs"
PATTERN = "wst_*_map_*.csv"   # pattern pour les fichiers CSV de WST

OUT_THETA_PT = "theta_wst.pt"  # (logA, Oc0h2)
OUT_X_PT     = "x_wst.pt"      # coefficients WST

# Que faire des NaN dans les coefficients ?
# "zero" -> remplacer par 0.0 ; "nan" -> laisser NaN (PyTorch n’aime pas trop)
NAN_POLICY = "zero"  # "zero" ou "nan"

# ==========================
# FONCTIONS UTILITAIRES
# ==========================
# Regex pour extraire logA, Oc0h2, map_id à partir du nom de fichier
# Exemple attendu : wst_logA=3.567216_Oc0h2=0.094511_map_309.csv
RE_FILENAME = re.compile(
    r"wst_(?:logA=)?(?P<logA>[-+0-9.eE]+)_Oc0h2=(?P<Oc0h2>[-+0-9.eE]+)_map_(?P<map>\d+)\.csv$"
)

def parse_params_from_filename(fname: str):
    """
    Extrait (logA, Oc0h2, map_id) d'un nom de fichier CSV.
    Lève une erreur si le pattern ne matche pas.
    """
    base = os.path.basename(fname)
    m = RE_FILENAME.match(base)
    if m is None:
        raise ValueError(f"Nom de fichier non reconnu pour les paramètres : {base}")
    logA  = float(m.group("logA"))
    Oc0h2 = float(m.group("Oc0h2"))
    map_id = int(m.group("map"))
    return logA, Oc0h2, map_id

def load_wst_coeffs_long_csv(path: str, feature_index=None):
    """
    Lit un CSV de WST au format 'long' (sample, kind, index, value).

    - feature_index = None :
        on le construit à partir de ce fichier et on renvoie :
        (values, feature_index)
        où :
          * values : np.ndarray [n_features]
          * feature_index : liste de tuples (kind, index)

    - feature_index != None :
        on réindexe ce fichier selon l'ordre donné par feature_index,
        et on renvoie :
        (values, feature_index) avec le même feature_index.
    """
    # On ignore la première ligne de méta (# n_samples=..., etc.) grâce à comment="#"
    df = pd.read_csv(path, comment="#")

    # On garde seulement sample = 0 (une carte par CSV dans ton cas)
    if "sample" not in df.columns:
        raise ValueError(f"Colonne 'sample' absente dans {path}")
    df = df[df["sample"] == 0].copy()

    # Vérifs basiques
    for col in ["kind", "index", "value"]:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' absente dans {path}")

    # On s'assure que 'index' est bien un entier
    df["index"] = df["index"].astype(int)

    # Si on n'a pas encore de feature_index, on le construit
    if feature_index is None:
        # On ordonne par type S0, puis S1, puis S2, et par index croissant
        kind_order_map = {"S0": 0, "S1": 1, "S2": 2}
        df["kind_order"] = df["kind"].map(kind_order_map)

        df = df.sort_values(["kind_order", "index"])

        feature_index = list(zip(df["kind"].values, df["index"].values))
    else:
        # On part d'un MultiIndex pour forcer l'ordre global déjà établi
        mi = pd.MultiIndex.from_tuples(feature_index, names=["kind", "index"])
        df = df.set_index(["kind", "index"]).reindex(mi).reset_index()

    values = df["value"].to_numpy(dtype=np.float64)

    # Gestion des NaN
    if NAN_POLICY == "zero":
        values = np.nan_to_num(values, nan=0.0)
    elif NAN_POLICY == "nan":
        # on laisse les NaN tels quels
        pass
    else:
        raise ValueError("NAN_POLICY doit être 'zero' ou 'nan'.")

    return values, feature_index

# ==========================
# SCRIPT PRINCIPAL
# ==========================
def main():
    # 1) Liste de tous les CSV
    pattern = os.path.join(WST_DIR, PATTERN)
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouvé avec le pattern : {pattern}")

    print(f"Trouvé {len(csv_files)} fichiers CSV de WST.")

    # On stocke ici :
    # - les paramètres (logA, Oc0h2)
    # - les vecteurs de coefficients x
    theta_list = []   # liste de [logA, Oc0h2]
    x_list = []       # liste de [n_features]
    map_ids = []      # pour info (ligne -> map_id)

    feature_index = None  # ordre de référence des (kind, index)

    # 2) Boucle sur les fichiers
    for i, path in enumerate(csv_files):
        logA, Oc0h2, map_id = parse_params_from_filename(path)

        # On lit les coefficients WST en respectant/initialisant l'ordre global
        coeffs, feature_index = load_wst_coeffs_long_csv(path, feature_index)

        theta_list.append([logA, Oc0h2])
        x_list.append(coeffs)
        map_ids.append(map_id)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"[{i+1}/{len(csv_files)}] {os.path.basename(path)}  -> logA={logA}, Oc0h2={Oc0h2}, map={map_id}")

    # 3) Conversion en arrays puis en tensors
    theta_arr = np.asarray(theta_list, dtype=np.float32)   # [n_maps, 2]
    x_arr     = np.stack(x_list, axis=0).astype(np.float32)  # [n_maps, n_features]

    print(f"theta_arr shape = {theta_arr.shape}  (n_maps, 2)")
    print(f"x_arr shape     = {x_arr.shape}      (n_maps, n_features)")
    print(f"Nombre de features WST = {x_arr.shape[1]}")

    theta = torch.from_numpy(theta_arr)
    x     = torch.from_numpy(x_arr)

    # 4) Sauvegarde en .pt
    torch.save(theta, OUT_THETA_PT)
    torch.save(x, OUT_X_PT)

    print(f"Sauvegardé : {OUT_THETA_PT}  (paramètres)")
    print(f"Sauvegardé : {OUT_X_PT}      (coefficients WST)")

    # Optionnel : sauvegarder le mapping feature_index et map_ids pour traçabilité
    meta = {
        "feature_index": feature_index,   # liste de (kind, index)
        "map_ids": map_ids,              # liste d'entiers (un par ligne)
        "nan_policy": NAN_POLICY,
    }
    torch.save(meta, "meta_wst.pt")
    print("Sauvegardé : meta_wst.pt  (feature_index, map_ids, nan_policy)")

if __name__ == "__main__":
    main()
