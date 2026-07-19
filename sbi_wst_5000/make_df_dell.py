#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob

import numpy as np
import pandas as pd
import torch


def parse_params_from_filename(fname):
    """
    Extrait (logA, Oc0h2) d'un nom de fichier du type :
    Dell_logA=3.220042_Oc0h2=0.099874.csv
    """
    base = os.path.basename(fname)
    pattern = r"Dell_logA=([0-9eE\+\-\.]+)_Oc0h2=([0-9eE\+\-\.]+)\.csv"
    m = re.match(pattern, base)
    if m is None:
        raise ValueError(f"Nom de fichier non reconnu pour les paramètres : {base}")
    logA = float(m.group(1))
    Oc0h2 = float(m.group(2))
    return logA, Oc0h2


def load_dell_from_csv(path, prefer_column="D_ell_mean"):
    """
    Lit un CSV D_ell (format produit par compute_dell_empirical)
    et renvoie (ell, D_ell) sous forme de np.ndarray.

    - ignore les lignes de commentaires (# ...)
    - essaye d'abord 'prefer_column' (par défaut 'D_ell_mean'),
      sinon tombe sur 'D_ell_patch0'.
    """
    # Les lignes qui commencent par "#" sont des commentaires
    df = pd.read_csv(path, comment="#")

    # Vérif colonnes disponibles
    if prefer_column in df.columns:
        dell = df[prefer_column].to_numpy(dtype=np.float64)
    elif "D_ell_patch0" in df.columns:
        dell = df["D_ell_patch0"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Aucune colonne D_ell trouvée dans {path} "
            f"(cherché '{prefer_column}' ou 'D_ell_patch0'). "
            f"Colonnes dispo : {list(df.columns)}"
        )

    ell = df["ell"].to_numpy(dtype=np.float64)

    return ell, dell


def build_dataset_from_folder(
    folder,
    output_pt="dell_dataset.pt",
    prefer_column="D_ell_mean",
    verbose=True,
):
    """
    Balaye 'folder' pour trouver tous les Dell_logA=..._Oc0h2=....csv,
    construit un tensor torch de taille (N_cosmo, 2 + N_ell) :
      [logA, Oc0h2, D_ell_0, ..., D_ell_{N_ell-1}]
    et le sauvegarde dans 'output_pt'.

    Retourne le tensor créé et l'array des ell (pour info).
    """
    pattern = os.path.join(folder, "Dell_logA=*_*Oc0h2=*.csv")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier trouvé avec le pattern : {pattern}")

    if verbose:
        print(f"Trouvé {len(files)} fichiers CSV dans {folder}")

    all_rows = []
    ell_ref = None

    for i, f in enumerate(files):
        if verbose:
            print(f"[{i+1}/{len(files)}] lecture : {os.path.basename(f)}")

        # 1) paramètres cosmologiques depuis le nom du fichier
        logA, Oc0h2 = parse_params_from_filename(f)

        # 2) lecture des D_ell
        ell, D_ell = load_dell_from_csv(f, prefer_column=prefer_column)

        # 3) vérif cohérence de la grille en ell
        if ell_ref is None:
            ell_ref = ell
        else:
            if len(ell_ref) != len(ell) or not np.allclose(ell_ref, ell, rtol=0, atol=1e-8):
                raise ValueError(
                    f"La grille en ell n'est pas la même pour tous les fichiers.\n"
                    f"Fichier problématique : {f}"
                )

        # 4) construit une ligne : [logA, Oc0h2, D_ell...]
        row = np.concatenate([[logA, Oc0h2], D_ell])
        all_rows.append(row)

    # Empilement en matrice (N_cosmo, 2 + N_ell)
    data = np.vstack(all_rows).astype(np.float32)  # float32 pour être plus léger
    tensor = torch.from_numpy(data)

    # Sauvegarde
    torch.save(tensor, output_pt)

    if verbose:
        print(f"\nDataset sauvegardé dans : {output_pt}")
        print(f"Shape = {tensor.shape}  (N_cosmo, 2 + N_ell) = ({tensor.shape[0]}, {tensor.shape[1]})")

    return tensor, ell_ref


if __name__ == "__main__":
    # Exemple d'utilisation :
    # python make_dell_dataset.py \
    #     --input-dir /chemin/vers/dell_outputs_max_el10000 \
    #     --output dell_dataset.pt
    import argparse

    parser = argparse.ArgumentParser(description="Construire un dataset .pt à partir des CSV D_ell.")
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Dossier contenant les fichiers Dell_logA=..._Oc0h2=....csv"
    )
    parser.add_argument(
        "--output", type=str, default="dell_dataset.pt",
        help="Chemin du fichier .pt de sortie (default: dell_dataset.pt)"
    )
    parser.add_argument(
        "--prefer-column", type=str, default="D_ell_mean",
        help="Colonne de D_ell à utiliser (D_ell_mean ou D_ell_patch0)."
    )
    args = parser.parse_args()

    build_dataset_from_folder(
        folder=args.input_dir,
        output_pt=args.output,
        prefer_column=args.prefer_column,
        verbose=True,
    )
