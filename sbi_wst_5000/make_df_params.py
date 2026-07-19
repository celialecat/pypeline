#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import os

def extract_cosmo_parameters(
    input_pt,
    output_pt="cosmo_parameters.pt",
    output_csv=None,
    param_names=("logA", "Oc0h2")
):
    """
    Charge un .pt au format (N, 2 + n_ell) : [logA, Oc0h2, D_ell...]
    Extrait uniquement les paramètres cosmologiques et crée un dataframe :
       col0 = logA
       col1 = Oc0h2

    Sauvegarde le dataframe au format .pt et éventuellement .csv.
    """

    # --- 1) Charger le fichier .pt (tensor)
    data = torch.load(input_pt)

    if data.ndim != 2:
        raise ValueError(f"Le tensor doit être 2D (N, M). Reçu shape={data.shape}")

    N, M = data.shape
    n_params = len(param_names)

    if M < n_params:
        raise ValueError(
            f"Le tensor ne contient pas assez de colonnes ({M}) "
            f"pour extraire {n_params} paramètres."
        )

    # --- 2) Extraire uniquement les paramètres
    # ici : colonnes [0, 1] = logA, Oc0h2
    params = data[:, :n_params].cpu().numpy()

    # --- 3) Construire le DataFrame pandas
    df = pd.DataFrame(params, columns=list(param_names))

    # --- 4) Sauvegarde au format .pt
    torch.save(df, output_pt)

    # --- 5) Option : sauvegarde CSV pour inspection humaine
    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    print(f"Param dataframe sauvegardé sous : {output_pt}")
    print(f"Shape = {df.shape}")
    return df


# ----------------------------------------------------------------------
# Exemple d'utilisation directe
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Chemin du dataset complet contenant paramètres + D_ell :
    input_pt = "dell_dataset_maxell10000.pt"

    extract_cosmo_parameters(
        input_pt,
        output_pt="cosmo_params_only.pt",
        output_csv="cosmo_params_only.csv",   # optionnel
        param_names=("logA", "Oc0h2")
    )
