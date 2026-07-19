"""
Script d'optimisation des hyperparamètres pour SBI (NPE) avec Optuna.
Ce script optimise l'architecture MAF et les paramètres d'entraînement
pour minimiser la validation loss.
"""
from __future__ import annotations
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Subset
import optuna
from optuna.samplers import TPESampler

# ---- Fix path: vers sbi_builder
sbi_builder_path = "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_backend"
sys.path.append(str(sbi_builder_path))

# ---- Import des utilitaires
from sbi_backend import (
    TszDataset,
    NPEConfig,
    NPETrainer,
    setup_scheduler,
    set_seed,
)

# ----------------------------------------------------------------------------
# 1. Configuration Globale et Chargement des Données (Exécuté une seule fois)
# ----------------------------------------------------------------------------

print("Utilisation de torch", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Chemins des données
theta_files = [
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/theta_wst.pt"    
]
x_files = [
    "/rds/rds-clecat/pipeline_alina_full/alina_paper/sbi_wst_5000/x_wst.pt"
]

# Dossier pour sauvegarder l'étude Optuna et les logs
STUDY_NAME = "optimize_npe_wst"
STORAGE_DIR = "/rds/rds-clecat/pipeline_alina_full/alina_paper/optuna_studies"
os.makedirs(STORAGE_DIR, exist_ok=True)
STORAGE_URL = f"sqlite:///{os.path.join(STORAGE_DIR, STUDY_NAME)}.db"

# Paramètres fixes (non optimisés)
TRAIN_SIZE = 5000
TRAIN_FRAC = 0.8
MAX_EPOCHS = 200 # Réduit pour l'optimisation, ou 2**20 avec early stopping agressif
STOP_AFTER_EPOCH = 20 # Patience pour l'early stopping (plus court pour aller vite)

# Chargement du dataset complet
print("Chargement des données...")
full_ds = TszDataset(theta_files, x_files, device=device)

# Création des indices fixes pour que tous les trials utilisent les mêmes données
idx = torch.arange(min(TRAIN_SIZE, len(full_ds)))
n_train = int(TRAIN_FRAC * len(idx))
train_idx = idx[:n_train]
val_idx   = idx[n_train:]

train_dataset = Subset(full_ds, train_idx)
val_dataset   = Subset(full_ds, val_idx)

print(f"Données chargées : Train={len(train_dataset)} | Val={len(val_dataset)}")


# ----------------------------------------------------------------------------
# 2. Définition de la fonction Objective
# ----------------------------------------------------------------------------

def objective(trial):
    """
    Fonction optimisée par Optuna.
    Elle configure un modèle, l'entraîne et retourne la meilleure validation loss.
    """
    
    # --- A. Echantillonnage des Hyperparamètres ---
    
    # 1. Architecture du Neural Spline Flow (MAF/NSF)
    hidden_features = trial.suggest_int("hidden_features", 32, 128, step=16)
    num_transforms  = trial.suggest_int("num_transforms", 3, 10)
    # num_components est souvent moins critique pour NSF, on peut le fixer ou l'optimiser
    num_components  = trial.suggest_int("num_components", 2, 8) 
    
    # 2. Régularisation et Entraînement
    dropout_p  = trial.suggest_float("dropout_probability", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # 3. Learning Rate et Scheduler
    initial_lr = trial.suggest_float("initial_lr", 1e-5, 1e-3, log=True)
    gamma      = trial.suggest_float("gamma", 0.8, 0.99)
    # step_size: tous les combien d'epochs on réduit le LR
    step_size  = trial.suggest_int("step_size", 10, 100)

    # 4. Fonction d'activation (Optionnel)
    activation_name = trial.suggest_categorical("activation", ["Tanh", "ReLU"])
    activation_fn = nn.Tanh if activation_name == "Tanh" else nn.ReLU

    # --- B. Configuration du modèle ---
    
    # On fixe une seed basée sur le numéro du trial pour la reproductibilité interne
    seed_trial = 42 + trial.number
    
    npe_config = NPEConfig(
        model="maf", # ou "nsf" selon ton backend, souvent "maf" dans sbi appelle un flow
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_components=num_components,
        dropout_probability=dropout_p,
        activation=activation_fn,
        grad_clip=5.0,
        stop_after_epoch=STOP_AFTER_EPOCH, # Early stopping patience
        seed=seed_trial,
    )

    # --- C. Entraînement ---
    
    trainer = NPETrainer(config=npe_config, device=device)
    
    # Définition du scheduler dynamique
    scheduler_fn = lambda opt: setup_scheduler(opt, step_size=step_size, gamma=gamma)

    try:
        # Lancement de l'entraînement
        run_info = trainer.fit(
            train=train_dataset,
            val=val_dataset,
            batch_size=batch_size,
            max_epochs=MAX_EPOCHS, 
            initial_lr=initial_lr,
            scheduler_fn=scheduler_fn,
            device=device,
            verbose_every=50, # Moins de logs
        )
        
        # Récupération de la meilleure validation loss
        # Note: run_info["val_losses"] contient l'historique. 
        # La dernière valeur est celle après early stopping.
        best_val_loss = min(run_info["val_losses"])
        
        # Optionnel: Pruning (arrêt prématuré des mauvais essais)
        # Comme ton trainer.fit fait toute la boucle, le pruning epoch par epoch 
        # nécessiterait de modifier sbi_backend. Pour l'instant, on optimise sur le résultat final.
        
        return best_val_loss

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # On retourne une valeur très haute pour dire à Optuna que c'est un échec
        return float('inf')


# ----------------------------------------------------------------------------
# 3. Lancement de l'étude Optuna
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Création ou chargement de l'étude
    # direction="minimize" car on veut réduire la validation loss (NLL)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="minimize",
        sampler=TPESampler(seed=42), # Sampler Bayésien efficace
        load_if_exists=True
    )

    print(f"Lancement de l'optimisation pour l'étude : {STUDY_NAME}")
    print(f"Base de données : {STORAGE_URL}")
    
    # Nombre d'essais total
    N_TRIALS = 50 
    
    # Lancement de l'optimisation
    # n_jobs=1 car PyTorch utilise déjà le GPU/CPU. 
    # Mettre n_jobs > 1 peut causer des erreurs CUDA out of memory.
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    # ----------------------------------------------------------------------------
    # 4. Résultats et Sauvegarde des meilleurs paramètres
    # ----------------------------------------------------------------------------

    print("\n------------------------------------------------")
    print("Optimisation terminée !")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Val Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Sauvegarde des meilleurs hyperparamètres dans un JSON pour réutilisation
    best_params_path = os.path.join(STORAGE_DIR, "best_hyperparams.json")
    with open(best_params_path, "w") as f:
        json.dump(trial.params, f, indent=4)
    
    print(f"Meilleurs hyperparamètres sauvegardés dans : {best_params_path}")
    
    # Visualisation (si possible)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(STORAGE_DIR, "history.png"))
        print("Graphique d'historique sauvegardé.")
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(STORAGE_DIR, "importance.png"))
        print("Graphique d'importance sauvegardé.")
    except Exception as e:
        print("Impossible de générer les graphiques (manque plotly/kaleido ?):", e)