from pathlib import Path
from pixell import enmap, enplot

def plot_fits_patches(folder_path, prefix="y_map_10deg_fast_", indices=None):
    """
    Affiche les patchs .fits contenus dans un dossier en utilisant pixell.enplot.

    Paramètres
    ----------
    folder_path : str ou Path
        Chemin vers le dossier contenant les patchs .fits
    prefix : str ou None
        Préfixe des fichiers (par défaut 'y_map_10deg_fast_'). Si None, prend tous les .fits.
    indices : list, range, ou None
        Indices des fichiers à afficher. Si None, tous les fichiers trouvés sont pris.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Le dossier {folder} n'existe pas.")

    # 1) Récupération des fichiers
    pattern = f"{prefix}*.fits" if prefix is not None else "*.fits"
    fits_files = sorted(folder.glob(pattern))
    if not fits_files:
        # Si le préfixe ne matche rien, on retente en prenant tous les .fits
        if prefix is not None:
            fits_files = sorted(folder.glob("*.fits"))
    if not fits_files:
        raise FileNotFoundError(
            f"Aucun fichier .fits trouvé dans {folder} "
            f"(pattern essayé: '{pattern}')."
        )

    # 2) Sélection des indices
    if indices is None:
        indices = range(len(fits_files))

    # 3) Boucle d'affichage
    for idx in indices:
        if idx < 0 or idx >= len(fits_files):
            print(f"⚠️ Index {idx} hors limites (0..{len(fits_files)-1}).")
            continue

        fname = fits_files[idx]
        print(f"→ Lecture : {fname.name}")
        m = enmap.read_map(str(fname))

        # Si carte multi-dimensionnelle, on prend le premier plan
        # (ex. [ncomp, ny, nx] ou [nu, ny, nx]).
        m_to_plot = m
        if hasattr(m, "ndim") and m.ndim >= 3:
            m_to_plot = m[0]
            print(f"   Carte multi-dimensionnelle détectée (shape {m.shape}), "
                  f"affichage du plan 0 → shape {m_to_plot.shape}")

        # 4) Plot robuste : on tente avec title, puis fallback sans title
        try:
            plots = enplot.plot(
                m_to_plot, colorbar=True, ticks=True, font_size=14, title=fname.name
            )
        except Exception as e:
            print(f"   ℹ️ 'title' non supporté par enplot.plot sur cette version "
                  f"(fallback sans title). Détail: {e}")
            plots = enplot.plot(
                m_to_plot, colorbar=True, ticks=True, font_size=14
            )

        # enplot.plot renvoie une liste de plots ; enplot.show accepte la liste
        enplot.show(plots)
