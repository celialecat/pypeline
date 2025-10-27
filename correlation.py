import numpy as np
import matplotlib.pyplot as plt

def covariance_to_correlation(C: np.ndarray) -> np.ndarray:
    """Convertit une covariance C en matrice de corrélation ρ."""
    d = np.sqrt(np.diag(C))
    D = np.outer(d, d)
    rho = C / D
    # Force la diagonale à 1 par précaution numérique
    #np.fill_diagonal(rho, 1.0)
    np.allclose(np.diag(rho), 1.0, atol=1e-8)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(rho, origin="lower", vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Corrélation ρ₍ᵢⱼ₎")
    plt.title("Matrice de corrélation (toutes statistiques combinées)")
    plt.xlabel("Indice j")
    plt.ylabel("Indice i")
    plt.tight_layout()
    plt.show()
    return rho
