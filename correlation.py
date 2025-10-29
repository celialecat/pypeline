import numpy as np
import matplotlib.pyplot as plt

def covariance_to_correlation(C: np.ndarray) -> np.ndarray:
    """Converts a covariance matrix C into a correlation matrix ρ."""
    d = np.sqrt(np.diag(C))
    D = np.outer(d, d)
    rho = C / D
    np.allclose(np.diag(rho), 1.0, atol=1e-8)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(rho, origin="lower", vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, label="Correlation ρ₍ᵢⱼ₎")
    plt.title("Correlation matrix (all statistics combined)")
    plt.xlabel("Index j")
    plt.ylabel("Index i")
    plt.tight_layout()
    plt.show()
    return rho
