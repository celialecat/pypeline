# analyse_updated.py
import getdist.plots as gdplots
import getdist
import matplotlib.pyplot as plt
import os

# 1. Mettre à jour le chemin vers la nouvelle sortie définie dans le .yml
# output: mcmc_out/chains/dell_b_3
chain_prefix = "mcmc_out/chains/run_emp_3"

print(f"Loading chains from: {chain_prefix}")

# 2. Charger les samples
try:
    # Si GetDist ne trouve pas les fichiers, il lèvera une exception ici
    samples = getdist.loadMCSamples(chain_prefix)
    
    # On retire 10% ou 20% de burn-in (le début de la chaîne avant convergence)
    samples.removeBurn(remove=0.2) 
    print(f"Samples loaded. Number of points after burn-in: {len(samples.samples)}")

except Exception as e:
    print(f"Error loading chains: {e}")
    print(f"Check that files like {chain_prefix}.1.txt exist.")
    exit()

# --- Analysis ---
try:
    # 3. Vérifier la convergence et les stats 1D
    stats = samples.getMargeStats()
    print("\nConvergence statistics & 1D Constraints:")
    
    # On utilise les noms exacts définis dans la section 'params' du YAML
    par_oc = stats.parWithName("Oc0h2")
    par_la = stats.parWithName("logA")
    
    print(par_oc)
    print(par_la)

    # 4. Calculer les moyennes et écarts-types manuellement si besoin
    mean_oc = samples.mean('Oc0h2')
    std_oc = samples.std('Oc0h2')
    mean_la = samples.mean('logA')
    std_la = samples.std('logA')

    print(f"\nResults (Gaussian approx):")
    print(f"Oc0h2 = {mean_oc:.5f} +/- {std_oc:.5f}")
    print(f"logA  = {mean_la:.5f} +/- {std_la:.5f}")

    # 5. Créer le "Triangle Plot"
    print("\nCreating triangle plot (contours_Oc0h2_logA_4.pdf)...")
    g = gdplots.get_subplot_plotter()
    
    # Configuration du plot
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    
    g.triangle_plot(
        [samples],
        params=["Oc0h2", "logA"], # Les paramètres à tracer
        filled=True,
        legend_labels=["MCMC Dell B"],
        contour_colors=['#0066CC'], # Bleu sympathique
        title_limit=1, # Affiche les contraintes (1 sigma) au-dessus des plots 1D
    )

    # 6. Sauvegarder
    plot_filename = "contours_Oc0h2_logA_1.pdf"
    plt.savefig(plot_filename)
    print(f"Plot saved to '{plot_filename}'")

except Exception as e:
    print(f"\n--- ANALYSIS ERROR ---")
    print(f"An error occurred: {e}")
    print("Note: If the chains are very short (sanity check), contours cannot be computed.")