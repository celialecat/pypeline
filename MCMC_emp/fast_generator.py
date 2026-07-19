#fast_generator.py

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Importation directe de cosmocnc (plus de rechargement dynamique)
# Assure-toi que les paths sont bons dans ton environnement ou sys.path
from cata_generator_en import load_local_cosmocnc
from cosmocnc import sr as scaling_relation_params_default

# Pour Julia via PythonCall (directement dans le processus principal)
from juliacall import Main as jl

class FastPatchGenerator:
    def __init__(self, 
                 cosmocnc_path, 
                 xgpaint_url, 
                 julia_env_path,
                 base_work_dir,
                 survey_sr_path=None,
                 survey_cat_path=None):
        """
        Initialise CosmoCNC et Julia UNE SEULE FOIS.
        """
        self.work_dir = Path(base_work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 1. Initialisation CosmoCNC (Persistante) ---
        print("[FastGen] Initialisation de CosmoCNC...")
        self.cnc, _ = load_local_cosmocnc(cosmocnc_path)
        
        # Configuration de base
        self.cnc_params = {
            "n_points": 3000, "n_z": 3000, 
            "z_min": 0.005, "z_max": 3.0,
            "M_min": 1e14, "M_max": 1e16,
            "cosmo_model": "lcdm", "hmf_type": "Tinker08",
            "mass_definition": "500c", "cosmology_tool": "classy_sz",
            "hmf_calc": "cnc", "hmf_type_deriv": "numerical",
            "power_spectrum_type": "cosmopower",
            "cosmo_amplitude_parameter": "logA", "Hubble_parameter": "h",
            "cosmo_param_density": "physical", "interp_tinker": "log",
            "class_sz_cosmo_model": "lcdm", "cosmocnc_verbose": "none",
            "load_catalogue": False, "class_sz_ndim_masses": 100,
            "class_sz_ndim_redshifts": 500, "class_sz_concentration_parameter": "B13",
            "class_sz_output": "mPk,m500c_to_m200c,m200c_to_m500c",
            "class_sz_hmf": "T08M500c",
            "class_sz_use_m500c_in_ym_relation": 1, "class_sz_use_m200c_in_ym_relation": 0,
            "observables": [["q_so_sim"]], "obs_select": "q_so_sim",
            "stacked_likelihood": False, "obs_select_min": 0.0, "obs_select_max": 0.0,
            "M_min_extended": None
        }
        
        # Chargement des fichiers survey si fournis (optionnel)
        if survey_sr_path: self.cnc_params["survey_sr"] = survey_sr_path
        if survey_cat_path: self.cnc_params["survey_cat"] = survey_cat_path

        # Paramètres cosmologiques fixes
        self.fixed_cosmo = {
            "h": 0.6766, "Ob0h2": 0.02242, 
            "n_s": 0.9665, "m_nu": 0.06, "tau_reio": 0.0544
        }
        
        # Initialisation de l'objet number_counts (le plus lourd)
        self.number_counts = self.cnc.cluster_number_counts(cnc_params=self.cnc_params)
        
        # On initialise avec une cosmo dummy pour précalculer les grilles
        dummy_cosmo = self.fixed_cosmo.copy()
        dummy_cosmo.update({"Oc0h2": 0.12, "logA": 3.0})
        self.scal_rel_params = self.cnc.scaling_relation_params_default.copy()
        
        self.number_counts.cosmo_params = dummy_cosmo
        self.number_counts.scal_rel_params = self.scal_rel_params
        self.number_counts.initialise() # <-- C'est l'étape qu'on ne veut faire qu'une fois !
        
        # --- 2. Initialisation Julia / XGPaint (Persistante) ---
        print("[FastGen] Initialisation de Julia/XGPaint...")
        
        # Config ENV
        os.environ["JULIA_PROJECT"] = julia_env_path
        
        # Chargement et compilation
        jl.seval("import Pkg")
        jl.Pkg.activate(julia_env_path)
        # On suppose que XGPaint est déjà "develop" dans cet env, sinon :
        # jl.Pkg.develop(path=xgpaint_url) 
        jl.seval("using XGPaint, CSV, DataFrames, Healpix, Pixell, ImageFiltering, Printf")
        
        # On définit la fonction de peinture directement dans la session Julia active
        # 
        self._define_julia_paint_func()
        
        print("[FastGen] Prêt.")

    def _define_julia_paint_func(self):
        """Injecte le code de peinture dans la session Julia active."""
        jl.seval(r"""
        function paint_patch_memory(ra_arr, dec_arr, z_arr, m_arr, 
                                    h, Ob0h2, Oc0h2, B,
                                    patch_size_deg, pix_res_arcmin, nx, beam_fwhm_arcmin,
                                    out_path)
            
            # Conversion params
            Ωb = Ob0h2 / h^2
            Ωc = Oc0h2 / h^2
            
            # Modèle tSZ
            a10_base = Arnauld10ThermalSZProfile(Omega_c = Ωc, Omega_b = Ωb, h = h, B = B)
            y_model  = build_interpolator(a10_base; Nx = nx)
            
            # Géométrie
            half = patch_size_deg / 2.0
            box  = [half -half; -half half] * Pixell.degree
            shape, wcs = geometry(CarClenshawCurtis{Float64}, box, pix_res_arcmin * Pixell.arcminute)
            
            # Painting
            sky_map64 = Enmap(zeros(shape), wcs)
            workspace = profileworkspace(shape, wcs)
            
            paint!(sky_map64, workspace, y_model, m_arr, z_arr, ra_arr, dec_arr)
            
            # Beam convolution
            if !isnothing(beam_fwhm_arcmin)
                σ_pix = (beam_fwhm_arcmin / (2*sqrt(2*log(2)))) / pix_res_arcmin
                k = ImageFiltering.KernelFactors.gaussian(σ_pix)
                sky_map64 .= ImageFiltering.imfilter(sky_map64, (k, k))
            end
            
            # Save as Float32 fits
            buf32 = Float32.(Array(sky_map64))
            sky_map32 = Enmap(buf32, wcs)
            write_map(out_path, sky_map32)
        end
        """)

    def generate(self, logA, Oc0h2, seed_offset=0, cleanup=True):
        """
        Génère un patch pour les paramètres donnés.
        """
        # 1. Mise à jour paramètres CosmoCNC
        current_cosmo = self.fixed_cosmo.copy()
        current_cosmo["logA"] = logA
        current_cosmo["Oc0h2"] = Oc0h2
        
        self.number_counts.update_params(current_cosmo, self.scal_rel_params)
        
        # 2. Génération Catalogue (En mémoire ou minimaliste)
        # On utilise le générateur interne pour éviter les I/O disque complexes
        unique_seed = int(seed_offset) + int(time.time()) # Seed simple
        np.random.seed(unique_seed)
        
        cat_gen = self.cnc.catalogue_generator(
            number_counts=self.number_counts,
            n_catalogues=1,
            seed=unique_seed,
            get_sky_coords=False, # On gère ça nous même plus vite
            sky_frac=self._compute_f_sky(10.0, 10.0) # Patch 10x10
        )
        cat_gen.generate_catalogues_hmf()
        raw_cat = cat_gen.catalogue_list # Dict {'M': [], 'z': []}
        
        if len(raw_cat['M']) == 0:
            print("[FastGen] Catalogue vide !")
            return None

        # 3. Attribution Coordonnées (Python pur, rapide)
        n_clus = len(raw_cat['M'])
        ra_rad, dec_rad = self._sample_coords(n_clus, 10.0, 10.0)
        
        # 4. Peinture (Appel direct Julia)
        # Préparation des vecteurs pour Julia (conversion numpy -> julia via PythonCall implicite)
        # Attention: XGPaint attend M en M_sun (pas 10^14), vérifier l'unité sortie par CNC.
        # Souvent CNC sort M500 en M_sun. XGPaint attend M500.
        
        # Définition chemin sortie
        run_id = f"run_{time.time_ns()}"
        out_dir = self.work_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fits_path = out_dir / "patch.fits"
        
        # Passage des données à Julia (PythonCall gère la conversion numpy->julia)
        # XGPaint attend: ra (rad), dec (rad), z, M (Msun)
        # Lat conversion: XGPaint attend souvent Dec, mais ta fonction julia convertissait lat->dec.
        # Ici _sample_coords donne (lon, colatitude).
        # Dec = pi/2 - colatitude.
        
        # Appel fonction Julia
        try:
            jl.paint_patch_memory(
                ra_rad,                 # ra_arr
                np.pi/2.0 - dec_rad,    # dec_arr (conv colat -> dec)
                raw_cat['z'],           # z_arr
                raw_cat['M'],           # m_arr
                current_cosmo['h'],
                current_cosmo['Ob0h2'],
                Oc0h2,
                1.35,                   # B
                10.0,                   # patch_size_deg
                0.5,                    # pix_res_arcmin
                int(10.0*60/0.5),       # nx
                None,                   # beam (None ou float)
                str(fits_path)
            )
        except Exception as e:
            print(f"[FastGen] Erreur Julia: {e}")
            return None
            
        return str(fits_path)

    def _compute_f_sky(self, w_deg, h_deg):
        return (w_deg * h_deg) / (4.0 * np.pi * (180.0 / np.pi) ** 2)

    def _sample_coords(self, n, w_deg, h_deg):
        rng = np.random.default_rng()
        ra = (rng.random(n) - 0.5) * np.deg2rad(w_deg)
        
        h = np.deg2rad(h_deg)
        theta0 = np.pi/2.0
        theta1 = theta0 - h/2.0
        theta2 = theta0 + h/2.0
        
        u = rng.random(n)
        cos_t = np.cos(theta2) + u * (np.cos(theta1) - np.cos(theta2))
        theta = np.arccos(cos_t)
        
        return ra, theta # lon, colatitude