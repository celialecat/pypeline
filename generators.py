import os
import numpy as np
import pandas as pd
import math
import logging
from typing import Optional, Tuple, Dict, Any, Literal
import scattering  # déjà importé dans ton code
setattr(scattering, "np", np)  # hotfix: fournit 'np' au module


# essaye d'importer les fonctions du dépôt ; si elles existent on les utilisera
try:
    import ST  # utilitaire du dépôt (get_random_data, power spectrum helpers)
except Exception:
    ST = None

try:
    import scattering  # package scattering du dépôt (synthesis, ...)
except Exception:
    scattering = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------
# CSV -> type & parsing helpers
# ------------------------------


def read_params_csv(csv_path: str) -> pd.DataFrame:
    """
    Lit un tableau de paramètres cosmologiques depuis:
      - un CSV (encodages usuels essayés), ou
      - un fichier NumPy (.npy/.npz) contenant un array structuré ou un dict.
    """
    import os
    ext = os.path.splitext(csv_path)[1].lower()

    if ext in (".npy", ".npz"):
        arr = np.load(csv_path, allow_pickle=True)
        # .npz -> mapping clé->array
        if hasattr(arr, "files"):
            # heuristique: s'il y a une clé unique avec tableau de records
            if len(arr.files) == 1 and arr[arr.files[0]].dtype.names:
                rec = arr[arr.files[0]]
                return pd.DataFrame.from_records(rec)
            # sinon, on tente d’empiler en colonnes
            data = {k: arr[k] for k in arr.files}
            return pd.DataFrame(data)
        else:
            # .npy
            if arr.dtype.names:  # array structuré
                return pd.DataFrame.from_records(arr)
            elif isinstance(arr, np.ndarray):
                # si c’est un array 2D, on fabrique des colonnes génériques
                if arr.ndim == 2:
                    cols = [f"col{i}" for i in range(arr.shape[1])]
                    return pd.DataFrame(arr, columns=cols)
                # si c’est un array d’objets/dicts
                if arr.dtype == object:
                    try:
                        return pd.DataFrame(list(arr))
                    except Exception:
                        pass
            raise ValueError("Format .npy/.npz non reconnu pour construire un DataFrame.")
    else:
        # CSV: on essaie plusieurs encodages courants
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(csv_path, encoding=enc, engine="python")
            except Exception as e:
                last_err = e
        raise last_err


def _read_csv_detect_type(csv_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Lit le CSV en ignorant les lignes commentées (#) et détecte s'il s'agit
    d'un fichier 'wst' (coefs scattering) ou d'un 'spectrum' (ell, D_ell, ...).

    Retour: (type_str, dataframe) où type_str ∈ {'wst','spectrum'}.
    """
    # On laisse pandas détecter le séparateur, garde header si présent.
    df = pd.read_csv(csv_path, comment='#', engine='python')

    # Normalise les noms de colonnes (strip)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    # Heuristique 'spectrum' : colonne 'ell' OU colonne contenant 'D_ell'/'Cl'/'power'
    low = [c.lower() for c in cols]
    if any(c == 'ell' for c in low):
        return 'spectrum', df
    if any(('d_ell' in c or 'dell' in c or 'power' in c or c == 'cl') for c in low):
        return 'spectrum', df

    # Heuristique 'wst' : présence d'une colonne (ou valeurs) de type 'S0','S1','S2', ...
    # Cas typique fourni: colonnes [sample, kind, index, value]
    if 'kind' in low:
        if df['kind' if 'kind' in cols else cols[low.index('kind')]].astype(str).str.match(r'^\s*S\d+').any():
            return 'wst', df
    # Sinon, chercher n'importe quelle colonne où les valeurs ressemblent à 'S\d'
    for c in cols:
        if df[c].astype(str).str.match(r'^\s*S\d+').any():
            return 'wst', df

    # Fallback: si première colonne est numérique et au moins deux colonnes -> spectrum
    if df.shape[1] >= 2:
        try:
            _ = pd.to_numeric(df.iloc[:, 0])
            return 'spectrum', df
        except Exception:
            pass

    raise ValueError("Impossible de détecter le format du CSV : vérifie les en-têtes ou le contenu.")


def _parse_spectrum_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrait le couple (ells, power) depuis un dataframe de type 'spectrum'.
    On cherche 'ell' (ou 1ère colonne) et 'D_ell_mean' / 'power' / 'Cl' (ou 2e colonne).
    """
    cols = list(df.columns)
    low = [c.lower() for c in cols]

    # colonne ell
    ell_col = None
    if 'ell' in low:
        ell_col = cols[low.index('ell')]
    else:
        ell_col = cols[0]

    # colonne power
    power_candidates = []
    for c in cols[1:]:
        lc = c.lower()
        if ('d_ell' in lc) or ('dell' in lc) or ('power' in lc) or (lc == 'cl'):
            power_candidates.append(c)
    power_col = power_candidates[0] if power_candidates else (cols[1] if len(cols) >= 2 else None)
    if power_col is None:
        raise ValueError("Impossible de trouver une colonne de puissance dans le CSV.")

    ells = pd.to_numeric(df[ell_col], errors='coerce').to_numpy(dtype=float)
    power = pd.to_numeric(df[power_col], errors='coerce').to_numpy(dtype=float)

    if np.any(~np.isfinite(ells)) or np.any(~np.isfinite(power)):
        raise ValueError("Colonnes ell/power contiennent des valeurs non finies.")

    return ells, power


def _parse_wst_df(
    df: pd.DataFrame,
    sample: Optional[int] = 0,
    aggregate: Literal['none','mean'] = 'none',
    kinds: Optional[Tuple[str, ...]] = ("S0","S1"),
) -> np.ndarray:
    """
    Transforme un CSV WST de forme type [sample, kind, index, value] en vecteur 1D.
    - sample: index d'échantillon à sélectionner (par défaut 0). Si None et aggregate='mean', on moyenne sur tous les samples.
    - aggregate: 'none' (sélection d'un sample) ou 'mean' (moyenne sur la dimension sample).
    - kinds: tuple des 'kinds' à garder et leur ordre (par défaut ('S0','S1')).

    Retourne un vecteur numpy 1D trié [S0(idx=0), S1(idx=0..), ...] pour un sample ou la moyenne.
    """
    # colonnes attendues (avec tolérance aux noms)
    cols = list(df.columns)
    low = [c.lower() for c in cols]

    # repère colonnes
    def _find(colname, default=None):
        return cols[low.index(colname)] if colname in low else default

    col_sample = _find('sample', None)
    col_kind   = _find('kind', None)
    col_index  = _find('index', None)

    # colonne valeur: dernière colonne numérique par défaut
    value_col = None
    for c in reversed(cols):
        if pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break
    value_col = value_col or cols[-1]

    # Filtre des kinds voulus
    if col_kind is None:
        raise ValueError("CSV WST: colonne 'kind' absente.")

    work = df.copy()
    work[col_kind] = work[col_kind].astype(str).str.strip()

    if kinds is not None:
        work = work[work[col_kind].isin(kinds)]
        if work.empty:
            raise ValueError(f"CSV WST: aucun des kinds {kinds} n'a été trouvé.")

    # Gestion du sample / moyenne
    if aggregate == 'mean':
        # moyenne par (kind, index)
        if col_index is None:
            # S0 only (index manquant) → ajoute index 0
            work['_idx'] = 0
            g = work.groupby([col_kind, '_idx'])[value_col].mean().reset_index()
            g = g.rename(columns={'_idx':'index'})
        else:
            g = work.groupby([col_kind, col_index])[value_col].mean().reset_index()
            g = g.rename(columns={col_index:'index'})
    else:
        # sélection d'un sample précis (défaut: 0)
        if col_sample is None:
            # pas de colonne sample → on considère l'ensemble comme un seul sample
            sub = work
        else:
            s = 0 if sample is None else int(sample)
            sub = work[work[col_sample] == s]
            if sub.empty:
                raise ValueError(f"CSV WST: sample={s} introuvable.")
        if col_index is None:
            sub = sub.assign(index=0)
        else:
            sub = sub.rename(columns={col_index:'index'})
        g = sub[[col_kind, 'index', value_col]].copy()

    # Ordonne: par ordre des kinds demandé puis par index croissant
    order_kind = list(kinds) if kinds is not None else sorted(g[col_kind].unique())
    g['_korder'] = g[col_kind].apply(lambda x: order_kind.index(x) if x in order_kind else len(order_kind))
    g = g.sort_values(['_korder','index']).reset_index(drop=True)

    vec = pd.to_numeric(g[value_col], errors='coerce').to_numpy(dtype=np.float32)
    if np.any(~np.isfinite(vec)):
        raise ValueError("CSV WST: des valeurs non finies ont été trouvées dans la colonne des coefficients.")

    return vec


# --------------------------------------------
# Spectrum -> Fourier amplitude & synthesis
# --------------------------------------------

def _build_fourier_amplitude_from_spectrum(
    ells: np.ndarray, power: np.ndarray, M: int, N: int, pixel_scale_arcmin: float
) -> np.ndarray:
    """
    Construit une grille d'amplitudes (M,N) en espace de Fourier à partir d'un vecteur (ell, power).
    - power: supposé ≥ 0; si D_ell, convertir en amont si nécessaire.
    - pixel_scale_arcmin : taille de pixel en arcmin → rad.
    amplitude = sqrt( power_interpolée(ell_grid) ).
    """
    pix_rad = pixel_scale_arcmin / 60.0 * (math.pi / 180.0)  # rad/pixel

    kx = np.fft.fftfreq(M, d=pix_rad)  # cycles / radian
    ky = np.fft.fftfreq(N, d=pix_rad)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell_grid = 2.0 * math.pi * np.sqrt(KX ** 2 + KY ** 2)

    # Interpolation 1D (linéaire) sur ell, maintien des bords
    sort_idx = np.argsort(ells)
    ells_s = np.maximum(ells[sort_idx], 0.0)
    power_s = np.maximum(power[sort_idx], 0.0)

    interp = np.interp(ell_grid.ravel(), ells_s, power_s, left=power_s[0], right=power_s[-1])
    interp = interp.reshape(ell_grid.shape)

    amplitude = np.sqrt(interp, dtype=np.float64)
    return amplitude


def _hermitian_symmetric_complex_field(amplitude: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Génère un champ complexe F (M,N) hermitien: F[-i,-j] = conj(F[i,j]) à partir d'une amplitude (M,N).
    """
    M, N = amplitude.shape
    F = np.zeros((M, N), dtype=np.complex128)

    def sym(i, j):
        return (-i) % M, (-j) % N

    for i in range(M):
        for j in range(N):
            si, sj = sym(i, j)
            if (si < i) or (si == i and sj < j):
                continue
            amp = float(amplitude[i, j])
            if amp < 0:
                amp = 0.0
            if (si == i) and (sj == j):
                # points auto-conjugués (DC/Nyquist) -> réel non négatif
                F[i, j] = amp
            else:
                phi = rng.uniform(0.0, 2.0 * math.pi)
                val = amp * np.exp(1j * phi)
                F[i, j] = val
                F[si, sj] = np.conj(val)
    return F


# ------------------------------
# Public API
# ------------------------------

def generate_patch_from_csv(
    method: str,
    patch_size: Tuple[int, int],
    pixel_scale_arcmin: float,
    csv_path: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = 'cpu',
    synthesis_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Génère un patch 2D à partir d'un CSV contenant soit un spectre (ell, D_ell_mean, ...)
    soit des coefficients WST (moments S0/S1, etc.).

    Args:
      - method: 'wst'/'Dell' pour scattering, ou 'spectrum' pour un champ gaussien à partir d'un spectre.
      - patch_size: (M, N) taille du patch de sortie.
      - pixel_scale_arcmin: taille de pixel en arcminutes.
      - csv_path: chemin vers le CSV.
      - output_path: si fourni, enregistre le patch .npy à ce chemin.
      - seed: graine RNG.
      - device: 'cpu' ou 'gpu'.
      - synthesis_kwargs: kwargs passés à scattering.synthesis (J, L, steps, learning_rate, estimator_name, ...).

    Retour: patch numpy float32 de shape (M, N).
    """
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 2 ** 31 - 1))
    method_norm = (method or '').strip().lower()
    if method_norm == 'dell':
        method_norm = 'wst'

    kind, df = _read_csv_detect_type(csv_path)
    M, N = patch_size
    synthesis_kwargs = dict(synthesis_kwargs or {})

    # ---------------- Spectrum path ----------------
    if method_norm in ('spectrum', 'power', 'wst_spectrum') or (kind == 'spectrum' and method_norm != 'wst'):
        logger.info("Génération par spectre de puissance (mode 'spectrum').")
        ells, power = _parse_spectrum_df(df)
        amplitude = _build_fourier_amplitude_from_spectrum(ells, power, M, N, pixel_scale_arcmin)
        F = _hermitian_symmetric_complex_field(amplitude, rng)
        field = np.fft.ifft2(F).real.astype(np.float32)
        # normalisation optionnelle: centrage et variance unité
        field -= field.mean()
        std = float(field.std())
        if std > 0:
            field /= std
        if output_path:
            np.save(output_path, field)
        return field

    # ---------------- WST path ----------------
    if method_norm == 'wst' or (kind == 'wst' and method_norm == 'wst'):
        if scattering is None:
            raise ImportError("Module 'scattering' introuvable : ajoute le package du dépôt au PYTHONPATH ou installe-le (pip install -e).")
        logger.info("Génération par Wavelet Scattering Transform (synthesis).")

        # --- Sélection d'un sample ou moyenne sur les 10 (évite le piège 80 vs 8) ---
        sample_id = synthesis_kwargs.pop('sample', 0)      # int ou None
        aggregate = synthesis_kwargs.pop('aggregate', 'none')  # 'none' | 'mean'
        kinds = synthesis_kwargs.pop('kinds', ("S0","S1"))   # garder S0/S1 par défaut

        coef_vec = _parse_wst_df(df, sample=sample_id, aggregate=aggregate, kinds=kinds)

                # --- Ajustement selon le type d’estimateur ---
        J = synthesis_kwargs.get("J", 7)
        L = synthesis_kwargs.get("L", 4)
        estimator = synthesis_kwargs.get("estimator_name", "wst")

        # Si on est en mode isotrope (wst_iso), on s’attend à un vecteur de longueur J
        if estimator == "wst_iso":
            if coef_vec.size != J:
                raise ValueError(f"wst_iso attend J={J} valeurs S0; trouvé {coef_vec.size}.")
            coef_vec = np.concatenate([[0.0], coef_vec])  # → taille 1+J

        # Si on est en mode wst complet (non isotrope), on s’attend à J*L valeurs
        elif estimator == "wst":
            expected = J * L
            if coef_vec.size != expected:
                raise ValueError(f"wst attend J*L={expected} valeurs S1(j,l); trouvé {coef_vec.size}.")
            coef_vec = np.concatenate([[0.0], coef_vec])  # → taille 1+J*L

        # --- Prépare la cible & arguments de synthèse ---
        import torch
        target_t = torch.as_tensor(coef_vec, dtype=torch.float32).view(1, -1)  # (1, K)

        # Nom d'estimateur par défaut pour S0/S1 (adapter selon la lib si besoin)
        estimator_name = synthesis_kwargs.pop('estimator_name', 'moments')

        synth_args = dict(
            estimator_name=estimator_name,
            target=target_t,
            mode='estimator',     # on fournit directement le vecteur cible
            M=M, N=N,
            device=device,
        )
        # Merge kwargs utilisateur (J, L, steps, learning_rate, etc.)
        synth_args.update(synthesis_kwargs)

        logger.info(f"Appel de scattering.synthesis avec args: {list(synth_args.keys())}")

        # --- Appel synthèse ---
        image_syn = scattering.synthesis(**synth_args)

        # --- Conversion numpy ---
        if isinstance(image_syn, (list, tuple)):
            image_syn = image_syn[0]
        if hasattr(image_syn, 'detach'):
            image_syn = image_syn.detach()
        if hasattr(image_syn, 'cpu'):
            image_syn = image_syn.cpu().numpy()

        # Attendu: (1,M,N) ou (M,N)
        if image_syn.ndim == 3 and image_syn.shape[0] == 1:
            patch = image_syn[0].astype(np.float32)
        elif image_syn.ndim == 2:
            patch = image_syn.astype(np.float32)
        else:
            patch = np.squeeze(image_syn).astype(np.float32)

        # Normalisation simple
        patch -= patch.mean()
        std = float(patch.std())
        if std > 0:
            patch /= std

        if output_path:
            np.save(output_path, patch)
        return patch

    # -------------------------------------------------
    raise ValueError(f"method '{method}' non supportée ou CSV type='{kind}' non compatible.")


# petit helper d'exemple pour notebook
def example_notebook_usage():
    """
    Exemple rapide (à adapter) pour un notebook :
    >>> from generators import generate_patch_from_csv
    >>> patch = generate_patch_from_csv(
    ...   'wst', patch_size=(256,256), pixel_scale_arcmin=0.5,
    ...   csv_path='coefs_wst.csv', seed=0, device='cpu',
    ...   synthesis_kwargs={'J':6,'L':4,'steps':400,'learning_rate':0.4,
    ...                     'sample':0, 'aggregate':'none', 'kinds':('S0','S1'),
    ...                     'estimator_name':'moments'}
    ... )
    >>> import matplotlib.pyplot as plt; plt.imshow(patch); plt.colorbar()
    """
    pass
