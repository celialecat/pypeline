from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, Union


"""
This script extracts cosmological parameters from CSV filenames.

Each filename is expected to include values for `logA` and `Oc0h2` 
encoded in its name (e.g., "logA=3.038456_Oc0h2=0.116816_0000.csv"). 
The script provides two main functions:

1. `extract_from_filename(path: Path) -> Dict[str, float]`
   - Extracts parameters directly from a given CSV file path.

2. `extract_params(path_like: Union[str, Path]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]`
   - If `path_like` is a single CSV file, returns a dictionary {param: value}.
   - If `path_like` is a directory, scans recursively for valid CSV files
     containing both `logA` and `Oc0h2` in their names, and returns a 
     dictionary mapping each file path to its parameters.

Fixed cosmological and nuisance parameters are predefined in `FIXED_PARAMS`. 
The script ensures that `logA` and `Oc0h2` are dynamically extracted 
from filenames, while all other parameters remain constant.

Usage examples:
- Extract parameters from one file
- Extract parameters from all matching files in a directory

Raises:
- `ValueError` if a valid filename pattern is not found.
- `FileNotFoundError` if the given path does not exist.
"""



# Fixed values (excluding logA and Oc0h2 which come from the filename)
FIXED_PARAMS = {
    "h": 0.6766,
    "n_s": 0.9665,
    "Ob0h2": 0.02242,
    "B": 1.35,
    "A_cib": 4.7,
    "A_ir": 3.2,
    "A_rs": 0.94,
}

# Regex to extract logA and Oc0h2
_LOGA_RE = re.compile(r"(?:^|[_-])logA=([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)")
_OC0H2_RE = re.compile(r"(?:^|[_-])Oc0h2=([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)")


def extract_from_filename(path: Path) -> Dict[str, float]:
    name = path.name
    m_logA = _LOGA_RE.search(name)
    m_oc   = _OC0H2_RE.search(name)
    if not m_logA or not m_oc:
        raise ValueError(f"Could not find logA/Oc0h2 in the filename: {name}")

    params = FIXED_PARAMS.copy()
    params["logA"] = float(m_logA.group(1))
    params["Oc0h2"] = float(m_oc.group(1))
    return params


def extract_params(path_like: Union[str, Path]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    - If path_like is a file: returns {param: value}
    - If path_like is a directory: returns {file_path: {param: value}}
    """
    p = Path(path_like)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    if p.is_file():
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {p}")
        return extract_from_filename(p)

    # Directory: search for all CSVs containing logA and Oc0h2
    out = {}
    for csv_path in sorted(p.rglob("*.csv")):
        name = csv_path.name
        if _LOGA_RE.search(name) and _OC0H2_RE.search(name):
            out[str(csv_path)] = extract_from_filename(csv_path)

    if not out:
        raise ValueError("No valid CSV files found in the directory.")
    return out


# --- Usage Example ---
if __name__ == "__main__":
    # Example file
    file_path = "/Users/celialecat/Desktop/Cesure/tsz_pipeline/front_end/on_test_cata/cosmo_000_df_row0000/logA=3.058456_Oc0h2=0.116816_0000.csv"
    print(extract_params(file_path))

    # Example directory
    # directory_path = "/Users/celialecat/Desktop/Cesure/tsz_pipeline/front_end/on_test_cata/"
    # print(extract_params(directory_path))