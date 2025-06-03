import os
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
from cache_decorator import Cache
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from matchms.filtering import default_filters
from matchms.logging_functions import set_matchms_logger_level
from pandarallel import pandarallel
from tqdm import tqdm

set_matchms_logger_level("ERROR")
# Initialize pandarallel (add progress bar if you want)
pandarallel.initialize(progress_bar=True)


def to_spectrum(row: pd.Series) -> Spectrum:
    """
    Convert a DataFrame row to a Spectrum object.
    """
    return Spectrum(
        mz=np.array(row["mzs"]),
        intensities=np.array(row["intensities"]),
        metadata={
            "identifier": row.name,
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "formula": row["formula"],
            "precursor_formula": row["precursor_formula"],
            "parent_mass": row["parent_mass"],
            "precursor_mz": row["precursor_mz"],
            "adduct": row["adduct"],
            "instrument_type": row["instrument_type"],
            "collision_energy": row["collision_energy"],
            "fold": row["fold"],
            "simulation_challenge": row["simulation_challenge"],
        },
    )


@Cache(
    cache_path="cache/{function_name}/{_hash}/spectra.pkl",
    use_approximated_hash=True,
)
def to_spectra(df: pd.DataFrame) -> T.List[Spectrum]:
    # Apply to_spectrum + default_filters in parallel
    spectra = df.parallel_apply(
        lambda row: default_filters(to_spectrum(row)), axis=1
    ).tolist()
    return spectra
