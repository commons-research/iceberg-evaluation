import typing as T

import numpy as np
import pandas as pd
from cache_decorator import Cache
from huggingface_hub import hf_hub_download


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.

    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
        filename="data/" + file_name,
        repo_type="dataset",
    )


@Cache(use_approximated_hash=True)
def load_massspecgym(fold: T.Optional[str] = None) -> pd.DataFrame:
    """
    Load the MassSpecGym dataset.

    Args:
        fold (str, optional): Fold name to load. If None, the entire dataset is loaded.
    """
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df["mzs"] = df["mzs"].apply(parse_spec_array)
    df["intensities"] = df["intensities"].apply(parse_spec_array)
    if fold is not None:
        df = df[df["fold"] == fold]

    df["spectrum"] = df.apply(
        lambda row: np.array([row["mzs"], row["intensities"]]), axis=1
    )
    return df
