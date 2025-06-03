import os
import typing as T
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dreams.api import dreams_embeddings
from dreams.utils.data import MSData
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from matchms.similarity import (
    BaseSimilarity,
    CosineGreedy,
    CosineHungarian,
    ModifiedCosine,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from iceberg_utils import predict


def similarity_function(
    reference: Spectrum, query: Spectrum, similarity_metric: BaseSimilarity
) -> Tuple[float, int]:
    """
    Calculate the similarity between two spectra using a specified similarity metric.
    """
    return similarity_metric.pair(reference, query).tolist()


def create_empty_dict(similarity_metrics: list) -> Dict[str, list]:

    dict_of_results = {
        "smiles": [],
        "inchikey": [],
        "adduct": [],
        "collision_energy": [],
        "precursor_mz": [],
        "instrument_type": [],
    }
    for metric in similarity_metrics:
        dict_of_results[metric.__class__.__name__ + "_score"] = []
        dict_of_results[metric.__class__.__name__ + "_matched_peaks"] = []

    return dict_of_results


def compute_all_similarities(
    df: pd.DataFrame,
    msdata: MSData,
    output_mgf_path: T.Union[str, Path],
) -> pd.DataFrame:
    if isinstance(output_mgf_path, str):
        output_mgf_path = Path(output_mgf_path)

    if not output_mgf_path.parent.exists():
        output_mgf_path.parent.mkdir()

    if output_mgf_path.exists():
        os.remove(output_mgf_path)

    similarity_metrics = [
        CosineGreedy(),
        ModifiedCosine(),
        CosineHungarian(),
    ]
    dict_of_results = create_empty_dict(similarity_metrics)

    for i, row in tqdm(df.reset_index().iterrows(), total=len(df)):
        if i == 20:
            break

        smiles = row["smiles"]
        inchikey = row["inchikey"]
        adduct = row["adduct"]
        collision_energy = row["collision_energy"]
        precursor_mz = row["precursor_mz"]
        instrument_type = row["instrument_type"]

        prediction = predict(
            smiles=smiles,
            adduct=adduct,
            device="cpu",
            max_nodes=128,
            binned_out=False,
            threshold=0.0,
        )
        save_as_mgf(prediction, str(output_mgf_path))

        true_spectrum = msdata.spec_to_matchms(i)
        for similarity_metric in similarity_metrics:
            score, matched_peaks = similarity_function(
                reference=true_spectrum,
                query=prediction,
                similarity_metric=similarity_metric,
            )
            dict_of_results[similarity_metric.__class__.__name__ + "_score"].append(
                score
            )
            dict_of_results[
                similarity_metric.__class__.__name__ + "_matched_peaks"
            ].append(matched_peaks)
        dict_of_results["smiles"].append(smiles)
        dict_of_results["inchikey"].append(inchikey)
        dict_of_results["adduct"].append(adduct)
        dict_of_results["collision_energy"].append(collision_energy)
        dict_of_results["precursor_mz"].append(precursor_mz)
        dict_of_results["instrument_type"].append(instrument_type)

    msdata_embeddings = compute_dreams_embeddings(msdata)
    iceberg_res = MSData.from_mgf(
        output_mgf_path,
        in_mem=True,
        prec_mz_col="PRECURSOR_MZ",
    )
    iceberg_res_embedding = compute_dreams_embeddings(iceberg_res)

    dict_of_results["dreams_embedding_cosine_similarity"] = (
        pair_similarity_from_dreams_embedding(msdata_embeddings, iceberg_res_embedding)
    )

    return pd.DataFrame(dict_of_results)


def compute_dreams_embeddings(
    data: T.Union[Path, str, MSData], **msdata_kwargs
) -> np.ndarray:
    if isinstance(data, MSData):
        if data.hdf5_pth.with_suffix(".npy").exists():
            return np.load(data.hdf5_pth.with_suffix(".npy"))

        else:
            embedding = dreams_embeddings(data)
            np.save(data.hdf5_pth.with_suffix(".npy"), embedding)
            return embedding

    else:
        if isinstance(data, str):
            data = Path(data)
        if data.with_suffix(".npy").exists():
            return np.load(data.with_suffix(".npy"))
        else:
            embedding = dreams_embeddings(data, **msdata_kwargs)
            np.save(data.with_suffix(".npy"), embedding)
            return embedding


def pair_similarity_from_dreams_embedding(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> List[np.float32]:
    ls = []
    for i in range(embedding1.shape[0]):
        ls.append(cosine_similarity([embedding1[i], embedding2[i]])[0][1])

    return ls
