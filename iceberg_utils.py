import typing as T
from collections import defaultdict

import numpy as np
from cache_decorator import Cache
from matchms import Spectrum
from matchms.filtering import default_filters, derive_smiles_from_inchi
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import (
    derive_precursor_mz_from_parent_mass,
)
from ms_pred.dag_pred import joint_model
from rdkit import Chem

# Get best models
INTEN_CKPT = "./models/iceberg/canopus_iceberg_score.ckpt"
GEN_CKPT = "./models/iceberg/canopus_iceberg_generate.ckpt"


MODEL = joint_model.JointModel.from_checkpoints(
    inten_checkpoint=INTEN_CKPT, gen_checkpoint=GEN_CKPT
)


def convert_mass_to_objects_to_spectrum(
    mass_to_obj: T.List[T.Tuple[float, T.Dict[str, T.Any]]],
    root_inchi: str,
    adduct: str,
) -> Spectrum:
    """
    Convert mass_to_obj dictionary to a Spectrum object.
    """
    _mass_to_obj = sorted(list(mass_to_obj.items()), key=lambda x: x[0])
    mz = np.array([k for k, v in _mass_to_obj])
    intensities = np.array([v["inten"] for k, v in _mass_to_obj])
    metadata = {
        "inchi": root_inchi,
        "inchikey": Chem.MolToInchiKey(Chem.MolFromInchi(root_inchi))[:14],
        "adduct": adduct,
        "parent_mass": Chem.rdMolDescriptors.CalcExactMolWt(
            Chem.MolFromInchi(root_inchi)
        ),
    }
    spectrum = default_filters(
        derive_smiles_from_inchi(
            Spectrum(mz=mz, intensities=intensities, metadata=metadata)
        )
    )

    precursor_mz = derive_precursor_mz_from_parent_mass(spectrum)
    if precursor_mz is not None:
        spectrum.set("precursor_mz", precursor_mz)
    else:
        spectrum.set("precursor_mz", np.nan)

    return spectrum


def convert_iceberg_to_spectrum(result: T.Dict[str, T.Any], adduct: str) -> Spectrum:
    root_inchi = result["root_inchi"]
    frags = result["frags"]
    # Convert from frags dict into a list of mz, inten
    mass_to_obj = defaultdict(lambda: {})
    for k, val in frags.items():
        masses, intens, form = val["mz_charge"], val["intens"], val["form"]
        for m, i in zip(masses, intens):
            if i <= 0:
                continue
            cur_obj = mass_to_obj[m]
            if cur_obj.get("inten", 0) > 0:
                # update
                if cur_obj.get("inten") < i:
                    cur_obj["frag_hash"] = k
                    cur_obj["form"] = form
                cur_obj["inten"] += i
            else:
                cur_obj["inten"] = i
                cur_obj["frag_hash"] = k
                cur_obj["form"] = form

    max_inten = max(*[i["inten"] for i in mass_to_obj.values()], 1e-9)
    mass_to_obj = {
        k: dict(inten=v["inten"] / max_inten, frag_hash=v["frag_hash"], form=v["form"])
        for k, v in mass_to_obj.items()
    }

    return convert_mass_to_objects_to_spectrum(mass_to_obj, root_inchi, adduct)


@Cache(use_approximated_hash=True)
def _predict_mol(smiles: str, adduct: str, **kwargs) -> T.Dict[str, T.Any]:
    return MODEL.predict_mol(
        smiles,
        adduct=adduct,
        **kwargs,
    )


def predict(
    smiles: str,
    adduct: str,
    **kwargs,
) -> Spectrum:
    """
    Predict the spectrum for a given SMILES string and adduct using the Iceberg model.

    Args:
        smiles (str): The SMILES representation of the molecule.
        adduct (str): The adduct to use for the prediction.
        model (joint_model.JointModel): The Iceberg model to use for prediction.

    Returns:
        Spectrum: The predicted spectrum.
    """
    result = _predict_mol(smiles, adduct, **kwargs)
    return convert_iceberg_to_spectrum(result, adduct)
