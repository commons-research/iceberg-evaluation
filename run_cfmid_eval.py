import os
import subprocess
import typing as T
from pathlib import Path

from cache_decorator import Cache
from dreams.utils.data import MSData
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from matchms.filtering import default_filters
from matchms.importing import load_from_mgf
from ms_pred import common
from tqdm import tqdm

from iceberg_evaluation_utils import compute_dreams_embeddings
from massspecgym import load_massspecgym

NUM_THREADS = 8


def main():
    res_folder = Path(f"results/cfm_id/")
    res_folder.mkdir(exist_ok=True)

    cfm_output_specs = res_folder / "cfm_out"
    cfm_batch_scripts = res_folder / "batches"
    cfm_batch_scripts.mkdir(exist_ok=True)

    df = load_massspecgym().drop_duplicates("inchikey")
    df = df[df["adduct"] == "[M+H]+"].iloc[:10]
    input_items = [f"{i} {j}" for i, j in df[["inchikey", "smiles"]].values]

    _ = run_cfmid(
        input_items=input_items,
        cfm_output_specs=str(cfm_output_specs),
        cfm_batch_scripts=str(cfm_batch_scripts),
        res_folder=str(res_folder),
        num_threads=NUM_THREADS,
    )

    spectra: T.List[Spectrum] = []
    for file in tqdm(
        os.listdir(cfm_output_specs), desc="Converting CFM-ID output to MGF"
    ):
        convert_to_mgf(cfm_output_specs / file)
        spectra.append(
            default_filters(
                list(load_from_mgf(cfm_output_specs / file))[0],
            ),
        )

    # Save the spectra to MGF
    save_as_mgf(spectra, "results/cfmid_res.mgf", file_mode="w")
    cfm_msdata = MSData.from_mgf(
        "results/cfmid_res.mgf",
        in_mem=True,
        prec_mz_col="PRECURSOR_MZ",
    )
    cfm_embeddings = compute_dreams_embeddings(cfm_msdata)


@Cache(use_approximated_hash=False, args_to_ignore=["num_threads"])
def run_cfmid(
    input_items: T.List[str],
    cfm_output_specs: T.Union[str, Path] = "results/cfm_id/cfm_out",
    cfm_batch_scripts: T.Union[str, Path] = "results/cfm_id/batches",
    res_folder: T.Union[str, Path] = "results/cfm_id",
    num_threads: int = NUM_THREADS,
):
    if isinstance(cfm_output_specs, str):
        cfm_output_specs = Path(cfm_output_specs)
    if isinstance(cfm_batch_scripts, str):
        cfm_batch_scripts = Path(cfm_batch_scripts)
    if isinstance(res_folder, str):
        res_folder = Path(res_folder)

    cfm_inputs = []
    batches = common.batches_num_chunks(input_items, num_threads)
    batches = list(batches)
    for batch_ind, batch in enumerate(batches):
        input_file = cfm_batch_scripts / f"cfm_input_{batch_ind}.txt"
        input_str = "\n".join(batch)
        with open(input_file, "w") as fp:
            fp.write(input_str)
        cfm_inputs.append(input_file)

    def make_cfm_command(cfm_input):
        return f"""cfm-predict '/cfmid/public/{cfm_input}' 0.001 \\
        /trained_models_cfmid4.0/[M+H]+/param_output.log \\
        /trained_models_cfmid4.0/[M+H]+/param_config.txt 0 \\
        /cfmid/public/{cfm_output_specs}"""

    cfm_commands = [make_cfm_command(i) for i in cfm_inputs]

    # Run in background
    full_cmd = "\n".join([f"{i} &" for i in cfm_commands])
    full_cmd += "\nwait\n"
    cmd_file = res_folder / "cfm_full_cmd.sh"

    # wait_forever_cmd = "\nwhile true; do\n\tsleep 100\ndone"
    with open(cmd_file, "w") as fp:
        fp.write(full_cmd)
        # fp.write(wait_forever_cmd)

    docker_str = f"""docker run --rm=true -v $(pwd):/cfmid/public/ -u {os.getuid()}:{os.getgid()} \\
    -i wishartlab/cfmid:latest  \\
    sh -c  ". /cfmid/public/{cmd_file}"
    """

    print(docker_str)
    subprocess.run(docker_str, shell=True)
    print("Done running CFM-ID evaluation. Check the results folder for output.")
    return (
        input_items,
        cfm_output_specs,
        cfm_batch_scripts,
        res_folder,
    )  # used only for caching otherwise the function would always run


def convert_to_mgf(file_path: T.Union[str, Path]) -> None:
    output_lines = []
    mass_intensity_dict = {}

    # Metadata lines to remove '#' from
    metadata_keys = ["#ID=", "#SMILES=", "#InChiKey=", "#Formula="]

    with open(file_path, "r") as file:
        for line in file:
            stripped_line = line.strip()

            # Check for metadata lines
            if any(stripped_line.startswith(key) for key in metadata_keys):
                # Remove the leading '#' character
                output_lines.append(stripped_line.lstrip("#"))
            elif stripped_line.startswith("#PMass="):
                # Replace with PEPMASS=
                output_lines.append(stripped_line.replace("#PMass=", "PEPMASS="))
            # Skip energy lines
            elif stripped_line.startswith("energy"):
                continue
            # Process mass-intensity pairs
            else:
                parts = stripped_line.split()
                if len(parts) == 2:
                    try:
                        mass = float(parts[0])
                        intensity = float(parts[1])
                        # Keep the highest intensity for each mass
                        if (
                            mass not in mass_intensity_dict
                            or intensity > mass_intensity_dict[mass]
                        ):
                            mass_intensity_dict[mass] = intensity
                    except ValueError:
                        # Skip lines that don't parse correctly
                        continue

    # Add the cleaned mass-intensity pairs
    for mass, intensity in sorted(mass_intensity_dict.items()):
        output_lines.append(f"{mass} {intensity}")

    output_lines = ["BEGIN IONS"] + ["ADDUCT=[M+H]+"] + output_lines + ["END IONS"]

    output_lines = "\n".join(output_lines)
    with open(file_path, "w") as file:
        file.write(output_lines)


if __name__ == "__main__":
    main()
