from cache_decorator import Cache

from evaluation_utils import compute_all_similarities
from massspecgym import load_massspecgym, load_msdata


def main():
    df = load_massspecgym()
    msdata = load_msdata(
        path=Cache.compute_path(load_massspecgym),
        in_mem=True,
    )

    df = compute_all_similarities(df, msdata, "results/iceberg_res.mgf")
    df.to_csv("results/evaluation_results.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
