import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml
import tqdm

datasets = [
    "kuhar",
    "motionsense",
    "wisdm",
    "uci",
    "realworld",
    "extrasensory"
]



def load_yaml(path: Union[Path, str]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main(results_dir: Path, output: Path):
    results = []
    files = list(results_dir.rglob("*.yaml"))
    for result_file in tqdm.tqdm(files, desc="Reading files"):
        result = load_yaml(result_file)
        fmt_result = {
            "execution_id": result["experiment"]["execution_id"],
            "classification_time": result["additional"]["classification_time"],
            "load_time": result["additional"]["load_time"],
            "reduce_time": result["additional"]["reduce_time"],
            "transform_time": result["additional"]["transform_time"],
            "reduce_on": result["experiment"]["extra"]["reduce_on"],
            "classifier": result["experiment"]["estimator"]["name"],
            "reducer": result["experiment"]["reducer"]["name"],
            "transforms": ", ".join(t["name"] for t in result["experiment"]["transforms"]),
            "number runs": result["experiment"]["number_runs"],
            "reducer_datasets": ", ".join(result["experiment"]["reducer_dataset"]),
            "train_datasets": ", ".join(result["experiment"]["train_dataset"]),
            "test_datasets": ", ".join(result["experiment"]["test_dataset"]),
            "sensors used": ", ".join(result["experiment"]["extra"]["in_use_features"]),
            "scaler": result["experiment"]["scaler"]["name"]
        }

        n_components = 0
        if result["experiment"]["reducer"]["algorithm"] == "umap":
            n_components = result["experiment"]["reducer"]["kwargs"]["n_components"]
        fmt_result["umap components"] = n_components

        # for sensor in ["accel", "gyro"]:
        #     for axis in ["x", "y", "z"]:
        #         fmt_result[f"use {sensor}-{axis}"] = f"{sensor}-{axis}" in result["experiment"]["extra"]["in_use_features"]

        # for in_use_set, name in [("reducer_dataset", "reduce"), ("train_dataset", "train"), ("test_dataset", "test")]:
        #     for dataset in datasets:
        #         fmt_result[f"{dataset} - {name}"] = dataset in result["experiment"][in_use_set]

        fmt_result["accuracy (mean)"] = np.mean([x["accuracy"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["accuracy (std)"] = np.std([x["accuracy"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["f1-score macro (mean)"] = np.mean([x["f1 score (macro)"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["f1-score macro (std)"] = np.std([x["f1 score (macro)"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["f1-score micro (mean)"] = np.mean([x["f1 score (micro)"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["f1-score micro (std)"] = np.std([x["f1 score (micro)"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["f1-score weighted (mean)"] = np.mean([x["f1 score (weighted)"] for r in result["results"]["runs"] for x in r["result"]])
        fmt_result["f1-score weighted (std)"] = np.std([x["f1 score (weighted)"] for r in result["results"]["runs"] for x in r["result"]])

        results.append(fmt_result)

    print("Generating Dataframe...")
    results = pd.DataFrame(results)
    results.to_csv(output, index=False)
    print(f"Results saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CSV Results Writer",
        description="Create the csv from a set of results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_path",
        action="store",
        help="Path with the results (in yaml format)",
        type=str
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        default="results.csv",
        help="Output file to store csv file",
        type=str,
        required=False
    )

    args = parser.parse_args()
    main(Path(args.results_path), Path(args.output))
