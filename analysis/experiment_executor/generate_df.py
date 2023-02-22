import argparse
from pathlib import Path
import traceback
from typing import Union

import numpy as np
import pandas as pd
import yaml
from tqdm.contrib.concurrent import thread_map


def load_yaml(path: Union[Path, str]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_result(result_file: Path):
    results = []
    result = load_yaml(result_file)
    transforms = result["experiment"]["transforms"] or []
    reducer_datasets  = result["experiment"]["reducer_dataset"] or []
    scaler = result["experiment"]["scaler"] or {}
    reducer = result["experiment"]["reducer"] or {}
    n_components = 0
    if reducer.get("algorithm", "") == "umap":
        n_components = result["experiment"]["reducer"]["kwargs"]["n_components"]

    fmt_result = {
        "experiment_name": result_file.parents[1].name,
        "run_name": result_file.parents[0].name,
        "config_id": result_file.stem,
        "reduce_size": result["additional"]["reduce_size"],
        "train_size": result["additional"]["train_size"],
        "test_size": result["additional"]["test_size"],
        "in_use_features": ", ".join(result["experiment"]["extra"]["in_use_features"]),
        "scale_on": result["experiment"]["extra"]["scale_on"],
        "reduce_on": result["experiment"]["extra"]["reduce_on"],
        "transforms": ", ".join(t["name"] for t in transforms),
        "scaler": scaler.get("name", ""),
        "reducer": reducer.get("name", ""),
        "umap components": n_components,
        "reducer_datasets": ", ".join(reducer_datasets),
        "train_datasets": ", ".join(result["experiment"]["train_dataset"]),
        "test_datasets": ", ".join(result["experiment"]["test_dataset"]),
    }

    for report in result["report"]:
        classifier_result = fmt_result.copy()
        classifier_result["estimator"] = report["estimator"]["name"]           
        classifier_result["accuracy (mean)"] = np.mean([x["accuracy"] for r in report["results"]["runs"] for x in r["result"]])
        classifier_result["accuracy (std)"] = np.std([x["accuracy"] for r in report["results"]["runs"] for x in r["result"]])
        classifier_result["f1-score macro (mean)"] = np.mean([x["f1 score (macro)"] for r in report["results"]["runs"] for x in r["result"]])
        classifier_result["f1-score macro (std)"] = np.std([x["f1 score (macro)"] for r in report["results"]["runs"] for x in r["result"]])
        classifier_result["f1-score weighted (mean)"] = np.mean([x["f1 score (weighted)"] for r in report["results"]["runs"] for x in r["result"]])
        classifier_result["f1-score weighted (std)"] = np.std([x["f1 score (weighted)"] for r in report["results"]["runs"] for x in r["result"]])
        results.append(classifier_result)
    return results

def get_result_wrapper(result_file: Path):
    try:
        return get_result(result_file)
    except Exception as e:
        print(f"Error processing {result_file}")
        print(f"{e.__class__.__name__}: {e}")
        return []



def main(results_dir: Path, output: Path, workers: int = None):
    results = []
    files = list(results_dir.rglob("*.yaml"))
    all_results = thread_map(get_result_wrapper, files, max_workers=workers, desc="Reading files")
    results = [r for rs in all_results for r in rs]

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
        "root_results_path",
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

    parser.add_argument(
        "-w",
        "--workers",
        action="store",
        type=int,
        default=None,
        help="Maximum number of workers to use",
        required=False
    )

    args = parser.parse_args()
    main(Path(args.root_results_path), Path(args.output), args.workers)
