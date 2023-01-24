# Python imports
import argparse
import json
import time
import uuid
import yaml
from dacite import from_dict

from itertools import product
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union
import traceback

# Third-party imports
import numpy as np
import pandas as pd
from umap import UMAP
import ray
from ray.util.multiprocessing import Pool

# Librep imports
from librep.base.transform import Transform
from librep.datasets.har.loaders import MegaHARDataset_BalancedView20Hz
from librep.datasets.multimodal import PandasMultiModalDataset, TransformMultiModalDataset, WindowedTransform
from librep.transforms.fft import FFT
from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
from librep.metrics.report import ClassificationReport

from config import *

import random
import tqdm

labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

def load_yaml(path: Union[Path, str]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.CLoader)

def load_mega(data_dir: Path, datasets: List[str] = None, label_columns: str = "standard activity code", features: List[str] = None):
    mega_dset = MegaHARDataset_BalancedView20Hz(data_dir, download=False)
    data = mega_dset.load(concat_all=True, label=label_columns, features=features)
    data.data.DataSet = data.data.DataSet.str.lower()

    if datasets is not None:
        data.data = data.data.loc[data.data["DataSet"].isin(datasets)]

    data.data['standard activity code'] = data.data['standard activity code'].astype('int')
    return data


# Non-parametric transform
def do_transform(train_dset, test_dset, transforms: List[TransformConfig]):
    transforms = []
    new_names = []
    for transform_config in transforms:
        the_transform = transforms_cls[transform_config.transform](**transform_config.kwargs)
        if transform_config.windowed:
            the_transform = WindowedTransform(
                transform=the_transform,
                fit_on=transform.windowed.fit_on,
                transform_on=transform.windowed.transform_on
            )
        transforms.append(the_transform)
        new_names.append(transform_config.name)

    transformer = TransformMultiModalDataset(transforms=transforms, new_window_name_prefix=".".join(new_names))
    train_dset = transformer(train_dset)
    test_dset = transformer(test_dset)
    return train_dset, test_dset


def do_reduce(reducer_dset, train_dset, test_dset, reducer_config, reduce_on: str = "all"):
    if reduce_on == "all":
        reducer = reducers_cls[reducer_config.algorithm](**reducer_config.kwargs)
        reducer.fit(reducer_dset[:][0])
        transform = WindowedTransform(
            transform=reducer,
            fit_on=reducer_config.windowed["fit_on"],
            transform_on=reducer_config.windowed["transform_on"],
        )
        transformer = TransformMultiModalDataset(transforms=[transform], new_window_name_prefix="reduced.")
        train_dset = transformer(train_dset)
        test_dset = transformer(test_dset)
        return train_dset, test_dset
    else:
        raise NotImplementedError(f"Reduce_on: {reduce_on} not implemented yet")


def _run(root_data_dir: str, output_dir: str, experiment_name: str, config: ExecutionConfig):
    output_dir = Path(output_dir)
    final_results = []
    additional_info = dict()

    features = config.extra.in_use_features
    # print(f"Running: {config}...")

    load_time = time.time()
    train_dset = load_mega(
        root_data_dir,
        datasets=config.train_dataset,
        features=features
    )
    additional_info["train_size"] = len(train_dset)
    test_dset = load_mega(
        root_data_dir,
        datasets=config.test_dataset,
        features=features
    )
    additional_info["test_size"] = len(test_dset)
    reducer_dset = load_mega(
        root_data_dir,
        datasets=config.reducer_dataset,
        features=features
    )
    additional_info["reduce_size"] = len(reducer_dset)
    additional_info["load_time"] = time.time()-load_time


    # print("Applying transforms...")
    # Transform
    transform_time = time.time()
    train_dset, test_dset = do_transform(train_dset, test_dset, config.transforms)
    additional_info["transform_time"] = time.time()-transform_time
    # Reduce
    # print("Applying reducer...")
    reduce_time = time.time()
    train_dset, test_dset = do_reduce(reducer_dset, train_dset, test_dset, config.reducer, config.extra.reduce_on)
    additional_info["reduce_time"] = time.time()-reduce_time

    # Create reporter
    reporter = ClassificationReport(
        use_accuracy=True,
        use_f1_score=True,
        use_classification_report=True,
        use_confusion_matrix=True,
        plot_confusion_matrix=False,
        #     normalize='true',
        #     display_labels=labels,
    )

    # Create Simple Workflow
    workflow = SimpleTrainEvalWorkflow(
        estimator=estimator_cls[config.estimator.algorithm],
        estimator_creation_kwags=config.estimator.kwargs,
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter
    )

    # Create a multi execution workflow
    num_runs = config.number_runs if config.estimator.allow_multirun else 1
    runner = MultiRunWorkflow(
        workflow=workflow,
        num_runs=num_runs
    )

    # print("Run...")
    # Run and collect results
    classification_time = time.time()
    results = runner(train_dset, test_dset)
    additional_info["classification_time"] = time.time()-classification_time

    # print("Saving...")
    # Create output directory
    output_file = output_dir / f"{config.execution_id}.yaml"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "experiment": asdict(config),
        "results": results,
        "additional": additional_info
    }

    with output_file.open("w") as f:
        yaml.dump(values, f, indent=4, sort_keys=True)

    return results

def run(args):
    root_data_dir: str = args[0]
    output_dir: str = args[1]
    experiment_name = args[2]
    file: Path = args[3]

    start = time.time()
    try:
        config = from_dict(data_class=ExecutionConfig, data=load_yaml(file))
        result = _run(root_data_dir, output_dir, experiment_name, config)
        result["exception"] = None
        result["ok"] = True
    except Exception as e:
        print(traceback.format_exc())
        result = {
            "experiment": asdict(config),
            "results": None,
            "exception": str(e),
            "additional": dict(),
            "ok": False
        }
    finally:
        result["additional"]["full_time"] = time.time()-start
        # print(f"Ended! Execution {config.execution_id} took {time.time()-start:.3f} seconds.")
        return result


if __name__ == "__main__":
    # ray.init(address="192.168.15.97:6379")

    parser = argparse.ArgumentParser(
        prog="Execute experiments in datasets",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "execution_configs_dir",
        action="store",
        help="Directory with execution configs",
        type=str
    )

    parser.add_argument(
        "--exp-name",
        action="store",
        default="experiment",
        help="Description of the experiment",
        type=str
    )

    parser.add_argument(
        "-d",
        "--data-path",
        action="store",
        help="Root data dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "-o",
        "--output-path",
        default="./results",
        action="store",
        help="Output path to store results",
        type=str,
    )

    parser.add_argument(
        "--address",
        action="store",
        default=None,
        help="Ray head node address. A local cluster will be started if false",
        type=str,
        required=False
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip executions that were already run"
    )

    # parser.add_argument(
    #     "--start",
    #     default=None,
    #     help="Start at execution config no..",
    #     type=int,
    #     required=False
    # )

    # parser.add_argument(
    #     "--end",
    #     default=None,
    #     help="End at execution config no..",
    #     type=int,
    #     required=False
    # )

    args = parser.parse_args()
    print(args)

    output_path = Path(args.output_path) / args.exp_name
    execution_configs = list(Path(args.execution_configs_dir).glob("*.yaml"))

    if args.skip_existing:
        to_keep_execution_ids = set([e.stem for e in execution_configs]).difference(set([o.stem for o in output_path.glob("*.yaml")]))
        execution_configs = [e for e in execution_configs if e.stem in to_keep_execution_ids]


    # experiments = load_yaml(args.experiment_file)
    # experiments = [from_dict(data_class=ExecutionConfig, data=e) for e in experiments]

    # exp_from = args.start or 0
    # exp_to = args.end or len(experiments)
    # experiments = experiments[exp_from:exp_to]
    # print(f"There are {len(experiments)} experiments")

    start = time.time()
    ray.init(args.address)
    print("Execution start...")

    pool = Pool()
    iterator = pool.imap_unordered(
        run, [(args.data_path, output_path, args.exp_name, e) for e in execution_configs],
    )
    final_res = list(tqdm.tqdm(iterator, total=len(execution_configs)))
    print(f"Finished! It took {time.time()-start:.3f} seconds!")
