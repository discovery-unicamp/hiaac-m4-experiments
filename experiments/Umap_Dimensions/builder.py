# Python imports
import argparse
import json
import time
import uuid
import yaml

from itertools import product
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

# Third-party imports
import numpy as np
import pandas as pd
from umap import UMAP
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

datasets_cls = [
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

def parse_configs(config_file: Path):
    def check_keys(d: dict, keys: list, complement: str = ""):
        for k in keys:
            if k not in d:
                if complement:
                    k = f"{complement}.{k}"
                raise KeyError(f"Key '{k}' not in dict")

    config = load_yaml(config_file)
    check_keys(config, ["reducers", "estimators", "transformers"])

    reducers = []
    transforms = []
    estimators = []

    # Parse reducers
    for reducer_name, reducer_dict in config["reducers"].items():
        check_keys(reducer_dict, ["algorithm"], "reducer")
        algorithm = reducer_dict["algorithm"]
        if algorithm not in reducers_cls:
            raise ValueError(f"Invalid reducer algorithm: {algorithm}")
        reducer_config = ReducerConfig(
            name=reducer_name,
            algorithm=algorithm,
            kwargs=reducer_dict.get("kwargs", {}),
            windowed=reducer_dict.get("windowed", {})
        )
        reducers.append(reducer_config)

    # Parse transforms
    for transform_chain_name, transform_list in config["transformers"].items():
        transform_chain = []
        for i, transform_dict in enumerate(transform_list):
            check_keys(transform_dict, ["transform"], f"transformers.{transform_chain_name}.{i}")
            transform = transform_dict["transform"]
            if transform not in transforms_cls:
                raise ValueError(f"Invalid transform: {transform}")
            transform_config = TransformConfig(
                name=f"{transform_chain_name}.{i}",
                transform=transform,
                kwargs=transform_dict.get("kwargs", {}),
                windowed=transform_dict.get("windowed", {})
            )
            transform_chain.append(transform_config)
        transforms.append(transform_chain)

    # Parse estimators
    for estimator_name, estimator_dict in config["estimators"].items():
        check_keys(estimator_dict, ["algorithm"], "estimators")
        algorithm = estimator_dict["algorithm"]
        if algorithm not in estimator_cls:
            raise ValueError(f"Invalid estimator: {algorithm}")
        estimator_config = EstimatorConfig(
            name=estimator_name,
            algorithm=algorithm,
            kwargs=estimator_dict.get("kwargs", {}),
            allow_multirun=reducer_dict.get("allow_multirun", True)
        )
        estimators.append(estimator_config)

    return reducers, transforms, estimators


def build_experiments_grid(reducers: List[ReducerConfig], transforms: List[TransformConfig], estimators: List[EstimatorConfig], experiment_name: str, runs:int):
    executions = []
    for i, (reducer_config, transform_config_list, estimator_config, dataset) in enumerate(product(reducers, transforms, estimators, datasets_cls)):
        experiment = ExecutionConfig(
            execution_id=str(i),
            experiment_name=experiment_name,
            run_id=1,
            number_runs=runs,
            reducer_dataset=[dataset],
            train_dataset=[dataset],
            test_dataset=[dataset],
            reducer=reducer_config,
            transforms=transform_config_list,
            estimator=estimator_config
        )
        print(experiment)
        print("-----")
        executions.append(experiment)
    return executions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Build experiment configutation",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        action="store",
        help="Configuration file to parse",
        type=str
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        help="Output file to store configs",
        type=str,
        required=True
    )

    parser.add_argument(
        "-e",
        "--experiment",
        default="A simple experiment",
        action="store",
        help="Name of the experiment",
        type=str,
    )

    parser.add_argument(
        "-r",
        "--number-runs",
        default=5,
        action="store",
        help="Number of runs",
        type=int
    )

    parser.add_argument(
        "-s",
        "--strategy",
        choices=["grid"],
        default="grid",
        action="store",
        help="Strategy to generate experiments",
        type=str
    )

    args = parser.parse_args()
    print(args)

    reducers, transforms, estimators = parse_configs(args.config_file)
    executions = build_experiments_grid(reducers, transforms, estimators, args.experiment, args.number_runs)
    print(f"There are {len(executions)} experiments!")

    with open(args.output, "w") as f:
        yaml.dump([asdict(e) for e in executions], f)
    print(f"Configs saved to {args.output}")
    # executions = [asdict(e) for e in executions]

    # executions = []
    # for i, (reducer_config, transform_config_list, estimator_config, dataset) in enumerate(product(reducers, transforms, estimators, datasets_cls)):
    #     experiment = ExecutionConfig(
    #         execution_id=str(i),
    #         experiment_name=args.experiment,
    #         run_id=1,
    #         number_runs=args.number_runs,
    #         reducer_dataset=dataset,
    #         train_dataset=dataset,
    #         test_dataset=dataset,
    #         reducer=reducer_config,
    #         transforms=transform_config_list,
    #         estimator=estimator_config
    #     )
    #     print(experiment)
    #     print("-----")
    #     executions.append(experiment)
    #
    # print(f"There are {len(executions)} experiments!")
