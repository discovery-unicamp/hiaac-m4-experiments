# Python imports
import argparse
import json
import time
import uuid
import yaml
from itertools import combinations, product

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
from librep.datasets.multimodal import (
    PandasMultiModalDataset,
    TransformMultiModalDataset,
    WindowedTransform,
)
from librep.transforms.fft import FFT
from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
from librep.metrics.report import ClassificationReport

from config import *

datasets_cls = ["kuhar", "motionsense", "wisdm", "uci", "realworld", "extrasensory"]


train_datasets = [
    "kuhar",
    "motionsense",
    "wisdm",
    "realworld",
]

test_datasets = ["kuhar", "motionsense", "wisdm", "uci", "realworld", "extrasensory"]

reducer_datasets = ["kuhar", "motionsense", "wisdm", "uci", "realworld", "extrasensory"]


intra_datasets = [
    {
        "train": ["kuhar.train", "kuhar.validation"],
        "test": ["kuhar.test"],
    },
    {
        "train": ["motionsense.train", "motionsense.validation"],
        "test": ["motionsense.test"],
    },
    {
        "train": ["uci.train", "uci.validation"],
        "test": ["uci.test"],
    },
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
    check_keys(config, ["reducers", "estimators", "transformers", "scalers"])

    reducers = []
    transforms = []
    estimators = []
    scalers = []

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
            windowed=reducer_dict.get("windowed", {}),
        )
        reducers.append(reducer_config)

    # Parse transforms
    for transform_chain_name, transform_list in config["transformers"].items():
        transform_chain = []
        for i, transform_dict in enumerate(transform_list):
            check_keys(
                transform_dict,
                ["transform"],
                f"transformers.{transform_chain_name}.{i}",
            )
            transform = transform_dict["transform"]
            if transform not in transforms_cls:
                raise ValueError(f"Invalid transform: {transform}")
            transform_config = TransformConfig(
                name=f"{transform_chain_name}.{i}",
                transform=transform,
                kwargs=transform_dict.get("kwargs", {}),
                windowed=transform_dict.get("windowed", {}),
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
            allow_multirun=reducer_dict.get("allow_multirun", True),
        )
        estimators.append(estimator_config)

    for scaler_name, scaler_dict in config["scalers"].items():
        check_keys(scaler_dict, ["algorithm"], "estimators")
        algorithm = scaler_dict["algorithm"]
        if algorithm not in scaler_cls:
            raise ValueError(f"Invalid scaler: {algorithm}")
        scaler_config = ScalerConfig(
            name=scaler_name,
            algorithm=algorithm,
            kwargs=scaler_dict.get("kwargs", {}),
        )
        scalers.append(scaler_config)

    return reducers, transforms, estimators, scalers


def build_experiments_grid(
    reducers: List[ReducerConfig],
    transforms: List[TransformConfig],
    estimators: List[EstimatorConfig],
    scalers: List[ScalerConfig],
    runs: int,
):
    executions = []
    count = 0
    for i, (reducer_config, transform_config_list, estimator_config) in enumerate(
        product(reducers, transforms, estimators)
    ):
        trains = list(combinations(train_datasets, r=1)) + list(
            combinations(train_datasets, r=2)
        )  # + list(combinations(train_datasets, r=3)) + list(combinations(train_datasets, r=4))
        tests = list(combinations(test_datasets, r=1))
        # reducers = list(combinations(train_datasets, r=1)) + list(combinations(train_datasets, r=2)) #+ list(combinations(train_datasets, r=3)) + list(combinations(train_datasets, r=4)) + list(combinations(train_datasets, r=5))
        in_use_features = [
            ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
            ["accel-x", "accel-y", "accel-z"],
            ["gyro-x", "gyro-y", "gyro-z"],
        ]
        reduce_on = [
            "all",
        ]  # "sensor", "axis"]
        scale_on = ["train"]

        for intra in intra_datasets:
            for _in_use_feat, _reduce_on, _reducers, _scaler, _scaler_on in product(
                in_use_features, reduce_on, intra_datasets, scalers, scale_on
            ):
                experiment = ExecutionConfig(
                    execution_id=f"{count}".zfill(6),
                    number_runs=runs,
                    # reducer_dataset=list(reducers),
                    reducer_dataset=list(_reducers["train"]),
                    train_dataset=list(intra["train"]),
                    test_dataset=list(intra["test"]),
                    reducer=reducer_config,
                    scaler=_scaler,
                    transforms=transform_config_list,
                    estimator=estimator_config,
                    extra=ExtraConfig(
                        in_use_features=_in_use_feat,
                        reduce_on=_reduce_on,
                        scale_on=_scaler_on,
                    ),
                )
                # print(experiment)
                # print("-----")
                count += 1
                executions.append(experiment)

        # for intra in combinations(intra_datasets, 2):
        #     for _in_use_feat, _reduce_on, _reducers in product(in_use_features, reduce_on, reducers):
        #         to_reduce = list(intra[0]["train"]) + list(intra[1]["train"])
        #         experiment = ExecutionConfig(
        #             execution_id=f"{count}",
        #             number_runs=runs,
        #             # reducer_dataset=list(reducers),
        #             reducer_dataset=to_reduce,
        #             train_dataset=list(intra[0]["train"]),
        #             test_dataset=list(intra[0]["test"]),
        #             reducer=reducer_config,
        #             transforms=transform_config_list,
        #             estimator=estimator_config,
        #             extra=ExtraConfig(_in_use_feat, _reduce_on)
        #         )
        #         # print(experiment)
        #         # print("-----")
        #         count += 1
        #         executions.append(experiment)

        # for j, (_train_datasets, _test_datasets, _reducer_datasets, _in_use_feat, _reduce_on) in enumerate(product(trains, tests, reducers, in_use_features, reduce_on)):
        #     experiment = ExecutionConfig(
        #         execution_id=f"{count}",
        #         number_runs=runs,
        #         reducer_dataset=list(_reducer_datasets),
        #         train_dataset=list(_train_datasets),
        #         test_dataset=list(_test_datasets),
        #         reducer=reducer_config,
        #         transforms=transform_config_list,
        #         estimator=estimator_config,
        #         extra=ExtraConfig(_in_use_feat, _reduce_on)
        #     )
        #     # print(experiment)
        #     # print("-----")
        #     count += 1
        #     executions.append(experiment)
    return executions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Build experiment configutation",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file", action="store", help="Configuration file to parse", type=str
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        help="Output directory to store configs",
        type=str,
        required=True,
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
        type=int,
    )

    parser.add_argument(
        "-s",
        "--strategy",
        choices=["grid"],
        default="grid",
        action="store",
        help="Strategy to generate experiments",
        type=str,
    )

    args = parser.parse_args()
    print(args)

    reducers, transforms, estimators, scalers = parse_configs(args.config_file)
    executions = build_experiments_grid(
        reducers, transforms, estimators, scalers, args.number_runs
    )
    print(f"There are {len(executions)} experiments!")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    for e in executions:
        output_file = output_path / (str(e.execution_id) + ".yaml")
        with output_file.open("w") as f:
            yaml.dump(
                asdict(e),
                f,
                encoding="utf-8",
                default_flow_style=False,
                Dumper=yaml.CDumper,
            )

    print(f"Configs saved to {output_path}")
