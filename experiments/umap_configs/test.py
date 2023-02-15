# Python imports
import argparse
import logging
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
from librep.datasets.har.loaders import (
    MegaHARDataset_BalancedView20Hz,
    PandasMultiModalLoader,
)
from librep.datasets.multimodal import (
    ArrayMultiModalDataset,
    PandasMultiModalDataset,
    TransformMultiModalDataset,
    WindowedTransform,
)
from librep.transforms.fft import FFT
from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
from librep.metrics.report import ClassificationReport

from config import *

import random
import tqdm

import warnings

warnings.filterwarnings("always")

labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}


datasets = {
    # KuHar
    "kuhar.raw_balanced": Path("experiments/umap_configs/data/processed/KuHar/raw_balanced"),
    "kuhar.standartized_balanced": Path("experiments/umap_configs/data/processed/KuHar/standartized_balanced"),
    # MotionSense
    "motionsense.raw_balanced": Path("experiments/umap_configs/data/processed/MotionSense/raw_balanced"),
    "motionsense.standartized_balanced": Path("experiments/umap_configs/data/processed/MotionSense/standartized_balanced"),
    # UCI
    "uci.raw_balanced": Path("data/processed/UCI/raw_balanced"),
    "uci.standartized_balanced": Path("data/processed/UCI/standartized_balanced"),
}



def load_yaml(path: Union[Path, str]) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


class Loader(PandasMultiModalLoader):
    train_file = "mega.csv"
    validation_file = None
    test_file = None


def load_mega(
    datasets_to_load: List[str] = None,
    label_columns: str = "standard activity code",
    features: List[str] = ("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"),
):

    # example:
    # datasets = [
    # "kuhar.standartized_balanced[train]",
    # "kuhar.standartized_balanced[validation]",
    # "motionsense.standartized_balanced[train]",
    # "motionsense.standartized_balanced[validation]",
    # "]
    dset_names = set()

    for dset in datasets_to_load:
        name = dset.split("[")[0]
        dset_names.add(name)
    # example:
    # dset_names = {"kuhar.standartized_balanced", "motionsense.standartized_balanced"}
    
    multimodal_datasets = dict()
    for name in dset_names:
        path = datasets[name]
        loader = PandasMultiModalLoader(
            root_dir=path
        )
        train, validation, test = loader.load(
            load_train=True,
            load_validation=True,
            load_test=True,
            as_multimodal=True,
            as_array=True,
            features=features,
            label=label_columns
        )
        multimodal_datasets[name] = {
            "train": train,
            "validation": validation,
            "test": test
        }

    name = datasets_to_load[0].split("[")[0]
    split = datasets_to_load[0].split("[")[1].split("]")[0]
    final_dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])

    for dset in datasets_to_load[1:]:
        name = dset.split("[")[0]
        split = dset.split("[")[1].split("]")[0]
        dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])
        final_dset = ArrayMultiModalDataset.concatenate(final_dset, dset)

    return final_dset


if __name__ == "__main__":
    load_mega(["kuhar.standartized_balanced[train]", "motionsense.standartized_balanced[validation]"])
