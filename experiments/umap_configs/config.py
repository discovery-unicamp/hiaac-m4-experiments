# Python imports
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

# Third-party imports
from umap import UMAP

# Librep imports
from librep.base.transform import Transform
from librep.transforms.fft import FFT
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier

@dataclass
class DatasetConfig:
    dataset: str        # Dataset name
    concat: str         # None, all, train/val
    label_columns: str  #

@dataclass
class ReducerConfig:
    name: str
    algorithm: str
    kwargs: dict
    windowed: dict

@dataclass
class TransformConfig:
    name: str
    transform: str
    kwargs: dict
    windowed: dict

@dataclass
class EstimatorConfig:
    name: str
    algorithm: str
    kwargs: dict
    allow_multirun: bool

@dataclass
class ExtraConfig:
    in_use_features: list
    reduce_on: str # valid values: all, sensor, axis

@dataclass
class ExecutionConfig:
    # Control variables
    execution_id: str
    number_runs: int
    # Dataset
    reducer_dataset: list # List[DatasetConfig]
    train_dataset: list # List[DatasetConfig]
    test_dataset: list # List[DatasetConfig]
    # Reducer
    reducer: ReducerConfig
    # Transforms
    transforms: list # List[TransformConfig]
    # Estimator
    estimator: EstimatorConfig
    # Extra
    extra: ExtraConfig


################################################################################

class Identity(Transform):
    def transform(self, X):
        return X

estimator_cls = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "RandomForest":RandomForestClassifier,
}

reducers_cls = {
    "identity": Identity,
    "umap": UMAP
}

transforms_cls = {
    "identity": Identity,
    "fft": FFT,
}
