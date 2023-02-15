# Python imports
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Third-party imports
from umap import UMAP

# Librep imports
from librep.base.transform import Transform
from librep.transforms.fft import FFT
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier

@dataclass
class WindowedConfig:
    fit_on: Optional[str]
    transform_on: Optional[str]

@dataclass
class ReducerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict]

@dataclass
class TransformConfig:
    name: str
    transform: str
    kwargs: Optional[dict]
    windowed: Optional[WindowedConfig]

@dataclass
class EstimatorConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict]
    allow_multirun: bool

@dataclass
class ScalerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict]

@dataclass
class ExtraConfig:
    in_use_features: list
    reduce_on: str # valid values: all, sensor, axis
    scale_on: str # valid values: self, train

@dataclass
class ExecutionConfig:
    # Control variables
    execution_id: str
    number_runs: int
    # Datasets to use
    reducer_dataset: Optional[List[str]]
    train_dataset: List[str]
    test_dataset: List[str]
    # Transforms
    transforms: Optional[List[TransformConfig]]
    # Reducer
    reducer: Optional[ReducerConfig]
    # Scaler
    scaler: Optional[ScalerConfig]
    # Estimator
    estimator: EstimatorConfig
    # Extra
    extra: ExtraConfig


################################################################################

class Identity(Transform):
    def __init__(self, *args, **kwargs) -> None:
        pass

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

scaler_cls = {
    "identity": Identity,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}