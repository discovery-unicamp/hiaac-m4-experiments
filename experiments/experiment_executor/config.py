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

################################################################################
# Configuration classes
################################################################################

# YAML valid confuguration keys. 
# The main experiment configuration class is `ExecutionConfig`

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
# Transforms
################################################################################

class Identity(Transform):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def transform(self, X):
        return X

################################################################################
# Constants (Valid keys)
################################################################################

# Dictionary with the valid estimators keys to use in experiment configuration 
# (under estimator.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Estimators must be a subclass of `librep.estimators.base.BaseEstimator` or implement
# the same interface (scikit-learn compatible, fit/predict methods)
estimator_cls = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "RandomForest":RandomForestClassifier,
}

# Dictionary with the valid reducer keys to use in experiment configuration
# (under reducer.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Reducers must be a subclass of `librep.reducers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
reducers_cls = {
    "identity": Identity,
    "umap": UMAP
}

# Dictionary with the valid transforms keys to use in experiment configuration
# (under transform.transform key).
# The key is the algorithm name and the value is the class to use.
# Transforms must be a subclass of `librep.transforms.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
transforms_cls = {
    "identity": Identity,
    "fft": FFT,
}

# Dictionary with the valid scalers keys to use in experiment configuration
# (under scaler.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Scalers must be a subclass of `librep.scalers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
scaler_cls = {
    "identity": Identity,
    "std_scaler": StandardScaler,
    "min_max_scaler": MinMaxScaler,
}

# Dictionary with standard labels for each activity code
standard_labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

# Dictionary with the valid datasets keys to use in experiment configuration
# (under reducer_dataset, train_dataset and test_dataset keys).
# The key is the dataset name and the value is the relative path to the dataset.
# The dataset must contain the following files: train.csv, validation.csv and test.csv
datasets = {
    # KuHar
    "kuhar.raw_balanced": Path("KuHar/raw_balanced"),
    "kuhar.standartized_balanced": Path("KuHar/standartized_balanced"),
    # MotionSense
    "motionsense.raw_balanced": Path("MotionSense/raw_balanced"),
    "motionsense.standartized_balanced": Path("MotionSense/standartized_balanced"),
    # UCI
    "uci.raw_balanced": Path("UCI/raw_balanced"),
    "uci.standartized_balanced": Path("UCI/standartized_balanced"),
}
