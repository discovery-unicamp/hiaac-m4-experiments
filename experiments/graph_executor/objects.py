from typing import List, Union

# Third-party imports
from umap import UMAP

from librep.config.type_definitions import PathLike
from librep.datasets.har.loaders import MegaHARDataset_BalancedView20Hz
from librep.base.transform import Transform
from librep.transforms.fft import FFT
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
from librep.metrics.report import ClassificationReport
from librep.datasets.multimodal import PandasMultiModalDataset
from librep.datasets.multimodal.operations import DatasetFitter, DatasetPredicter, DatasetTransformer, DatasetCombiner, \
    DatasetWindowedTransform, DatasetSplitTransformCombine, DatasetEvaluator, DatasetX, DatasetY, DatasetWindow, Watcher


class WrapperNode:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class DataLoader:
    def __init__(self, path: PathLike, download=False, **load_kwargs):
        self.path = path
        self.download = download
        self.load_kwargs = load_kwargs

    def load(self, datasets: Union[str, List[str]] = None):
        # Update datasets variable
        datasets = datasets if isinstance(datasets, list) else [datasets]
        datasets = [d.lower() for d in datasets]

        # Load data
        mega_dataset = MegaHARDataset_BalancedView20Hz(self.path, download=self.download).load(**self.load_kwargs)
        
        # --- Useful manipulations ---
        mega_dataset.data.Dataset = mega_dataset.data.Dataset.str.lower()
        # Convert 'standard activity code' to int
        mega_dataset.data['standard activity code'] = mega_dataset.data['standard activity code'].astype(int)
        # Select columns with 'float64' dtype and convert tofloat32
        float64_cols = list(mega_dataset.data.select_dtypes(include='float64'))
        mega_dataset.data[float64_cols] = mega_dataset.data[float64_cols].astype('float32')

        # Select the datasets if not None
        if datasets is not None:
            mega_dataset.data = mega_dataset.data.loc[mega_dataset.data.Dataset.isin(datasets)]
        
        return mega_dataset

    def __call__(self, *args, **kwds) -> PandasMultiModalDataset:
        return self.load(*args, **kwds)


the_objects = {
    "DataLoader": DataLoader,
    "DatasetFitter": DatasetFitter,
    "DatasetPredicter": DatasetPredicter,
    "DatasetTransformer": DatasetTransformer,
    "DatasetCombiner": DatasetCombiner,
    "DatasetWindowedTransform": DatasetWindowedTransform,
    "DatasetSplitTransformCombine": DatasetSplitTransformCombine,
    "DatasetEvaluator": DatasetEvaluator,
    "DatasetX": DatasetX,
    "DatasetY": DatasetY,
    "DatasetWindow": DatasetWindow,
    "Watcher": Watcher,

    # Non-graph operations
    "fft": FFT,
    "umap": UMAP,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "classification_report": ClassificationReport,

    # Other functions
    "print": WrapperNode(print)
}