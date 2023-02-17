# Python imports
import argparse
import logging
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
import coloredlogs
import ray
import tqdm
import yaml
from config import *
from dacite import from_dict

# Librep imports
from librep.config.type_definitions import PathLike
from librep.datasets.har.loaders import PandasMultiModalLoader
from librep.datasets.multimodal import (
    ArrayMultiModalDataset,
    MultiModalDataset,
    TransformMultiModalDataset,
    WindowedTransform,
)
from librep.metrics.report import ClassificationReport
from librep.utils.workflow import MultiRunWorkflow, SimpleTrainEvalWorkflow
from ray.util.multiprocessing import Pool

"""This module is used to execute the experiments based on configuration files,
written in YAML. The configuration files are writen in YAML and the valid keys 
are defined in the ExecutionConfig in config.py file. Each YAML configuration 
file is loaded using the dacite library, which converts it into a ExecutionConfig 
dataclass. The ExecutionConfig is used to configure the experiment and then run it.

The experiment is run using the ray library, which allows to run the experiment
in parallel. The number of parallel processes is, by default, defined by the 
number of CPUs available on the machine.

The valid values for the configuration file are defined in the config.py file.
This includes:
- Estimator names and classes (in the estimator_cls variable)
- Scaler names and classes (in the scaler_cls variable)
- Reducer names and classes (in the reducer_cls variable)
- Transform names and classes (in the transform_cls variable)
- Dataset names and paths (in the datasets variable)

The code is divided into four main parts:
1. The main function (at the end), which is used to parse the command line arguments 
    and  call the `run_single_thread` (for sequential execution) or `run_ray` 
    (for parallel execution).
2. The `run_single_thread` and the `run_ray` functions calls the `run_wrapper` function.
    The `run_single_thread` function calls the `run_wrapper` function sequentially.
    The `run_ray` function calls the `run_wrapper` function in parallel, for each 
    configuration file, using ray.
3. The `run_wrapper` function, put a exception handling before calling the `run_experiment`
    function. 
4. The `run_experiment` function is the main function of the module, that actually runs
    the experiment. It is responsible for loading the configuration file, loading the 
    datasets, running the experiment and saving the results. 
    This experiment is controled by a `ExecutionConfig` object, which is passed to the
    function as a parameter. This object is created from the YAML configuration file.
    The `run_experiment` function calls the utilitary functions defined in this module.
"""

# Uncomment to remove warnings
# warnings.filterwarnings("always")

class catchtime:
    """Utilitary class to measure time in a `with` python statement."""

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.e = time.time()

    def __float__(self):
        return float(self.e - self.t)

    def __coerce__(self, other):
        return (float(self), other)

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return str(float(self))


def load_yaml(path: PathLike) -> dict:
    """Utilitary function to load a YAML file.

    Parameters
    ----------
    path : PathLike
        The path to the YAML file.

    Returns
    -------
    dict
        A dictionary with the YAML file content.
    """
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def load_datasets(
    dataset_locations: Dict[str, PathLike],
    datasets_to_load: List[str],
    label_columns: str = "standard activity code",
    features: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
) -> ArrayMultiModalDataset:
    """Utilitary function to load the datasets.
    It load the datasets from specified in the `datasets_to_load` parameter.
    The datasets specified are concatenated into a single ArrayMultiModalDataset.
    This dataset is then returned.

    Parameters
    ----------
    datasets_to_load : List[str]
        A list of datasets to load. Each dataset is specified as a string in the
        following format: "dataset_name.dataset_view[split]". The dataset name is the name
        of the dataset as specified in the `datasets` variable in the config.py
        file. The split is the split of the dataset to load. It can be either
        "train", "validation" or "test".
    label_columns : str, optional
        The name of column that have the label, by default "standard activity code"
    features : List[str], optional
        The features to load, from datasets
        by default ( "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z", )

    Returns
    -------
    ArrayMultiModalDataset
        An ArrayMultiModalDataset with the loaded datasets (concatenated).

    Examples
    --------
    >>> load_datasets(
    ...     dataset_locations={
    ...         "kuhar.standartized_balanced": "data/kuhar",
    ...         "motionsense.standartized_balanced": "data/motionsense",
    ...     },
    ...     datasets_to_load=[
    ...         "kuhar.standartized_balanced[train]",
    ...         "kuhar.standartized_balanced[validation]",
    ...         "motionsense.standartized_balanced[train]",
    ...         "motionsense.standartized_balanced[validation]",
    ...     ],
    ... )
    """
    # Transform it to a Path object
    dset_names = set()

    # Remove the split from the dataset name
    # dset_names will contain the name of the datasets to load
    # it is used to index the datasets variable in the config.py file
    for dset in datasets_to_load:
        name = dset.split("[")[0]
        dset_names.add(name)

    # Load the datasets
    multimodal_datasets = dict()
    for name in dset_names:
        # Define dataset path. Join the root_dir with the path of the dataset
        path = dataset_locations[name]
        # Load the dataset
        loader = PandasMultiModalLoader(root_dir=path)
        train, validation, test = loader.load(
            load_train=True,
            load_validation=True,
            load_test=True,
            as_multimodal=True,
            as_array=True,
            features=features,
            label=label_columns,
        )
        # Store the multiple MultimodalDataset in a dictionary
        multimodal_datasets[name] = {
            "train": train,
            "validation": validation,
            "test": test,
        }

    # Concatenate the datasets

    # Pick the name and the split of the first dataset to load
    name = datasets_to_load[0].split("[")[0]
    split = datasets_to_load[0].split("[")[1].split("]")[0]
    final_dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])

    # Pick the name and the split of the other datasets to load and
    # Concatenate the other datasets
    for dset in datasets_to_load[1:]:
        name = dset.split("[")[0]
        split = dset.split("[")[1].split("]")[0]
        dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])
        final_dset = ArrayMultiModalDataset.concatenate(final_dset, dset)

    return final_dset


# Non-parametric transform
def do_transform(
    datasets: List[MultiModalDataset],
    transform_configs: List[TransformConfig],
    keep_suffixes: bool = True,
) -> List[MultiModalDataset]:
    """Utilitary function to apply a list of transforms to a list of datasets

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        List of the datasets to transform.
    transform_configs : List[TransformConfig]
        List of the transforms to apply. Each transform it will be instantiated
        based on the TransformConfig object and each one will be applied to the
        datasets.
    keep_suffixes : bool, optional
        Keep the window name suffixes, by default True

    Returns
    -------
    List[MultiModalDataset]
        The transformed datasets.
    """
    new_datasets = []
    # Loop over the datasets
    for dset in datasets:
        transforms = []
        new_names = []

        # Loop over the transforms and instantiate them
        for transform_config in transform_configs:
            # Get the transform class and kwargs and instantiate the transform
            kwargs = transform_config.kwargs or {}
            the_transform = transforms_cls[transform_config.transform](**kwargs)
            # If the transform is windowed, instantiate the WindowedTransform
            # with the defined fit_on and transform_on.
            if transform_config.windowed:
                the_transform = WindowedTransform(
                    transform=the_transform,
                    fit_on=transform_config.windowed.fit_on,
                    transform_on=transform_config.windowed.transform_on,
                )
            # Else instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            else:
                the_transform = WindowedTransform(
                    transform=the_transform,
                    fit_on=None,
                    transform_on="window",
                )
            # Create the list of transforms to apply to the dataset
            transforms.append(the_transform)
            if keep_suffixes:
                new_names.append(transform_config.name)

        # Instantiate the TransformMultiModalDataset with the list of transforms
        transformer = TransformMultiModalDataset(
            transforms=transforms, new_window_name_prefix=".".join(new_names)
        )
        # Apply the transforms to the dataset
        dset = transformer(dset)
        # Append the transformed dataset to the list of new datasets
        new_datasets.append(dset)

    # Return the list of transformed datasets
    return new_datasets


# Parametric transform
def do_reduce(
    datasets: List[MultiModalDataset],
    reducer_config: ReducerConfig,
    reduce_on: str = "all",
    suffix: str = "reduced.",
) -> List[MultiModalDataset]:
    """Utilitary function to perform dimensionality reduce to a list of
    datasets. The first dataset will be used to fit the reducer. And the
    reducer will be applied to the remaining datasets.

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        List of the datasets to perform the dimensionality reduction.
        The first dataset will be used to fit the reducer. And the
        reducer will be applied to the remaining datasets.
    reducer_config : ReducerConfig
        The reducer configuration, used to instantiate the reducer.
    reduce_on : str, optional
        How reduce will perform, by default "all".
        It can have the following values:
        - "all": the reducer will be applied to the whole dataset.
        - "sensor": the reducer will be applied to each sensor, and then,
            the datasets will be concatenated.
        - "axis": the reducer will be applied to each axis of each sensor,
            and then, the datasets will be concatenated.
    suffix : str, optional
        The new suffix to be appended to the window name, by default "reduced."

    Returns
    -------
    List[MultiModalDataset]
        The list of datasets with the dimensionality reduction applied.
        **Note**: the first will not be transformed (and not returned)
    Raises
    ------
    ValueError
        - If the number of datasets is less than 2.
        - If the reduce_on value is invalid.

    NotImplementedError
        If the reduce_on is not implemented yet.
    """
    # Sanity check
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required to reduce")

    # Get the reducer kwargs
    kwargs = reducer_config.kwargs or {}
    if reduce_on == "all":
        # Get the reducer class and instantiate it using the kwargs
        reducer = reducers_cls[reducer_config.algorithm](**kwargs)
        # Fit the reducer on the first dataset
        reducer.fit(datasets[0][:][0])
        # Instantiate the WindowedTransform with fit_on=None and
        # transform_on="all", i.e. the transform will be applied to
        # whole dataset.
        transform = WindowedTransform(
            transform=reducer,
            fit_on=None,
            transform_on="all",
        )
        # Instantiate the TransformMultiModalDataset with the list of transforms
        # and the new suffix
        transformer = TransformMultiModalDataset(
            transforms=[transform], new_window_name_prefix=suffix
        )
        # Apply the transform to the remaining datasets
        datasets = [transformer(dataset) for dataset in datasets[1:]]
        return datasets

    elif reduce_on == "axis":
        raise NotImplementedError(f"Reduce_on: {reduce_on} not implemented yet")
    elif reduce_on == "sensor":
        raise NotImplementedError(f"Reduce_on: {reduce_on} not implemented yet")
    else:
        raise ValueError(
            "Invalid reduce_on value. Must be one of: 'all', 'axis', 'sensor"
        )


# Scaling transform
def do_scaling(
    datasets: List[MultiModalDataset],
    scaler_config: ScalerConfig,
    scale_on: str = "self",
    suffix: str = "scaled.",
) -> List[MultiModalDataset]:
    """Utilitary function to perform scaling to a list of datasets.
    If scale_on is "self", the scaling will be fit and transformed applied
    to each dataset. If scale_on is "train", the scaling will be fit to the
    first dataset and then, the scaling will be applied to all the datasets.
    (including the first one, that is used to fit the model).

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        The list of datasets to scale. The first dataset will be used to fit
        the scaler if scale_on is "train".
    scaler_config : ScalerConfig
        The scaler configuration, used to instantiate the scaler.
    scale_on : str, optional
        How scaler will perform, by default "self".
        It can have the following values:
        - "self": the scaler will be fit and transformed applied to each dataset.
        - "train": the scaler will be fit to the first dataset and then, the
            scaling will be applied to all the datasets.
    suffix : str, optional
        The new suffix to be appended to the window name, by default "scaled."

    Returns
    -------
    List[MultiModalDataset]
        The list of datasets with the scaling applied.

    Raises
    ------
    ValueError
        - If the scale_on value is invalid.
    """
    #
    kwargs = scaler_config.kwargs or {}
    if scale_on == "self":
        new_datasets = []
        # Loop over the datasets
        for dataset in datasets:
            # Get the scaler class and instantiate it using the kwargs
            transform = scaler_cls[scaler_config.algorithm](**kwargs)
            # Fit the scaler usinf the whole dataset and (i.e., fit_on="all")
            # and then, apply the transform to the whole dataset (i.e.,
            # transform_on="all")
            windowed_transform = WindowedTransform(
                transform=transform,
                fit_on="all",
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            transformer = TransformMultiModalDataset(
                transforms=[windowed_transform], new_window_name_prefix=suffix
            )
            # Apply the transform to the dataset
            dataset = transformer(dataset)
            # Append the dataset to the list of new datasets
            new_datasets.append(dataset)
        return new_datasets

    elif scale_on == "train":
        new_datasets = []
        # Get the scaler class and instantiate it using the kwargs
        transform = scaler_cls[scaler_config.algorithm](**kwargs)
        # Fit the scaler on the first dataset
        transform.fit(datasets[0][:][0])
        for dataset in datasets:
            # Instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            windowed_transform = WindowedTransform(
                transform=transform,
                fit_on=None,
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            transformer = TransformMultiModalDataset(
                transforms=[windowed_transform], new_window_name_prefix=suffix
            )
            # Apply the transform to the dataset
            dataset = transformer(dataset)
            # Append the dataset to the list of new datasets
            new_datasets.append(dataset)
        return new_datasets
    else:
        raise ValueError(f"scale_on: {scale_on} is not valid")


# Function that runs the experiment
def run_experiment(
    dataset_locations: Dict[str, PathLike],
    experiment_output_file: PathLike,
    config_to_execute: ExecutionConfig,
) -> dict:
    """This function is the wrapper that runs the experiment.
    The experiment is defined by the config_to_execute parameter,
    which controls the experiment execution.

    This code runs the following steps (in order):
    1. Load the datasets
    2. Perform the non-parametric transformations, if any, using `do_transform`
        function. The transforms are specified by `config_to_execute.transforms`
        which is a list of `TransformConfig` objects.
    3. Perform the parametric transformations, if any, using `do_reduce` function.
        The reducer algorithm and parameters are specified by
        `config_to_execute.reducers` which `ReducerConfig` object.
    4. Perform the scaling, if any, using `do_scaling` function. The scaler
        algorithm and parameters are specified by `config_to_execute.scaler`
        which is a `ScalerConfig` object.
    5. Perform the training and evaluation of the model.
    6. Save the results to a file.

    Parameters
    ----------
    dataset_locations :  Dict[str, PathLike],
        Dictionary with dataset locations. Key is the dataset name and value
        is the path to the dataset.
    experiment_output_file : PathLike
        Path to the file where the results will be saved.
    config_to_execute : ExecutionConfig
        The configuration of the experiment to be executed.

    Returns
    -------
    dict
        Dictionary with the results of the experiment.

    Raises
    ------
    ValueError
        If the reducer is specified but the reducer_dataset is not specified.
    """
    # Some sanity checks
    if (
        config_to_execute.reducer is not None
        and config_to_execute.reducer_dataset is None
    ):
        raise ValueError(
            "If reducer is specified, reducer_dataset must be specified as well"
        )

    experiment_output_file = Path(experiment_output_file)

    if config_version != config_to_execute.version:
        raise ValueError(
            f"Config version ({config_to_execute.version}) "
            f"does not match the current version ({config_version})"
        )

    # Useful variables
    additional_info = dict()
    start_time = time.time()

    # ----------- 1. Load the datasets -----------
    with catchtime() as loading_time:
        # Load train dataset
        train_dset = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.train_dataset,
            features=config_to_execute.extra.in_use_features,
        )
        # Load test dataset
        test_dset = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.test_dataset,
            features=config_to_execute.extra.in_use_features,
        )
        # If there is any reducer dataset speficied, load reducer
        if config_to_execute.reducer_dataset:
            reducer_dset = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.reducer_dataset,
                features=config_to_execute.extra.in_use_features,
            )
        else:
            reducer_dset = None

    # Add some meta information
    additional_info["load_time"] = float(loading_time)
    additional_info["train_size"] = len(train_dset)
    additional_info["test_size"] = len(test_dset)
    additional_info["reduce_size"] = len(reducer_dset) if reducer_dset else 0

    # ----------- 2. Do the non-parametric transform on train, test and reducer datasets ------------

    with catchtime() as transform_time:
        # Is there any transform to do?
        if config_to_execute.transforms is not None:
            # If there is a reducer dataset, do the transform on all of them
            if reducer_dset is not None:
                train_dset, test_dset, reducer_dset = do_transform(
                    datasets=[train_dset, test_dset, reducer_dset],
                    transform_configs=config_to_execute.transforms,
                    keep_suffixes=True,
                )
            # If there is no reducer dataset, do the transform only on train and test
            else:
                train_dset, test_dset = do_transform(
                    datasets=[train_dset, test_dset],
                    transform_configs=config_to_execute.transforms,
                    keep_suffixes=True,
                )
    additional_info["transform_time"] = float(transform_time)

    # ----------- 3. Do the parametric transform on train and test, using the reducer dataset to fit the transform ------------

    with catchtime() as reduce_time:
        # Is there any reducer to do?
        if config_to_execute.reducer is not None:
            train_dset, test_dset = do_reduce(
                datasets=[reducer_dset, train_dset, test_dset],
                reducer_config=config_to_execute.reducer,
                reduce_on=config_to_execute.extra.reduce_on,
            )
    additional_info["reduce_time"] = float(reduce_time)

    # ----------- 4. Do the scaling on train and test, using the train dataset to fit the scaler ------------

    with catchtime() as scaling_time:
        # Is there any scaler to do?
        if config_to_execute.scaler is not None:
            train_dset, test_dset = do_scaling(
                datasets=[train_dset, test_dset],
                scaler_config=config_to_execute.scaler,
                scale_on=config_to_execute.extra.scale_on,
            )

    additional_info["scaling_time"] = float(scaling_time)

    # ----------- 5. Do the training, testing and evaluate ------------

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

    all_results = []

    # Create Simple Workflow
    for estimator_cfg in config_to_execute.estimators:
        results = dict()

        workflow = SimpleTrainEvalWorkflow(
            estimator=estimator_cls[estimator_cfg.algorithm],
            estimator_creation_kwags=estimator_cfg.kwargs or {},
            do_not_instantiate=False,
            do_fit=True,
            evaluator=reporter,
        )

        # Create a multi execution workflow
        runner = MultiRunWorkflow(workflow=workflow, num_runs=estimator_cfg.num_runs)
        with catchtime() as classification_time:
            results["results"] = runner(train_dset, test_dset)

        results["classification_time"] = float(classification_time)
        results["estimator"] = asdict(estimator_cfg)
        all_results.append(results)

    end_time = time.time()
    additional_info["total_time"] = end_time - start_time
    additional_info["start_time"] = start_time
    additional_info["end_time"] = end_time

    # ----------- 6. Save results ------------
    values = {
        "experiment": asdict(config_to_execute),
        "report": all_results,
        "additional": additional_info,
    }

    with experiment_output_file.open("w") as f:
        yaml.dump(values, f, indent=4, sort_keys=True)

    return results


def run_wrapper(args) -> dict:
    """Run a single experiment. This is the function that is parallelized, if needed.
    It is a wrapper around run_experiment, that takes the arguments as a list.

    Parameters
    ----------
    args : _type_
        A list of arguments, in the following order:
        - dataset_locations: Dict[str, PathLike] (locations of the datasets)
        - output_dir: Path (the directory where the results will be stored)
        - yaml_config_file: Path (the path to the yaml file containing the experiment configuration)

    Returns
    -------
    dict
        A dict with the results of the experiment and additional information.
    """
    # Unpack arguments
    dataset_locations: Dict[str, PathLike] = args[0]
    output_dir: Path = Path(args[1])
    yaml_config_file: Path = Path(args[2])
    experiment_id = yaml_config_file.stem
    result = None
    try:
        # Load config
        config = from_dict(data_class=ExecutionConfig, data=load_yaml(yaml_config_file))
        # Create output file
        experiment_output_file = output_dir / f"{experiment_id}.yaml"
        logging.info(f"Starting execution {experiment_id}. Output at {experiment_output_file}")

        # Run experiment
        result = run_experiment(dataset_locations, experiment_output_file, config)
    except Exception as e:
        logging.exception(f"Error while running experiment: {yaml_config_file}")
    finally:
        return result


def run_single_thread(
    args: Any, dataset_locations: Dict[str, PathLike], execution_config_files: List[PathLike], output_path: PathLike
):
    """Runs the experiments sequentially, without parallelization.

    Parameters
    ----------
    args : Any
        The arguments passed to the script
    dataset_locations: Dict[str, PathLike]
        A dictionary with the dataset names and their locations.
    execution_config_files : List[PathLike]
        List of configuration files to execute.
    output_path : PathLike
        Output path where the results will be stored.
    """
    results = []
    for e in tqdm.tqdm(execution_config_files, desc="Executing experiments"):
        r = run_wrapper((dataset_locations, output_path, e))
        results.append(r)
    return results


def run_ray(args: Any, dataset_locations: Dict[str, PathLike], execution_config_files: List[PathLike], output_path: PathLike):
    """Runs the experiments in parallel, using Ray.

    Parameters
    ----------
    args : Any
        The arguments passed to the script
    dataset_locations: Dict[str, PathLike]
        A dictionary with the dataset names and their locations.
    execution_config_files : List[PathLike]
        List of configuration files to execute.
    output_path : PathLike
        Output path where the results will be stored.
    """
    ray.init(args.address)
    pool = Pool()
    iterator = pool.imap(
        run_wrapper,
        [(dataset_locations, output_path, e) for e in execution_config_files],
    )
    results = list(
        tqdm.tqdm(
            iterator, total=len(execution_config_files), desc="Executing experiments"
        )
    )
    return results


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
        help="Directory with execution configuration files in YAML format",
        type=str,
    )

    parser.add_argument(
        "--run-name",
        action="store",
        default="execution",
        help="Description of the experiment run",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--data-path",
        action="store",
        help="Root data dir where the datasets are stored",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-l",
        "--dataset-locations",
        action="store",
        help="Dataset locations YAML file",
        type=str,
        required=False,
        default="./dataset_locations.yaml"
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
        "--ray", action="store_true", help="Run using ray (parallel/distributed execution)"
    )

    parser.add_argument(
        "--address",
        action="store",
        default=None,
        help="Ray head node address (cluster). A local cluster will be started if nothing is informed",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip executions that were already run, that is, have something in the output path",
    )

    parser.add_argument(
        "--start",
        default=None,
        help="Number of execution config to start",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Number of execution config to end",
        type=int,
        required=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Enable verbosity. Multiples -v increase level: 1=INFO, 2=Debug",
        default=0,
    )

    args = parser.parse_args()
    print(args)

    # ------ Enable logging ------
    log_level = logging.WARNING
    log_format = "[%(asctime)s] [%(hostname)s] [%(name)s] [%(levelname)s]: %(message)s"
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    coloredlogs.install(
        level=log_level, fmt=log_format, encoding="utf-8", milliseconds=True
    )

    # ------ Create output path ------
    output_path = Path(args.output_path) / args.run_name
    output_path.mkdir(parents=True, exist_ok=True)

    # ------ Load dataset locations ------
    data_path = Path(args.data_path)
    dataset_locations = load_yaml(args.dataset_locations)
    for k, v in dataset_locations.items():
        dataset_locations[k] = data_path / v

    # ------ Read and filter execution configs ------
    # Load configs from directory (sorted)
    execution_config_files = sorted(
        list(Path(args.execution_configs_dir).glob("*.yaml"))
    )
    logging.info(f"There are {len(execution_config_files)} configs (total)!")

    # Filter configs
    exp_from = args.start or 0
    exp_to = args.end or len(execution_config_files)
    execution_config_files = execution_config_files[exp_from:exp_to]

    # Skip existing?
    if args.skip_existing:
        # Calculate the difference between the execution configs and the output files (configs already executed)
        # Note, here we assume that the execution id is the same as the output file name
        to_keep_execution_ids = set(
            [e.stem for e in execution_config_files]
        ).difference(set([o.stem for o in output_path.glob("*.yaml")]))
        # Filter execution configs
        execution_config_files = [
            e for e in execution_config_files if e.stem in to_keep_execution_ids
        ]
    logging.info(f"There are {len(execution_config_files)} to execute!")

    # ------ Run experiments ------
    with catchtime() as total_time:
        # Run single
        if not args.ray:
            logging.warning("Running in single mode! (slow)")
            results = run_single_thread(args, dataset_locations, execution_config_files, output_path)
        else:
            results = run_ray(args, dataset_locations, execution_config_files, output_path)
            # ray.shutdown()

    if None in results:
        logging.error("Finished with errors!")
        print(f"\tFinished with errors! It took {float(total_time):.4f} seconds!")
        sys.exit(1)
    else:
        print(f"\tFinished without errors! It took {float(total_time):.4f} seconds!")
        sys.exit(0)
