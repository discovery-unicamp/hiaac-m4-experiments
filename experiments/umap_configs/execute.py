# Python imports
import argparse
import logging
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List

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
    "kuhar.raw_balanced": Path("KuHar/raw_balanced"),
    "kuhar.standartized_balanced": Path("KuHar/standartized_balanced"),
    # MotionSense
    "motionsense.raw_balanced": Path("MotionSense/raw_balanced"),
    "motionsense.standartized_balanced": Path("MotionSense/standartized_balanced"),
    # UCI
    "uci.raw_balanced": Path("UCI/raw_balanced"),
    "uci.standartized_balanced": Path("UCI/standartized_balanced"),
}


class catchtime:
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
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def load_datasets(
    root_dir: PathLike,
    datasets_to_load: List[str] = None,
    label_columns: str = "standard activity code",
    features: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
):
    root_dir = Path(root_dir)

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
        path = root_dir / datasets[name]
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
        multimodal_datasets[name] = {
            "train": train,
            "validation": validation,
            "test": test,
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


# Non-parametric transform
def do_transform(
    datasets: List[MultiModalDataset],
    transform_configs: List[TransformConfig],
    keep_suffixes: bool = True,
):
    transforms = []
    new_names = []
    for transform_config in transform_configs:
        kwargs = transform_config.kwargs or {}
        the_transform = transforms_cls[transform_config.transform](
            **kwargs
        )
        if transform_config.windowed:
            the_transform = WindowedTransform(
                transform=the_transform,
                fit_on=transform_config.windowed.fit_on,
                transform_on=transform_config.windowed.transform_on,
            )
        else:
            the_transform = WindowedTransform(
                transform=the_transform,
                fit_on=None,
                transform_on="all",
            )
        transforms.append(the_transform)
        if keep_suffixes:
            new_names.append(transform_config.name)

    transformer = TransformMultiModalDataset(
        transforms=transforms, new_window_name_prefix=".".join(new_names)
    )

    transformed_datasets = [transformer(dataset) for dataset in datasets]
    return transformed_datasets


def do_reduce(
    datasets: List[MultiModalDataset],
    reducer_config: ReducerConfig,
    reduce_on: str = "all",
    suffix: str = "reduced.",
):
    kwargs = reducer_config.kwargs or {}
    if reduce_on == "all":
        reducer = reducers_cls[reducer_config.algorithm](**kwargs)
        reducer.fit(datasets[0][:][0])
        transform = WindowedTransform(
            transform=reducer,
            fit_on=None,
            transform_on="all",
        )
        transformer = TransformMultiModalDataset(
            transforms=[transform], new_window_name_prefix=suffix
        )
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


def do_scaling(
    datasets: List[MultiModalDataset],
    scaler_config: ScalerConfig,
    scale_on: str = "self",
    suffix: str = "scaled.",
):
    kwargs = scaler_config.kwargs or {}
    if scale_on == "self":
        datasets = [
            TransformMultiModalDataset(
                transforms=[
                    WindowedTransform(
                        transform=scaler_cls[scaler_config.algorithm](
                            **kwargs
                        ),
                        fit_on="all",
                        transform_on="all",
                    )
                ],
                new_window_name_prefix=suffix,
            )(dataset)
            for dataset in datasets
        ]
        return datasets
    elif scale_on == "train":
        transform = scaler_cls[scaler_config.algorithm](**kwargs)
        transform.fit(datasets[0][:][0])
        datasets = [
            TransformMultiModalDataset(
                transforms=[
                    WindowedTransform(
                        transform=transform,
                        fit_on=None,
                        transform_on="all",
                    )
                ],
                new_window_name_prefix=suffix,
            )(dataset)
            for dataset in datasets
        ]
        return datasets
    else:
        raise ValueError(f"scale_on: {scale_on} is not valid")


def _run_wrapper(
    root_data_dir: PathLike,
    experiment_output_file: PathLike,
    config_to_execute: ExecutionConfig,
):
    # Some sanity checks
    if (
        config_to_execute.reducer is not None
        and config_to_execute.reducer_dataset is None
    ):
        raise ValueError(
            "If reducer is specified, reducer_dataset must be specified as well"
        )

    root_data_dir = Path(root_data_dir)
    experiment_output_file = Path(experiment_output_file)

    # Useful variables
    additional_info = dict()
    start_time = time.time()

    # Load datasets
    with catchtime() as loading_time:
        # Load train dataset
        train_dset = load_datasets(
            root_dir=root_data_dir,
            datasets_to_load=config_to_execute.train_dataset,
            features=config_to_execute.extra.in_use_features,
        )
        # Load test dataset
        test_dset = load_datasets(
            root_dir=root_data_dir,
            datasets_to_load=config_to_execute.test_dataset,
            features=config_to_execute.extra.in_use_features,
        )
        # If there is any reducer dataset speficied, load reducer
        if config_to_execute.reducer_dataset:
            reducer_dset = load_datasets(
                root_dir=root_data_dir,
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

    # The workflow is divided in 5 steps:
    # 1. Do the non-parametric transform on train, test and reducer datasets
    # 2. Do the parametric transform on train and test, using the reducer dataset to fit the transform
    # 3. Do the scaling on train and test, using the train dataset to fit the scaler
    # 4. Do the training and testing
    # 5. Save the results

    # ----------- 1. Do the non-parametric transform on train, test and reducer datasets ------------

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

    # ----------- 2. Do the parametric transform on train and test, using the reducer dataset to fit the transform ------------

    with catchtime() as reduce_time:
        # Is there any reducer to do?
        if config_to_execute.reducer is not None:
            train_dset, test_dset = do_reduce(
                datasets=[reducer_dset, train_dset, test_dset],
                reducer_config=config_to_execute.reducer,
                reduce_on=config_to_execute.extra.reduce_on,
            )
    additional_info["reduce_time"] = float(reduce_time)

    # ----------- 3. Do the scaling on train and test, using the train dataset to fit the scaler ------------

    with catchtime() as scaling_time:
        # Is there any scaler to do?
        if config_to_execute.scaler is not None:
            train_dset, test_dset = do_scaling(
                datasets=[train_dset, test_dset],
                scaler_config=config_to_execute.scaler,
                scale_on=config_to_execute.extra.scale_on,
            )

    additional_info["scaling_time"] = float(scaling_time)

    # ----------- 4. Do the training and testing ------------

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
        estimator=estimator_cls[config_to_execute.estimator.algorithm],
        estimator_creation_kwags=config_to_execute.estimator.kwargs or {},
        do_not_instantiate=False,
        do_fit=True,
        evaluator=reporter,
    )

    # Create a multi execution workflow
    num_runs = (
        config_to_execute.number_runs
        if config_to_execute.estimator.allow_multirun
        else 1
    )
    runner = MultiRunWorkflow(workflow=workflow, num_runs=num_runs)

    with catchtime() as classification_time:
        results = runner(train_dset, test_dset)
    additional_info["classification_time"] = float(classification_time)

    additional_info["total_time"] = time.time() - start_time
    additional_info["num_runs"] = num_runs

    # ----------- 5. Save results ------------
    values = {
        "experiment": asdict(config_to_execute),
        "results": results,
        "additional": additional_info,
    }

    with experiment_output_file.open("w") as f:
        yaml.dump(values, f, indent=4, sort_keys=True)

    return results


def run_experiment(args):
    root_data_dir: Path = Path(args[0])
    output_dir: Path = Path(args[1])
    experiment_name: str = args[2]
    yaml_config_file: Path = Path(args[3])
    result = None
    try:
        # Load config
        config = from_dict(data_class=ExecutionConfig, data=load_yaml(yaml_config_file))
        logging.info(f"Starting execution {config.execution_id}...")

        # Create output file
        output_dir = output_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        experiment_output_file = output_dir / f"{config.execution_id}.yaml"
        result = _run_wrapper(root_data_dir, experiment_output_file, config)
    except Exception as e:
        logging.exception("Error while running experiment!")
    finally:
        # print(f"Ended! Execution {config.execution_id} took {time.time()-start:.3f} seconds.")
        return result


def run_single(args, execution_config_files: List[PathLike], output_path: PathLike):
    for e in tqdm.tqdm(execution_config_files, desc="Executing experiments"):
        run_experiment((args.data_path, output_path, args.exp_name, e))


def run_ray(args, execution_config_files: List[PathLike], output_path: PathLike):
    ray.init(args.address)
    pool = Pool()
    iterator = pool.imap(
        run_experiment,
        [
            (args.data_path, output_path, args.exp_name, e)
            for e in execution_config_files
        ],
    )
    final_res = list(
        tqdm.tqdm(
            iterator, total=len(execution_config_files), desc="Executing experiments"
        )
    )
    return final_res  # ignored


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
        type=str,
    )

    parser.add_argument(
        "--exp-name",
        action="store",
        default="experiment",
        help="Description of the experiment",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--data-path",
        action="store",
        help="Root data dir",
        type=str,
        required=True,
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
        "--ray", action="store_true", help="Run using ray (parallel execution)"
    )

    parser.add_argument(
        "--address",
        action="store",
        default=None,
        help="Ray head node address. A local cluster will be started if false",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip executions that were already run",
    )

    parser.add_argument(
        "--start",
        default=None,
        help="Start at execution config no..",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--end",
        default=None,
        help="End at execution config no..",
        type=int,
        required=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Enable verbosity: 1=INFO, 2=Debug",
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
    coloredlogs.install(level=log_level, fmt=log_format, encoding="utf-8", milliseconds=True)

    # ------ Create output path ------
    output_path = Path(args.output_path) / args.exp_name

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
            run_single(args, execution_config_files, output_path)
        else:
            run_ray(args, execution_config_files, output_path)
    print(f"Finished! It took {float(total_time):.4f} seconds!")

    # Return OK
    sys.exit(0)
