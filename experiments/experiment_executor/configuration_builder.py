import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path

import tqdm
import yaml
from config import *
from utils import load_yaml


def main(args):
    """
    Builds configuration files for the experiment executor, based on a template.
    """
    template = load_yaml(args.template)
    dataset_locations = load_yaml(args.locations)
    view = str(args.view)
    output_dir = Path(args.output_dir)
    prefix = str(args.prefix)
    intra_train_combinations = args.intra_train
    intra_test_combinations = args.intra_test
    intra_reduce_combinations = args.intra_reduce
    same_train_test = args.same_train_test
    same_reducer_train = args.same_reducer_train

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter datasets with the given view name
    datasets_to_use = sorted(
        [k for k in dataset_locations.keys() if view in ".".join(k.split(".")[1:])]
    )

    if (
        max(intra_train_combinations) > len(datasets_to_use)
        or max(intra_test_combinations) > len(datasets_to_use)
        or max(intra_reduce_combinations) > len(datasets_to_use)
    ):
        raise ValueError(
            "The maximum intra-dataset combination is larger than the number "
            + "of datasets to use"
        )

    train_combinations = []
    for intra in intra_train_combinations:
        for dset_list in itertools.combinations(datasets_to_use, intra):
            combinations = []
            for dset in dset_list:
                combinations.append(f"{dset}[train]")
                combinations.append(f"{dset}[validation]")
            train_combinations.append(combinations)

    test_combinations = []
    for intra in intra_test_combinations:
        for dset_list in itertools.combinations(datasets_to_use, intra):
            combinations = []
            for dset in dset_list:
                combinations.append(f"{dset}[test]")
            test_combinations.append(combinations)

    reducer_combinations = []
    for intra in intra_reduce_combinations:
        for dset_list in itertools.combinations(datasets_to_use, intra):
            combinations = []
            for dset in dset_list:
                combinations.append(f"{dset}[train]")
                combinations.append(f"{dset}[validation]")
            reducer_combinations.append(combinations)

    dataset_combinations = []
    for dset_list_train, dset_list_test, dset_list_reducer in itertools.product(
        train_combinations, test_combinations, reducer_combinations
    ):
        train_set = set([i.split("[")[0] for i in dset_list_train])
        test_set = set([i.split("[")[0] for i in dset_list_test])
        reducer_set = set([i.split("[")[0] for i in dset_list_reducer])

        if same_train_test and same_reducer_train:
            if train_set == test_set == reducer_set:
                dataset_combinations.append(
                    [dset_list_train, dset_list_test, dset_list_reducer]
                )
            continue
        if same_train_test:
            if train_set == test_set:
                dataset_combinations.append(
                    [dset_list_train, dset_list_test, dset_list_reducer]
                )
            continue
        if same_reducer_train:
            if train_set == reducer_set:
                dataset_combinations.append(
                    [dset_list_train, dset_list_test, dset_list_reducer]
                )
            continue
        if not same_train_test and not same_reducer_train:
            dataset_combinations.append(
                [dset_list_train, dset_list_test, dset_list_reducer]
            )
            continue

    all_combinations = list(
        itertools.product(
            template["reducers"],
            template["transform_list"],
            template["scalers"],
            template["in_use_features_list"],
            template["reduce_on"],
            template["scale_on"],
            dataset_combinations,
        )
    )

    for i, (
        reducer,
        transforms,
        scaler,
        in_use_features,
        reduce_on,
        scale_on,
        (train_dset, test_dset, reducer_dset),
    ) in tqdm.tqdm(
        enumerate(all_combinations),
        total=len(all_combinations),
        desc="Building configurations",
    ):
        estimators = [EstimatorConfig(**e) for e in template["estimators"]]
        reducer_config = ReducerConfig(**reducer) if reducer else None
        scaler_config = ScalerConfig(**scaler) if scaler else None
        transform_list = (
            [TransformConfig(**t) for t in transforms] if transforms else None
        )
        extra_config = ExtraConfig(
            in_use_features=in_use_features, reduce_on=reduce_on, scale_on=scale_on
        )

        config = ExecutionConfig(
            version=template["version"],
            reducer_dataset=reducer_dset,
            train_dataset=train_dset,
            test_dataset=test_dset,
            transforms=transform_list,
            reducer=reducer_config,
            scaler=scaler_config,
            estimators=estimators,
            extra=extra_config,
        )

        config = asdict(config)
        output_file = output_dir / f"{prefix}{str(i).zfill(5)}.yaml"
        with open(output_file, "w") as f:
            yaml.dump(config, f, sort_keys=True, indent=4, default_flow_style=False)

    print(f"Built {len(all_combinations)} configurations and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically builds configuration files for the "
        + "experiment executor, based on a template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--template",
        type=str,
        help="The template file to use, in YAML format",
        required=True,
    )

    parser.add_argument(
        "-l",
        "--locations",
        type=str,
        help="The file containing the dataset locations, in YAML format",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="The directory to output the configuration files to",
        required=True,
    )

    parser.add_argument(
        "-v",
        "--view",
        action="store",
        type=str,
        help="View of dataset to use",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--prefix",
        action="store",
        type=str,
        help="Prefix to add to the output files",
        required=False,
        default="",
    )

    parser.add_argument(
        "-i-train",
        "--intra-train",
        action="store",
        help="Intra-dataset train combinations",
        required=False,
        type=int,
        default=[1],
        nargs="+",
    )

    parser.add_argument(
        "-i-test",
        "--intra-test",
        action="store",
        help="Intra-dataset test combinations",
        required=False,
        type=int,
        default=[1],
        nargs="+",
    )

    parser.add_argument(
        "-i-reduce",
        "--intra-reduce",
        action="store",
        help="Intra-dataset reduce combinations",
        required=False,
        type=int,
        default=[1],
        nargs="+",
    )

    parser.add_argument(
        "-s",
        "--same-train-test",
        action="store_true",
        help="Use the same train and test datasets",
        required=False,
    )

    parser.add_argument(
        "-sr",
        "--same-reducer-train",
        action="store_true",
        help="Use the same reducert and train datasets",
        required=False,
    )

    args = parser.parse_args()
    main(args)
    sys.exit(0)
