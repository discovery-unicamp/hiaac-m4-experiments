# Experiment executor

This is an experiment executor that allows running different experiment configurations by using execution config files.

## Installation

### Using pip

You may install the dependencies using the `requirements.txt` file.

```
pip install -r requirements.txt
```

### Using docker image

You may use the docker image to run the experiments. It can be built using the `Dockerfile` file.

```
docker build -t experiment-executor .

```

## Execution

The execution consists of two steps:

1. Writing configuration files in [YAML](https://yaml.org/). Each configuration file corresponds to a single experiment. They are put into a user-defined directory.

2. Execute experiment configuration files in a local machine or distributed.

Once the experiments are written, the easiest way to execute the script is using:

```
python execute.py <experiments_dir> --data-path <path_to_data_root> --run-name my-experiment-run-1 --skip-existing
```

Where the `experiments_dir` is the path where configuration files are stored and the `path_to_data_root` is the path to the root of the datasets. The `--skip-existing` option allows skipping the execution of the experiment if the results already exist. Finally, the `--run-name` is the symbolic name of the execution run of the experiment. The `dataset_locations.yaml` file contains the paths to the datasets.

The script will execute each configuration file sequentially or in parallel if using `--ray` option (it also allows distributed execution in ray clusters). The results will be stored in the `results` folder. 

More options and information about the execution can be found by executing `python execute.py --help`. And more information about the execution workflow can be found in the `execute.py` file.


## Experiment configuration files

Each YAML configuration file represents one experiment and has all information to execute it (such as the datasets to be used, the transforms to be applied, and the classification algorithms). The executor script (`execute.py`) reads a folder with several experiment configuration files and executes each one sequentially or in parallel. Usually, the name of the configuration file is also the experiment ID (in the YAML file).

The `execute.py` script will perform the following steps:

1. Load the datasets (reducer, train and test datasets)
2. Apply the non-parametric transforms (*e.g.*, FFT, etc)
3. Apply the reducer algorithm (if defined). The reducer algorithm is fit to the reducer dataset, and then the train and test datasets are transformed.
4. Apply the scaler algorithm (if defined).
5. Apply the estimator algorithm
6. Save the results

The configuration file controls the behavior of execution and has the following structure:


```yaml
estimators:                       # List of estimators to be executed
-   algorithm: RandomForest       # (estimator 0) Algorithm to be used. 
                                  # Valid algorithms can be found in the
                                  # config.py file, in the `estimator_cls`
                                  # dictionary.
    kwargs:                       # (estimator 0) Algorithm crreation parameters
        n_estimators: 100         # (estimator 0) Number of estimators (RF)
    name: randomforest-100        # (estimator 0) Symbolic name of the estimator
    num_runs: 10                  # (estimator 0) Number of runs to be executed
                                  # for this estimator
-   algorithm: KNN                # (estimator 1) Algorithm to be used
    kwargs:                       # ...
        n_neighbors: 5
    name: KNN-5
    num_runs: 10
-   algorithm: SVM                # (estimator 2) Algorithm to be used
    kwargs:                       # ...
        C: 1.0
        kernel: rbf
    name: SVM-rbf-C1.0
    num_runs: 10

extra:                            # Extra information to be configure the 
                                  # experiment execution
    in_use_features:              # List of features to be used.
    - accel-x
    - accel-y
    - accel-z
    - gyro-x
    - gyro-y
    - gyro-z
    reduce_on: sensor             # How dimensionality reduction algorithms
                                  # will be applied. Valid values are:
                                  # - all: apply the reducer to the whole
                                  # dataset
                                  # - sensor: apply the reducer to each sensor
                                  # separately, then concatenate the arrays
                                  # - axis: apply the reducer to each axis
                                  # of each sensor separately, 
                                  # then concatenate the arrays
    scale_on: train               # How the scaler will be applied. Valid
                                  # values are:
                                  # - train: fit the scaler on the train
                                  # dataset and transform the train and
                                  # test datasets
                                  # - self: fit and transform on train
                                  # and test datasets separately

reducer_dataset:                  # List of datasets to be used to fit the
                                  # reducer algorithm (it can be null). 
                                  # The name of the datasets must be defined 
                                  # in the `dataset_locations.yaml` file. 
                                  # All datasets follows the format:
                                  # <dataset_name>.<view>[<split>] where:
                                  # - dataset_name: name of the dataset
                                  # - view: view of the dataset
                                  # - split: split of the dataset (train, 
                                  # validation or test).
                                  # If multiple datasets are defined, they
                                  # will be concatenated, before applying
                                  # the reducer algorithm.
- motionsense.standartized_intra_balanced[train]
- motionsense.standartized_intra_balanced[validation]
test_dataset:                     # List of datasets that will be used to
                                  # test the estimators. Follows the same
                                  # format as the reducer_dataset.
- motionsense.standartized_intra_balanced[test]
train_dataset:                    # List of datasets that will be used to
                                  # train the estimators. Follows the same
                                  # format as the reducer_dataset.
- motionsense.standartized_intra_balanced[train]
- motionsense.standartized_intra_balanced[validation]

reducer:                          # Information about the reducer algorithm
                                  # (it can be null).
    algorithm: umap               # Algorithm to be used. Valid algorithms
                                  # can be found in the config.py file, in
                                  # the `reducer_cls` dictionary.
    kwargs:                       # Algorithm creation parameters 
                                  # (it can be null)
        n_components: 5           # Number of components to be reduced
    name: umap-5                  # Symbolic name of the reducer

scaler:                           # Information about the scaler algorithm
                                  # (it can be null).
    algorithm: StandardScaler     # Algorithm to be used. Valid algorithms
                                  # can be found in the config.py file, in
                                  # the `scaler_cls` dictionary.
    kwargs: null                  # Algorithm creation parameters
                                  # (it can be null)
    name: StandardScaler          # Symbolic name of the scaler

transforms:                       # List of transforms to be applied (in order)
    transform: fft                # (transform 0) Transform to be applied. Valid
                                  # transforms can be found in the config.py
                                  # file, in the `transform_cls` dictionary.
-   kwargs:                       # (transform 0) Algorithm creation parameters
        centered: true            # (transform 0) If the FFT should be centered
    name: FFT-centered            # (transform 0) Symbolic name of the transform
    windowed: null                # Windowed transform controls.
                                  # It may be null (equals to fit_on=null
                                  # transform_on=window)
                                  # or a dictionary with the two keys:
                                  # - fit_on: null (do not do fit) or 
                                  #     all (fit on the whole dataset)
                                  # - transform_on: null (do not do transform) or
                                  #     all (transform on the whole dataset) or
                                  #     window (apply the transform to each window)
version: '1.0'                    # Version of the configuration file 
                                  # (it must be a string).

```

To work, users must first download the datasets and extract them in a folder as they wish. The valid dataset names and relative path are defined an external YAML file (`dataset_locations.yaml`), where the key is the dataset name and view (used in the datasets sections in the YAML file) and the value is the path to the dataset, relative to the `--data-root` argument. 

It is assumed that all datasets will have the `train.csv`, `validation.csv`, and `test.csv` files. Besides that, the datasets must have `accel-x`, `accel-y`, `accel-z`, `gyro-x`, `gyro-y`, and `gyro-z` columns. 

More examples can be found in the `examples` directory and the respective results in `examples/results/execution` directory. They can be executed (parallel) with the following command:

```bash
python execute.py examples/experiment_configurations/ -o examples/results/ -d data/processed/ --ray --skip-existing
```

The `-d` option is used to specify the path to the datasets and should point to the dataset root directory (where have `raw_balanced`, `standartized_balanced` datasets). The `--ray` option is used to execute the experiments in parallel using Ray.

## How to alter the execution flow and add new options

You may want to modify the execution of the script to add more options or change the execution flow by rewriting some parts of the `execute.py` script, in special, the `run_experiment` function that runs an experiment based on a configuration file.

The valid values for configuration files are defined in the `config.py` file, in the `ExecutionConfig` class. This is a Python's dataclass that models the YAML dictionary. YAML configuration files are loaded (and populated) into objects of `ExecutionConfig` before executing `run_experiment`. You may want to add more options to the configuration files, by editing this class.


## Running experiments in a distributed environment

We use Ray to run experiments in a distributed environment. Each machine will be a worker and will receive a configuration to execute the `run_experiment` function in parallel, with all available cores. The workers will be connected to a head node, which will be responsible for distributing the work.

We expect all workers have the same configuration files and datasets, all at same location (this is easier using Docker). Also, we expect all workers have the same Python environment, with all dependencies installed. Finally, all workers must be able to communicate with the head node.

### Starting the head node

First, log into a node and start the head node, using the following command:


```bash
ray start --head --block --include-dashboard true
```

This will start the head node and block the terminal. The `--block` parameter is optional, and it allows to stop it using SIGINT (i.e., control+C). You can access the dashboard by opening the URL printed in the terminal (usually `http://localhost:8265`).

The head node address is printed in the terminal. It will be used to connect the workers. It is something like:

```bash

Next steps
  To connect to this Ray runtime from another node, run
    ray start --address='192.168.1.112:6379'

...
```

In this case the head node operates at address: `192.168.1.112` and port: `6379`.


### Starting the workers

Log into the other nodes and start the workers, using the following command:

```bash
ray start --address 192.168.1.112:6379 --block
```

The `--address` parameter must be the address of the head node. The `--block` parameter is optional, and it allows to stop it using SIGINT (i.e., control+C).

### Submitting the experiments

Now, you can submit the experiments to the cluster, passing the `--ray` and `--address` parameters to the `execute.py` script. The `--address` parameter must be the address of the head node. The `--skip-existing` parameter is optional, and it allows to skip experiments that already have results. The `--data-root` parameter must be the path to the datasets root directory.:

```bash
python execute.py examples/experiment_configurations/ -o examples/results/ -d data/processed/ --ray --address 192.168.1.112:6379 --skip-existing
```

The experiments will be distributed among the workers. You can monitor the execution in the dashboard (usually `http://localhost:8265`).