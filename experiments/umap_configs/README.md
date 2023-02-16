# Experiment executor

This is an experiment executor that allows running different experiment configurations by using execution config files.

## Installation

### Using pip

You may install the dependencies using the `requirements.txt` file.

```
pip install -r requirements.txt
```

### Using docker image

TO BE DONE

## Execution

The execution consists in two steps:

1. Writing configuration files in [YAML](https://yaml.org/). Each configuration file corresponds to a single experiment. They are put into a user-defined directory.

2. Execute experiment configuration files in a local machine or distributed.

Once the experiments are written, the easiest way to execute the script is using:

```
python execute.py <experiments_dir> --data-path <path_to_data_root> --exp-name my-experiment-1 --skip-existing 
```

Where the `experiments_dir` is the path where configuration files are stored and the `path_to_data_root` is the path to the root of the datasets. The `--skip-existing` option will skip the execution of the experiment if the results already exists. Finally, the `--exp-name` is the symbolic name of the execution of the experiment.

The script will execute each configuration file sequentially or parallel, if using `--ray` option(it also allows distributed execution in ray clusters). The results will be stored in the `results` folder. More options and information about the execution can be found executing `python execute.py --help`.


## Experiment configuration files

Each YAML configuration file represent one experiment, and has all information in order to execute it (such as, the datasets to be used, the transforms to be applied, and the classification algorithms). The executor script (`execute.py`) reads a folder with several experiment configuration files and execute each one sequentially or parallel. Usually, the name of the configuration file it is also the experiment id.

The `execute.py` script will perform the following steps:

1. Load the datasets
2. Apply the non-parametric transforms
3. Apply the reducer algorithm
4. Apply the scaler algorithm
5. Apply the estimator algorithm
6. Save the results

The configuration file controls the behavior of execution and have the following structure:


```yaml
execution_id: '000000'            # Symbolic execution id (it is optional to match file name)
number_runs: 5                    # Number of times the estimator will run (fit and predict)


reducer_dataset:                                
- motionsense.standartized_balanced[train]   # List of datasets used in reducer algorithm (in order). The datasets will be merged into a single dataset. The dataset name must be in the format <dataset_name>.<dataset_version>[<dataset_split>] where:
                                            # - dataset_name: name of the dataset (must be in the datasets folder)
                                            # - dataset_version: version of the dataset (must be in the datasets folder)
                                            # - dataset_split: split of the dataset (must be in the datasets folder)
                                            # The dataset must be in the format <dataset_name>.<dataset_version>.<dataset_split>.csv
                                            # The dataset must have the following columns: "sensor", "axis", "timestamp", "value"
                                            # The dataset must be standartized (mean=0, std=1)
                                            # The dataset must be balanced (same number of samples per class)
- motionsense.standartized_balanced[validation]
test_dataset:
- kuhar.standartized_balanced[test]
train_dataset:
- kuhar.standartized_balanced[train]
- kuhar.standartized_balanced[validation]

transforms:                         # List of non-parametric transforms to be applyied in order)
- kwargs: null                      # Parameters for transform creation
  name: fft_transform.0             # Symbolic transform name
  transform: fft                    # Name of the transform. Valid transform names is under transforms_cls in file config.py
  windowed: null                    # Windowed transform controls

reducer:                            # Especifies the reducer algorithm
  algorithm: umap                   # Name of the reducer. Valid values names is under reducers_cls in the file config.py
  kwargs:                           # Parameters for algorithm's creation
    n_components: 25
  name: umap-25-all                 # Symbolic reducer name

scaler:                             # Especifies the scaler algorithm
  algorithm: identity               # Name of the scaler. Valid values names is under scalers_cls in the file config.py
  kwargs: null                      # Parameters for algorithm's creation
  name: no_scaler                   # Symbolic scaler name

estimator:                          # Information about the classification algorithm (for step 5)
  algorithm: RandomForest           # Name of the algorithm. Valid algorithm names is under estimator_cls in file config.py
  allow_multirun: true              # If the algorithm is non-deterministic (must be run many times)
  kwargs:                           # Parameters for algorithm creation
    n_estimators: 100               # Number of trees
  name: randomforest-100            # Symbolic estimator name

extra:                              # Extra options for execution
  in_use_features:                  # List of features to be used fro loading datasets. The dataframe columns will be filtred with columns that have the prefix in this list 
  - accel-x
  - accel-y
  - accel-z
  - gyro-x
  - gyro-y
  - gyro-z
  reduce_on: all                    # It can be: "all": if the reducer algorithm will be applyied over all features
                                    # "sensor": if the reducer will be applyed one per sensor (will have multiple reducers)
                                    # "axis": if the reducer will be applyied one per axis of sensor (will have multiple reducers)
  scale_on: train                   # It can be: "train" or "self". "train" means that the scaler will be fit on the train dataset and then applied on the train and test datasets. "self" means that the scaler will be fit and applyyed to each dataset (train, test).
```

In order to work, users must first download the datasets and extract in a folder as you wish.

More examples can be found in the `examples` directory.

## Altering the execution

You may want to modify the execution of the script to add more options or change the execution flow by rewriting some parts of the `execute.py` script, in special, the `run_experiment` function that runs an experiment based on a configuration file.

The valid values for configuration files are defined in the `config.py` file, in the `ExecutionConfig` class. This is a dataclass that models the YAML dictionary. YAML configuration files are loaded (and populated) into objects of `ExecutionConfig` before executing `run_experiment`. You may want to add more options to the configuration files, editing this class.


## Running experiments in a distributed environment

TO BE DONE