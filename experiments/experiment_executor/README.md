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

Each YAML configuration file represents one experiment and has all information to execute it (such as the datasets to be used, the transforms to be applied, and the classification algorithms). The executor script (`execute.py`) reads a folder with several experiment configuration files and executes each one sequentially or in parallel. Usually, the name of the configuration file is also the experiment id (in the YAML file).

The `execute.py` script will perform the following steps:

1. Load the datasets
2. Apply the non-parametric transforms
3. Apply the reducer algorithm
4. Apply the scaler algorithm
5. Apply the estimator algorithm
6. Save the results

The configuration file controls the behavior of execution and has the following structure:


```yaml
reducer_dataset:                                
- motionsense.standartized_balanced[train]  # List of datasets used in reducer algorithm (in order). 
                                            # The datasets will be merged into a single dataset. 
                                            # The dataset name must be in the format 
                                            # <dataset_name>.<dataset_view>[<dataset_split>] 
                                            # where:
                                            # - dataset_name: name of the dataset
                                            # - dataset_view: view of the dataset
                                            # - dataset_split: split of the dataset
                                            #    (train, validation, or test)
                                            # Valid dataset values are found in the file
                                            # dataset_locations.yaml
- motionsense.standartized_balanced[validation]
test_dataset:                               # List of datasets used in the test (in order). 
                                            # The datasets will be merged into a single dataset. 
                                            # The dataset name must be in the format 
                                            # <dataset_name>.<dataset_view>[<dataset_split>] 
                                            # Valid dataset values are found in the file
                                            # dataset_locations.yaml
- kuhar.standartized_balanced[test]
train_dataset:                              # List of datasets used in train (in order).                                     
                                            # The datasets will be merged into a single dataset. 
                                            # The dataset name must be in the format 
                                            # <dataset_name>.<dataset_view>[<dataset_split>] 
                                            # Valid dataset values are found in the file
                                            # dataset_locations.yaml
- kuhar.standartized_balanced[train]
- kuhar.standartized_balanced[validation]

transforms:                         # List of non-parametric transforms to be applied in order)
- kwargs:                           # Parameters for transform creation 
                                    # (can be null or a dictionary)
    centered: true                      
  name: fft_transform.0             # Symbolic transform name
  transform: fft                    # Name of the transform. 
                                    # Valid transform names are under transforms_cls in file config.py
  windowed: null                    # Windowed transform controls.
                                    # It may be null (equals to fit_on=null, transform_on=window)
                                    # or a dictionary with the two keys:
                                    # - fit_on: null (do not do fit) or 
                                    #     all (fit on the whole dataset)
                                    # - transform_on: null (do not do transform) or
                                    #     all (transform on the whole dataset) or
                                    #     window (apply the transform to each window)

reducer:                            # Especifies the reducer algorithm
  algorithm: umap                   # Name of the reducer. Valid values names are under 
                                    # reducers_cls in the file config.py
  kwargs:                           # Parameters for algorithm's creation
    n_components: 25
  name: umap-25-all                 # Symbolic reducer name

scaler:                             # Especifies the scaler algorithm
  algorithm: std_scaler             # Name of the scaler. Valid values names are 
                                    # under scalers_cls in the file config.py
  kwargs: null                      # Parameters for algorithm's creation
                                    # (can be null or a dictionary)
  name: StandardScalerUse           # Symbolic scaler name

estimator:                          # Information about the classification algorithm (for step 5)
  algorithm: RandomForest           # Name of the algorithm. Valid algorithm names are under 
                                    # estimator_cls in file config.py
  kwargs:                           # Parameters for algorithm creation
    n_estimators: 100               # Number of trees
  name: randomforest-100            # Symbolic estimator name

extra:                              # Extra options for execution
  in_use_features:                  # List of features to be used for loading datasets.
                                    # The dataframe columns will be filtred with columns
                                    # starts with any of the prefixes in this list 
  - accel-x
  - accel-y
  - accel-z
  - gyro-x
  - gyro-y
  - gyro-z
  reduce_on: all                    # It can be: "all": if the reducer algorithm will be
                                    # applied over all features
                                    # "sensor": if the reducer will be applied one per 
                                    # sensor (will have multiple reducers)
                                    # "axis": if the reducer will be applied one per 
                                    # axis of the sensor (will have multiple reducers)
  scale_on: train                   # It can be: "train" or "self". "train" means that
                                    # the scaler will fit on the training dataset and
                                    # then applied to the train and test datasets. 
                                    # "self" means that the scaler will be fit and 
                                    # applied to each dataset (train, test).
  estimator_runs: 5                 # Number of times the estimator will run (fit and predict)
  estimator_deterministic: false    # If the algorithm is deterministic (if true, the 
                                    # estimator_runs will be ignored, and the estimator will
                                    # be fit and predict only once)
```

To work, users must first download the datasets and extract them in a folder as they wish. The valid dataset names are defined an external YAML file (`dataset_locations.yaml`), where the key is the dataset name and view (used in the datasets sections in the YAML file) and the value is the path to the dataset, relative to the `--data-root` argument. 

It is assumed that all datasets will have the `train.csv`, `validation.csv`, and `test.csv` files. Besides that, the datasets must have `accel-x`, `accel-y`, `accel-z`, `gyro-x`, `gyro-y`, and `gyro-z` columns. 

More examples can be found in the `examples` directory. They can be executed (parallel) with the following command:

```bash
python execute.py examples/experiment_configurations/ -o examples/results/ -d data/processed/ --ray --skip-existing
```

**NOTE**: The `-d` option is used to specify the path to the datasets and should point to the dataset root directory.

## How to alter the execution flow and add new options

You may want to modify the execution of the script to add more options or change the execution flow by rewriting some parts of the `execute.py` script, in special, the `run_experiment` function that runs an experiment based on a configuration file.

The valid values for configuration files are defined in the `config.py` file, in the `ExecutionConfig` class. This is a Python's dataclass that models the YAML dictionary. YAML configuration files are loaded (and populated) into objects of `ExecutionConfig` before executing `run_experiment`. You may want to add more options to the configuration files, by editing this class.


## Running experiments in a distributed environment

TO BE DONE