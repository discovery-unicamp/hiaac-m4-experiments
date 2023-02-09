# Parallel experiment executor

This is an experiment executor that allows running different experiment configurations by using execution config files.

## Installation

### Using pip

First check the latest version of [`librep`](https://github.com/otavioon/hiaac-librep). This can be done by using:

```bash
pip install git+https://github.com/otavioon/hiaac-librep.git
```

After that, you must install the dependencies. To do this, run:

```
pip install -r requirements.txt
```

### Using docker image

TO BE DONE

## Execution

The execution consists in two steps

1. Writing configuration files in [YAML](https://yaml.org/) format
2. Executing the configuration files in a local machine or distributed

### About the configuration files

Each YAML configuration file consists in one execution. The executor script reads a folder with several configuration files, where each one will become an experiment. The name of the configuration file it is the experiment id.

The folder `executions_config2` contains a lot of experiments to be executed. The YAML structure follows the following format:

```yaml
estimator:                      # Information about the classification algorithm
  algorithm: RandomForest       # Name of the algorithm. Valid algorithm names is under estimator_cls in file config.py
  allow_multirun: true          # If the algorithm is non-deterministic (must be run many times)
  kwargs:                       # Parameters for algorithm creation
    n_estimators: 100
  name: randomforest-100        # Symbolic estimator name
execution_id: '0'               # Symbolic execution id (matches file name). It will be removed
extra:                          # Extra options for execution
  in_use_features:              # Features to be used
  - accel-x
  - accel-y
  - accel-z
  - gyro-x
  - gyro-y
  - gyro-z
  reduce_on: all                # It can be: "all": if the reducer algorithm will be applyied over all features
                                # "sensor": if the reducer will be applyed one per sensor (will have multiple reducers)
                                # "axis": if the reducer will be applyied one per axis of sensor (will have multiple reducers) 
number_runs: 5                  # Number of times the estimator will run (fit and predict)
reducer:                        # Especifies the reducer algorithm
  algorithm: umap               # Name of the reducer. Valid values names is under reducers_cls in the file config.py
  kwargs:                       # Parameters for algorithm's creation
    n_components: 2
  name: umap-2-all              # Symbolic reducer name
  windowed:                     # Windowed transform controls (not used)
    fit_on: null
    transform_on: all
reducer_dataset:                # List of datasets used in reducer algorithm (in order). The datasets will be merged into a single dataset
- kuhar
test_dataset:                   # List of datasets used to test (in order). The datasets will be merged into a single dataset
- kuhar
train_dataset:                  # List of datasets used to train (in order). The datasets will be merged into a single dataset
- kuhar
transforms:                     # List of non-parametric transforms to be applyied in order)
- kwargs:                       # Transform kwargs
    centered: true                    
  name: fft_transform.0         # Symbolic transforrm name
  transform: fft                # Transformer algorithm. Valid names is under transforms_cls in config.py file
  windowed: {}

```

Once a configuration file is written, it must be saved inside a directory (where user wishes) in order to be fetched and executed 


### Executing the configuration files

In order to work, users must first download the [Megazord Dataset](https://drive.google.com/file/d/1h6CD9B8Tx3XXXLlsGaawxj_0c4eSCEPb/view?usp=share_link) and extract in a folder as they wish.

Supposing the data where extracted to `data/balanced_20Hz_filtered` and the configuration files are saved to `configurations`/, the user may launch the executor, by using the following command:

```
python execute.py configurations/ -d "data/balanced_20Hz_filtered" -o "./results" --skip-existing 
```

Where the `results` is the directory where results will be stored and the flag `--skip-existing` will skip configurations that have already run (that is, one the has a result produced in the results folder).

Finally, for the experiments already run, some information can be viewed in the `executions_config2_results`


## Altering the execution

You may want to modify the execution of the script by: adding more options, rewriting the runner, or something else. The root script is the `execute.py` script, in special, the `_run` function.

The `_run` function is called by `run` which, in turn, is parallelized. The `run` function is called for each configuration file in the folder (which is received as parameter) and it is executed distributively.

The `_run` function (that you may want to modify) receives:
- `root_data_dir`: a string with the root path to the dataset
- `output_dir`: a string with the output results directory
- `experiment_name`: the id of the experiment
- `config`: The `ExecutionConfig` associated. The `ExecutionConfig` it exactly equals the configuration YAML, except that instead of a dictionary, it is a python Dataclass. Every change in the dataclass, you may change the YAML files. The dataclass can be found in `config.py` file.


The `_run` method must execute based on the `ExecutionConfig` which is the YAML file. You may want to alter this function and do whatever you want here.
For now, this function behaves the following way:

1. Load the train datasets (into `train_dset` variable), test datasets (into `test_dset` variable) and the reducer datasets (into `reducer_dset` variable).
2. Apply the list of non-parametric transforms (like FFT), based on the list of `transforms` listed in the configuration file. The `do_transform` function receives the `train_dset`, the `test_dset` and the list of transforms to be applied.
3. Apply the reducer, based on the `reducer` algorithm listed in the configuration file. It will use the `do_reduce` function to apply the reduction. This functions receives the `reducer_dset` where reducer.fit will be applyied. The `train` and `test` datasets, which will be transformed, the reducer config (to instantiate the object) and the `reduce_on` parameter, telling where the reducer will be applyied ('all', 'sensor' or 'axis')
4. Execute the train of the algorithm in the train dataset
5. Evaluate the algorithm over the test dataset
6. Store the results and additional information under the results directory, as a YAML file.
7. Return the results

The `additional_info` variable saves additional information to be stored in the experiment results (like, timings, sizes, etc.) 