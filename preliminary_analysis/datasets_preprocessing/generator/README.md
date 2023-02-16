# Dataset Views Generator

This folder contains the code to generate the dataset views from the original datasets.

## Install Dependencies

You need to install the dependencies listed in the `requirements.txt` file. You can do it by running the following command:

```bash
pip install -r requirements.txt
```

You also must have jupyter notebook installed to render notebooks that are used to generate the dataset views.

## Datasets Directory Structure

From the root's git repository path, the datasets must be placed in the `preliminary_analisys/datasets_preprocessing/data` directory.
The datasets must have the following structure:

```
preliminary_analisys
└── datasets_preprocessing
    └── data
        ├── original
        │   ├── KuHar
        │   ├── MotionSense
        │   ├── RealWord
        │   ├── UCI
        │   └── WISDM
        └── raw_balanced
        │   ├── KuHar
        │   ├── MotionSense
        │   ├── RealWord
        │   ├── UCI
        │   └── WISDM
        └── standartized_balanced
        │   ├── KuHar
        │   ├── MotionSense
        │   ├── RealWord
        │   ├── UCI
        │   └── WISDM
        └── unbalanced
            ├── KuHar
            ├── MotionSense
            ├── RealWord
            ├── UCI
            └── WISDM
```

Where the `original` directory contains the original datasets (as they are downloaded), the `raw_balanced` will be generated by the notebooks and the directory contains the raw balanced datasets, that is, without standardization. The `standartized_balanced` directory contains the standardized balanced datasets and the `unbalanced` directory contains the unbalanced datasets, that is, standardized datasets, but before train/validation/test splits.

---

**NOTE**

- To generate datasets, the original data must be placed in the `preliminary_analisys/datasets_preprocessing/data/original` directory.
- The `raw_balanced` and `standartized_balanced` directories will be generated by the notebooks.
- Do not add the datasets to the git repository. The datasets are too big, and it is not necessary to add them to the repository. The datasets can be downloaded from the sources or downloaded from [HIAAC M4 Datasets Drive](https://drive.google.com/drive/u/1/folders/1NF63hQu1hCpVU2GlxjLuE5EcxYm9i0EP).
- The `preliminary_analisys/datasets_preprocessing/data` directory is ignored by git.

---

## Dataset Generation Process

To generate the datasets, you can run the notebooks, which are in this directory. Notebooks whose file name starts with `1` must be run first, then the notebooks whose file name starts with `2`, and so on. Notebooks with the same initial number can be run in parallel, in any order.
The `1` notebooks generate the `raw balanced` datasets, `standardized balanced` datasets, and `unbalanced` datasets. The notebooks will generate the datasets in the `preliminary_analisys/datasets_preprocessing/data` directory.


## Standardization Process

The standardization process also called the standardization pipeline, comprises the execution of several steps (operations) per dataset.
Each dataset has its standardization pipeline, as they are different from each other. 
The operators are all defined in the `dataset_processor.py` file. The operators are defined as classes, and each class has a `__call__` method, which receives a pandas dataframe and returns a pandas dataframe. The `__call__` method is the operator's implementation.

---

**NOTE**

- The order of the operators is important, as some operators may require columns that may be added from other operators (which must run before).
- Seldom, some operators may return multiple pandas Dataframes. 

---

The standardization codes from notebooks usually comprise the following steps (**this is not a strict rule**):

1. Load the datasets and generate a single pandas dataframe with all the data, where each row represents a single instant of capture (and now a window). The loading is a dataset-specific process. The dataframe generated **usually** have the following columns (**this is not a rule**):
- A column for the x-axis acceleration (`accel-x` column); y-axis acceleration (`accel-y` column); z-axis acceleration (`accel-z` column); gyroscope x-axis (`gyro-x` column); gyroscope y-axis (`gyro-y` column); gyroscope z-axis (`gyro-z` column); and the timestamp from the accelerometer (`accel-timestamp`) and gyroscope (`gyro-timestamp`), if provided.
- A column for the label (`activity code` column).
- A column for the user id (`user` column), if provided.
- A column for x-axis gravity (`gravity-x` column); y-axis gravity (`gravity-y` column); and z-axis gravity (`gravity-z` column) if provided.
- A serial column, which represents the attempt that the collection was made (`serial` column), if provided. For instance, if the user has a time series running in the morning and another in the afternoon, it will be two different serial numbers.
- A CSV or file column, which represents the file that the row was extracted from (`csv` column). 
- An index column, that is, a column that represents the row index from the CSV file (`index` column).
- Any other column that may be useful for the standardization process or metadata.
2. Create the operator objects.
3. Crete the pipeline object, passing the operator object list as parameters.
4. Execute the pipeline, passing the dataframe as a parameter.

The following code snippet illustrates a fictitious standardization process for the `KuHar` dataset that resamples it to 20Hz and creates 3-second windows (**using this kind of code and operators is not a rule**):

```python

def read_kuhar(kuhar_dir_path: str) -> pd.DataFrame:
    # This is dataset-specific code. It reads the CSV files and generates a single dataframe with all the data.
    ... 
    return dataframe

# -----------------------------------------------------------------------------
# 1. Load the datasets and generate a single pandas dataframe with all the data
# -----------------------------------------------------------------------------

dataframe = read_kuhar("../data/original/KuHar/1.Raw_time_domian_data")

# -----------------------------------------------------------------------------
# 2. Create operaators
# -----------------------------------------------------------------------------

# Lista com as colunas que são features
feature_columns = [
    "accel-x",
    "accel-y",
    "accel-z",
    "gyro-x",
    "gyro-y",
    "gyro-z",
]


# Instacia o objeto que reamostra os dados para 20Hz 
# (supondo que o dataset original é 100Hz, constante)
resampler = ResamplerPoly(
    features_to_select=feature_columns, # Nome das colunas que serão usadas 
                                        # como features
    up=2,                               # O fator de upsampling.
    down=10,                            # O fator de downsampling.
    groupby_column="csv",               # Agrupa pela coluna csv. 
                                        # A reamostragem é feita para cada 
                                        # grupo da coluna csv
)

# Instancia o objeto que cria as janelas
windowizer = Windowize(
    features_to_select=feature_columns, # Nome das colunas que serão usadas 
                                        # como features
    samples_per_window=60,              # Numero de amostras por janela
    samples_per_overlap=0,              # Numero de amostras que se sobrepõem
    groupby_column="csv",               # Agrupa pela coluna csv. 
                                        # A reamostragem é feita para cada 
                                        # grupo da coluna csv
)

# -----------------------------------------------------------------------------
# 3. Create the pipeline object, passing the operator object list as parameters
# -----------------------------------------------------------------------------

# Cria o pipeline
# 1. Reamostra os dados
# 2. Cria as janelas
# 3. Adiciona a coluna com o código da atividade
pipeline = Pipeline(
    [
        differ,
        resampler,
        windowizer,
        standard_label_adder
    ]
)

# -----------------------------------------------------------------------------
# 4. Execute the pipeline, passing the dataframe as a parameter
# -----------------------------------------------------------------------------

dataset_padronizado = pipeline(dataframe)
```

The pipeline operators are usually shared with other datasets, as they are generic.
