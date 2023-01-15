import numpy as np
import pandas as pd
from umap import UMAP

import time
import warnings
warnings.filterwarnings('ignore')

# Librep imports
from librep.utils.dataset import PandasDatasetsIO          # For quick load train, test and validation CSVs
from librep.datasets.har.loaders import (
    MegaHARDataset_BalancedView20Hz
)
from librep.datasets.multimodal import PandasMultiModalDataset, TransformMultiModalDataset, WindowedTransform
from librep.transforms.fft import FFT
from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
from librep.metrics.report import ClassificationReport

Root = "../../../.."

dimensions_umap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 150, 180]

labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

datasets = ['KuHar', 'RealWorld', 'MotionSense', 'WISDM', 'UCI']

classes = list(labels_activity.keys())

labels_dataset = {
    'KuHar': 'KuHar',
    'RealWorld': 'RealWorld',
    'MotionSense': 'MotionSense',
    'ExtraSensory': 'ExtraSensory',
    'WISDM': 'WISDM',
    'UCI': 'UCI',
}

# Load all datasets, creating PandasMultiModalDatasets with the correct pre-defined windows
loader = MegaHARDataset_BalancedView20Hz(
    Root+"/data/views/AllDatasets/balanced_20Hz_filtered",
#     Root+"/data/views/KuHar/balanced_20Hz_motionsense_equivalent",
    download=False)
train_data, test_data = loader.load(concat_train_validation=True, label="standard activity code")
train_data, test_data

columns = ['Classifier', 'Umap dimension', 'Dataset']

metrics = ['accuracy', 'f1 score (weighted)']
stats = ['mean', 'std']
columns += [metric + ' - ' + stat
            for metric in metrics
            for stat in stats]

metrics_class = ['f1-score', 'precision', 'recall', 'support']
columns += [
    metric + ' - ' + stat + ' - ' + activity
    for metric in metrics_class
    for stat in stats
    for activity in labels_activity.values()
]

columns, len(columns)
df_results = {column: [] for column in columns}

results_dict = {
    'RandomForest': {},
    'SVC': {},
    'KNN': {}
}
for classifier in results_dict.keys():
    results_dict[classifier] = {
        'Umap dimension': [],
        'Dataset': [],
        'result': []
    }

def create_data_multimodal(data):
    # Features to select
    features = [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z"
    ]

    # Creating the datasets

    # Data
    data_multimodal = PandasMultiModalDataset(
        data,
        feature_prefixes=features,
        label_columns="standard activity code",
        as_array=True
    )

    return data_multimodal

def evaluate(dimension, dataset, train_fft, test_fft, evaluators, df, results_dict, labels_activity, metrics_class,
             reporter):
# The reporter will be the same

    if dimension != 180:
        umap = UMAP(n_components=dimension, random_state=42)
        umap.fit(train_fft[:][0])

        umap_transform = WindowedTransform(
            transform=umap, fit_on=None, transform_on="all"
        )
        transformer = TransformMultiModalDataset(transforms=[umap_transform], new_window_name_prefix="reduced.")
        train_fft = transformer(train_fft)
        test_fft = transformer(test_fft)

    for estimator, evaluator in evaluators.items():
        multi_run_experiment = MultiRunWorkflow(
            workflow=evaluator['experiment'],
            num_runs=evaluator['num_runs'],
            debug=False)

        results = multi_run_experiment(train_fft, test_fft)
        results_dict[estimator]['Umap dimension'].append(dimension)
        results_dict[estimator]['Dataset'].append(dataset)
        results_dict[estimator]['result'].append(results)

        df['Classifier'].append(estimator)
        df['Umap dimension'].append(dimension)
        df['Dataset'].append(dataset)

        df['accuracy - mean'].append(
            np.average(
                [res["result"][0]["accuracy"] for res in results["runs"]]
            )
        )
        df['accuracy - std'].append(
            np.std(
                [res["result"][0]["accuracy"] for res in results["runs"]]
            )
        )
        df['f1 score (weighted) - mean'].append(
            np.average(
                [res["result"][0]["f1 score (weighted)"] for res in results["runs"]]
            )
        )
        df['f1 score (weighted) - std'].append(
            np.std(
                [res["result"][0]["f1 score (weighted)"] for res in results["runs"]]
            )
        )

        labels = test.data['standard activity code'].unique()
        for metric in metrics_class:
            for index, activity in labels_activity.items():
                df[f'{metric} - mean - {activity}'].append(
                    np.average(
                        [res['result'][0]['classification report'][str(index)][metric] for res in results["runs"]]
                    )
                ) if index in labels else  df[f'{metric} - mean - {activity}'].append(np.nan)

                df[f'{metric} - std - {activity}'].append(
                    np.std(
                        [res['result'][0]['classification report'][str(index)][metric] for res in results["runs"]]
                    )
                ) if index in labels else  df[f'{metric} - std - {activity}'].append(np.nan)
    return df, results_dict

reporter = ClassificationReport(
    use_accuracy=True,
    use_f1_score=True,
    use_classification_report=True,
    use_confusion_matrix=True,
    plot_confusion_matrix=False,
#     normalize='true',
#     display_labels=labels,
)

evaluators = {
    'RandomForest':
    {
        'experiment':
        SimpleTrainEvalWorkflow(
            estimator=RandomForestClassifier,
            estimator_creation_kwags ={'n_estimators':100},
            do_not_instantiate=False,
            do_fit=True,
            evaluator=reporter),
        'num_runs':
        10

    },
    'SVC':
    {
        'experiment':
        SimpleTrainEvalWorkflow(
            estimator=SVC,
            estimator_creation_kwags ={'C':3.0, 'kernel':"rbf"} ,
            do_not_instantiate=False,
            do_fit=True,
            evaluator=reporter),
        'num_runs':
        1
    },
    'KNN':
    {
        'experiment':
        SimpleTrainEvalWorkflow(
            estimator=KNeighborsClassifier,
            estimator_creation_kwags={'n_neighbors' :1},
            do_not_instantiate=False,
            do_fit=True,
            evaluator=reporter),
        'num_runs':
        1
    }
}
start = time.time()

fft_transform = FFT(centered=True)
transformer_fft = TransformMultiModalDataset(transforms=[fft_transform], new_window_name_prefix="fft.")

train_data.data['standard activity code'] = train_data.data['standard activity code'].astype('int')
test_data.data['standard activity code'] = test_data.data['standard activity code'].astype('int')

k=1
for dataset in datasets:

    train = train_data.data[train_data.data['DataSet'].isin([dataset])]
    train = create_data_multimodal(train)

    test = test_data.data[test_data.data['DataSet'].isin([dataset])]
    test = create_data_multimodal(test)

    train_fft = transformer_fft(train)
    test_fft = transformer_fft(test)

    new_start = time.time()
    for dimension in dimensions_umap:
        df_results, results_dict = evaluate(dimension, dataset, train_fft, test_fft, evaluators, df_results,
                                            results_dict, labels_activity, metrics_class, reporter)
        new_end = time.time()
        print(f'Iteration: {k} \t Time of execution: {int(new_end - new_start) // 60} minutes and {int(new_end - new_start) % 60} seconds')
        k+=1

end = time.time()
total = int(end - start)
print(f'Time of execution: {total} seconds')
print(f'Time of execution: {total // 60} minutes and {total % 60} seconds')
print(f'Time of execution: {(total // 3600) % 24} hours, {(total // 60) % 60} minutes and {total % 60} seconds')

df_results = pd.DataFrame(df_results)

# Save results
import json

with open('results/results_df_umap_dimension_frequency.json', 'w') as file:
    json.dump(df_results.to_dict(), file)

with open('results/results_dict_umap_dimension_frequency.json', 'w') as file:
    json.dump(results_dict, file)