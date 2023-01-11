#!/usr/bin/env python
# coding: utf-8

# # Plot datasets
# 
# This notebook will visualize the data from diferent datasets (KuHar, MotionSense, and UCI-HAR). The data is without gravity and it was removed with a high-pass filter.
# 
# 1. Apply DFT over dataset windows
# 3. Plot UMAP and T-SNE


from pathlib import Path  # For defining dataset Paths
import sys
Root = "../../../.."
sys.path.append("../../../..")

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
#from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import plotly.express as px
import plotly.graph_objects as go
import itertools
from itertools import combinations
import pickle

# Librep imports
from librep.utils.dataset import PandasDatasetsIO          # For quick load train, test and validation CSVs
from librep.datasets.har.loaders import (
    KuHar_BalancedView20HzMotionSenseEquivalent,
    MotionSense_BalancedView20HZ,
    ExtraSensorySense_UnbalancedView20HZ,
    CHARM_BalancedView20Hz,
    WISDM_UnbalancedView20Hz,
    UCIHAR_UnbalancedView20Hz
)
from librep.datasets.multimodal import PandasMultiModalDataset, TransformMultiModalDataset, WindowedTransform
from librep.transforms.fft import FFT
from librep.transforms. stats import StatsTransform
from librep.utils.workflow import SimpleTrainEvalWorkflow, MultiRunWorkflow
from librep.estimators import RandomForestClassifier, SVC, KNeighborsClassifier
from librep.metrics.report import ClassificationReport

# # Datasets to train the manifold

labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

classes = list(labels_activity.keys())
# print(labels_activity)

labels_dataset = {
    'KuHar': 'KuHar', 
    'MotionSense': 'MotionSense',
    'ExtraSensory': 'ExtraSensory',
    'WISDM': 'WISDM',
    'WISDM_V2': 'WISDM_V2',
    'UCI': 'UCI',
    'RealWorld': 'RealWorld',
}

# Features to select
features = [
    "accel-x",
    "accel-y",
    "accel-z",
    "gyro-x",
    "gyro-y",
    "gyro-z"
]

dimension = 10

# Create the objects

fft_transform = FFT(centered=True)

# Compose the transform
# First apply the normalizer over whole dataset and then apply FFT over each window
transformer = TransformMultiModalDataset(
    transforms=[fft_transform], new_window_name_prefix="fft."
)

data_umap_name = ['KuHar', 'MotionSense', 'ExtraSensory', 'WISDM', 'UCI', 'RealWorld']

data_train_name = ['KuHar', 'MotionSense', 'WISDM', 'UCI', 'RealWorld']

data_test_name = ['KuHar', 'MotionSense', 'WISDM', 'UCI', 'RealWorld']

combinations_sets = { 
    f'Umap - {dimension}': [],
    'Train': [],
    'Test': []
}

from itertools import combinations

tam = len(data_umap_name)
combination_umap = [list(combinations(data_umap_name, i+1)) for i in range(tam)]
combination_umap[0].insert(0, ('-',))

tam = len(data_train_name)
combination_train = [list(combinations(data_train_name, i+1)) for i in range(tam)]

combination_test = list(combinations(data_test_name, 1))

un = 0
for c in combination_umap:
    un += len(c)
    
tr = 0
for c in combination_train:
    tr += len(c)
ts = 0
for c in combination_test:
    ts += len(c)
print(f'Total of combinations: {un * tr * ts}',
      f'\nTotal of combinations - umap: {un}',
      f'\nTotal of combinations - train: {tr}',
      f'\nTotal of combinations - test: {ts}')

for comb_umap in combination_umap:
    for set_umap in comb_umap:
        for comb_train in combination_train:
            for set_train in comb_train:
                for comb_test in combination_test:
                    for set_test in comb_test:
                        if len(set_train) != 1:
                            combinations_sets[f'Umap - {dimension}'].append(set_umap)
                            combinations_sets['Train'].append(set_train)
                            combinations_sets['Test'].append(set_test)   

n = len(combinations_sets[f'Umap - {dimension}'])
print(f'Total of combinations without only ExtraSensory as train data: {n}')

umap_models = {combination: None for combination in combinations_sets[f'Umap - {dimension}']}

columns = ['Classifier', f'Umap - {dimension}', 'Train', 'Test']

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
        f'Umap - {dimension}': [],
        'Train': [],
        'Test': [],
        'result': []
    }
umap_dict = {combination: None for combination in combination_umap}
        
# metrics_class

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

def evaluate(umap, train, test, evaluators, df, results_dict, umap_name, train_name, test_name, labels_activity, metrics_class, reporter, umap_dict):
# The reporter will be the same

    fft_transform = FFT(centered=True)
    if umap_name == ['-']:
        transformer = TransformMultiModalDataset(transforms=[fft_transform], 
                                                 new_window_name_prefix="fft.")

    elif umap_dict[umap_name[0]] == None:
        transformer_fft = TransformMultiModalDataset(transforms=[fft_transform], 
                                                 new_window_name_prefix="fft.")

        umap = UMAP(n_components=10, random_state=42)
        train_fft = transformer_fft(train)

        umap.fit(train_fft[:][0])
        umap_dict[umap_name[0]] = umap

        umap_transform = WindowedTransform(
            transform=umap, fit_on=None, transform_on="all"
        )

        transformer = TransformMultiModalDataset(transforms=[fft_transform, umap_transform], 
                                                 new_window_name_prefix="reduced.")

    else:
        umap = umap_dict[umap_name[0]]
        
        umap_transform = WindowedTransform(
            transform=umap, fit_on=None, transform_on="all"
        )

        transformer = TransformMultiModalDataset(transforms=[fft_transform, umap_transform], 
                                                 new_window_name_prefix="reduced.")

    train_fft = transformer(train)
    test_fft = transformer(test)

    for estimator, evaluator in evaluators.items():
        multi_run_experiment = MultiRunWorkflow(
            workflow=evaluator['experiment'], 
            num_runs=evaluator['num_runs'],
            debug=False)

        results = multi_run_experiment(train_fft, test_fft)
        results_dict[estimator][f'Umap - {dimension}'].append(umap_name)
        results_dict[estimator]['Train'].append(train_name)
        results_dict[estimator]['Test'].append(test_name)
        results_dict[estimator]['result'].append(results)

        df['Classifier'].append(estimator)
        df[f'Umap - {dimension}'].append(umap_name)
        df['Train'].append(train_name)
        df['Test'].append(test_name)

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
    return df, results_dict, umap_dict

start = time.time()
reporter = ClassificationReport(
    use_accuracy=True,
    use_f1_score=True,
    use_classification_report=True,
    use_confusion_matrix=True,
    plot_confusion_matrix=False,
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

train_data.data['standard activity code'] = train_data.data['standard activity code'].astype('int')
test_data.data['standard activity code'] = test_data.data['standard activity code'].astype('int')

k = 1
for umap_name, train_name, test_name in zip(combinations_sets[f'Umap - {dimension}'], combinations_sets['Train'], combinations_sets['Test']):
    umap_name, train_name = list(umap_name), list(train_name)
    
    if umap_name != ['-']:
        umap = train_data.data[train_data.data['DataSet'].isin(umap_name)]
        umap = create_data_multimodal(umap)
    else:
        umap = None
                           
    train = train_data.data[train_data.data['DataSet'].isin(train_name)]
    train = create_data_multimodal(train)
                           
    test = test_data.data[test_data.data['DataSet'].isin([test_name])]
    test = create_data_multimodal(test)
    
    new_start = time.time()
    df_results, results_dict, umap_dict = evaluate(umap, train, test, evaluators, df_results, results_dict, 
                                        umap_name, train_name, test_name, labels_activity, 
                                        metrics_class, reporter)
    new_end = time.time()
    print(f'Combination: {k} \t Time of execution: {int(new_end - new_start) // 60} minutes and {int(new_end - new_start) % 60} seconds')
    k+=1
    
    # Save results
    df_results_csv = pd.DataFrame(df_results)
    df_results_csv.to_csv('df_results.csv')
    with open('df_results_fixed.pkl', 'wb') as file:
        pickle.dump(df_results, file)

    with open('results_dict_fixed.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

end = time.time()
total = int(end - start)
print(f'Time of execution: {total} seconds')
print(f'Time of execution: {total // 60} minutes and {total % 60} seconds')
print(f'Time of execution: {(total // 86400)} days, {(total // 3600) % 24} hours, {(total // 60) % 60} minutes and {total % 60} seconds')

# df_results = pd.DataFrame(df_results)
# df_results

# Save results

with open('df_results_fixed.pkl', 'wb') as file:
    pickle.dump(df_results, file)
    
with open('results_dict_fixed.pkl', 'wb') as file:
    pickle.dump(results_dict, file)