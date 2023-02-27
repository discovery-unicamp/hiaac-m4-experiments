#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from scipy.fftpack import fft

views = [
    "raw_balanced",
    "standartized_balanced",
    "standartized_intra_balanced",
]

def read_sets(path: Path):
    d = defaultdict(dict)
    for p in path.rglob('*.csv'):
        d[p.parent.name][p.stem] = pd.read_csv(p)

    return d

standartized_codes = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down"
}

def main(root_dir: Path):
    global views

    the_sets = {
        view: read_sets(root_dir / view)
        for view in views
    }
    
    for (view, datasets) in the_sets.items():
        print("===============================")
        for (dataset, split) in datasets.items():
            train_val_df = pd.concat((split['train'], split['validation']))
            test_df = split['test']
            
            features_list = [[f for f in test_df.columns if s in f] for s in ["accel", "gyro"]]
            features = [f for k in features_list for f in k]
            
            X_train = train_val_df[features]
            y_train = train_val_df["standard activity code"]
            users_train = train_val_df["user"]
            X_test = test_df[features]
            y_test = test_df["standard activity code"]
            users_test = test_df["user"]

            ffts_train = []
            ffts_test = []
            for f_list in features_list:
                x = []
                for data in train_val_df[f_list].values:
                    data = fft(data)
                    data = np.abs(data)
                    data = data[:len(data)//2]
                    x.append(data)
                ffts_train.append(np.array(x))

                x = []
                for data in test_df[f_list].values:
                    data = fft(data)
                    data = np.abs(data)
                    data = data[:len(data)//2]
                    x.append(data)
                ffts_test.append(np.array(x))

            X_train_fft = np.concatenate(ffts_train, axis=1)
            X_test_fft = np.concatenate(ffts_test, axis=1)

            reducer  = UMAP(n_components=10, random_state=42)
            reducer.fit(X_train)
            X_train = reducer.transform(X_train)
            X_test = reducer.transform(X_test)

            reducer_fft = UMAP(n_components=10, random_state=42)
            reducer_fft.fit(X_train_fft)
            X_train_fft = reducer_fft.transform(X_train_fft)
            X_test_fft = reducer_fft.transform(X_test_fft)

            scores = []
            scores_fft = []
            for i in range(10):
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                scores.append(acc)

                model_fft = RandomForestClassifier()
                model_fft.fit(X_train_fft, y_train)
                y_pred_fft = model_fft.predict(X_test_fft)
                acc_fft = accuracy_score(y_test, y_pred_fft)
                scores_fft.append(acc_fft)

            print(f"(TIME UMAP-10) {view} {dataset}. ACC: {np.mean(scores):.3f} +-{np.std(scores):.3f}")
            print(f"(FFT  UMAP-10) {view} {dataset}. ACC: {np.mean(scores_fft):.3f} +-{np.std(scores_fft):.3f}")            
            print(f"X_train.shape={X_train.shape}; y_train_features: {list(y_train.value_counts().sort_index().items())}; X_test.shape={X_test.shape}; y_test_features: {list(y_test.value_counts().sort_index().items())};")
            print(f"X_train_fft.shape={X_train_fft.shape}; y_train_features: {list(y_test.value_counts().sort_index().items())}; X_test_fft.shape={X_test_fft.shape}; y_test_features: {list(y_test.value_counts().sort_index().items())};")
            print(f"users_train: {list(users_train.value_counts().sort_index().items())}; users_test: {list(users_test.value_counts().sort_index().items())};")
            print("---")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    root_dir = Path(args.root_dir)

    the_sets = main(root_dir)