from pathlib import Path

import pandas as pd

datasets = {
    # KuHar
    "kuhar.train": Path("./data/processed/KuHar/standartized_balanced/train.csv"),
    "kuhar.validation": Path("./data/processed/KuHar/standartized_balanced/validation.csv"),
    "kuhar.test": Path("./data/processed/KuHar/standartized_balanced/test.csv"),
    # MotionSense
    "motionsense.train": Path("./data/processed/MotionSense/standartized_balanced/train.csv"),
    "motionsense.validation": Path("./data/processed/MotionSense/standartized_balanced/validation.csv"),
    "motionsense.test": Path("./data/processed/MotionSense/standartized_balanced/test.csv"),
    # UCI
    "uci.train": Path("./data/processed/UCI/standartized_balanced/train.csv"),
    "uci.validation": Path("./data/processed/UCI/standartized_balanced/validation.csv"),
    "uci.test": Path("./data/processed/UCI/standartized_balanced/test.csv"),
}

columns = [
    [f"accel-x-{i}" for i in range(60)], 
    [f"accel-y-{i}" for i in range(60)], 
    [f"accel-z-{i}" for i in range(60)], 
    [f"gyro-x-{i}" for i in range(60)], 
    [f"gyro-y-{i}" for i in range(60)],
    [f"gyro-z-{i}" for i in range(60)],
] 
columns = [i for l in columns for i in l] + ["standard activity code", "user", "window","activity code"]

dfs = []

for name, path in datasets.items():
    print("--------- " + name + " ---------")
    df = pd.read_csv(path, header=0)
    df = df[columns]
    df["dataset"] = name
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
df.to_csv("./data/processed/mega.csv", index=False)
print(df)