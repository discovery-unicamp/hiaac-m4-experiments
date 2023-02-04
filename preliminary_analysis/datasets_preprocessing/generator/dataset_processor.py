from abc import ABC
from typing import Callable, List, Union
import numpy as np

import pandas as pd
from scipy import signal
from scipy import constants
import tqdm


class WindowReconstruction:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Interpolate:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class AddGravityColumn:
    def __init__(self, axis_columns: List[str], gravity_columns: List[str]) -> None:
        self.axis_columns = axis_columns
        self.gravity_columns = gravity_columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for axis_col, gravity_col in zip(self.axis_columns, self.gravity_columns):
            df[axis_col] = df[axis_col] - df[gravity_col]
        return df


class Convert_G_to_Ms2:
    def __init__(self, axis_columns: List[str], g_constant: float = constants.g):
        self.axis_columns = axis_columns
        self.gravity_constant = g_constant

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for axis_col in self.axis_columns:
            df[axis_col] = df[axis_col] *self. gravity_constant
        return df


class ButterworthFilter:
    def __init__(self, axis_columns: List[str], fs: float):
        self.axis_columns = axis_columns
        self.fs = fs

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        h = signal.butter(3, 0.3, "hp", fs=self.fs, output="sos")
        for axis_col in self.axis_columns:
            df[axis_col] = signal.sosfiltfilt(h, df[axis_col].values)
        return df


class CalcTimeDiffMean:
    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        column_to_diff: str,
        new_column_name: str = "diff",
        filter_predicate: Callable[[pd.DataFrame], pd.DataFrame] = None,
    ):
        self.groupby_column = groupby_column
        self.column_to_diff = column_to_diff
        self.new_column_name = new_column_name
        self.filter_predicate = filter_predicate

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.new_column_name] = df.groupby(self.groupby_column)[
            self.column_to_diff
        ].diff()
        df = df.dropna(subset=[self.new_column_name])
        if self.filter_predicate:
            df = df.groupby(self.groupby_column).filter(self.filter_predicate)
        return df.reset_index(drop=True)


class Resampler:
    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        original_fs: float,
        target_fs: float,
    ):
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.original_fs = original_fs
        self.target_fs = target_fs

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        for key, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Resampling"
        ):
            for column in self.features_to_select:
                time = len(grouped_df) / self.original_fs
                arr = np.array([np.nan] * len(grouped_df))
                resampled = signal.resample(
                    grouped_df[column].values, int(time * self.target_fs)
                )
                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class Windowize:
    def __init__(
        self,
        features_to_select: List[str],
        samples_per_window: int,
        samples_per_overlap: int,
    ):
        self.features_to_select = (
            features_to_select
            if isinstance(features_to_select, list)
            else [features_to_select]
        )
        self.samples_per_window = samples_per_window
        self.samples_per_overlap = samples_per_overlap

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        values = []
        other_columns = set(df.columns) - set(self.features_to_select)
        for key, grouped_df in tqdm.tqdm(df.groupby("csv"), desc="Creating windows"):
            for start in range(
                0, len(grouped_df), self.samples_per_window - self.samples_per_overlap
            ):
                window_df = grouped_df[start : start + self.samples_per_window]
                features = window_df[self.features_to_select].unstack()
                features.index = features.index.map(
                    lambda a: f"{a[0]}-{(a[1])%(self.samples_per_window)}"
                )
                for column in other_columns:
                    features[column] = window_df[column].iloc[0]
                values.append(features)
        return pd.concat(values, axis=1).T.dropna().reset_index(drop=True)


class AddStandardActivityCode:
    def __init__(self, codes_map: dict):
        self.codes_map = codes_map

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df["standard activity code"] = df["activity code"].map(self.codes_map)
        return df


class RenameColumns:
    def __init__(self, columns_map: dict):
        self.columns_map = columns_map

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns=self.columns_map, inplace=True)
        return df


class Pipeline:
    def __init__(self, transforms: Callable[[pd.DataFrame], pd.DataFrame]):
        self.transforms = transforms

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for transform in self.transforms:
            print(f"Executing {transform.__class__.__qualname__}")
            df = transform(df)
        return df
