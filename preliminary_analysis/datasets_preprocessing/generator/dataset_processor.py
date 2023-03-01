from typing import Callable, List, Union
import numpy as np

import pandas as pd
from scipy import signal, interpolate
from scipy import constants
import tqdm
import plotly.express as px

from typing import Tuple
import random


class FilterByCommonRows:
    def __init__(self, match_columns: Union[str, List[str]]):
        self.match_columns = (
            match_columns if isinstance(match_columns, list) else [match_columns]
        )

    def __call__(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        common_rows = set(
            df1[self.match_columns].itertuples(index=False, name=None)
        ) & set(df2[self.match_columns].itertuples(index=False, name=None))
        df1_filtered = df1[
            df1[self.match_columns].apply(tuple, axis=1).isin(common_rows)
        ]
        df2_filtered = df2[
            df2[self.match_columns].apply(tuple, axis=1).isin(common_rows)
        ]
        return df1_filtered, df2_filtered


class SplitGuaranteeingAllClassesPerSplit:
    def __init__(
        self,
        column_to_split: str = "user",
        class_column: str = "standard activity code",
        train_size: float = 0.8,
        random_state: int = None,
        retries: int = 10,
    ):
        self.column_to_split = column_to_split
        self.class_column = class_column
        self.train_size = train_size
        self.random_state = random_state
        self.retries = retries

    def __call__(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        random.seed(self.random_state)
        split_values = dataframe[self.column_to_split].unique() # user ids
        class_values = dataframe[self.class_column].unique() # activity codes

        for _ in range(self.retries):
            random.shuffle(split_values)
            train_values = split_values[: int(len(split_values) * self.train_size)]
            test_values = split_values[int(len(split_values) * self.train_size) :]

            train_df = dataframe.loc[dataframe[self.column_to_split].isin(train_values)]
            test_df = dataframe.loc[dataframe[self.column_to_split].isin(test_values)]

            if len(train_df[self.class_column].unique()) != len(class_values):
                continue
            if len(test_df[self.class_column].unique()) != len(class_values):
                continue
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

        raise ValueError(
            "Could not split dataframe in a way that all classes are present in both splits"
        )


class BalanceToMinimumClass:
    def __init__(
        self,
        class_column: str = "standard activity code",
        filter_column: str = "user",
        random_state: int = 42,
        min_value: int = None,
    ):
        self.class_column = class_column
        self.random_state = random_state
        self.min_value = min_value
        self.filter_column = filter_column

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        class_values = dataframe[self.class_column].unique()
        min_value_size = self.min_value

        # First we need to filter the dataframe to only have filter column values that are present in all classes
        filter_values = dataframe[self.filter_column].unique()
        filter_values_to_use = []

        for filter_value in filter_values:
            if sorted(dataframe[dataframe[self.filter_column] == filter_value][self.class_column].unique()) == sorted(class_values):
                filter_values_to_use.append(filter_value)
        df = dataframe[dataframe[self.filter_column].isin(filter_values_to_use)].copy()
        
        # Now we can balance the dataframe
        if self.min_value is None:
            min_value_size = min(
                [
                    len(df.loc[(df[self.class_column] == class_value) & (df[self.filter_column] == filter_value)])
                    for class_value in class_values
                    for filter_value in filter_values_to_use
                ]
            )
        balanced_df = pd.concat(
            [
                df.loc[(df[self.class_column] == class_value) & (df[self.filter_column] == filter_value)].sample(
                    min_value_size, random_state=self.random_state
                )
                for class_value in class_values for filter_value in filter_values_to_use
            ]
        )
        return balanced_df


class WindowReconstruction:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Interpolate:
    """Interpola colunas do dataframe assumindo que os dados estão em uma frequência fixa.
    Usa a função `scipy.interpolate` para interpolar os dados.
    """

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        original_fs: float,
        target_fs: float,
        kind: str = "cubic",
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Nome da(s) coluna(s) a ser agrupada para reamostrar.
            Normalmente agrupa-se por evento do usuário
            (senão reamostra o dataframe todo, com amostras de diferentes eventos e usuários)
        features_to_select : Union[str, List[str]]
            Nome da(s) coluna(s) a ser reamostrada.
        original_fs : float
            Frequência de amostragem original.
        target_fs : float
            Frequência de amostragem desejada.
        kind : str, optional.
            Tipo de interpolação a ser usada, por padrão 'cubic'.
        """
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.kind = kind

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reamostra as colunas do dataframe.
        Parameters
        ----------
        df : pd.DataFrame
            O dataframe a ser reamostrado.
        Returns
        -------
        pd.DataFrame
            O dataframe com as colunas desejadas, reamostrado.
        """
        df = df.reset_index()
        for _, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Interpoling"
        ):
            for column in self.features_to_select:

                signal = grouped_df[column].values
                arr = np.array([np.nan] * len(grouped_df))
                time = np.arange(0, len(signal), 1) / self.original_fs
                interplator = interpolate.interp1d(
                    time,
                    signal,
                    kind=self.kind,
                )
                new_time = np.arange(0, time[-1], 1 / self.target_fs)
                resampled = interplator(new_time)

                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class AddGravityColumn:
    """Adiciona uma coluna com a gravidade em cada eixo."""

    def __init__(self, axis_columns: List[str], gravity_columns: List[str]):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Nome das colunas que contém os dados de aceleração.
        gravity_columns : List[str]
            Nome da coluna com que contém os dados de gravidade.
        """
        self.axis_columns = axis_columns
        self.gravity_columns = gravity_columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona uma coluna com a gravidade em cada eixo.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe a ser utilizado.

        Returns
        -------
        pd.DataFrame
            Dataframe com os dados de gravidade adicionados.
        """
        for axis_col, gravity_col in zip(self.axis_columns, self.gravity_columns):
            df[axis_col] = df[axis_col] + df[gravity_col]
        return df


class Convert_G_to_Ms2:
    """Converte a aceleração de g para m/s²."""

    def __init__(self, axis_columns: List[str], g_constant: float = constants.g):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Nome das colunas que contém os dados de aceleração.
        g_constant : float, optional
            Valor da gravidade a ser adicionado, por padrão `scipy.constants.g`
        """
        self.axis_columns = axis_columns
        self.gravity_constant = g_constant

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a conversão de g para m/s².

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe a ser utilizado.

        Returns
        -------
        pd.DataFrame
            Dataframe com os dados de aceleração convertidos.
        """
        for axis_col in self.axis_columns:
            df[axis_col] = df[axis_col] * self.gravity_constant
        return df


class ButterworthFilter:
    """Aplica o filtro Butterworth para remoção da gravidade."""

    def __init__(self, axis_columns: List[str], fs: float):
        """
        Parameters
        ----------
        axis_columns : List[str]
            Nome das colunas que contém os dados de aceleração.
        fs : float
            Frequencia original do conjunto de dados
        """
        self.axis_columns = axis_columns
        self.fs = fs

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica o filtro Butterworth para remoção da gravidade.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe a ser utilizado.

        Returns
        -------
        pd.DataFrame
            Dataframe com os dados de aceleração filtrados (filtro passado).
        """
        h = signal.butter(3, 0.3, "hp", fs=self.fs, output="sos")
        for axis_col in self.axis_columns:
            df[axis_col] = signal.sosfiltfilt(h, df[axis_col].values)
        return df


class CalcTimeDiffMean:
    """Calcula a differença entre os intervalos de tempo."""

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        column_to_diff: str,
        new_column_name: str = "diff",
        filter_predicate: Callable[[pd.DataFrame], pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Nome da(s) coluna(s) a ser agrupadas para calcular da differença.
            Normalmente agrupa-se por evento do usuário
            (senão calcula a diferença usando o dataframe todo, com amostras de diferentes eventos e usuários)
        column_to_diff : str
            Nome da coluna a ser utilizada para calcular a differença.
        new_column_name : str, optional
            Nome da coluna onde a diferença será armazenada, por padrão "diff"
        filter_predicate : Callable[[pd.DataFrame], pd.DataFrame], optional
            Função que filtra o dataframe, por padrão None
        """
        self.groupby_column = groupby_column
        self.column_to_diff = column_to_diff
        self.new_column_name = new_column_name
        self.filter_predicate = filter_predicate

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula a differença entre os intervalos de tempo.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe a ser utilizado.

        Returns
        -------
        pd.DataFrame
            Dataframe com a coluna com a diferença entre os intervalos de tempo.
            Caso `filter_predicate` não seja None, o dataframe será filtrado.
        """
        df[self.new_column_name] = df.groupby(self.groupby_column)[
            self.column_to_diff
        ].diff()
        df = df.dropna(subset=[self.new_column_name])
        if self.filter_predicate:
            df = df.groupby(self.groupby_column).filter(self.filter_predicate)
        return df.reset_index(drop=True)


class PlotDiffMean:
    """Imprime o histograma da diferença entre os intervalos de tempo."""

    def __init__(self, column_to_plot: str = "diff"):
        """
        Parameters
        ----------
        column_to_plot : str, optional
            Coluna para ser utilizada para imprimir o histograma, por padrão "diff"
        """
        self.column_to_plot = column_to_plot

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imprime o histograma da diferença entre os intervalos de tempo.

        Returns
        -------
        _type_
            O próprio dataframe.
        """

        fig = px.histogram(df, x=self.column_to_plot)
        fig.show("png")
        return df


class Resampler:
    """Reamostra colunas do dataframe assumindo que os dados estão em uma frequência fixa.
    Usa a função `scipy.signal.resample` para reamostrar os dados.
    """

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        original_fs: float,
        target_fs: float,
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Nome da(s) coluna(s) a ser agrupada para reamostrar.
            Normalmente agrupa-se por evento do usuário
            (senão reamostra o dataframe todo, com amostras de diferentes eventos e usuários)
        features_to_select : Union[str, List[str]]
            Nome da(s) coluna(s) a ser reamostrada.
        original_fs : float
            Frequencia original do conjunto de dados.
        target_fs : float
            Frequencia desejada para o conjunto de dados.
        """
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.original_fs = original_fs
        self.target_fs = target_fs

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reamostra as colunas do dataframe.
        Parameters
        ----------
        df : pd.DataFrame
            O dataframe a ser reamostrado.
        Returns
        -------
        pd.DataFrame
            O dataframe com as colunas desejadas, reamostrado.
        """
        df = df.reset_index()
        for key, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Resampling"
        ):
            for column in self.features_to_select:
                time = len(grouped_df) // self.original_fs
                arr = np.array([np.nan] * len(grouped_df))
                resampled = signal.resample(
                    grouped_df[column].values, int(time * self.target_fs)
                )
                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class ResamplerPoly:
    """Reamostra colunas do dataframe assumindo que os dados estão em uma frequência fixa.
    Usa a função `scipy.signal.resample` para reamostrar os dados.
    """

    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        features_to_select: Union[str, List[str]],
        up: float,
        down: float,
        padtype: str = "mean",
    ):
        """
        Parameters
        ----------
        groupby_column : Union[str, List[str]]
            Nome da(s) coluna(s) a ser agrupada para reamostrar.
            Normalmente agrupa-se por evento do usuário
            (senão reamostra o dataframe todo, com amostras de diferentes eventos e usuários)
        features_to_select : Union[str, List[str]]
            Nome da(s) coluna(s) a ser reamostrada.
        up : float
            Fator de aumento da frequencia.
        down : float
            Fator de redução da frequencia.
        padtype : str, optional
            Tipo de preenchimento, por padrão 'mean'.
        """
        self.groupby_column = groupby_column
        self.features_to_select = (
            [features_to_select]
            if isinstance(features_to_select, str)
            else features_to_select
        )
        self.up = up
        self.down = down
        self.padtype = padtype

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reamostra as colunas do dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            O dataframe a ser reamostrado.

        Returns
        -------
        pd.DataFrame
            O dataframe com as colunas desejadas, reamostrado.
        """
        df = df.reset_index()
        for _, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column, group_keys=True), desc="Resampling"
        ):
            for column in self.features_to_select:
                arr = np.array([np.nan] * len(grouped_df))
                resampled = signal.resample_poly(
                    grouped_df[column].values,
                    up=self.up,
                    down=self.down,
                    padtype=self.padtype,
                )
                arr[: len(resampled)] = resampled
                df.loc[grouped_df.index, column] = arr
        return df.dropna().reset_index(drop=True)


class Windowize:
    """Realiza o janelamento dos dados em janelas de tamanho fixo.
    O janelamento será feito com amostras consecutivas do dataframe e a ultima janela será descartada.
    As colunas desejadas serão transpostas (de linha para coluna) no tamanho da janela desejada.
    Para as colunas remanescentes, será mantido o primeiro elemento da janela.
    Nota: assume-se aqui que a janela não possui sobreposição e que a taxa de amostragem é constante.
    """

    def __init__(
        self,
        features_to_select: List[str],
        samples_per_window: int,
        samples_per_overlap: int,
        groupby_column: Union[str, List[str]],
        divisible_by: int = None,
    ):
        """_summary_

        Parameters
        ----------
        features_to_select : List[str]
            Features que serão utilizadas para realizar o janelamento
            (serão transpostas de linhas para colunas e adicionado um sufixo de indice).
        samples_per_window : int
            Numero de amostras consecutivas que serão utilizadas para realizar o janelamento.
        samples_per_overlap : int
            Numero de amostras que serão sobrepostas entre janelas consecutivas.
        groupby_column : Union[str, List[str]]
            Nome da(s) coluna(s) a ser agrupadas para realizar o janelamento.
            Normalmente agrupa-se por evento do usuário
            (senão calcula a diferença usando o dataframe todo, com amostras de diferentes eventos e usuários).
        """
        self.features_to_select = (
            features_to_select
            if isinstance(features_to_select, list)
            else [features_to_select]
        )
        self.samples_per_window = samples_per_window
        self.samples_per_overlap = samples_per_overlap
        self.groupby_column = groupby_column
        self.divisible_by = divisible_by

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realiza o janelamento nos das colunas do dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            O dataframe a ser janelado.

        Returns
        -------
        pd.DataFrame
            O dataframe com janelas de tamanho fixo.
        """
        values = []
        other_columns = set(df.columns) - set(self.features_to_select)

        for key, grouped_df in tqdm.tqdm(
            df.groupby(self.groupby_column), desc="Creating windows"
        ):
            for i, start in enumerate(
                range(
                    0,
                    len(grouped_df),
                    self.samples_per_window - self.samples_per_overlap,
                )
            ):
                window_df = grouped_df[
                    start : start + self.samples_per_window
                ].reset_index(drop=True)
                if len(window_df) != self.samples_per_window:
                    continue
                if window_df.isnull().values.any():
                    continue

                features = window_df[self.features_to_select].unstack()
                features.index = features.index.map(
                    lambda a: f"{a[0]}-{(a[1])%(self.samples_per_window)}"
                )
                for column in other_columns:
                    features[column] = window_df[column].iloc[0]
                features["window"] = i
                values.append(features)
        return pd.concat(values, axis=1).T.reset_index(drop=True)


class AddStandardActivityCode:
    """Adiciona a coluna "standard activity code" ao dataframe."""

    def __init__(self, codes_map: dict):
        """
        Parameters
        ----------
        codes_map : dict
            Dicionário com o código da atividade (do conjunto de dados original)
            como chave e o código da atividade padrão como valor
        """
        self.codes_map = codes_map

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona a coluna "standard activity code" ao dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            O dataframe a ser adicionada a coluna.

        Returns
        -------
        pd.DataFrame
            O dataframe com a coluna "standard activity code" adicionada.
        """
        df["standard activity code"] = df["activity code"].map(self.codes_map)
        return df


class RenameColumns:
    """Renomeia colunas do dataframe."""

    def __init__(self, columns_map: dict):
        """
        Parameters
        ----------
        columns_map : dict
            Dicionário com os nome das colunas originais como chave e o novo nome como valor.
        """
        self.columns_map = columns_map

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia as colunas do dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            O dataframe com as colunas a serem renomeadas.

        Returns
        -------
        pd.DataFrame
            O dataframe com as colunas renomeadas.
        """
        df.rename(columns=self.columns_map, inplace=True)
        return df


class Pipeline:
    """Pipeline de transformações de dados."""

    def __init__(self, transforms: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        Parameters
        ----------
        transforms : Callable[[pd.DataFrame], pd.DataFrame]
            Lista de transformações a serem executadas.
            As transformações devem ser objetos chamáveis, isto é, que implementam o método __call__.
            O método __call__ deve receber um dataframe como parâmetro e retornar um dataframe.
        """
        self.transforms = transforms

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica as transformações na ordem em que foram adicionadas.

        Parameters
        ----------
        df : pd.DataFrame
            O dataframe a ser transformado.

        Returns
        -------
        pd.DataFrame
            O dataframe transformado.
        """
        for transform in self.transforms:
            print(f"Executing {transform.__class__.__qualname__}")
            df = transform(df)
        return df
