from typing import Callable, List, Union
import numpy as np

import pandas as pd
from scipy import signal
from scipy import constants
import tqdm
import plotly.express as px

from typing import Tuple
import random


class SplitGuaranteeingAllClassesPerSplit:
    def __init__(
        self,
        column_to_split: str = "user",
        class_column: str = "standard activity code",
        train_size: float = 0.8,
        random_state: int = 42,
        retries: int = 10,
    ):
        self.column_to_split = column_to_split
        self.class_column = class_column
        self.train_size = train_size
        self.random_state = random_state
        self.retries = retries

    def __call__(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        random.seed(self.random_state)
        split_values = dataframe[self.column_to_split].unique()
        class_values = dataframe[self.class_column].unique()

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
        self, class_column: str = "standard activity code", random_state: int = 42
    ):
        self.class_column = class_column
        self.random_state = random_state

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        class_values = dataframe[self.class_column].unique()
        min_class_size = min(
            [
                len(dataframe.loc[dataframe[self.class_column] == class_value])
                for class_value in class_values
            ]
        )
        balanced_df = pd.concat(
            [
                dataframe.loc[dataframe[self.class_column] == class_value].sample(
                    min_class_size, random_state=self.random_state
                )
                for class_value in class_values
            ]
        )
        return balanced_df

class WindowReconstruction:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Interpolate:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class AddGravityColumn:
    """Adiciona uma coluna com a gravidade em cada eixo.
    """
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
    """Converte a aceleração de g para m/s².
    """
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
    """Aplica o filtro Butterworth para remoção da gravidade.
    """
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
    """Calcula a differença entre os intervalos de tempo.
    """
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
    """Imprime o histograma da diferença entre os intervalos de tempo.
    """
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
        for key, grouped_df in tqdm.tqdm(df.groupby(self.groupby_column), desc="Creating windows"):
            for start in range(
                0, len(grouped_df), self.samples_per_window - self.samples_per_overlap
            ):
                window_df = grouped_df[start : start + self.samples_per_window].reset_index(drop=True)
                features = window_df[self.features_to_select].unstack()
                features.index = features.index.map(
                    lambda a: f"{a[0]}-{(a[1])%(self.samples_per_window)}"
                )
                for column in other_columns:
                    features[column] = window_df[column].iloc[0]
                values.append(features)
        return pd.concat(values, axis=1).T.dropna().reset_index(drop=True)


class AddStandardActivityCode:
    """Adiciona a coluna "standard activity code" ao dataframe.
    """
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
    """Renomeia colunas do dataframe.
    """
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
    """Pipeline de transformações de dados.
    """
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
