import numpy as np
import pandas as pd
from pathlib import Path
import os, shutil
from natsort import natsorted
from scipy import interpolate
from zipfile import ZipFile

from dataset_processor import (
    AddGravityColumn,
    ButterworthFilter,
    CalcTimeDiffMean,
    Convert_G_to_Ms2,
    PlotDiffMean,
    ResamplerPoly,
    Windowize,
    AddStandardActivityCode,
    SplitGuaranteeingAllClassesPerSplit,
    BalanceToMinimumClass,
    BalanceToMinimumClassAndUser,
    FilterByCommonRows,
    RenameColumns,
    Pipeline
)

maping = [4, 3, -1, -1, 5, 0, 1, 2]
tarefas = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
standard_activity_code_realworld_map = {
    activity: maping[tarefas.index(activity)] for activity in tarefas 
}

datasets = [
    "KuHar",
    "MotionSense",
    "UCI",
    "WISDM",
    "RealWorld",
]

column_group = {
    "KuHar": "csv",
    "MotionSense": "csv",
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "activity code", "serial"],
    "RealWorld": ["user", "activity code", "position"],
}

standard_activity_code_map = {
    "KuHar": {
        0: 1,
        1: 0,
        2: -1,
        3: -1,
        4: -1,
        5: -1,
        6: -1,
        7: -1,
        8: -1,
        9: -1,
        10: -1,
        11: 2,
        12: -1,
        13: -1,
        14: 5,
        15: 3,
        16: 4,
        17: -1,
    },
    "MotionSense": {
        0: 4,
        1: 3,
        2: 0,
        3: 1,
        4: 2,
        5: 5
    },
    "WISDM": {
        "A": 2,
        "B": 5,
        "C": 6,
        "D": 0,
        "E": 1,
        "F": -1,
        "G": -1,
        "H": -1,
        "I": -1,
        "J": -1,
        "K": -1,   
        "L": -1,
        "M": -1,
        "O": -1,
        "P": -1,
        "Q": -1,
        "R": -1,
        "S": -1,
    },
    "UCI": {
        1: 2, # walk
        2: 3, # stair up
        3: 4, # stair down
        4: 0, # sit
        5: 1, # stand
        6: -1, # Laying
        7: -1, # stand to sit
        8: -1, # sit to stand
        9: -1, # sit to lie
        10: -1, # lie to sit
        11: -1, # stand to lie
        12: -1 # lie to stand
    },
    "RealWorld": standard_activity_code_realworld_map
}

columns_to_rename = {
    "KuHar": None,
    "MotionSense": {
        "userAcceleration.x": "accel-x",
        "userAcceleration.y": "accel-y",
        "userAcceleration.z": "accel-z",
        "rotationRate.x": "gyro-x",
        "rotationRate.y": "gyro-y",
        "rotationRate.z": "gyro-z",
    },
    "WISDM": None,
    "UCI": None,
}

feature_columns = {
    "KuHar": [  
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ],
    "MotionSense": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
        "attitude.roll",
        "attitude.pitch",
        "attitude.yaw",
        "gravity.x",
        "gravity.y",
        "gravity.z",
    ],
    "WISDM": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ],
    "UCI": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "RealWorld": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
}

match_columns = {
    "KuHar": ["user", "serial", "window", "activity code"],
    "MotionSense": ["user", "serial", "window"],
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "serial", "window", "activity code"],
    "RealWorld": ["user", "window", "activity code", "position"],
    "RealWorld_thigh": ["user", "window", "activity code", "position"],
    "RealWorld_upperarm": ["user", "window", "activity code", "position"],
    "RealWorld_waist": ["user", "window", "activity code", "position"],
}

# Create functions to read datasets 

def read_kuhar(kuhar_dir_path: str) -> pd.DataFrame:
    """Le o dataset Kuhar e retorna um DataFrame com os dados (vindo de todos os arquivos CSV)
    O dataframe contém as seguintes colunas:
    - accel-x: Aceleração no eixo x
    - accel-y: Aceleração no eixo y
    - accel-z: Aceleração no eixo z
    - gyro-x: Velocidade angular no eixo x
    - gyro-y: Velocidade angular no eixo y
    - gyro-z: Velocidade angular no eixo z
    - accel-start-time: Tempo de início da janela de aceleração
    - gyro-start-time: Tempo de início da janela de giroscópio
    - activity code: Código da atividade
    - index: Índice da amostra vindo do csv
    - user: Código do usuário
    - serial: Número de serial da atividade
    - csv: Nome do arquivo CSV

    Parameters
    ----------
    kuhar_dir_path : str
        Caminho para o dataset Kuhar

    Returns
    -------
    pd.DataFrame
        Dataframe com os dados do dataset Kuhar
    """
    kuhar_dir_path = Path(kuhar_dir_path)

    # Cria um dicionário com os tipos de dados de cada coluna
    feature_dtypes = {
        "accel-start-time": np.float32,
        "accel-x": np.float32,
        "accel-y": np.float32,
        "accel-z": np.float32,
        "gyro-start-time": np.float32,
        "gyro-x": np.float32,
        "gyro-y": np.float32,
        "gyro-z": np.float32,
    }

    dfs = []
    for i, f in enumerate(sorted(kuhar_dir_path.rglob("*.csv"))):
        # Pega o nome da atividade (nome da pasta, ex.: 5.Lay)
        # Pega o nome do arquivo CSV (ex.: 1052_F_1.csv)
        # Separa o número da atividade e o nome (ex.: [5, 'Lay'])
        activity_no, activity_name = f.parents[0].name.split(".")
        activity_no = int(activity_no)

        # Divide o código do usuário, o tipo de atividade e o número de serial (ex.: [1055, 'G', 1])
        csv_splitted = f.stem.split("_")
        user = int(csv_splitted[0])
        serial = "_".join(csv_splitted[2:])

        # Le o arquivo CSV
        df = pd.read_csv(f, names=list(feature_dtypes.keys()), dtype=feature_dtypes)
        
        # Remove dataframes que contenham NaN
        if df.isnull().values.any():
            continue

        # Apenas reordenando as colunas (não é removida nenhuma coluna)
        df = df[
            [
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
                "accel-start-time",
                "gyro-start-time",
            ]
        ]

        # ----- Adiciona colunas auxiliares e meta-dados ------
        # Como é um simples instante de tempo (sem duração), o tempo de início e fim são iguais
        df["accel-end-time"] = df["accel-start-time"]
        df["gyro-end-time"] = df["gyro-start-time"]
        # Adiciona a coluna com o código da atividade
        df["activity code"] = activity_no
        # Adiciona a coluna do índice (qual é o numero da linha da amostra no dataframe)
        df["index"] = range(len(df))
        # Adiciona a coluna do usuário
        df["user"] = user
        # Adiciona a coluna do serial (a vez que o usuário praticou)
        df["serial"] = serial
        # Adiciona a coluna com o caminho do csv
        df["csv"] = "/".join(f.parts[-2:])
        # ----------------------------------------------------
        dfs.append(df)
    return pd.concat(dfs)

def read_motionsense(motionsense_path: str) -> pd.DataFrame:
    """Le o dataset do motionsense e retorna um dataframe com os dados (vindos de todos os arquivos CSV)
    O dataframe retornado possui as seguintes colunas:
    - attitude.roll: Rotação em torno do eixo x
    - attitude.pitch: Rotação em torno do eixo y
    - attitude.yaw: Rotação em torno do eixo z
    - gravity.x: Gravidade em torno do eixo x
    - gravity.y: Gravidade em torno do eixo y
    - gravity.z: Gravidade em torno do eixo z
    - rotationRate.x: Velocidade angular em torno do eixo x
    - rotationRate.y: Velocidade angular em torno do eixo y
    - rotationRate.z: Velocidade angular em torno do eixo z
    - userAcceleration.x: Aceleração no eixo x
    - userAcceleration.y: Aceleração no eixo y
    - userAcceleration.z: Aceleração no eixo z
    - activity code: Código da atividade
    - index: Índice da amostra vindo do csv
    - user: Usuário que realizou a atividade
    - serial: Número de série da atividade
    - csv: Caminho do csv que contém a atividade

    Parameters
    ----------
    motionsense_path : str
        Caminho para o dataset MotionSense

    Returns
    -------
    pd.DataFrame
        Dataframe com os dados do dataset MotionSense
    """

    motionsense_path = Path(motionsense_path)
    activity_names = {0: "dws", 1: "ups", 2: "sit", 3: "std", 4: "wlk", 5: "jog"}
    activity_codes = {v: k for k, v in activity_names.items()}

    feature_dtypes = {
        "attitude.roll": np.float32,
        "attitude.pitch": np.float32,
        "attitude.yaw": np.float32,
        "gravity.x": np.float32,
        "gravity.y": np.float32,
        "gravity.z": np.float32,
        "rotationRate.x": np.float32,
        "rotationRate.y": np.float32,
        "rotationRate.z": np.float32,
        "userAcceleration.x": np.float32,
        "userAcceleration.y": np.float32,
        "userAcceleration.z": np.float32,
    }

    dfs = []
    for i, f in enumerate(sorted(motionsense_path.rglob("*.csv"))):
        # Pegando o nome da atividade
        activity_name = f.parents[0].name
        # Pariticiona o nome da atividade em o cóigo da corrida
        activity_name, serial = activity_name.split("_")
        activity_code = activity_codes[activity_name]

        user = int(f.stem.split("_")[1])
        df = pd.read_csv(
            f, names=list(feature_dtypes.keys()), dtype=feature_dtypes, skiprows=1
        )

        if df.isnull().values.any():
            continue

        # ----- Adiciona colunas auxiliares e meta-dados ------
        df["activity code"] = activity_code
        df["index"] = range(len(df))
        df["user"] = user
        df["serial"] = serial
        df["csv"] = "/".join(f.parts[-2:])
        # ----------------------------------------------------
        dfs.append(df)

    return pd.concat(dfs)

def read_uci(uci_path):
    """Le o dataset do motionsense e retorna um dataframe com os dados (vindos de todos os arquivos CSV)
    O dataframe retornado possui as seguintes colunas:
    - attitude.roll: Rotação em torno do eixo x
    - attitude.pitch: Rotação em torno do eixo y
    - attitude.yaw: Rotação em torno do eixo z
    - gravity.x: Gravidade em torno do eixo x
    - gravity.y: Gravidade em torno do eixo y
    - gravity.z: Gravidade em torno do eixo z
    - rotationRate.x: Velocidade angular em torno do eixo x
    - rotationRate.y: Velocidade angular em torno do eixo y
    - rotationRate.z: Velocidade angular em torno do eixo z
    - userAcceleration.x: Aceleração no eixo x
    - userAcceleration.y: Aceleração no eixo y
    - userAcceleration.z: Aceleração no eixo z
    - activity code: Código da atividade
    - index: Índice da amostra vindo do txt
    - user: Usuário que realizou a atividade
    - serial: Número de série da atividade
    - txt: Caminho do txt que contém a atividade

    Parameters
    ----------
    uci_path : str
        Caminho para o dataset MotionSense

    Returns
    -------
    pd.DataFrame
        Dataframe com os dados do dataset UCI-HAR
    """
    activity_names = {
        1: "WALKING", 
        2: "WALKING_UPSTAIRS", 
        3: "WALKING_DOWNSTAIRS", 
        4: "SITTING", 
        5: "STANDING", 
        6: "LAYING",
        7: "STAND_TO_SIT",
        8: "SIT_TO_STAND",
        9: "SIT_TO_LIE",
        10: "LIE_TO_SIT",
        11: "STAND_TO_LIE",
        12: "LIE_TO_STAND"
    }
    activity_codes = {v: k for k, v in activity_names.items()}
    
    feature_columns = [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]
    
#     df_labels = pd.read_csv("data/RawData/labels.txt", header=None, sep=" ")
    df_labels = pd.read_csv(uci_path / "labels.txt", header=None, sep=" ")
    df_labels.columns=["serial", "user", "activity code", "start", "end"]
    
    uci_path = Path(uci_path)
    
    dfs = []
    data_path = list(uci_path.glob("*.txt"))
    new_data_path = [elem.name.split("_")+[elem] for elem in sorted(data_path)]
    df = pd.DataFrame(new_data_path, columns=["sensor", "serial", "user", "file"])
    for key, df2 in df.groupby(["serial", "user"]):
        acc, gyr = None, None
        for row_index, row in df2.iterrows():
            data = pd.read_csv(row["file"], header=None, sep=" ")
            if row["sensor"] == "acc":
                acc = data
            else:
                gyr = data
        new_df = pd.concat([acc, gyr], axis=1)
        new_df.columns = feature_columns
        
        user = int(key[1].split(".")[0][4:])
        serial = int(key[0][3:])
        
        new_df['txt'] = row["file"]
        
        new_df["user"] = user
        new_df["serial"] = serial
#         new_df["activity code"] = -1
        
        for row_index, row in df_labels.loc[(df_labels["serial"] == serial) & (df_labels["user"] == user)].iterrows():
            start = row['start']
            end = row["end"]+1
            activity = row["activity code"]
            resumed_df = new_df.loc[start:end].copy()
            resumed_df["index"] = [i for i in range(start, end+1)]
            resumed_df["activity code"] = activity

            # Drop samples with NaN
            if resumed_df.isnull().values.any():
                continue
            
            dfs.append(resumed_df)
    
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    return df

def read_wisdm(wisdm_path: str, interpol = True) -> pd.DataFrame:
    """Le o dataset do motionsense e retorna um dataframe com os dados (vindos de todos os arquivos txt)
    O dataframe retornado possui as seguintes colunas:
    - activity code: Código da atividade
    - user: Usuário que realizou a atividade
    - timestamp-accel: Timestamp da aceleração
    - accel-x: Aceleração no eixo x
    - accel-y: Aceleração no eixo y
    - accel-z: Aceleração no eixo z
    - timestamp-gyro: Timestamp do giroscópio
    - gyro-x: Giroscópio no eixo x
    - gyro-y: Giroscópio no eixo y
    - gyro-z: Giroscópio no eixo z

    Parameters
    ----------
    wisdm_path : str
        Caminho para o dataset WISDM

    Returns
    -------
    pd.DataFrame
        Dataframe com os dados do dataset WISDM
    """
    
    feature_columns_acc = [
        "user",
        "activity code",
        "timestamp-accel",
        "accel-x",
        "accel-y",
        "accel-z",
    ]
    feature_columns_gyr = [
        "user",
        "activity code",
        "timestamp-gyro",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]

    # Lista com letras maiúsculas de A até S sem o N
    labels = [chr(i) for i in range(65, 84) if chr(i) != "N"]

    dfs = []
    window = 1
    for user in range(1600,1651):
        window = 1
        df_acc = pd.read_csv(wisdm_path / f"accel/data_{user}_accel_phone.txt", sep=",|;", header=None, engine="python")
        df_acc = df_acc[df_acc.columns[0:-1]]
        df_acc.columns = feature_columns_acc
        df_acc["timestamp-accel"] = df_acc["timestamp-accel"].astype(np.int64)


        df_gyr = pd.read_csv(wisdm_path / f"gyro/data_{user}_gyro_phone.txt", sep=",|;", header=None, engine="python")
        df_gyr = df_gyr[df_gyr.columns[0:-1]]
        df_gyr.columns = feature_columns_gyr
        df_gyr["timestamp-gyro"] = df_gyr["timestamp-gyro"].astype(np.int64)

        for activity in labels:
            acc = df_acc[df_acc["activity code"] == activity].copy()
            gyr = df_gyr[df_gyr["activity code"] == activity].copy()

            time_acc = np.array(acc["timestamp-accel"])
            time_gyr = np.array(gyr["timestamp-gyro"])

            if interpol:
                # Setando o tempo inicial para 0
                if len(time_acc) > 0 and len(time_gyr) > 0:
                    time_acc = (time_acc - time_acc[0]) / 1000000000
                    time_gyr = (time_gyr - time_gyr[0]) / 1000000000

                    ### Retirando os intervalos sem amostra (periodos vazios)
                    if np.any(np.diff(time_acc)<0):
                        pos = np.nonzero(np.diff(time_acc)<0)[0].astype(int)
                        for k in pos:
                            time_acc[k+1:] = time_acc[k+1:]+time_acc[k]+1/20
                    if np.any(np.diff(time_gyr)<0):
                        pos = np.nonzero(np.diff(time_gyr)<0)[0].astype(int)
                        for k in pos:
                            time_gyr[k+1:] = time_gyr[k+1:]+time_gyr[k]+1/20

                    # Interpolando os dados
                    sigs_acc = []
                    sigs_gyr = []
                    for sig_acc, sig_gyr in zip(acc[feature_columns_acc[2:]], gyr[feature_columns_gyr[2:]]):
                        fA = np.array(acc[sig_acc])
                        fG = np.array(gyr[sig_gyr])

                        intp1 = interpolate.interp1d(time_acc, fA, kind='cubic')
                        intp2 = interpolate.interp1d(time_gyr, fG, kind='cubic')
                        nt1 = np.arange(0,time_acc[-1],1/20)
                        nt2 = np.arange(0,time_gyr[-1],1/20)
                        sigs_acc.append(intp1(nt1))
                        sigs_gyr.append(intp2(nt2))

                    tam = min(len(nt1), len(nt2))

                    new_acc = pd.DataFrame()
                    new_gyr = pd.DataFrame()

                    for x, y in zip(sigs_acc, sigs_gyr):
                        x = x[:tam]
                        y = y[:tam]
                    
                    new_acc["timestamp-accel"] = nt1[:tam]
                    new_gyr["timestamp-gyro"] = nt2[:tam]

                    for sig_acc, sig_gyr, column_acc, column_gyr in zip(sigs_acc, sigs_gyr, feature_columns_acc[2:], feature_columns_gyr[2:]):
                        new_acc[column_acc] = sig_acc[:tam]
                        new_gyr[column_gyr] = sig_gyr[:tam]
            else:
                tam = min(len(time_acc), len(time_gyr))
                new_acc = acc[feature_columns_acc[2:]].iloc[:tam]
                new_gyr = gyr[feature_columns_gyr[2:]].iloc[:tam]
                
            # Criando um dataframe com os dados de aceleração e giroscópio
            df = pd.concat([new_acc, new_gyr], axis=1)
            df["activity code"] = activity
            df["user"] = user
            df["window"] = window
            df = df.dropna()

            dfs.append(df)

    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)

    for column in feature_columns_acc[2:] + feature_columns_gyr[2:]:
        df[column] = df[column].astype(np.float32)
    df["user"] = df["user"].astype(np.int32)

    return df.dropna().reset_index(drop=True)

def read_realworld(workspace, users):
    """Le o dataset RealWorld e retorna um DataFrame com os dados (vindo de todos os arquivos CSV)
    O dataframe contém as seguintes colunas:
    - user: usuário
    - activity: atividade
    - sensor: sensor (acc ou gyr)
    - position: posição do sensor
    - index: índice do arquivo
    - accel-start-time: tempo de início da leitura do acelerômetro
    - accel-x: leitura do acelerômetro no eixo x
    - accel-y: leitura do acelerômetro no eixo y
    - accel-z: leitura do acelerômetro no eixo z
    - gyro-start-time: tempo de início da leitura do giroscópio
    - gyro-x: leitura do giroscópio no eixo x
    - gyro-y: leitura do giroscópio no eixo y
    - gyro-z: leitura do giroscópio no eixo z

    Parâmetros
    ----------
    workspace: Path
        Caminho para a pasta de trabalho
    users: list
        Lista com os usuários

    Retorno
    -------
    DataFrame
        DataFrame com os dados
    """

    # Agora vamos nos preparar para criar as views
    tarefas = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']

    # Por enquanto só vamos criar as views das seguintes posições
    posicao = ['thigh', 'upperarm', 'waist']

    # Lista de features
    feature_acc = ["index", "accel-start-time", "accel-x", "accel-y", "accel-z"]
    feature_gyr = ["index", "gyro-start-time", "gyro-x", "gyro-y", "gyro-z"]

    dfs = []

    for p in posicao:
        for user in users:
            filesacc = sorted(os.listdir(workspace / "realworld2016_dataset_organized" / user / "acc"))
            filesgyr = sorted(os.listdir(workspace / "realworld2016_dataset_organized" / user / "gyr"))

            pos = []
            for i in range(len(filesacc)):
                if filesacc[i].find(p)>-1:
                    pos.append(i)
            
            for i in pos:
                acc = pd.read_csv(workspace / "realworld2016_dataset_organized" / user / "acc" / filesacc[i])
                acc.columns = feature_acc
                gyr = pd.read_csv(workspace / "realworld2016_dataset_organized" / user / "gyr" / filesgyr[i])
                gyr.columns = feature_gyr
                for activity in tarefas:
                    if filesacc[i].find(activity)>-1:
                        break

                if not abs(acc.shape[0]-gyr.shape[0])<200:
                    # Remove todas as linhas dos dataframes
                    acc.drop(acc.index, inplace=True)
                    gyr.drop(gyr.index, inplace=True)

                tam = min(acc.shape[0],gyr.shape[0])
                
                new_acc = acc[feature_acc].iloc[:tam]
                new_gyr = gyr[feature_gyr[1:]].iloc[:tam]
                
                # Criando um dataframe com os dados de aceleração e giroscópio
                df = pd.concat([new_acc, new_gyr], axis=1)
                df['user'] = user
                df['position'] = p
                df['activity code'] = activity
                # df['activity code'] = map[tarefas.index(activity)]

                # Drop samples with NaN
                if df.isnull().values.any():
                    continue
                
                dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    return df

# Function to sanity check the balance of the dataset
def sanity_function(train_df, val_df, test_df):
    train_size = train_df.shape[0]
    val_size = val_df.shape[0]
    test_size = test_df.shape[0]
    total = train_size + val_size + test_size

    print(f"Train size: {train_size} ({train_size/total*100:.2f}%)")
    print(f"Validation size: {val_size} ({val_size/total*100:.2f}%)")
    print(f"Test size: {test_size} ({test_size/total*100:.2f}%)")

    print(f"Train activities: {train_df['standard activity code'].unique()}")
    print(f"Validation activities: {val_df['standard activity code'].unique()}")
    print(f"Test activities: {test_df['standard activity code'].unique()}")

    dataframes = {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df
    }
    for name, df in dataframes.items():
        users = df['user'].unique()
        activities = df['standard activity code'].unique()

        tam = len(df[(df["user"] == users[0]) & (df["standard activity code"] == activities[0])])
        flag = True
        for user in users:
            for activity in activities:
                if len(df[(df["user"] == user) & (df["standard activity code"] == activity)]) != tam:
                    print(f"User {user} has different size for activity {activity}")
                    flag = False
        if flag:
            print(f"All users have the same size per activity in {name} dataset - Samples per user and activity: {tam}")

    users = train_df['user'].unique()
    activities = train_df['standard activity code'].unique()

    print(f"Users in train: {train_df['user'].unique()}")
    print(f"Users in validation: {val_df['user'].unique()}")
    print(f"Users in test: {test_df['user'].unique()}\n")

# Crate pipeline for data processing
pipelines = {
    "KuHar": {
        "raw_dataset":
            Pipeline([
                CalcTimeDiffMean(
                    groupby_column=column_group["KuHar"],            # Agrupa pela coluna do CSV. Os tempos de início e fim são calculados para cada grupo da coluna CSV
                    column_to_diff="accel-start-time",      # Coluna para calcular a diferença
                    new_column_name="timestamp diff",       # Nome da coluna com a diferença
                ),
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=300,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"]
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar"])
            ]),
        "standartized_dataset": 
            Pipeline([
                CalcTimeDiffMean(
                    groupby_column=column_group["KuHar"],            # Agrupa pela coluna do CSV. Os tempos de início e fim são calculados para cada grupo da coluna CSV
                    column_to_diff="accel-start-time",      # Coluna para calcular a diferença
                    new_column_name="timestamp diff",       # Nome da coluna com a diferença
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["KuHar"],
                    up=2,
                    down=10,
                    groupby_column=column_group["KuHar"]
                ),
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"],
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar"])
            ])
    },
    "MotionSense": {
        "raw_dataset": 
            Pipeline([
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(standard_activity_code_map["MotionSense"]),
            ]),
        "standartized_dataset":
            Pipeline([
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                AddGravityColumn(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    gravity_columns=["gravity.x", "gravity.y", "gravity.z"],
                ),
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["MotionSense"],
                    up=2,
                    down=5,
                    groupby_column=column_group["MotionSense"],
                ),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(standard_activity_code_map["MotionSense"]),
            ])
    },
    "WISDM": {
        "raw_dataset":
            Pipeline([
                CalcTimeDiffMean(
                    groupby_column=column_group["WISDM"],            # Agrupa pela coluna do CSV. Os tempos de início e fim são calculados para cada grupo da coluna CSV
                    column_to_diff="timestamp-accel",                # Coluna para calcular a diferença
                    new_column_name='accel-timestamp-diff',          # Nome da coluna com a diferença
                ),
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM"]),
            ]),
        "standartized_dataset": 
            Pipeline([
                CalcTimeDiffMean(
                    groupby_column=column_group["WISDM"],            # Agrupa pela coluna do CSV. Os tempos de início e fim são calculados para cada grupo da coluna CSV
                    column_to_diff="timestamp-accel",                # Coluna para calcular a diferença
                    new_column_name='accel-timestamp-diff',          # Nome da coluna com a diferença
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=20,
                ),
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM"]),
            ])
    },
    "UCI": {
        "raw_dataset":
            Pipeline([
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI"]),
            ]),
        "standartized_dataset": 
            Pipeline([
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["UCI"],
                    up=2,
                    down=5,
                    groupby_column=column_group["UCI"],
                ),
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI"]),
            ]),
    },
    "RealWorld": {
        "raw_dataset":
            Pipeline([
                CalcTimeDiffMean(
                    groupby_column=column_group["RealWorld"],            # Agrupa pela coluna do CSV. Os tempos de início e fim são calculados para cada grupo da coluna CSV
                    column_to_diff="accel-start-time",                # Coluna para calcular a diferença
                    new_column_name="timestamp diff",          # Nome da coluna com a diferença
                ),
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(standard_activity_code_map["RealWorld"]),
            ]),
        "standartized_dataset":
            Pipeline([
                CalcTimeDiffMean(
                    groupby_column=column_group["RealWorld"],            # Agrupa pela coluna do CSV. Os tempos de início e fim são calculados para cada grupo da coluna CSV
                    column_to_diff="accel-start-time",                # Coluna para calcular a diferença
                    new_column_name="timestamp diff",          # Nome da coluna com a diferença
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["RealWorld"],
                    up=2,
                    down=5,
                    groupby_column=column_group["RealWorld"],
                ),
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(standard_activity_code_map["RealWorld"]),
            ]),
    },
}

# Crating a list of functuions to read the datasets
functions = {
    "KuHar": read_kuhar,
    "MotionSense": read_motionsense,
    "WISDM": read_wisdm,
    "UCI": read_uci,
    "RealWorld": read_realworld,
}

dataset_path = {
    "KuHar": "KuHar/1.Raw_time_domian_data",
    "MotionSense": "MotionSense/A_DeviceMotion_data",
    "WISDM": "WISDM/wisdm-dataset/raw/phone",
    "UCI": "UCI/RawData",
    "RealWorld": "RealWorld/realworld2016_dataset",
}

# Preprocess the datasets

# Pathes to save the datasets
output_path_unbalanced = Path("../data/unbalanced")

output_path_balanced = Path("../data/raw_balanced")
output_path_balanced_standartized = Path("../data/standartized_balanced")

output_path_balanced_user = Path("../data/raw_balanced_user")
output_path_balanced_standartized_user = Path("../data/standartized_balanced_user")

balancer_activity = BalanceToMinimumClass(class_column="standard activity code")
balancer_activity_and_user = BalanceToMinimumClassAndUser(class_column="standard activity code", filter_column="user")

split_data = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.8,
    random_state=42,
)

split_data_train_val = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.9,
    random_state=42,
)

def balance_per_activity(dataset, dataframe, output_path):

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity(train_df)
    val_df = balancer_activity(val_df)
    test_df = balancer_activity(test_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Raw data balanced per activity saved at {output_dir}")

    return train_df, val_df, test_df

def balance_per_user_and_activity(dataset, dataframe, output_path):
    new_df_balanced = balancer_activity_and_user(dataframe[dataframe["standard activity code"] != -1])
    train_df, test_df = split_data(new_df_balanced)
    train_df, val_df = split_data_train_val(train_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Raw data balanced per user and activity saved at {output_dir}")

    return train_df, val_df, test_df


def getfiles(user, activity, workspace, root):
    # Essa função vai descompactar os arquivos na pasta junk
    # e em seguida mover os csv para a pasta realworldcsvs
    folder = workspace / "realworld2016_dataset_organized"


    for sensor in ["acc", "gyr"]:
        file = root / user / f"data/{sensor}_{activity}_csv.zip"
        with ZipFile(file, 'r') as zip:
            zip.extractall(workspace / "junk")

        for i in os.listdir(workspace / "junk"):
            if i.find('zip')>-1:
                file = workspace / "junk" / i
                with ZipFile(file, 'r') as zip:
                    zip.extractall(workspace / "junk")

        for files in os.listdir(workspace / "junk"):
            if os.path.isfile(workspace / "junk" / files):
                if files.find(activity)>-1 and files.find('zip')<0:
                    os.rename(workspace / "junk" / files, folder / user / files)
                else:
                    os.remove(workspace / "junk" / files)

        os.rmdir(workspace / "junk")

def real_world_organize():
    # Let's organize the real world dataset
    workspace = Path("../data/processed/RealWorld")
    root = Path("../data/original/RealWorld/realworld2016_dataset")

    # List of users and activities
    users = natsorted(os.listdir(root))
    tarefas = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
    SAC = ['sitting', 'standing', 'walking', 'climbingup', 'climbingdown', 'running']

    # Let's create a folder to unzip the files .zip
    if not os.path.isdir(workspace / "junk"):
        os.makedirs(workspace / "junk")
    os.path.isdir(workspace / "junk")
    # and the same folder to organize the unzipped files in a friendly way
    if not os.path.isdir(workspace / "realworld2016_dataset_organized"):
        os.mkdir(workspace / "realworld2016_dataset_organized")
    os.path.isdir(workspace / "realworld2016_dataset_organized")

    # Let's create folders for each user
    for i in users:
        if not os.path.isdir(workspace / "realworld2016_dataset_organized" / i):
            os.mkdir(workspace / "realworld2016_dataset_organized" / i)

    # Let's interate over the files that we want to unzip
    for user in users:
        for activity in tarefas:
            getfiles(user, activity, workspace, root)
    # Now we will create a folder for the accelerometer and gyroscope data
    for user in users:
        if not os.path.isdir(workspace / "realworld2016_dataset_organized" / user / "acc"):
            os.mkdir(workspace / "realworld2016_dataset_organized" / user / "acc")
        if not os.path.isdir(workspace / "realworld2016_dataset_organized" / user / "gyr"):
            os.mkdir(workspace / "realworld2016_dataset_organized" / user / "gyr")
    # And we'll move the files to the right folder
    for user in users:
        for files in os.listdir(workspace / "realworld2016_dataset_organized" / user):
            if files.find('acc')>-1 and os.path.isfile(workspace / "realworld2016_dataset_organized" / user / files):
                origin = workspace / "realworld2016_dataset_organized" / user / files
                destiny = workspace / "realworld2016_dataset_organized" / user / "acc" / files
                os.rename(origin, destiny)
            if files.find('Gyr')>-1 and os.path.isfile(workspace / "realworld2016_dataset_organized" / user / files):
                origin = workspace / "realworld2016_dataset_organized" / user / files
                destiny = workspace / "realworld2016_dataset_organized" / user / "gyr" / files
                os.rename(origin, destiny)
    # Let's verify if the folders have the same number of files
    flag = 1
    for user in users:
        files_acc = os.listdir(workspace / "realworld2016_dataset_organized" / user / "acc")
        files_gyr = os.listdir(workspace / "realworld2016_dataset_organized" / user / "gyr")
        if len(files_acc) != len(files_gyr):
            flag = 0
            print(f"User {user} has {len(files_acc)} acc files and {len(files_gyr)} gyr files")
            flag = -1
    if flag == 1:
        print("All users have the same number of acc and gyr files")

    return workspace, users

def generate_views(new_df, new_df_standartized, dataset):

    # Filter the datasets by equal elements
    filter_common = FilterByCommonRows(match_columns=match_columns[dataset])
    new_df, new_df_standartized = filter_common(new_df, new_df_standartized)

    # Save the unbalanced dataset
    output_dir = output_path_unbalanced / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_dir / "unbalanced.csv", index=False)

    # Preprocess and save the raw balanced dataset per user and activity
    train_df, val_df, test_df = balance_per_user_and_activity(dataset, new_df, output_path_balanced_user)
    sanity_function(train_df, val_df, test_df)

    #Preprocess and save the raw balanced dataset per activity
    train_df, val_df, test_df = balance_per_activity(dataset, new_df, output_path_balanced)
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the standartized balanced dataset per user and activity
    train_df, val_df, test_df = balance_per_user_and_activity(dataset, new_df_standartized, output_path_balanced_standartized_user)
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the standartized balanced dataset per activity
    train_df, val_df, test_df = balance_per_activity(dataset, new_df_standartized, output_path_balanced_standartized)
    sanity_function(train_df, val_df, test_df)

# Creating the datasets
for dataset in datasets:

    # Verify if the dataset is already created
    if os.path.isdir(output_path_unbalanced / dataset):
        print(f"The dataset {dataset} is already created")

    else:
        print(f"Preprocess the dataset {dataset} ...\n")

        reader = functions[dataset]

        # Read the raw dataset
        # Organize the RealWorld dataset
        if dataset == "RealWorld":
            print("Organizing the RealWorld dataset ...\n")
            workspace, users = real_world_organize()
            path = workspace
            raw_dataset = reader(path, users)
            # Preprocess the raw dataset
            new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
            # Preprocess the standartized dataset
            new_df_standartized = pipelines[dataset]["standartized_dataset"](raw_dataset)
            # Remove activities that are equal to -1
            new_df = new_df[new_df["standard activity code"] != -1]
            new_df_standartized = new_df_standartized[new_df_standartized["standard activity code"] != -1]
            generate_views(new_df, new_df_standartized, dataset)
            positions = new_df["position"].unique()
            for position in list(positions):
                new_df_filtered = new_df[new_df["position"] == position]
                new_df_standartized_filtered = new_df_standartized[new_df_standartized["position"] == position]
                new_dataset = dataset + "_" + position
                generate_views(new_df_filtered, new_df_standartized_filtered, new_dataset)

        else:
            path = Path(f"../data/original/{dataset_path[dataset]}")
            raw_dataset = reader(path)
            # Preprocess the raw dataset
            new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
            # Preprocess the standartized dataset
            new_df_standartized = pipelines[dataset]["standartized_dataset"](raw_dataset)
            # Remove activities that are equal to -1
            new_df = new_df[new_df["standard activity code"] != -1]
            new_df_standartized = new_df_standartized[new_df_standartized["standard activity code"] != -1]
            generate_views(new_df, new_df_standartized, dataset)


# Remove the junk folder
workspace = Path("../data/processed")
if os.path.isdir(workspace):
    shutil.rmtree(workspace)