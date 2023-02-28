# Importing the libraries
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px

from IPython.display import display, HTML
import plotly.graph_objects as go

import json 
from dash import dash_table, Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = Dash(__name__, external_stylesheets=external_stylesheets)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


# Let's define some constants
datasets = [
    "All",
    "kuhar",
    "motionsense",
    "uci",
    "wisdm",
    "realworld"
]

domains = [
    "All",
    "FFT-centered",
    "Time",
]

labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}

# Let's define some filters
def only_fft(df):
    return df[df["transforms"].str.contains("FFT")]


def only_time(df):
    return df.loc[df["transforms"] == ""]


def using_all_features(df):
    return df[
        df["in_use_features"].str.contains("accel-x")
        & df["in_use_features"].str.contains("accel-y")
        & df["in_use_features"].str.contains("accel-z")
        & df["in_use_features"].str.contains("gyro-x")
        & df["in_use_features"].str.contains("gyro-y")
        & df["in_use_features"].str.contains("gyro-z")
    ]


def using_only_accel(df):
    return df[
        df["in_use_features"].str.contains("accel-x")
        & df["in_use_features"].str.contains("accel-y")
        & df["in_use_features"].str.contains("accel-z")
        & ~df["in_use_features"].str.contains("gyro-x")
        & ~df["in_use_features"].str.contains("gyro-y")
        & ~df["in_use_features"].str.contains("gyro-z")
    ]


def using_only_gyro(df):
    return df[
        ~df["in_use_features"].str.contains("accel-x")
        & ~df["in_use_features"].str.contains("accel-y")
        & ~df["in_use_features"].str.contains("accel-z")
        & df["in_use_features"].str.contains("gyro-x")
        & df["in_use_features"].str.contains("gyro-y")
        & df["in_use_features"].str.contains("gyro-z")
    ]


def only_reduce_on_all(df):
    return df[df["reduce_on"] == "all"]


def only_reduce_on_sensor(df):
    return df[df["reduce_on"] == "sensor"]


def only_reduce_on_axis(df):
    return df[df["reduce_on"] == "axis"]


def only_rf(df):
    return df[df["estimator"].str.lower().str.contains("randomforest")]


def only_svm(df):
    return df[df["estimator"].str.lower().str.contains("svm")]


def only_knn(df):
    return df[df["estimator"].str.lower().str.contains("knn")]


def no_scaler(df):
    return df[df["scaler"] == ""]

def min_max_scaler(df):
    return df[df["scaler"].str.lower().str.contains("minmaxscaler")]

def standard_scaler(df):
   return df[df["scaler"].str.lower().str.contains("standardscaler")]

def only_reducer_equals_train(df):
    return df[df["reducer_datasets"] == df["train_datasets"]]

def only_reducer_equals_train_or_no_reduce(df):
    return df[(df["reducer_datasets"] == df["train_datasets"]) | (df["reducer_datasets"] == "")]


def rename_datasets(
    df, columns: List[str] = ("reducer_datasets", "train_datasets", "test_datasets")
):
    def rename_row(row):
        for col in columns:
            names = set()
            for name in row[col].split(","):
                name = name.strip()
                names.add(name.split(".")[0])
            row[col] = ", ".join(sorted(names))
        return row

    df = df.apply(rename_row, axis=1)
    return df


def add_view_name(df, new_column_name: str = "view"):
    df[new_column_name] = df["config_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
    return df


def match_configs(df, new_column_name: str = "config_group"):
    group_no = 0
    for k, subdf in df.groupby(
        [
            "in_use_features",
            "scale_on",
            "reduce_on",
            "transforms",
            "scaler",
            "reducer",
            "umap components",
            "reducer_datasets",
            "train_datasets",
            "test_datasets",
            "estimator",
        ]
    ):
        if len(subdf) == 2:
            df.loc[subdf.index, new_column_name] = group_no
            group_no += 1
    return df

def only_standardized_view(df):
    return df[df["view"] == "standartized_intra_balanced"]


def only_raw_view(df):
    return df[df["view"] == "raw_balanced"]


# x = improvement_over_baseline(results.copy())

# Read the results
results_file = Path("results.csv")
results = pd.read_csv(results_file).fillna("")

results[(results["umap components"] == 0) & (results["transforms"] == "")] == 360
results[(results["umap components"] == 0) & (results["transforms"] == "FFT-centered")] == 180

# Preprocess the results
results = rename_datasets(results)
results = add_view_name(results)
results = match_configs(results)

classifiers = ['All']
classifiers += list(results['estimator'].unique())

# Create the aplicaiton
controls = dbc.Card(
    [
    #     dbc.FormGroup(
    #         [
    #             dbc.Label("Classifier"),
    #             dcc.RadioItems(
    #                 id="classifier",
    #                 inline=True,
    #                 options=classifiers,
    #                 value="All"
    #             ),
    #         ]
    #     ),
    #     dbc.FormGroup(
    #         [
    #             dbc.Label("Dataset"),
    #             dcc.RadioItems(
    #                 id="dataset",
    #                 inline=True,
    #                 options=datasets,
    #                 value="kuhar"
    #             ),
    #         ]
    #     ),
    # ],
        html.Div([
        # html.H1(children='Results'),
        # html.H6(children='Add a description here'),
        html.H2(children='Pergunta 1: Qual é o impacto do UMAP na capacidade de discriminação dos modelos de ML na tarefa de HAR?'),
        html.H6(children='1.1. O desempenho dos 3 modelos de ML com o experimento realizado com e sem o UMAP'),
        html.H6(children='1.2. O impacto da normatização no resultado'),
        
        html.H4('Estimator: '),
        dcc.RadioItems(
            id='classifier',
            inline=True,
            options=classifiers,
            value='All'
        ),

        html.H4('DataSet: '),
        dcc.RadioItems(
            id='dataset',
            inline=True,
            options=datasets,
            value="All"
        ),
        html.H4('Domain: '),
        dcc.RadioItems(
            id='domain',
            inline=True,
            options=domains,
            value="All"
        ),
        # dcc.Graph(id="graph"),
    ]),

])

app.layout = dbc.Container(
    [
        html.H1("Perguntas de pesquisa"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=14),
                dbc.Col(dcc.Graph(id="graph"), md=10),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    Output("graph", "figure"),
    Input("classifier", "value"),
    Input("dataset", "value"),
    Input("domain", "value"),
)

def improvement_over_baseline(classifier, dataset, domain):

    df = results.copy()
    df = only_reduce_on_all(df)
    df = using_all_features(df)
    df = no_scaler(df)
    df = only_reducer_equals_train_or_no_reduce(df)
    df = only_standardized_view(df)

    if classifier != "All":
        df = df[df["estimator"] == classifier]

    if dataset != "All":
        df = df[(df["train_datasets"] == dataset) & (df["test_datasets"] == dataset)]

    if domain != "All":
        df = df[df["transforms"] == domain] if domain == "FFT-centered" else df[df["transforms"] != "FFT-centered"]

    fig = go.Figure()
    for (estimator, transform, dataset_name), subdf in df.groupby(
        ["estimator", "transforms", "train_datasets"]
    ):
        # subdf = subdf[
        #     ~(subdf["reducer_datasets"] == "")
        #     | (subdf["umap components"] == 0)
        # ]
        subdf = subdf.sort_values("umap components")
        transform = transform if transform else 'Time'
        fig.add_trace(
            go.Scatter(
                x=subdf["umap components"],
                y=subdf["accuracy (mean)"]*100,
                mode="lines+markers",
                name=f"{estimator} {transform if transform else 'Time'} {dataset_name}",
                legendgroup=transform,
                error_y=dict(
                    type="data",
                    array=subdf["accuracy (std)"]*100,
                    visible=True,
                    color="black",
                    thickness=1,
                    width=2,
                ),
            )
        )

    fig.update_layout(
        title=f"Datasets: {dataset}",
        xaxis_title="UMAP components",
        yaxis_title="Accuracy (mean 10 runs)",
        yaxis_range=[0, 100],
        xaxis = dict(
            tickmode = 'linear',
        ),
        autosize=True,
        width=1400,
        height=700,
    )
    fig.update_layout(hovermode="x unified")
#     fig.update_layout(plot_bgcolor = 'white',
# #     font = {'family': 'Arial','size': 16,'color': 'black'},
#     colorway=["red", "green", "blue"])
    return fig
 
if __name__ == '__main__':
    app.run_server(port=8050, debug=True, use_reloader=True)