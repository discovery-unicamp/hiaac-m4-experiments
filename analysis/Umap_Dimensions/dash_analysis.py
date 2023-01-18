import numpy as np
import pandas as pd
import json 
import plotly.express as px
from plotly.subplots import make_subplots
# from pyngrok import ngrok

from dash import Dash, dcc, html, Input, Output, State, callback_context
# import dash_table
# import dash_core_components as dash_core
# import dash_html_components as dash_html
# from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
# 

# Inicialize some variables

datasets = ['All', 'KuHar', 'RealWorld', 'MotionSense', 'WISDM', 'UCI']
# datasets = ['KuHar', 'RealWorld', 'MotionSense', 'WISDM', 'UCI']

classifiers = [
    'All',
    'RandomForest',
    'SVC',
    'KNN'
]
metrics = [
    'accuracy - mean',
    'f1 score (weighted) - mean',
    'f1-score - mean - sit', 
    'f1-score - mean - stand',
    'f1-score - mean - walk', 
    'f1-score - mean - stair up',
    'f1-score - mean - stair down', 
    'f1-score - mean - run',
    'f1-score - mean - stair up and down',
]
features = [
    'Classifier', 
    'Umap dimension', 
    'Dataset',
    'accuracy - mean',
    'f1 score (weighted) - mean',
    'f1-score - mean - sit', 
    'f1-score - mean - stand',
    'f1-score - mean - walk', 
    'f1-score - mean - stair up',
    'f1-score - mean - stair down', 
    'f1-score - mean - run',
    'f1-score - mean - stair up and down',
]
domains = ['Time', 'Frequency']
losts = [i*0.01 for i in range(21)]

def generate_fig(result_filtered, dataset, metric, domain):

    min_x = 1
    if domain == 'Time':
        max_x = 360
        step = 5

    elif domain == 'Frequency':
        max_x = 180
        step = 2

    fig = make_subplots(rows=1, cols=1)

    if dataset == 'All':
        result = result_filtered.loc[result_filtered['Umap dimension'] % step == 0]
        result = pd.concat([result_filtered.loc[result_filtered['Umap dimension'] == 1], result])

        fig = px.line(
            result, 
            x="Umap dimension", 
            y=metric, 
            color="Dataset", #hover_name="Test", 
            symbol="Classifier", 
            range_x=[min_x, max_x],
            range_y=[0, 1]
        )

    else:
        result_aux = result_filtered.loc[result_filtered['Dataset'] == dataset]
        result = result_aux.loc[result_aux['Umap dimension'] % step == 0]
        result = pd.concat([result_aux.loc[result_aux['Umap dimension'] == 1], result])

        fig = px.line(
            result, 
            x="Umap dimension", 
            y=metric, 
            color="Classifier", 
            # error_y=error,
            range_x=[min_x, max_x],
            range_y=[0, 1]
        )
        
    fig.update_xaxes( title_text = "Umap dimension", showgrid=True, gridwidth=1, gridcolor='lightgray', 
                    showline=True, linewidth=1, linecolor='black', rangemode='tozero')
    fig.update_yaxes( title_text = metric, showgrid=True, gridwidth=1, gridcolor='lightgray', 
                    showline=True, linewidth=1, linecolor='black')
    fig.update_layout(hovermode="x unified")
    fig.update_layout(plot_bgcolor = 'white',
#     font = {'family': 'Arial','size': 16,'color': 'black'},
    colorway=["red", "green", "blue"])
    fig.update_layout(title_text="Time: "+dataset, title_x=0.5)

    return fig

# Create the application

app.layout = html.Div([
    html.H1('Umap Results Analysis'),
    html.H6('This dashboord is responsible to show results from umap dimensions results. The experients is based in reduce the dimenson data with Umap from time and frequency domain from 5 diferents datasets (KuHar, RealWorld, MotionSense, WISDM, and UCI) and find the lowest dimension that we can reduce the data and lost the minimun of classifier\' perfomance. The chart below represent a resume from results and you can select some parameters selecting the options below.'),
    # justify="center",
    html.H4('Metric: '),
    dcc.Dropdown(id="metric", options=metrics, value='accuracy - mean'),
    
    html.H4('Domain: '),
    dcc.RadioItems(
        id='domain',
        inline=True,
        options=domains,
        value='Time'
    ),

    html.H4('Classifier: '),
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
    dcc.Graph(id="graph"),

])

@app.callback(
    Output("graph", "figure"),
    Input("metric", "value"),
    Input("domain", "value"),
    Input("classifier", "value"),
    Input("dataset", "value")
)

def update_chart(metric, domain, classifier, dataset):

    if domain == 'Time':
        Root = '/home/patrick/Documents/Repositories/hiaac-m4-experiments/experiments/Umap_Dimensions/results/results_df_umap_dimension_time.json' 
        max_x = 360
        step = 5

    elif domain == 'Frequency':
        Root = '/home/patrick/Documents/Repositories/hiaac-m4-experiments/experiments/Umap_Dimensions/results/results_df_umap_dimension_frequency.json' 
        max_x = 180
        step = 2

    with open(Root, 'r') as f:
        result_load = json.load(f)
    
    result = pd.DataFrame(result_load)
    if classifier != 'All':
        result = result.loc[result['Classifier'] == classifier]
    result_filtered = result[features]

    fig = generate_fig(result_filtered, dataset, metric, domain)

    return fig
 
if __name__ == '__main__':
    app.run_server(port=8050, debug=True, use_reloader=True)