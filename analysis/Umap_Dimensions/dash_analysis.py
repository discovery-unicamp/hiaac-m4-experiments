import numpy as np
import pandas as pd
import json 
import plotly.express as px
from plotly.subplots import make_subplots
# from pyngrok import ngrok

from dash import dash_table, Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output, State

# import dash_core_components as dash_core
# import dash_html_components as dash_html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

df['id'] = df['country']
df.set_index('id', inplace=True, drop=False)

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
table_columns = [
    'Classifier',
    'Dataset'
]
for i in range(1,11):
    table_columns.append(f'Umap dimension - {i}')
table_columns.append('Original dimension')
table_columns.append('Umap dimension - Max')
table_columns.append('Umap dimension - Max score')
table_columns.append('Umap dimension - Best dimension - 1:10')
table_columns.append('Loss %')

domains = ['Time', 'Frequency']
lost_results = {
    "Classifier":[],
    "Metric": [],
    "DataSet": [],
    "Best UMAP dimension": [],
    "Lost %": []
}
def analysis(domain, result_df, metric):
    if domain == 'Time':
        end = 360
    elif(domain == 'Frequency'):
        end = 180
    else:
        print('Error')
        return
    
    datasets = list(result_df['Dataset'].unique())
    
    columns = {column:[] for column in table_columns}

    classifiers = list(result_df['Classifier'].unique())
    dimensions = [i for i in range(1,11)] + [end]

    for dataset in datasets:
        for classifier in classifiers:
            columns['Classifier'].append(classifier)
            columns['Dataset'].append(dataset)
            for dimension in dimensions:
                result_filtered = result_df.loc[(result_df['Classifier'] == classifier) & (result_df['Dataset'] == dataset) & (result_df['Umap dimension'] == dimension)]
                final_score = result_df.loc[(result_df['Classifier'] == classifier) & (result_df['Dataset'] == dataset) & (result_df['Umap dimension'] == end)][metric][0]
                columns[f'Original dimension'].append(result_filtered[metric][0])
            scores = list(result_df.loc[(result_df['Dataset'] == dataset) & (result_df['Classifier'] == classifier)][metric])
            best_dimension = scores.index(max(scores[:10]))
            columns[f'Umap dimension - Best dimension - 1:10'].append(best_dimension)
            result_filtered = result_df.loc[(result_df['Classifier'] == classifier) & (result_df['Dataset'] == dataset)]
            data = list(result_filtered[metric])[:-1]
            columns[f'Umap dimension - Max score'].append(max(data))
            columns[f'Umap dimension - Max'].append(data.index(max(data))+1)

    df = pd.DataFrame(columns)
    df['Lost %'] = df[f'Umap dimension - {end}'] - df[f'Umap dimension - Max score']
    
    return df

def generate_fig(result_filtered, dataset, metric, domain):

    min_x = 1
    if domain == 'Time':
        max_x = 360
        step = 5

    elif domain == 'Frequency':
        max_x = 180
        step = 1

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
    dash_table.DataTable(
        id='table-info',
        columns=(
            [{'id': p, 'name': p} for p in table_columns]
        ),
        data=[
            # dict(Model=i, **{param: 0 for param in params}) for i in range(1, 5)
        ],
        editable=True
    )
])

@app.callback(
    Output("graph", "figure"),
    Output("table-info", "data"),
    Input("metric", "value"),
    Input("domain", "value"),
    Input("classifier", "value"),
    Input("dataset", "value")
)

def update_chart(metric, domain, classifier, dataset):

    if domain == 'Time':
        Root = '../../experiments/Umap_Dimensions/results/results_df_umap_dimension_time.json' 
        max_x = 360
        step = 5

    elif domain == 'Frequency':
        Root = '../../experiments/Umap_Dimensions/results/results_df_umap_dimension_frequency.json' 
        max_x = 180
        step = 2

    with open(Root, 'r') as f:
        result_load = json.load(f)
    
    result = pd.DataFrame(result_load)
    if classifier != 'All':
        result = result.loc[result['Classifier'] == classifier]
    result_filtered = result[features]
    print(result_filtered)
    # print([col for col in result_filtered.columns])
    fig = generate_fig(result_filtered, dataset, metric, domain)
    
    filter_for_table = result_filtered.copy()
    if dataset != 'All':
        filter_for_table = filter_for_table[filter_for_table['Dataset'] == dataset]
    data = []
    for row_tuple in filter_for_table.itertuples(index=False, name=None):
        obj = {}
        for feature_index in range(len(features)):
            obj[features[feature_index]] = row_tuple[feature_index]
        data.append(obj)

    # data = analysis(domain, filter_for_table, metric).to_dict()
    return fig, data
 
if __name__ == '__main__':
    app.run_server(port=8050, debug=True, use_reloader=True)