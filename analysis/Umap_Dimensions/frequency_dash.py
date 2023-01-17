import numpy as np
import pandas as pd
import json 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from pyngrok import ngrok

from dash import Dash, dcc, html, Input, Output, State, callback_context
# from jupyter_dash import JupyterDash








Root = '../../experiments/Umap_Dimensions/'
with open(Root+'results/results_df_umap_dimension_frequency.json', 'r') as f:
    result = json.load(f)
result = pd.DataFrame(result)

result_filtered = result[['Classifier', 'Umap dimension', 'Dataset', 'accuracy - mean', 'accuracy - std',
                          'f1 score (weighted) - mean', 'f1 score (weighted) - std']]

datasets = list(result_filtered['Dataset'].unique())

#gerando subplots
rows=5
subplot_titles = tuple(dataset for dataset in datasets)
min_x, max_x = 0, 180
metric = "accuracy - mean"
error = "accuracy - std"

for k in range(rows):
    dataset = datasets[k]
    result = result_filtered.loc[result_filtered['Dataset'] == dataset]
    
    fig = px.line(
        result, 
        x="Umap dimension", 
        y=metric, 
        color="Classifier", 
        error_y=error,
        range_x=[min_x, max_x],
        range_y=[0, 1]
    )
    
    fig.update_xaxes( title_text = "Umap dimension", showgrid=True, gridwidth=1, gridcolor='lightgray', 
                     showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes( title_text = metric, showgrid=True, gridwidth=1, gridcolor='lightgray', 
                     showline=True, linewidth=1, linecolor='black')
    fig.update_layout(hovermode="x unified")

# o comando fig.update_layout nos permite alterar o layout do gráfico
    fig.update_layout(plot_bgcolor = 'white',
#     font = {'family': 'Arial','size': 16,'color': 'black'},
    colorway=["red", "green", "blue"])
    fig.update_layout(title_text="Frequency: "+dataset, )
    fig.write_image(f'results/png/umap_dimensions_{dataset}_{metric}_frequency.png')
    fig.write_image(f'results/svg/umap_dimensions_{dataset}_{metric}_frequency.svg')
    fig.write_html(f'results/html/umap_dimensions_{dataset}_{metric}_frequency.html')
    fig.write_json(f'results/json/umap_dimensions_{dataset}_{metric}_frequency.json')
    fig.show()

#gerando subplots
rows=5
subplot_titles = tuple(dataset for dataset in datasets)
min_x, max_x = 0, 180
metric = "f1 score (weighted) - mean"
error = "f1 score (weighted) - std"

for k in range(rows):
    dataset = datasets[k]
    result = result_filtered.loc[result_filtered['Dataset'] == dataset]
    
    fig = px.line(
        result, 
        x="Umap dimension", 
        y=metric, 
        color="Classifier", 
        error_y=error,
        range_x=[min_x, max_x],
        range_y=[0, 1]
    )
    
    fig.update_xaxes( title_text = "Umap dimension", showgrid=True, gridwidth=1, gridcolor='lightgray', 
                     showline=True, linewidth=1, linecolor='black', rangemode='tozero')
    fig.update_yaxes( title_text = metric, showgrid=True, gridwidth=1, gridcolor='lightgray', 
                     showline=True, linewidth=1, linecolor='black')
    fig.update_layout(hovermode="x unified")

# o comando fig.update_layout nos permite alterar o layout do gráfico
    fig.update_layout(plot_bgcolor = 'white',
#     font = {'family': 'Arial','size': 16,'color': 'black'},
    colorway=["red", "green", "blue"])
    fig.update_layout(title_text="Frequency: "+dataset, title_x=0.5)
    fig.write_image(f'results/png/umap_dimensions_{dataset}_{metric}_frequency.png')
    fig.write_image(f'results/svg/umap_dimensions_{dataset}_{metric}_frequency.svg')
    fig.write_html(f'results/html/umap_dimensions_{dataset}_{metric}_frequency.html')
    fig.write_json(f'results/json/umap_dimensions_{dataset}_{metric}_frequency.json')
    fig.show()

lost_results = {
    "Classifier":[],
    "Metric": [],
    "DataSet": [],
    "Best UMAP dimension": [],
    "Lost": []
}
erros = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,  0.1]
for dataset in datasets:
    result = result_filtered.loc[result_filtered['Dataset'] == dataset]
    
    for classifier in ['RandomForest', 'SVC', 'KNN']:
        for lost in erros:
            for metric in ["accuracy - mean", "f1 score (weighted) - mean"]:
                datas = result.loc[result['Classifier'] == classifier]
                data = np.array(datas[metric])
                best_dimension = 0
                for i in range(180):
                    if data[i] - data[best_dimension] >= lost:
                        best_dimension = i
                best_dimension+=1
                lost_results['Classifier'].append(classifier)
                lost_results['Metric'].append(metric)
                lost_results['DataSet'].append(dataset)
                lost_results["Best UMAP dimension"].append(best_dimension)
                lost_results['Lost'].append(lost)
lost_results_df = pd.DataFrame(lost_results)
lost_results_df.loc[(lost_results_df['Metric'] =='f1 score (weighted) - mean') & (lost_results_df['Lost'] == 0.03)]

##################################### Creating a Dash app ###################################################3

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

app = Dash(__name__)

app.layout = html.Div([
    html.H2('Umap Results Analysis'),
])


#     html.Div(children='O gŕafico abaixo apresenta uma visualização dos resultados dos experimentos cross datasets com a identificação do conjunto de dados utilizados para o umap (que reduz a dimensão do dado para 10), treino, teste, o tipo de métrica desejada e classificador.'),   
#     html.H4('Métrica'),
#     dcc.Dropdown(id="metric", options=metrics, value='accuracy - mean'),
#     html.H4('Classificador: '),
#     dcc.RadioItems(
#         id='classifier',
#         inline=True,
#         options=classifiers,
#         value='All'
#     ),
#     html.H4('Dados de Teste:'),
#     dcc.RadioItems(
#         id='checklist',
#         inline=True,
#         options=datasets,
#         value='All'
#     ),
#     dcc.Graph(id="graph"),
    
# ])

# @app.callback(
#     Output("graph", "figure"),
#     Input("checklist", "value"),
#     Input("metric", "value"),
#     Input("classifier", "value")
# )
    
# def update_chart(test_data, metric, classifier):
    
#     new_df_filtered = df_filtered if classifier =='All' else df_filtered[df_filtered["Classifier"].isin([classifier])]            
#     new_df_filtered = new_df_filtered if test_data =='All' else new_df_filtered[new_df_filtered["Test"].isin([test_data])]          
#     new_df_filtered = new_df_filtered.sort_values(by=metric, ascending=True)
#     fig = px.scatter(
#         new_df_filtered, 
#         x="Train", 
#         y=metric, 
#         color="Umap - 10", #hover_name="Test", 
#         symbol="Classifier", 
#         hover_data=hover_data1 if metric == 'f1 score (weighted) - mean' 
#         else hover_data2 if metric == 'f1-score - mean - without stair up/down' 
#         else 
#         {
#             'Classifier': True,
#             'Umap - 10': True,
#             'Train': True,
#             'Test':True,
#             f'{metric}': ':.2f',
#         }
#     )
    
#     return fig

if __name__ == "__main__":
    app.run_server(port=8050, debug=True, use_reloader=False, mode = "external")