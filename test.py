import dash
# from dash import html
import plotly.express as px
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def dummy_image():
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                ], dtype=np.uint8)
    
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = dash.html.Div(
    [
        dash.html.Div([
            dash.dcc.Graph(
                id="bar_chart",
                figure = px.scatter(x=np.random.rand(10), y=np.random.rand(10),
                        opacity = 0.4,
                title = 'UMAP Embedding for',
                height = 900
                ),
            )
        ], style={'width': '60%', 'display': 'inline-block'}),
        dash.html.Div([
            # dash.html.Div([
                dash.html.H2(children='Hello Dash', style = {}),
                dash.html.Div(id='graph_heading', children='file: ...'),
                dash.html.Button(id="button1", children="Click me for sound", 
                                n_clicks = 0),
                dash.dcc.Graph(	id="table_container", figure = px.imshow(
                    dummy_image(), height = 700)),
            # ]),
        ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top'})
    ]
)
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),
#     html.Div([
#     html.Audio(src = "/home/vincent/Code/workshop/data_exploration/info_achim.mp3")
#     ])
#     # html.Audio(src = "https://drive.google.com/file/d/1pNOpPFOXEfbyTkVcd_9LW4Wk7iKw4Dop/view?usp=sharing", 
#     #            controls=True)
#     # html.Audio(src='url/amazon/s3/bucket/%s.wav' % hover_data_index, controls=False, autoplay=True)
# ])

if __name__ == '__main__':
    app.run_server(debug=True)