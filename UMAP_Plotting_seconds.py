# Written by Avery Bick, June 2021
# Adapted from UMAP documentation: https://umap-learn.readthedocs.io/en/latest/supervised.html
"""
Apply UMAP dimensionality reduction to Audioset embeddings
"""

import umap
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import sys
from itertools import chain
from datetime import date, time, datetime
import glob
import librosa as lb
from librosa.display import specshow
 
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

def splitEmbeddingsByColumnValue(df, column):
	dfs = [d for _, d in df.groupby([column])]
	values = df[column].unique()
	return(dfs, values)

def createTimeLabelsList(length, percentiles):
	outList = []
	labelCount = int(length/percentiles)

	for percentile in list(range(1,percentiles)):
		tmpList = [percentile]*labelCount
		outList.extend(tmpList)

	if len(outList) < length:
		nextNumber = outList[-1]+1
		print(nextNumber)
		while len(outList) < length:
			outList.append(nextNumber)

	if len(outList) > length:
		while len(outList) > length:
			outList = outList[:-1]

	outArray = np.array(outList)

	return(outArray)

def calculateCentroids_Continuous(embeddings, timeLabels):
	classCentroids = []
	arr = np.column_stack((embeddings,timeLabels))
	split_arrs = np.split(arr, np.where(np.diff(arr[:,2]))[0]+1)
	for array in split_arrs:
		length = array.shape[0]
		sum_x = np.sum(array[:, 0])
		sum_y = np.sum(array[:, 1])
		centroid = [sum_x/length,sum_y/length]
		classCentroids.append(centroid)
	outArray = np.array(classCentroids)
	return outArray

def calculateCentroids_Classes(l):
	array = np.array(l)
	length = array.shape[0]
	sum_x = np.sum(array[:, 0])
	sum_y = np.sum(array[:, 1])
	centroid = [sum_x/length,sum_y/length]
	return centroid

def plot_wo_specs(data, timeLabels, title, centroids, classes):
    fig = px.scatter(data, x='x', y='y', color=timeLabels, opacity = 0.4,
                    hover_data = ['time_within_file', 'filename'],
                    title = 'UMAP Embedding for {}'.format(title))
    fig.add_trace(
        go.Scatter(
            x = centroids[:,0], y= centroids[:,1], mode = 'markers',
            marker = dict(
                color = classes,
                size = [20]*10
            ) ) )
    
    fig.show()
    fig.write_html('Hydro22.html')
 
def plotUMAP_Continuous_plotly(audioEmbeddingsList, percentiles, title, 
                        colormap, files, lengths, classNames=None):

    embeddings = umap.UMAP(n_neighbors=10).fit_transform(audioEmbeddingsList)
    timeLabels = createTimeLabelsList(len(audioEmbeddingsList), percentiles)
    classes = list(set(timeLabels))
    if classNames == None:
        classNames = classes

    centroids = calculateCentroids_Continuous(embeddings,timeLabels)

    embeddings = np.array(embeddings)
    centroids = np.array(centroids)

    lin_array = np.linspace(0, 599, lengths[2])
    # divisions = lin_array // 60 + np.mod(lin_array, 60)/100
    # divisions = np.round(divisions, 3)
    # div_strings = divisions.astype(str)

    img_rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                    ], dtype=np.uint8)
    
    
    files_array = []
    divisions_array = []
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            files_array.append(files[i])
            divisions_array.append(
                f'{int(lin_array[j]/60)}:{np.mod(lin_array[j], 60):.1f}s')
    
    
    data = pd.DataFrame({'x' : embeddings[:,0], 'y':embeddings[:,1],
                        'time_within_file' : divisions_array,
                        'filename' : files_array})

    app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
    app.layout = dash.html.Div(
        [
            dash.html.Div([
                dash.dcc.Graph(
                    id="bar_chart",
                    figure = px.scatter(data, x='x', y='y', color=timeLabels, 
                            opacity = 0.4,
                    hover_data = ['time_within_file', 'filename'],
                    title = 'UMAP Embedding for {}'.format(title),
                    height = 900
                    ),
                )
            ], style={'width': '70%', 'display': 'inline-block'}),
            dash.html.Div([
                # dash.html.Div([
                    dash.dcc.Graph(	
                    id="table_container",
                    figure = px.imshow(
                        img_rgb, height = 700))
                # ]),
                # dash.html.Div([
                # 	dash.html.Button(id="button1", children="Click me for sound"),
                # 	dash.html.Audio(id='audio-player', src=''),
                #           controls=True,
                #           autoPlay=False,
                # 	])
            ], style={'width': '25%', 'display': 'inline-block'})
                    
        ]
    )

    @app.callback(
        Output("table_container", "figure"),
        Input("bar_chart", "clickData"))
    def fig_click(clickData):
        if not clickData:
            pass

        data = clickData['points'][0]

        return 	create_specs(data['customdata'][1], data['customdata'][0])


    app.run_server(debug = False)
 
def create_specs(file, time):
	# def plot_spec(spec_data, file_path, prediction, start,
	#                       fft_window_length, sr, cntxt_wn_sz, fmin, fmax,
	#                       mod_name, noise=False, **_):
	file_stem = file.split('data')[-1].split('.pickle')[0][1:]
	main_path = '/home/vincent/Music/conf/'
 
	minu = int(time.split(':')[0])
	sec = int(time.split(':')[1].split('.')[0])/60
	ttt = minu+sec
	data, sr = lb.load(main_path + file_stem, sr=16000, offset=ttt, duration = 1)
	# fig, ax = plt.subplots(figsize = [6, 4])
	S = np.abs(lb.stft(data, win_length = 128))
    # fig, ax = plt.subplots(figsize = [6, 4])
    # # limit S first dimension from [10:256], thatway isolating frequencies
    # # (sr/2)/1025*10 = 48.78 to (sr/2)/1025*266 = 1297.56 for visualization
    # fmin = sr/2/S.shape[0]*10
    # fmax = sr/2/S.shape[0]*266
	S_dB = lb.amplitude_to_db(S, ref=np.max)
	# img = specshow(S_dB, x_axis = 's', y_axis = 'linear', 
	# 				sr = sr, win_length = 1024, ax=ax) 
				# x_coords = np.linspace(0, cntxt_wn_sz/sr, spec_data.shape[1]),
				# 	y_coords = np.linspace(fmin, fmax, spec_data.shape[0]))
	# fig.colorbar(img, ax=ax, format='%+2.1f dB')
	# file_name = f'{Path(file_path).stem}_spec_w_label.png'
	# ax.set(title=f'spec. of random sample | prediction: {prediction:.4f}\n'\
	# 		f'file: {Path(file_path).stem}.wav | start = {get_time(start)}')
	fig = px.imshow(S_dB, origin='lower', 
			title = "Spectrogram of file <br>"+
   					f"{file_stem} <br>" +
					f"at {time}",
   			labels = {'x' : 'time in ms', 'y' : 'frequency in Hz'},
      		height = 700)
	return fig


def get_embeddings():
    folders = glob.glob('../data/*Moth*')
    df_all = pd.DataFrame()
    file_list= []
    lenghts = []
    for folder in folders:
        files = glob.glob(folder + '/*.pickle')
        files = np.sort(files).astype(list)
        for file in files:
            with open(file, 'rb') as savef:
                audioEmbeddings = pickle.load(savef)
                # print(audioEmbeddings[0])
                audioEmbeddingsList = [arr.tolist() for arr in audioEmbeddings]
                df = pd.DataFrame({'embeddings':audioEmbeddingsList})
                print(df)
                '''
                audioEmbeddings, labs, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)
                print(audioEmbeddings)
                print(audioEmbeddings.shape, labs.shape, datetimes.shape, recorders.shape, unique_ids.shape)
                audioEmbeddingsList = [arr.tolist() for arr in audioEmbeddings]
                print('listCreated')
                df = pd.DataFrame({'embeddings':audioEmbeddingsList,'labs':labs, 'datetimes':datetimes, 'recorders':recorders,'unique_ids':unique_ids})
                df['datetimes'] = pd.to_datetime(df['datetimes'])
                df['year'], df['month'], df['day'], df['hour'] = df['datetimes'].dt.year, df['datetimes'].dt.month, df['datetimes'].dt.day, df['datetimes'].dt.hour

                df = df.loc[df['hour'].isin([4,5,6,7])]
                '''
                df_all = df_all.append(df, ignore_index = True)
                lenghts.append(len(df))
                file_list.append(file)
    
    return df_all, folders, file_list, lenghts
	


if __name__ == "__main__":

    df_all, folders, file_list, lenghts = get_embeddings()
    percentiles = 24
    plotUMAP_Continuous_plotly(df_all['embeddings'].tolist(), 
                            percentiles, f'Aug2-Aug3 + {folders}', 
                            'plasma', file_list, lenghts)

    sys.exit()

# download checkpoint: 
# https://drive.google.com/file/d/1k1UpQFKSMkmdjYlm1GphjP-nW1uMHEiU/view?usp=sharing