from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from backend.ievad.plot_helpers import load_audio
from backend.ievad.dash_plot import create_specs2
import numpy as np

global_path = Path('frontend/public')

class Item(BaseModel):
    x: float
    y: float
    z: float
    source_file: str
    meta: dict
    index: int

class DataPath(BaseModel):
    path: str
    model_name: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    # Adjust the origin to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/getEmbedPaths")
async def get_folders(path: DataPath):
    # jsons = [str(p.relative_to(global_path))
    #          for p in global_path.joinpath(path.path).rglob('*umap.json')]
    dirs = [str(p.relative_to(global_path)) 
            for p in global_path.joinpath(path.path).rglob('*umap.json')
            if p.parent.is_dir() and not p.parent.name.startswith('.')]
    jsons = {
        Path(d).parent.stem.split('-')[-1]:
            d for d in dirs
    }
    # sort the dict my model name
    # also send label_dictionaries
    # print(jsons)
    return {'message': 'dictionaries successfully retrieved', 
            'dict': jsons}
    
@app.post("/getLabels")
async def get_labels(path: DataPath):
    eval_path = (global_path
                 .joinpath(path.path)
                 .parent
                 .joinpath('task_results')
                 .joinpath(path.model_name))
    label_dict = {}
    
    get_default_labels = np.load(
        eval_path
        .joinpath('labels')
        .joinpath('default_labels.npy'),
        allow_pickle=True
    ).item()
    for key in get_default_labels.keys():
        label_dict[key] = get_default_labels[key]
    
    ground_truth_file = (
        eval_path
        .joinpath('labels')
        .joinpath('ground_truth.npy')
    )
    if ground_truth_file.exists():
        ground_truth = np.load(
            ground_truth_file,
            allow_pickle=True
        ).item()
        inv = {v: k for k, v in ground_truth['label_dict'].items()}
        
        label_dict['ground_truth'] = [
            inv[v] if v != -1.0 else 'noise' for v in ground_truth['labels'] 
            ]
    
    clust_label_file = (
        eval_path
        .joinpath('clustering')
        .joinpath('clust_label__no_noise.npy')
    )
    if clust_label_file.exists():
        clust_labels = np.load(
            clust_label_file,
            allow_pickle=True
        ).item()
        for key in clust_labels.keys():
            label_dict[key] = clust_labels[key].tolist()
        
    return {'message': 'labels successfully retrieved',
            'dict': label_dict}
    
    

@app.post("/getDataPoint")
async def create_spectrogram(item: Item):
    
    path = (
        global_path
        .joinpath('files')
        .joinpath('audio')
        .joinpath(Path(item.meta['audio_dir']).stem)
        .joinpath(item.source_file)
    )
    sr = item.meta['sample_rate (Hz)']
    segment_length = item.meta['segment_length (samples)'] / sr
        
    audio, sr, file_stem = load_audio(item.z, 
                                      path,
                                      sr,
                                      segment_length)
    spec = create_specs2(audio)

    return {'message': 'values successfully received', 
            'spectrogram_data': spec.tolist()}

@app.get("/")

async def read_item():
    return {'message': 'laeuft'}