# Projects/Risk_Assessment_Models/backend/api/main.py

import os
import json
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ..core.graph import Graph
from ..core.solver import solve_var
from .models import SolverRequest, SolverResponse

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frontend'))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))


def load_json_file(file_path: str) -> dict:
    """
    Load and check JSON files from the specified path
    """
    if not os.path.commonpath([data_dir, os.path.realpath(file_path)]) == os.path.realpath(
        data_dir
    ):
        raise HTTPException(status_code=400, detail='Invalid file path')
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f'File not found: {file_path}')
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f'Invalid JSON format in {file_path}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error loading file: {str(e)}')


@app.get('/api/data_files', response_model=List[str])
async def get_data_files():
    """
    Obtain a list of all JSON files in the data directory
    """
    try:
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if not json_files:
            raise HTTPException(status_code=404, detail='No JSON files found in data directory')
        return json_files
    except Exception as e:
        logger.error(f'\nError accessing data directory: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to list files: {str(e)}')


@app.get('/api/graph')
async def get_graph_data(file: str = Query(...)):
    """
    Load graph data from the specified JSON file.
    """
    file_path = os.path.join(data_dir, file)
    try:
        network_data = load_json_file(file_path)
        graph = Graph.from_json(network_data)
        return {'nodes': graph.nodes, 'edges': graph.edges}
    except HTTPException as e:
        logger.error(f'\nError Loading graph data: {str(e.detail)}')
        raise HTTPException(status_code=422, detail=f'Failed to load graph data: {str(e.detail)}')


@app.post('/api/solve', response_model=SolverResponse)
async def solve_var_endpoint(request: SolverRequest):
    """
    Get the optimal path and cost based on a specified algorithm.
    """
    if not request.selected_file:
        raise HTTPException(status_code=400, detail='No file selected')
    file_path = os.path.join(data_dir, request.selected_file)
    network_data = load_json_file(file_path)
    graph = Graph.from_json(network_data)
    result = solve_var(graph, request.src, request.dest, request.alpha)
    return result


# Mount static files to /frontend
app.mount('/', StaticFiles(directory=frontend_dir, html=True), name='frontend_app')
