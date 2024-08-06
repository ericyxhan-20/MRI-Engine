from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

from mri_engine import MRIEngine

app = Flask(__name__)

ROOT_PATH = "/Users/erichan/Documents/Projects/MRI-Engine"

def create_combined_html(file1, file2):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Combined View</title>
        <style>
            body, html {{
                margin: 0;
                height: 100%;
                overflow: hidden;
                background-color: black;
            }}
            h1 {{
                color: white;
            }}
            #title {{
                margin-top: 5%;
                margin-left: 5%;
            }}
            .iframe-container {{
                height: 60%;
                display: flex;
                flex-direction: row;
                justify-content: space-between;
            }}
            .iframe-container > iframe {{
                width: 50%;
                height: 100%;
                border: none;
            }}
            .slider-container > iframe {{
                width: 100%;
                height: 100%;
                border: none;
            }}
            .image-container {{
                height: 30%;
            }}
            .image-container img {{
                width: 100%;
                height: 100%;
                border: none;
            }}
            .slider-container {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                width: 50%;
            }}
            .slider-container input {{
                width: 80%;
                padding: 10px;
            }}
        </style>
    </head>
    <body>
        <div id="header">
            <h1 id="title">Nathan's Glioma Detection Engine</h1>
        </div>
        <div class="iframe-container">
            <iframe id="pyvista-frame" src=".{file1}"></iframe>
            <div class="slider-container">
                <iframe src=".{file2}"></iframe>
                <input type="range" id="slider" min="0" max="100" value="0" step="1">
            </div>
        </div>
        <div class="image-container">
            <img id="dynamic-image" src="./BraTS20_Training_006_result.png" alt="Dynamic Image">
        </div>
    </body>
    </html>
    """
    combined_html_path = 'visualization.html'
    with open(combined_html_path, 'w') as f:
        f.write(html_content)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/run_engine')
def run_engine():
    print("Starting Up Nathan's Glioma Detection Engine...")
    engine = MRIEngine()
    engine.read_flair()
    print("Creating Visualizations...")
    # vis_paths = engine.visualize()
    print("Opening Visualizations In Browser...")
    engine.load_sequences()
    resnet = engine.init_model()
    engine.train_and_predict(resnet, TRUE_LABEL=0)
    # create_combined_html(vis_paths[0], vis_paths[1])
    return render_template("visualization.html")
    # webbrowser.open_new_tab(f"file://{ROOT_PATH}/visualization.html")

if __name__ == '__main__':
    app.run(debug=True)
