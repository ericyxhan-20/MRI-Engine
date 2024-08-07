def create_home_html():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Combined View</title>
        <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body, html {{
                margin: 0;
                height: 100%;
                overflow: hidden;
                background-color: black;
            }}
            h1, h3 {{
                color: white;
                margin-left: 5%;
                font-family: 'Open Sans', sans-serif
            }}
            h1 {{
                margin-top: 5%;
            }}
            h3 {{
                margin-top: 0;
            }}
            #title {{
                margin-top: 5%;
                margin-left: 5%;
            }}
            .iframe-container1 {{
                height: 40%;
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: center;
            }}
            .iframe-container2 {{
                height: 100%;
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: center;
            }}
            .iframe-container1 > iframe, .iframe-container1 > img {{
                width: 40%;
                height: 100%;
            }}
            .iframe-container2 > iframe {{
                width: 50%;
                height: 100%;
            }}
            .image-container {{
                height: 30%;
            }}
            .image-container img {{
                width: 100%;
                height: 100%;
            }}
        </style>
    </head>
    <body>
        <div id="header">
            <h1 id="title">Heka: Glioma Detection Engine</h1>
        </div>
        <div class="iframe-container1">
            <img src=""></img>
            <iframe src=""></iframe>
        </div>
        <div class="iframe-container2">
            <iframe src=""></iframe>
        </div>
    </body>
    </html>
    """
    combined_html_path = 'visualization.html'
    with open(combined_html_path, 'w') as f:
        f.write(html_content)

def create_visualization_html(patient_file, file1, file2):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Combined View</title>
        <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body, html {{
                margin: 0;
                height: 100%;
                overflow: hidden;
                background-color: black;
                display: flex;
                flex-direction: column;
            }}
            h1, h3 {{
                color: white;
                margin-left: 5%;
                font-family: 'Open Sans', sans-serif
            }}
            h1 {{
                margin-top: 5%;
            }}
            h3 {{
                margin-top: 0;
            }}
            #title {{
                margin-bottom: 15px;
            }}
            .iframe-container1 {{
                flex: 1;
                display: flex;
                height: 40%;
                flex-direction: row;
                justify-content: center;
                align-items: center
            }}
            .iframe-container2 {{
                flex: 1;
                display: flex;
                height: 100%;
                flex-direction: row;
                justify-content: center;
                align-items: center;
            }}
            .iframe-container1 > iframe, .iframe-container1 > img {{
                width: 40%;
                height: 100%;
                border: none;
            }}
            .iframe-container2 > iframe {{
                width: 50%;
                height: 100%;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div id="header">
            <h1 id="title">Heka: Glioma Detection Engine</h1>
        </div>
        <h3 id="patient">{patient_file}</h3>
        <div class="iframe-container1">
            <img src=".{file1}">
            <iframe src=".{file2}"></iframe>
        </div>
        <div class="iframe-container2">
            <iframe src=""></iframe>
        </div>
    </body>
    </html>
    """
    combined_html_path = 'visualization.html'
    with open(combined_html_path, 'w') as f:
        f.write(html_content)

def create_combined_html(patient_file, file1, file2, file3):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Combined View</title>
        <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body, html {{
                margin: 0;
                height: 100%;
                overflow: hidden;
                background-color: black;
                display: flex;
                flex-direction: column;
            }}
            h1, h3 {{
                color: white;
                margin-left: 5%;
                font-family: 'Open Sans', sans-serif
            }}
            h1 {{
                margin-top: 5%;
            }}
            h3 {{
                margin-top: 0;
            }}
            #title {{
                margin-bottom: 15px;
            }}
            .iframe-container1 {{
                flex: 1;
                display: flex;
                height: 40%;
                flex-direction: row;
                justify-content: center;
                align-items: center
            }}
            .iframe-container2 {{
                flex: 1;
                display: flex;
                height: 100%;
                flex-direction: row;
                justify-content: center;
                align-items: center;
            }}
            .iframe-container1 > iframe, .iframe-container1 > img {{
                width: 40%;
                height: 100%;
                border: none;
            }}
            .iframe-container2 > iframe {{
                width: 50%;
                height: 100%;
                border: none;
            }}
        </style>
    </head>
    <body>
        <div id="header">
            <h1 id="title">Heka: Glioma Detection Engine</h1>
        </div>
        <h3 id="patient">{patient_file}</h3>
        <div class="iframe-container1">
            <img src=".{file1}">
            <iframe src=".{file2}"></iframe>
        </div>
        <div class="iframe-container2">
            <iframe src=".{file3}"></iframe>
        </div>
    </body>
    </html>
    """
    combined_html_path = 'visualization.html'
    with open(combined_html_path, 'w') as f:
        f.write(html_content)