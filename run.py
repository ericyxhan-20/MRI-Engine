import webbrowser
import os
from mri_engine import MRIEngine
from render_html import create_combined_html, create_home_html, create_visualization_html

ROOT_PATH = "/Users/erichan/Documents/Projects/MRI-Engine"

running = True
while(running):
    print("Starting Up Nathan's Glioma Detection Engine...")

    create_home_html()
    webbrowser.open_new_tab(f"file://{ROOT_PATH}/visualization.html")

    print()
    print("Patient Directory")
    file_names = os.listdir("./patient_data_files")
    for file_name in file_names:
        print(f"  - {file_name}")

    patient_num = input("Enter the patient number: ")
    if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}_flair.nii"):
        print(f"    Reading data from BraTS20_Training_{patient_num}_flair.nii")
    if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}_seg.nii"):
        print(f"    Reading data from BraTS20_Training_{patient_num}_seg.nii")
    if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}_t1.nii"):
        print(f"    Reading data from BraTS20_Training_{patient_num}_t1.nii")
    if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}_t1ce.nii"):
        print(f"    Reading data from BraTS20_Training_{patient_num}_t1ce.nii")
    if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}_t2.nii"):
        print(f"    Reading data from BraTS20_Training_{patient_num}_t2.nii")
    
    engine = MRIEngine(f'./patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}')
    engine.read_flair()

    print()
    while True:
        print("Enter a x, y, and z coordinates of desired cross-section")
        x_slice = input("x-coordinate [55, 180]: ")
        y_slice = input("y-coordinate [40, 215]: ")
        z_slice = input("z-coordinate [0, 140]: ")
        try:
            x_slice = int(x_slice)
            y_slice = int(y_slice)
            z_slice = int(z_slice)
            if x_slice >= 55 and x_slice <= 180 and y_slice >= 40 and y_slice <= 215 and z_slice >= 0 and z_slice <= 140:
                break
        except ValueError:
            print("Unexpected input format!")
    
    print()
    print("Creating Visualizations...")
    full_brain_path = engine.visualize_full_brain()
    brain_slice_path = engine.visualize_slice(x_slice, y_slice, z_slice)
    create_visualization_html(f"BraTS20 Patient {patient_num}", full_brain_path, brain_slice_path)
    webbrowser.open_new_tab(f"file://{ROOT_PATH}/visualization.html")

    print("Analyzing and Making Prediction...")
    engine.load_sequences()
    engine.train_and_predict(TRUE_LABEL=1, num_epochs=50)
    print("Opening Visualizations In Browser...")
    create_combined_html(f"BraTS20 Patient {patient_num}", full_brain_path, brain_slice_path, f"/patient_data_files/patient_{patient_num}/BraTS20_Training_{patient_num}_result.html")
    webbrowser.open_new_tab(f"file://{ROOT_PATH}/visualization.html")

    while True:
        command = input("Enter \'q\' to close or \'r\' to run again: ")
        if command == 'q':
            running = False
            break
        if command == 'r':
            break

if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_result.html"):
    os.remove(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_brain.gif")
if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_result.html"):
    os.remove(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_result.html")
if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_flair.html"):
    os.remove(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_flair.html")
if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_flair_cross_sections.html"):
    os.remove(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_flair_cross_sections.html")
if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_92.pt"):
    os.remove(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_92.pt")
if os.path.exists(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_92_normalized.pt"):
    os.remove(f"./patient_data_files/patient_{patient_num}/BraTS20_Training_006_92_normalized.pt")
print("Shutting Down.")