import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import plotly.io as pio

import pyvista as pv
from vtk import vtkNIFTIImageReader

class NIFTIReader(pv.BaseReader):
    _class_reader = vtkNIFTIImageReader

class MRIEngine:
    def __init__(self, mri_prefix):
        self.mri_prefix = mri_prefix
        self.flair_path = self.mri_prefix + '_flair.nii'
        self._92 = self.mri_prefix + '_92.pt'
        self.normalized_92 = self.mri_prefix + '_92_normalized.pt'
    
    def read_flair(self):
        # Upload image file to Colab's local runtime files first
        # Load the NIfTI file using nibabel
        TRUE_LABEL = 1
        img = nib.load(self.flair_path) 
        type(img) # nibabel.nifti1.Nifti1Image

        # Read the NIfTI image data as a numpy array
        img_npy = img.get_fdata()
        # type(img_npy) # numpy.memmap
        # img_npy.shape # (240, 240, 155)
        return img_npy
    
    def load_sequences(self):
        # List of sequences to load
        load_sequence = ['flair','t1ce','t1','t2']

        # Stack the sequences as channels
        stacked_tensor = torch.stack([
            # Load the slice 92 for each sequence
            torch.from_numpy(
                nib.load(f'{self.mri_prefix}_{seq}.nii')
                    .get_fdata()[:,:,92]
            )
            for seq in load_sequence # Iterate through each sequence
        ])
        # Normalize the stacked tensor (example normalization)
        mean = np.mean(stacked_tensor.numpy(), axis=(1, 2))
        std = np.std(stacked_tensor.numpy(), axis=(1, 2))
        stacked_tensor_normalized = (stacked_tensor - mean[:, None, None]) / std[:, None, None]
        # Save the normalized tensor
        torch.save(stacked_tensor, self._92)
        torch.save(stacked_tensor_normalized, self.normalized_92)

    def init_model(self):
        resnet = models.resnet18()
        # Update the first convolutional layer to accept 4 input channels
        resnet.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Initialize the new convolutional layer
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Update the output layer to have 2 output classes
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=2)
        # Initialize the new fully connected layer
        nn.init.normal_(resnet.fc.weight, 0, 0.01)
        nn.init.constant_(resnet.fc.bias, 0)
        resnet.eval()
        return resnet

    def train_and_predict(self, TRUE_LABEL, num_epochs=10):
        losses = []
        predictions = []
        for epoch in range(num_epochs):
            resnet = self.init_model()
            # Load image data
            img = torch.load(f'{self.mri_prefix}_92_normalized.pt')
            assert(img.shape == torch.Size([4, 240, 240]))
            # Convert the 3D tensor to 4D by adding the batch dimension
            img_unsqueezed = img.unsqueeze(dim=0)
            img_unsqueezed.shape # Shape after adding the batch dimension [1,4,240,240]
            # Create label tensor
            label = torch.Tensor([1])
            # Check for GPU availability
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img_unsqueezed = img_unsqueezed.to(device,dtype = torch.float32)
            label = label.to(device,dtype=torch.long)
            resnet = resnet.to(device)
            resnet.eval()
            # Get predictions from the model for the input image
            with torch.no_grad():
                pred = resnet(img_unsqueezed)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(pred, label)
                predicted_label = pred.argmax(dim=1).item()
                predicted_prob = torch.softmax(pred, dim=1).squeeze().cpu().numpy()

                losses.append(loss)
                predictions.append([predicted_label, predicted_prob[0], predicted_prob[1]])
                # print(f'Prediction: {predicted_label}, Loss: {loss.item()}')

        losses = np.array(losses)
        predictions = np.array(predictions)

        pred_0_sum = 0
        pred_0_prob = 0
        pred_0_count = 0
        pred_1_sum = 0
        pred_1_prob = 0
        pred_1_count = 0
        for i in range(len(losses)):
            pred = predictions[i]
            loss = losses[i]
            if pred[0] == 0:
                pred_0_sum += loss
                pred_0_prob += pred[1]
                pred_0_count += 1
            else:
                pred_1_sum += loss
                pred_1_prob += pred[1]
                pred_1_count += 1
        pred_0_avg = pred_0_sum / pred_0_count
        pred_1_avg = pred_1_sum / pred_1_count
        if pred_0_avg > pred_1_avg:
            final_prediction = 1
            final_loss = pred_1_avg
            pred_1_prob = pred_1_prob / pred_1_count
            final_predicted_prob = [pred_1_prob, 1-pred_1_prob]
        else:
            final_prediction = 0
            final_loss = pred_0_avg
            pred_0_prob = pred_0_prob / pred_0_count
            final_predicted_prob = [pred_0_prob, 1-pred_0_prob]

        # Create a Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Class 0', 'Class 1'],
                y=final_predicted_prob,
                marker=dict(color=['rgb(55, 83, 109)', 'rgb(26, 118, 255)']),
                text=[f'{p*100:.2f}%' for p in final_predicted_prob],
                textposition='auto'
            )
        ])
        
        # Customize the layout
        fig.update_layout(
            title=f'Prediction Probabilities<br>True Label: {"High" if TRUE_LABEL==1 else "Low"} | Predicted Label: {"High" if final_prediction==1 else "Low"}',
            xaxis_title='Class',
            yaxis_title='Probability',
            font=dict(family="Arial", size=14, color="white"),
            plot_bgcolor='rgb(0, 0, 0)',
            paper_bgcolor='rgb(0, 0, 0)',
        )

        # Add loss information as an annotation
        fig.add_annotation(
            x=0.5, y=-0.2,
            text=f'Loss: {final_loss.item():.4f}',
            showarrow=False,
            font=dict(size=14, color="white"),
            xref='paper', yref='paper'
        )
        # Save the Plotly figure as an HTML file
        fig.write_html(self.mri_prefix + '_result.html')

        print(f'Prediction: {final_prediction}, Loss: {final_loss.item()}')

    def visualize_full_brain(self):
        # Load NIfTI data
        reader = NIFTIReader(self.flair_path)
        mesh = reader.read()

        # Convert to PyVista dataset
        data = pv.wrap(mesh)

        # Apply thresholding to remove zero values
        thresholded = data.threshold(0.01)  # Adjust threshold value as needed

        # Create a PyVista plotter object
        plotter = pv.Plotter(off_screen=True)

        plotter.background_color = 'black'
        # Add the thresholded mesh to the plotter
        plotter.add_mesh(thresholded, opacity=0.3, color='white', show_edges=True, edge_color='white')

        # Save the visualization to an HTML file
        # html_file = self.mri_prefix + '_flair.html'
        # plotter.export_html(html_file)
        gif_file = self.mri_prefix + "_brain.gif"
        path = plotter.generate_orbital_path(n_points=50, shift=mesh.length)
        plotter.open_gif(self.mri_prefix + "_brain.gif")
        plotter.orbit_on_path(path, write_frames=True)
        plotter.close()

        print("Exported Full Brain Map!")
        return gif_file[1:]
    
    def visualize_slice(self, x_slice, y_slice, z_slice):
        # Load NIfTI data
        pv.global_theme.allow_empty_mesh = True
        reader = NIFTIReader(self.mri_prefix + '_flair.nii')
        mesh = reader.read()
        # Convert to PyVista dataset
        data = pv.wrap(mesh)

        # Apply thresholding to remove zero values
        thresholded = data.threshold(0.01)  # Adjust threshold value as needed
        # Create a PyVista plotter object
        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = 'black'

        # Function to create and add a slice to the plotter
        def add_slice_to_plotter(axis, slice_index):
            if axis == 'x':
                slice_data = thresholded.slice(normal='x', origin=(slice_index, 0, 0))
            elif axis == 'y':
                slice_data = thresholded.slice(normal='y', origin=(0, slice_index, 0))
            elif axis == 'z':
                slice_data = thresholded.slice(normal='z', origin=(0, 0, slice_index))
            else:
                raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

            plotter.add_mesh(slice_data, cmap='bone', opacity=0.8)  # Adjust opacity as needed
            # Configure the color bar to be visible
            plotter.add_scalar_bar(title='Intensity', vertical=True, 
                                   color='white', label_font_size=16, 
                                   title_font_size=20)
            plotter.view_isometric()

        # Add slices
        add_slice_to_plotter('x', x_slice)  # Add x slice
        add_slice_to_plotter('y', y_slice)  # Add y slice
        add_slice_to_plotter('z', z_slice)  # Add z slice

        # Save the visualization to an HTML file
        html_file = self.mri_prefix + '_flair_cross_sections.html'
        plotter.export_html(html_file)
        print("Exported Cross Section Map!")
        return html_file[1:]
