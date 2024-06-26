import sys
import os
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize
import vtk
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit, QLabel,
    QHBoxLayout, QGroupBox
)
from PyQt5.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class ImageReportGenerator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lung Nodule Details")
        self.setGeometry(200, 200, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Button Layout
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        # Load Images Button
        self.load_button = QPushButton("Load Images")
        self.load_button.clicked.connect(self.load_images)
        button_layout.addWidget(self.load_button)

        # Generate Nodule Details Button
        self.generate_nodule_details = QPushButton("Generate Nodule Details")
        self.generate_nodule_details.clicked.connect(self.generate_report)
        button_layout.addWidget(self.generate_nodule_details)

        # Generate 3D Model Button
        self.generate_3d_button = QPushButton("Generate 3D Model")
        self.generate_3d_button.clicked.connect(self.generate_3d_model)
        button_layout.addWidget(self.generate_3d_button)

        # Report Group Box
        report_group_box = QGroupBox("Nodule Details")
        layout.addWidget(report_group_box)

        report_layout = QVBoxLayout()
        report_group_box.setLayout(report_layout)

        # Report Text Area
        self.report_text = QTextEdit()
        report_layout.addWidget(self.report_text)

        # VTK Group Box
        vtk_group_box = QGroupBox("3D Model")
        layout.addWidget(vtk_group_box)

        vtk_layout = QVBoxLayout()
        vtk_group_box.setLayout(vtk_layout)

        # VTK Renderer Widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        vtk_layout.addWidget(self.vtk_widget)

        self.loaded_images = []

    def load_images(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Numpy Files (*.npy)")
        if file_dialog.exec_():
            self.loaded_images = file_dialog.selectedFiles()
            if self.loaded_images:
                print("Loaded images:", self.loaded_images)
            
            else:
                print("No images selected.")
                self.report_text.append("No images selected.")

    
    def generate_report(self):
        if not self.loaded_images:
            print("No images loaded.")
            self.report_text.append("No images loaded.")
            return
        
        report_image = None
        for image_path in self.loaded_images:
            if "_0_prediction.npy" in image_path:
                report_image = image_path
                break
    
        if report_image:
            try:
                # Load the binary image from .npy file
                binary_image = np.load(report_image)
                print(f"Loaded image shape: {binary_image.shape}")

                # Remove singleton dimensions
                binary_image = np.squeeze(binary_image)
                print(f"Binary image shape after squeezing: {binary_image.shape}")

                # Ensure the image is binary
                binary_image = (binary_image > 0).astype(np.uint8)
                print(f"Binary image shape after thresholding: {binary_image.shape}")

                # Calculate features
                nodule_features = self.calculate_nodule_features(binary_image)

                # Display calculated features in the GUI
                self.report_text.clear()
                self.report_text.append("Nodule Features:")
                for nodule_id, feature_dict in nodule_features.items():
                    self.report_text.append(f"\nNodule {nodule_id} Features:")
                    for feature_name, value in feature_dict.items():
                        self.report_text.append(f"{feature_name}: {value}")

                # Plot the image
                plt.imshow(binary_image, cmap='gray')
                plt.title("Binary Mask of the Lung Nodule")
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Error processing the report image: {e}")
                self.report_text.append(f"Error processing the report image: {e}")
        else:
            print("No image with '_0_prediction.npy' found.")
            self.report_text.append("No image with '_0_prediction.npy' found.")


    def generate_3d_model(self):
        if not self.loaded_images:
            print("No images loaded.")
            self.report_text.append("No images loaded.")
            return
        
        try:
            # File paths to segmented images
            file_paths = list(self.loaded_images)

            # Load segmented images
            segmented_images = self.load_segmented_files(file_paths)
            print(f"Segmented images shape: {segmented_images.shape}")

            # Ensure correct dimensions (num_slices, height, width)
            if segmented_images.ndim == 5 and segmented_images.shape[1] == 1 and segmented_images.shape[-1] == 1:
                segmented_images = np.squeeze(segmented_images, axis=(1, -1))
                print(f"Segmented images shape after squeezing: {segmented_images.shape}")

            # Perform surface reconstruction
            mesh = self.surface_reconstruction(segmented_images)

            # Visualize the mesh
            self.visualize_mesh(mesh)
        except Exception as e:
            print(f"Error generating 3D model: {e}")
            self.report_text.append(f"Error generating 3D model: {e}")



    




    def calculate_nodule_features(self, binary_image):
        # Fill holes in the binary image using morphological closing
        binary_image_filled = closing(binary_image, square(5))

            # Label connected components
        labeled_image = label(binary_image_filled)

            # Extract region properties
        props = regionprops(labeled_image)

            # Initialize features dictionary
        features = {}

        for prop in props:
            # Calculate area
            area = "{:.3f} mm²".format(prop.area)

            # Calculate perimeter
            perimeter = "{:.3f} mm".format(prop.perimeter)

            # Calculate aspect ratio
            aspect_ratio = "{:.3f}".format(prop.major_axis_length / prop.minor_axis_length)

            # Calculate eccentricity
            eccentricity = "{:.3f}".format(prop.eccentricity)

            # Calculate solidity
            hull_area = ConvexHull(prop.coords).volume
            solidity = "{:.3f}".format(prop.area / hull_area)

            # Add features to dictionary
            features[prop.label] = {
                'Area': area,
                'Perimeter': perimeter,
                'Aspect Ratio': aspect_ratio,
                'Eccentricity': eccentricity,
                'Solidity': solidity,
            }

        return features

    def load_segmented_files(self, file_paths):
        segmented_images = []
        for file_path in file_paths:
            segmented_image = np.load(file_path)
            print(f"Loaded segmented image shape: {segmented_image.shape}")
            segmented_images.append(segmented_image)
        segmented_images = np.array(segmented_images)
        print(f"Stacked segmented images shape: {segmented_images.shape}")
        return segmented_images

    def surface_reconstruction(self, segmented_images):
        # Create a vtkImageData object to store segmented images
        imageData = vtk.vtkImageData()
        depth, height, width = segmented_images.shape
        imageData.SetDimensions(width, height, depth)
        imageData.SetSpacing(1.0, 1.0, 1.0)
        imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Fill vtkImageData with segmented image data
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    value = segmented_images[z, y, x]
                    imageData.SetScalarComponentFromDouble(x, y, z, 0, value)

        # Perform surface reconstruction using Marching Cubes
        surface = vtk.vtkMarchingCubes()
        surface.SetInputData(imageData)
        surface.ComputeNormalsOn()
        surface.SetValue(0, 1)  # Threshold value for surface reconstruction

        # Generate the mesh
        surface.Update()

        return surface.GetOutput()

        

    def smooth_mesh(self, mesh):
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(100)  
        smoother.SetRelaxationFactor(0.1)    
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.Update()
        return smoother.GetOutput()

    def visualize_mesh(self, mesh):
        # Create a renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)  

        # Add the renderer to the vtk widget
        self.vtk_widget.GetRenderWindow().AddRenderer(renderer)

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set the color of the actor's surface to gray
        actor.GetProperty().SetColor(0.5, 0.5, 0.5)

        # Add the actor to the renderer
        renderer.AddActor(actor)

        # Set camera position and orientation
        renderer.ResetCamera()
        renderer.GetActiveCamera().Azimuth(30)
        renderer.GetActiveCamera().Elevation(30)

        # Start the render loop
        self.vtk_widget.Start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageReportGenerator()
    window.show()
    sys.exit(app.exec_())
