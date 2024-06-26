# 3D Reconstruction of Lung Nodules

Lung nodules are small masses of tissue in the lungs that are typically detected on imaging tests like chest X-rays or CT scans. They can vary in size from a few millimeters to a few centimeters in diameter. Majority of the Lung nodules are non-cancerous (benign). The importance of lung nodules lies in their potential as early indicators of lung cancer. Detecting lung cancer at an early stage significantly improves the chances of successful treatment and survival. Several factors are considered when assessing lung nodules, including their size, shape, density, and growth rate. 

3D Reconstruction can help with the following:
1. Better Visualization: 
   Due to their small size, a 3D Reconstructed model of the lung nodule can aid in better understanding their characteristics.
2. Accurate Measurements: 
   Measurement of lung nodules can be crucial for monitoring changes in nodule size over time, which can help in distinguishing between benign and malignant nodules and in determining the appropriate course of 
   action.
3. Treatment Planning: 
   3D reconstruction helps in surgical planning where the reconstructed images can be used to ensure optimal outcomes.

## Dataset

The LUNA16 dataset, a subset of the LIDC-IDRI dataset is used. It consists of CT Scan data in the mhd/raw format. An annotations file is also provided that includes details such as X, Y, Z coordinates and the diameter of the nodule.

## Methodology

1. Image preprocessing:

Two main pre-processing steps are performed:
- Nodule Mask Generation: 
  Using the annotations and diameter provided in the dataset, nodule masks are extracted by performing a threshold-based segmentation, where the threshold is chosen using K-means clustering. 
- Lung Segmentation: 
  A similar threshold-based segmentation is performed along with applying some post processing to extract segmented lung images.

The extracted Nodule Masks and Segmented Lung Images are stored as .npy files. They serve as input to the UNet model.

2. Automatic Segmentation of Lung Nodules
- The UNet architecture is employed in this step. The predicted masks are also saved as .npy files.
  
4. Nodule Sizing
- Parameters such as Area, Perimeter, Aspect Ratio, Eccentricity and Solidity of the nodule are calculated from the segmented nodules. These can help in the better understanding of the shape and size of the nodule.

5. 3D Reconstruction
- A total of Nine images are considered for 3D reconstruction.
- Marching Cubes algorithm has been employed for the same.

6. Graphical User Interface
- A GUI is created using PyQt5, where the segmented images can be chosen, parameters can be calculated and an interactive 3D model can be visualised.

## Results

- The trained UNet model achieved a Dice coefficient of 0.5753 on the test dataset.

## Acknowledgments

- This project utilizes the [LUNA16 dataset](https://luna16.grand-challenge.org/Data/).
