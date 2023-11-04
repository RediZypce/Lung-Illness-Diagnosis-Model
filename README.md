# Lung Illness Diagnosis Model - Convolutional Neural Network (CNN)
This Jupyter notebook focuses on developing a deep learning model for diagnosing lung illnesses from X-ray images, including pneumonia, Covid-19, and normal scans. The goal is to assist medical professionals in interpreting X-ray scans effectively. The notebook uses the Keras module in TensorFlow and employs a multi-class classification approach.
![image](https://github.com/RediZypce/Lung-Illness-Diagnosis-Model/assets/109640560/b3a99ff7-9925-49bf-b797-866ec40af4a3)

# Dataset
The dataset used in this project is the [Covid-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) which has been sourced from Kaggle. The dataset contains grayscale X-ray images, already split into training and testing sets. It is a multi-class classification problem, as it involves three different classes: Covid, Normal, and Pneumonia.
You can either download the dataset from the link provided on Kaggle or access it [here].
[Link to Jupyter notebook](Lung-Illness-Diagnosis-With-X-ray-(CNN).ipynb)
# Requirements
To run this project, you'll need the following libraries and tools:

* Python (3.6+)
* Jupyter Notebook
* TensorFlow
* Keras
* scikit-learn
* Matplotlib
* NumPy
* os module
* random module

# Project Structure
The project is organized into various sections within the Jupyter notebook:

* Importing Libraries

* Constructing ImageDataGenerator
  * Preparation of the data for model training.
  * Data augmentation techniques.

* Building the Convolutional Neural Network (CNN) Model
  * Compilation of the model with specified metrics and loss functions.

* Training the Model with Early Stopping
  * Implementation of early stopping to prevent overfitting.

* Model training with visualizations.
  * Visualizing training progress, including loss and accuracy curves.
  * Visualizing the confusion matrix.

# Usage
* Ensure you have all the required libraries and tools installed.
* Open the Jupyter notebook provided in this project.
* Execute each cell sequentially by running the notebook.
* You can customize and experiment with the model architecture, hyperparameters, and dataset as needed.
