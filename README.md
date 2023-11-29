# Mask-Detection

This repository contains code to train a deep learning model to detect whether a person is wearing a mask or not. 
It utilizes the MobileNetV2 architecture pre-trained on ImageNet and fine-tunes it on a dataset containing images of people with and without masks.

## Introduction

The model architecture comprises MobileNetV2 as the base model, followed by a Flatten layer and a Dense layer with 'sigmoid' activation for binary classification ('With Mask' and 'Without Mask').

### Requirements

- Python 3.9
- TensorFlow
- Pillow
- Numpy

## Dataset
The dataset directory structure should follow:

├── <REPO-NAME>
    ├── Train
    ├── Test
    └── Validation
Ensure that the 'Train', 'Validation', and 'Test' directories contain subdirectories for 'With Mask' and 'Without Mask' images.

OR    

You can download the dataset from here:
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

Clone the repository:
git clone https://github.com/ketanbhole/Mask-Detection.git
cd mask-detection    


#Install dependencies:
pip install -r requirements.txt

Model Details
The model is trained with data augmentation techniques on the 'Train' dataset and validated on the 'Validation' dataset. Evaluation metrics (loss and accuracy) are computed on the 'Test' dataset.

Contributing
Contributions are welcome! Fork the repository, make changes, and create a pull request.
