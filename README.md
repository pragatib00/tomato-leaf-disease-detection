## Tomato Leaf Disease Classification System
A deep learning–based image classification system that detects tomato leaf diseases using a pretrained convolutional neural network (MobileNetV2) and deploys predictions through an interactive Streamlit web application.

## Overview
This project classifies tomato leaf images into four categories:

- Early Blight
- Healthy
- Late Blight
- Septoria Leaf Spot

The model uses Transfer Learning with MobileNetV2 pretrained on ImageNet and fine-tuned for tomato disease detection.The system allows users to upload an image and receive real-time disease predictions with confidence scores.

## How it Works
- Load and preprocess tomato leaf dataset
- Apply image augmentation (rotation, flipping, zooming)
- Normalize pixel values (rescale = 1./255)
- Use MobileNetV2 (pretrained on ImageNet)
- Freeze base model and train custom classification head
- Evaluate model using validation accuracy and confusion matrix
- Deploy trained model using Streamlit for real-time predictions

## Model Architecture
- Base Model: MobileNetV2 (ImageNet weights)
- GlobalAveragePooling2D
- Dense Layer (ReLU)
- Dropout (0.5)
- Output Layer (Softmax – 4 classes)

## Live App

- 

## Evaluation
- Validation Accuracy: ~92%
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

These metrics measure recommendation quality and prediction accuracy.

## Tech Stack
- Python
- Tensorflow/Keras
- Numpy
- Scikit-learn
- Streamlit
- Git & GitHub

## Run Locally
git clone https://github.com/pragatib00/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection
pip install -r requirements.txt
streamlit run app.py

## Author
Pragati Basnet
BSc. CSIT | Data Science & Machine Learning
GitHub: https://github.com/pragatib00
LinkedIn: https://www.linkedin.com/in/pragati-basnet-1595492a7/