# Deep-Learning-Based-Colon-Tissue-Recognition

## Overview

Deep Learning-Based Colon Tissue Recognition is a deep learning and computer vision project developed to classify histopathological colon tissue images into different tissue categories.

The project uses deep learning-based image classification techniques to automatically recognize tissue patterns from microscopic histopathology images. It includes image preprocessing, data augmentation, transfer learning, model training, evaluation, and prediction.

The project also incorporates Explainable AI techniques such as Grad-CAM and SHAP to provide better understanding of the regions contributing to model predictions.

## Features

* Deep learning-based colon tissue recognition
* Histopathological image classification
* Multi-class tissue classification
* Image preprocessing and normalization
* Data augmentation
* Transfer learning
* EfficientNet-based image classification
* Model training and fine-tuning
* Model performance evaluation
* Accuracy, precision, recall, and F1-score calculation
* Confusion matrix generation
* Single image prediction
* Grad-CAM visualization
* SHAP-based model interpretation
* 5-fold cross-validation
* Interactive Streamlit web application
* Prediction confidence display
* Organized model and result storage

## Technologies Used

* Python
* PyTorch
* Torchvision
* EfficientNet
* Scikit-learn
* NumPy
* Matplotlib
* OpenCV
* SHAP
* Grad-CAM
* Streamlit

## Dataset

The project uses the **NCT-CRC-HE-100K** histopathology image dataset.

The dataset contains 100,000 colorectal histopathology images categorized into nine different tissue classes.

### Dataset Details

| Property          | Details              |
| ----------------- | -------------------- |
| Dataset           | NCT-CRC-HE-100K      |
| Total Images      | 100,000              |
| Number of Classes | 9                    |
| Image Size        | 224 × 224            |
| Image Format      | RGB                  |
| Domain            | Colon Histopathology |

The dataset is organized into separate classes representing different types of colon tissue.

## Tissue Classes

The model recognizes the following nine tissue classes:

| Class | Tissue Type                          |
| ----- | ------------------------------------ |
| ADI   | Adipose                              |
| BACK  | Background                           |
| DEB   | Debris                               |
| LYM   | Lymphocytes                          |
| MUC   | Mucus                                |
| MUS   | Smooth Muscle                        |
| NORM  | Normal Colon Mucosa                  |
| STR   | Cancer-Associated Stroma             |
| TUM   | Colorectal Adenocarcinoma Epithelium |

## Project Workflow

The overall workflow of the project is:

```text
Histopathology Images
        ↓
Dataset Preparation
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
Feature Extraction
        ↓
Transfer Learning
        ↓
EfficientNet Model
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Tissue Classification
        ↓
Prediction & Visualization
```

## Image Preprocessing

The input histopathology images are processed before being provided to the deep learning model.

The preprocessing pipeline includes:

* Image resizing
* RGB image conversion
* Image normalization
* Dataset organization
* Training and validation preparation

The images are resized to **224 × 224 pixels** to match the input requirements of the selected model architecture.

## Data Augmentation

Data augmentation techniques are applied to increase the diversity of training images and improve model generalization.

The augmentation pipeline includes:

* Random horizontal flip
* Random vertical flip
* Random rotation
* Color jitter
* Image resizing
* Image normalization

These transformations help the model learn robust visual features from histopathological images.

## Model Architecture

The project uses **EfficientNet with Transfer Learning** for multi-class colon tissue classification.

The architecture includes:

* Pretrained EfficientNet backbone
* Transfer learning
* Fine-tuning
* Batch normalization
* Dropout regularization
* Fully connected classification layer
* Softmax-based multi-class prediction

The pretrained model is adapted to recognize the nine colon tissue classes in the dataset.

## Model Training

The model is trained using the prepared histopathological image dataset.

The training process includes:

* Loading the dataset
* Applying preprocessing and augmentation
* Loading the pretrained EfficientNet model
* Modifying the classification layer
* Training the model
* Validating model performance
* Saving the best-performing model

The trained model is stored in the `saved_models/` directory.

## Model Evaluation

The trained model is evaluated using standard classification metrics.

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

These metrics are used to analyze the performance of the model across different tissue categories.

## Single Image Prediction

The project supports classification of individual histopathology images.

The prediction process:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Trained EfficientNet Model
     ↓
Class Prediction
     ↓
Prediction Confidence
```

Run the prediction script using:

```bash
python predict.py
```

The system provides:

* Predicted tissue class
* Prediction confidence

## Explainable AI

The project includes Explainable AI techniques to understand model predictions.

### Grad-CAM

Grad-CAM is used to generate visual heatmaps that highlight important regions of the histopathology image contributing to the model's prediction.

Run:

```bash
python gradcam.py
```

The generated visualization helps identify the image regions that influenced the predicted tissue class.

### SHAP

SHAP is used to analyze the contribution of different image regions toward the model's prediction.

Run:

```bash
python shap_analysis.py
```

These techniques provide additional insight into the decision-making process of the deep learning model.

## 5-Fold Cross-Validation

The project supports stratified **5-fold cross-validation** to evaluate model performance across multiple data splits.

Run:

```bash
python train_kfold.py
```

The cross-validation process calculates performance metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score

The average performance across the folds can be used to assess the consistency of the model.

## Streamlit Web Application

An interactive Streamlit application is included for image classification.

Launch the application using:

```bash
streamlit run streamlit_app.py
```

Users can upload a colon histopathology image through the interface.

The application displays:

* Uploaded image
* Predicted tissue class
* Prediction confidence

## Project Structure

```text
Deep-Learning-Based-Colon-Tissue-Recognition/
│
├── dataset.py
├── model.py
├── train.py
├── train_kfold.py
├── test.py
├── predict.py
├── utils.py
├── gradcam.py
├── shap_analysis.py
├── streamlit_app.py
├── config.py
├── requirements.txt
├── README.md
│
├── database/
│   └── NCT-CRC-HE-100K/
│
├── saved_models/
│
└── results/
```

## How to Run the Project

### Step 1

Clone the repository:

```bash
git clone <your-repository-url>
```

### Step 2

Navigate to the project directory:

```bash
cd Deep-Learning-Based-Colon-Tissue-Recognition
```

### Step 3

Create a virtual environment:

```bash
python -m venv .venv
```

### Step 4

Activate the virtual environment on Windows:

```bash
.\.venv\Scripts\Activate.ps1
```

### Step 5

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 6

Prepare the dataset in the required directory structure.

### Step 7

Train the model:

```bash
python train.py
```

### Step 8

Evaluate the trained model:

```bash
python test.py
```

### Step 9

Run single-image prediction:

```bash
python predict.py
```

### Step 10

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Model Output

For an input histopathology image, the system produces:

```text
Input Image
     ↓
Predicted Tissue Class
     ↓
Prediction Confidence
```

Example tissue predictions include:

* Adipose
* Background
* Debris
* Lymphocytes
* Mucus
* Smooth Muscle
* Normal Colon Mucosa
* Cancer-Associated Stroma
* Colorectal Adenocarcinoma Epithelium

## Project Objectives

The main objectives of this project are:

* To develop a deep learning-based colon tissue recognition system.
* To classify histopathological images into different tissue categories.
* To apply transfer learning for medical image classification.
* To perform image preprocessing and augmentation.
* To train and evaluate a deep learning classification model.
* To analyze model performance using standard evaluation metrics.
* To explore Explainable AI techniques for model interpretation.
* To develop an interactive interface for image prediction.

## Learning Outcomes

Through this project, the following concepts are demonstrated:

* Deep Learning
* Computer Vision
* Histopathological Image Analysis
* Image Classification
* Transfer Learning
* EfficientNet
* Image Preprocessing
* Data Augmentation
* Feature Extraction
* Model Training
* Model Evaluation
* Explainable AI
* Grad-CAM
* SHAP
* PyTorch
* Streamlit

## Future Enhancements

The project can be enhanced in the future by adding:

* Vision Transformer models
* Ensemble deep learning architectures
* External dataset validation
* Advanced stain normalization
* Test-Time Augmentation
* Improved Explainable AI techniques
* ONNX model deployment
* Cloud-based deployment
* Real-time image classification
* Mobile-based image analysis

## Purpose of the Project

The purpose of this project is to explore the application of deep learning and computer vision techniques for automated recognition of colon tissue from histopathological images.

The project provides practical experience in preparing medical image datasets, applying transfer learning, developing image classification models, evaluating model performance, and using Explainable AI techniques to understand deep learning predictions.

## Internship Information

**Project:** Deep Learning-Based Colon Tissue Recognition

**Organization:** SASTRA Deemed to be University

**Department:** Department of Information Technology, School of Computing

**Duration:** July 01, 2026 – July 30, 2026

## Author

**PRITHYADARSHAN T**

Computer Science and Engineering (Artificial Intelligence & Machine Learning)

**GitHub:** https://github.com/prithyadarshan

**LinkedIn:** https://www.linkedin.com/in/prithyadarshan-thiyagarajan-379a4b2a3
