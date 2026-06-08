# Pneumonia_Prediction_CNN
A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify chest X-ray images for pneumonia detection.
# Pneumonia Detection in Chest X-Rays Using Convolutional Neural Networks (CNN)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Project Overview
This project targets the automated detection of Pneumonia from chest X-ray images using custom Deep Learning architectures. Developed as an Undergraduate project, it utilizes a 3-layer Convolutional Neural Network (CNN) with customized Dropout regularizations built over **TensorFlow/Keras**. 

The goal is to provide binary image classification mapping to either **NORMAL** or **PNEUMONIA** categories to assist in clinical screening procedures.

---

## 📊 Dataset Profile
The model is trained and validated using a structured subset of chest X-ray images:
* **Total Training Images:** 5,216 images split across 2 target classes.
* **Total Evaluation / Test Images:** 624 images.
* **Pre-processing:** All images are rescaled down to normalization channels ($1./255$) and dynamically scaled down to uniform dimensions of $150 \times 150 \times 3$.
* **Data Augmentation Strategies:** Applied zoom configurations (`zoom_range=0.3`) and random vertical transformations (`vertical_flip=True`) on training data streams to prevent overfitting profiles.

---

## 🏗️ Model Architecture
The network pipeline features structural sequence groupings optimized for pattern spatial translation:

| Layer (Type) | Specification / Hyperparameters | Output Shape |
| :--- | :--- | :--- |
| **Conv2D (Input Layer)** | 32 Filters, $3 \times 3$ Kernel, ReLU, `he_normal` | (None, 148, 148, 32) |
| **MaxPooling2D** | Pool Size $2 \times 2$ | (None, 74, 74, 32) |
| **Dropout** | Rate = 0.25 | (None, 74, 74, 32) |
| **Conv2D** | 64 Filters, $3 \times 3$ Kernel, ReLU | (None, 72, 72, 64) |
| **MaxPooling2D** | Pool Size $2 \times 2$ | (None, 36, 36, 64) |
| **Dropout** | Rate = 0.25 | (None, 36, 36, 64) |
| **Conv2D** | 128 Filters, $3 \times 3$ Kernel, ReLU | (None, 34, 34, 128) |
| **Dropout** | Rate = 0.40 | (None, 34, 34, 128) |
| **Flatten** | Dimensional Flattening | (None, 147968) |
| **Dense** | 128 Units, ReLU Activation | (None, 128) |
| **Dropout** | Rate = 0.30 | (None, 128) |
| **Dense (Output Layer)**| 1 Unit, Sigmoid Activation (Binary Class) | (None, 1) |

* **Loss Function:** `binary_crossentropy`
* **Optimization Algorithm:** `adam`
* **Batch Size Configuration:** 8

---

## 📉 Execution Metrics & Performance Analytics

The model was configured to train over 10 execution epochs. The testing metrics on the validation stream demonstrated regularized stabilization over the validation curves:

* **Final Model Classification Accuracy:** **`89.26%`**

### Epoch Execution Performance Log
```text
Epoch 1/10  - Loss: 0.5485 - Accuracy: 0.7920 - Val_Loss: 0.5799 - Val_Accuracy: 0.6506
Epoch 5/10  - Loss: 0.2859 - Accuracy: 0.8821 - Val_Loss: 0.4069 - Val_Accuracy: 0.8381
Epoch 10/10 - Loss: 0.2005 - Accuracy: 0.9201 - Val_Loss: 0.3794 - Val_Accuracy: 0.8894
