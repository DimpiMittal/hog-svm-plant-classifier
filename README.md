# 🌿 Plant Leaf Disease Classifier using HOG + SVM
A machine learning project that uses Histogram of Oriented Gradients (HOG) for feature extraction and Support Vector Machine (SVM) for classifying plant leaf diseases.

## 📌 Table of Contents

### About the Project
This project aims to classify plant leaf images into different disease categories using machine learning. It uses the HOG (Histogram of Oriented Gradients) method to extract features from grayscale images and then trains a Support Vector Machine (SVM) to classify them.

The goal is to assist in early plant disease detection using simple, interpretable, and efficient machine learning techniques.

### 🧰 Technologies Used
Python 🐍

NumPy

Matplotlib

Scikit-learn (SVM, Metrics, Model Persistence)

scikit-image (for HOG feature extraction)

OpenCV (for image processing)


### ⚙️ How It Works
Loads training images from the dataset.

Converts images to grayscale.

Extracts features using HOG (Histogram of Oriented Gradients).

Trains an SVM classifier on the extracted features.

Saves the trained model as a .pkl file.

Loads the model to predict a test image and displays the result.

### 🛠 Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/DimpiMittal/hog-svm-plant-classifier.git
cd hog-svm-plant-classifier

### Install the dependencies:
bash
Copy
Edit
pip install numpy matplotlib scikit-learn scikit-image opencv-python
▶️ Usage
