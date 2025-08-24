# 🌱 Plant Disease Classifier

A web application built with **Streamlit** that uses a trained **Deep Learning model (TensorFlow/Keras)** to classify plant leaf diseases from uploaded images.  

*Website* :- [https://plant-disease-prediction05.streamlit.app/](https://plant-disease-prediction05.streamlit.app/)

*Dataset* :- [https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---
<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  </p>

---

## 📖 Table of Contents

- [🚀 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [📊 Model Overview](#-model-overview)
- [✅ Performance Metrics](#-performance-metrics)
- [🛠️ Installation](#️-installation)
- [🌿 Usgae](#️-usage)
- [💻 Tech Stack](#-tech-stack)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🚀 Features

- Upload images of plant leaves (`.jpg`, `.jpeg`, `.png`).
- Classifies the uploaded image into the correct plant disease category.
- Displays the predicted class with a **confidence score**.
- Simple and interactive **UI built with Streamlit**.
- 📊 Validation Accuracy: 90.65%

---

## 📁 Project Structure
```md
plant_disease_prediction/
│── Models/
│ │── plant_model.keras # Trained Keras model
│── Notebooks/
│── Streamlit/
│ │── app.py # Streamlit app
│── requirements.txt # Dependencies
│── README.md # Project documentatio

```

📊 Model Overview

- Model Type: Convolutional Neural Network (CNN)
- Architecture:
  -- Multiple convolution + pooling layers for feature extraction
  -- Fully connected layers for classification
- Input Features:
  -- Leaf images of different plants (RGB format, resized to fixed dimensions)
- Output:
  -- Predicted plant disease category (e.g., healthy, bacterial spot, late blight, etc.)
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy

Goal: Accurately classify plant leaf images into disease categories to help farmers and researchers with early detection and treatment.

### ✅ Performance Metrics

Validation Accuracy: 90.65%

---

## 🛠️ Installation

```bash
# 1️⃣ Clone the repo
git clone https://github.com/subham-28/plant_disease_prediction.git
cd plant_disease_prediction

# 2️⃣ Set up environment
pip install -r requirements.txt

# 3️⃣ Pull model tracked with Git LFS
git lfs install
git lfs pull

# 4️⃣ Run the Streamlit app
streamlit run Streamlit/app.py

```

## 🌿 Usage

1) Upload a leaf image.
2) Click Classify.
3) View predicted disease with confidence score.

---

## 💻 Technologies Stack

| Layer               | Technology                                                                 |
|---------------------|----------------------------------------------------------------------------|
| **Frontend**        | Streamlit                                                                 |
| **Model**           | Convolutional Neural Network (CNN) built with TensorFlow / Keras           |
| **Preprocessing**   | ImageDataGenerator (rescaling, augmentation), NumPy, OpenCV                |
| **Data Visualization** | Matplotlib, Seaborn                                                     |
| **Experimentation** | Jupyter Notebooks                                                         |
| **Deployment**      | Streamlit / Streamlit Cloud                                                |
| **Version Control** | Git, GitHub, Git LFS (for large model files)                               |
  

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch
3. Make your changes
4. Submit a pull request!

---

## 📜 License
MIT License. Feel free to use, modify, and share!

---

## 🙌 Acknowledgements
* TensorFlow/Keras for model training.
* Streamlit for interactive UI.
* Plant disease dataset used for training (Kaggle / public datasets).

---

Made by Subham Mohanty

