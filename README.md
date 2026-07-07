# MoringaLeaf_Classifier 🌿

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A robust, research-grade deep learning framework for classifying the health of Moringa leaves. This repository contains the complete experimental pipeline used to benchmark multiple deep learning architectures and generate Explainable AI (XAI) visualizations for precision agriculture.

---

## 🌟 Key Features

1. **Multi-Architecture Benchmarking**: Support for training and evaluating 6 state-of-the-art architectures (InceptionV3, ResNet50, DenseNet121, EfficientNetB0, MobileNetV3, ViT).
2. **Explainable AI (XAI)**: Integrated Grad-CAM and LIME for visual model interpretability, ensuring models focus on genuine disease symptoms rather than background noise.
3. **Rigorous Validation**: Built-in 5-fold stratified cross-validation for statistically sound performance metrics.
4. **Comprehensive Evaluation**: Automated generation of ROC-AUC curves, normalized confusion matrices, and per-class classification reports.
5. **Systematic Ablation Studies**: Scripts to quantify the impact of data augmentation and fine-tuning strategies.
6. **Web Application**: A Flask-based web interface for real-time inference and XAI visualization.

---

## 🚀 Getting Started / Installation

### 1. Prerequisites

- Python 3.8 or later
- CUDA Toolkit (if using GPU acceleration)

### 2. Clone the Repository

```bash
git clone https://github.com/c-onfused69/MoringaLeaf_Classifier.git
cd MoringaLeaf_Classifier
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Setup

Ensure your dataset is organized inside the `dataset/` directory as defined in `config.py`. The structure should be:

```
dataset/
├── traning_set/
│   ├── diseased/
│   └── healthy/
├── validation_set/
│   ├── diseased/
│   └── healthy/
└── testing_set/
    ├── diseased/
    └── healthy/
```

---

## 📊 Experimental Pipeline

The framework is driven by a central configuration file (`config.py`). All paths, hyperparameters, and augmentation settings can be modified there.

### Training

Train models using cross-validation. The script supports two-phase training (frozen backbone followed by fine-tuning).

```bash
# Train models on all 5 folds
python train.py --model inception_v3
python train.py --model resnet50
python train.py --model densenet121
python train.py --model efficientnetb0
python train.py --model mobilenetv3
python train.py --model vit

# Example: Train DenseNet121 on a single fold for a quick test
python train.py --model densenet121 --folds 1 --quick-test
```

### Evaluation

Generate comprehensive metrics (Accuracy, F1, Kappa, AUC) and publication-quality plots.

```bash
# Aggregate results across all folds for each model
python evaluate.py --model-name inception_v3 --all-folds
python evaluate.py --model-name resnet50 --all-folds
python evaluate.py --model-name densenet121 --all-folds
python evaluate.py --model-name efficientnetb0 --all-folds
python evaluate.py --model-name mobilenetv3 --all-folds
python evaluate.py --model-name vit --all-folds

# Example: Evaluate a specific fold
python evaluate.py --model-name inception_v3 --fold 0
```

### Ablation Studies

Run controlled experiments to test specific design choices.

```bash
# Test the impact of data augmentation for each model
python ablation.py --experiment augmentation --model inception_v3
python ablation.py --experiment augmentation --model resnet50
python ablation.py --experiment augmentation --model densenet121
python ablation.py --experiment augmentation --model efficientnetb0
python ablation.py --experiment augmentation --model mobilenetv3
python ablation.py --experiment augmentation --model vit

# Test freezing strategies
python ablation.py --experiment freezing --model inception_v3
python ablation.py --experiment freezing --model resnet50
python ablation.py --experiment freezing --model densenet121
python ablation.py --experiment freezing --model efficientnetb0
python ablation.py --experiment freezing --model mobilenetv3
python ablation.py --experiment freezing --model vit
```

### Publication Assets

Generate the final efficiency metrics (FLOPs, latency) and export publication-ready LaTeX tables for your manuscript.

```bash
# Run model efficiency profiling (simulates CPU edge deployment)
python efficiency.py

# Export all metrics to LaTeX tables
python export_tables.py
```

---

## 💻 Web Application

To deploy the local web application for testing predictions with Dual XAI overlays (Grad-CAM & LIME):

```bash
# 1. Activate the environment
venv\Scripts\activate

# 2. Run the application
python app.py
```

Then navigate to `http://127.0.0.1:5000/` in your browser.

### 🐳 Docker Deployment

For 100% reproducibility and cross-platform compatibility, you can run the web app using Docker:

```bash
docker compose up --build
```

This will automatically build the environment with all dependencies and serve the application on port 5000.

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras, timm (Transformers)
- **Computer Vision**: OpenCV, scikit-image
- **Data & Evaluation**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Flask, Bootstrap 5
- **Explainability**: tf-keras-vis (Grad-CAM), LIME

---

## 📝 Legacy Scripts

Earlier iterations of procedural scripts (`main.py`, `evaluation.py`, etc.) have been moved to the `legacy/` directory for historical reference. The new modular pipeline (`train.py`, `evaluate.py`) replaces them.

---

## 📫 Contact / Connect

Name: Md Nahijul Islam Niloy  
Email: nniloy888@gmail.com  
GitHub: [Nahijul Islam Niloy](https://github.com/c-onfused69)

_If you use this framework in your research, please consider citing this repository._
