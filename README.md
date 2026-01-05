
# Image Classifier with TensorFlow (Transfer Learning)

End-to-end image classification project using **TensorFlow** and **MobileNetV2**.  
This project demonstrates a complete deep learning workflow: data ingestion, preprocessing, baseline modeling, transfer learning, evaluation, model export, and inference.

---

## Project Overview

**Objective:**  
Build an accurate image classifier using a pretrained convolutional neural network and evaluate its performance on unseen data.

**Key Techniques:**
- Convolutional Neural Networks (CNNs)
- Transfer Learning with MobileNetV2 (ImageNet weights)
- `tf.data` pipelines for efficient input processing
- Model evaluation with accuracy, confusion matrix, and classification report
- Exporting models for reuse and inference

**Dataset:**  
CIFAR-10 (10 image classes, loaded via `tensorflow_datasets`)

---

## Repository Structure

```text
image-classifier-tensorflow/
│
├── notebooks/
│   └── 01_image_classifier_end_to_end.ipynb
│
├── src/
│   └── inference.py
│
├── models/
│   ├── keras/                # exported .keras models (ignored by git)
│   └── class_names.json
│
├── reports/
│   ├── figures/              # confusion matrix, training curves, prediction grids
│   └── metrics/              # classification report + metrics JSON
│
├── requirements.txt
├── .gitignore
└── README.md
```
---
## Workflow Summary

### Data Loading
- CIFAR-10 loaded using `tensorflow_datasets`
- Train / validation / test splits created

### Preprocessing
- Resize images to model input size
- Normalize pixel values to `[0, 1]`
- Build efficient `tf.data` pipelines

### Baseline Model
- Lightweight CNN trained from scratch
- Used as a performance reference

### Transfer Learning
- MobileNetV2 pretrained on ImageNet
- Backbone frozen, custom classification head trained
- Callbacks for early stopping and learning-rate scheduling

### Evaluation
- Accuracy on validation and test sets
- Confusion matrix
- Precision / recall / F1-score per class

### Export & Inference
- Model exported in `.keras` format
- CLI inference script for predictions on new images

---

## Results (CIFAR-10)

**Transfer Learning – MobileNetV2 (Frozen Backbone)**
- Validation Accuracy: **high 80s / low 90s (varies by run)**
- Test Accuracy: **comparable to validation**

### Saved Artifacts
- Confusion Matrix:  
  `reports/figures/confusion_matrix_mobilenetv2_frozen.png`

- Training Curves:  
  - `reports/figures/mobilenetv2_frozen_accuracy.png`  
  - `reports/figures/mobilenetv2_frozen_loss.png`

- Classification Report:  
  `reports/metrics/classification_report_mobilenetv2_frozen.txt`

---

## How to Run

### 1. Set up environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

```
### 2. Run the notebook (training + evaluation)

Open and run:

```text
notebooks/01_image_classifier_end_to_end.ipynb
```
This notebook:

- trains the baseline and transfer learning models

- evaluates performance

- exports the trained model

- saves all figures and metrics
---
## Run Inference (CLI)
Use the exported .keras model to run predictions from the command line.
```bash
python src/inference.py \
  --model models/keras/<your_model>.keras \
  --num 9 \
  --save-grid reports/figures/pred_grid.png
```
### What this does
- loads the trained model

- runs inference on CIFAR-10 test images

- prints predictions + confidence scores

- saves a grid of predicted images (optional)

## Technologies Used

- Python

- TensorFlow / Keras

- TensorFlow Datasets

- NumPy

- Matplotlib

- scikit-learn

## Next Improvements

- Fine-tune upper MobileNetV2 layers for additional accuracy gains

- Add custom dataset support for real-world images

- Package inference as a lightweight API or web demo

- Experiment with alternative pretrained backbones (EfficientNet, ResNet)

## Author


**Juante Wilson**
Aspiring Applied Data Scientist / Machine Learning Engineer
Focused on end-to-end ML projects and AWS-centric workflows



  

