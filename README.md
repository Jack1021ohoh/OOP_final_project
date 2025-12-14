# Stroke Prediction Using Machine Learning

A comprehensive machine learning project that predicts stroke likelihood using multiple algorithms, advanced preprocessing techniques, and anomaly detection methods.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project implements a machine learning pipeline for stroke prediction, developed as part of an Object-Oriented Programming course. The system employs advanced data preprocessing techniques, handles class imbalance, and utilizes both predictive modeling and anomaly detection approaches to identify individuals at risk of stroke.

## Features

- **Advanced Data Preprocessing**
  - Hash encoding for categorical variables
  - Standardization for numerical features (age, BMI, average glucose level)
  - Efficient handling of diverse data types

- **Imbalanced Data Handling**
  - Combined SMOTE (Synthetic Minority Over-sampling Technique) and ENN (Edited Nearest Neighbors)
  - Addresses severe class imbalance (stroke ratio: 4.2%)

- **Multiple Modeling Approaches**
  - Supervised learning: Random Forest and LightGBM
  - Anomaly detection: One-Class SVM and XGBoost-based outlier detection

- **Model Interpretability**
  - SHAP (SHapley Additive exPlanations) values for feature importance
  - Comprehensive performance metrics and visualizations

## Dataset

**Source**: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

**Dataset Characteristics**:
- Total samples: 5,110
- Stroke cases: 249 (4.9%)
- Non-stroke cases: 4,861 (95.1%)

**Features**:
- **Categorical**: gender, hypertension, heart_disease, ever_married, work_type, Residence_type, smoking_status
- **Numerical**: age, avg_glucose_level, BMI
- **Target**: stroke (binary: 0/1)

## Methodology

### 1. Data Preprocessing
- **Numerical Variables**: StandardScaler normalization
- **Categorical Variables**: HashingEncoder for efficient encoding
- **Missing Values**: Appropriate imputation strategies

### 2. Model Training

#### Without Resampling
- **Random Forest**: Accuracy 95%, AUC 0.82
- **LightGBM**: Accuracy 93%, AUC 0.83

#### With SMOTE + ENN Resampling
- **Random Forest**: Accuracy 79%, AUC 0.66
- **LightGBM**: Accuracy 72%, AUC 0.81

### 3. Anomaly Detection
- **One-Class SVM**: Precision 0.10, AUC 0.70
- **XGBoost Outlier Detection**: Precision 0.20, AUC 0.84

## Results

### Key Findings

| Model | Metric | Before Resampling | After Resampling |
|-------|--------|-------------------|------------------|
| **Random Forest** | Accuracy | 0.95 | 0.79 |
|  | AUC | 0.82 | 0.66 |
| **LightGBM** | Accuracy | 0.93 | 0.72 |
|  | AUC | 0.83 | 0.81 |

**Best Model**: LightGBM with SMOTE+ENN achieved the highest AUC (0.81), providing better balance between sensitivity and specificity for stroke prediction.

## Technologies

### Core Technologies
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)

### Libraries & Frameworks
- **scikit-learn**: Machine learning algorithms and preprocessing
- **LightGBM**: Gradient boosting framework
- **imbalanced-learn**: SMOTE and ENN resampling
- **category_encoders**: Hash encoding for categorical variables
- **PyOD**: Anomaly detection algorithms
- **SHAP**: Model interpretability and feature importance
- **matplotlib & seaborn**: Data visualization
- **pandas & numpy**: Data manipulation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OOP_final_project.git
cd OOP_final_project

# Install required packages
pip install scikit-learn lightgbm imbalanced-learn category_encoders pyod shap matplotlib seaborn pandas numpy
```

## Usage

```python
# Open the Jupyter notebook
jupyter notebook OOP_finalterm.ipynb

# Or run the notebook cells sequentially to:
# 1. Load and preprocess the data
# 2. Train models with/without resampling
# 3. Evaluate model performance
# 4. Generate SHAP visualizations
```

## Model Performance

### Classification Metrics (LightGBM with SMOTE+ENN)
- **Accuracy**: 0.72
- **AUC-ROC**: 0.81
- **Precision**: Optimized for minority class detection
- **Recall**: Improved through resampling techniques

### ROC Curves
The project includes comprehensive ROC curve analysis for all models, demonstrating the trade-offs between true positive rate and false positive rate.

## Feature Importance

Based on **SHAP analysis** and **LightGBM feature importance (Gain)**:

1. **Age**: Most significant predictor (highest gain: 93,510)
2. **Average Glucose Level**: Second most important (gain: 56,097)
3. **BMI**: Third ranking feature (gain: 44,889)
4. **Ever Married**: Moderate importance (gain: 5,859)
5. **Hypertension**: Contributing factor (gain: 5,800)

Other features include heart disease, residence type, gender, and smoking status.

## Contributors

This project was developed as part of an Object-Oriented Programming course final project.

## License

This project is available for educational purposes. Please refer to the dataset source for data usage terms.
