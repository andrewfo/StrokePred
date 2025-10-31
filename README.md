
# Stroke Risk Prediction with Advanced Machine Learning

## Project Overview

This repository contains a comprehensive machine learning pipeline designed to predict the likelihood of a patient having an **ischemic** or **hemorrhagic stroke** based on a diverse set of health metrics and lifestyle factors.

The project addresses the critical challenge of **class imbalance** (low occurrence of stroke cases) by employing the **SMOTE** (Synthetic Minority Over-sampling Technique) algorithm. It systematically evaluates four distinct classification models (**Logistic Regression**, **Random Forest**, **Neural Network (MLP)**, and **Support Vector Machine (SVM)**) and culminates in a **Stacking Ensemble model** to achieve optimal predictive performance.

---

## Dataset and Preprocessing

The model is trained on the `stroke_data.csv` (or a similar dataset).

### Key Preprocessing Steps

* **Missing Value Imputation:**
  Missing values in the `bmi` column are filled with the mean of the column.

* **Feature Encoding:**
  Categorical features (e.g., `work_type`, `smoking_status`) are converted into numerical format using `OneHotEncoder`.

* **Data Scaling:**
  Numerical features are standardized using `StandardScaler` to ensure equal contribution from all variables.

* **Imbalance Handling:**
  The training data is balanced using the **SMOTE** technique to mitigate bias toward the majority class.

---

## Modeling and Evaluation

Four base models and one ensemble model were implemented and benchmarked using metrics critical for imbalanced datasets, including:

* **Accuracy**
* **F1-Score**
* **ROC-AUC**
* **Precision-Recall AUC (PR-AUC)**

---

## Model Details

### Logistic Regression

* Baseline performance
* `class_weight="balanced"`

### Random Forest

* Feature importance analysis
* Hyperparameter tuning

### Neural Network (MLP)

* Multi-layer Perceptron architecture
* Early stopping for regularization

### Support Vector Machine (SVM)

* `probability=True`
* `class_weight="balanced"`

### Stacking Ensemble

* Combines predictions from all four base models
* Uses a final **Logistic Regression meta-classifier** for robust generalization

---

## Model Interpretability with SHAP

Understanding why a model makes a specific prediction is crucial in healthcare applications.
This project integrates **SHAP** (SHapley Additive exPlanations) to provide both **local** and **global** interpretability for the **Logistic Regression** and **Neural Network** models.

These insights highlight which health factors most influence stroke risk predictions for individual patients.

---

## How to Run

### Prerequisites

* **Python:** 3.8+
* **Required Libraries:**
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imbalanced-learn`, `shap`

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn shap
```

### Execution

1. Ensure you have `stroke_data.csv` in your project directory.
2. Run the main script:

```bash
python stroke.py
```

The script outputs model performance metrics and classification reports, and displays visualizations for:

* Confusion Matrices
* ROC Curves
* SHAP Plots

