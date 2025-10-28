Stroke Risk Prediction with Advanced Machine Learning

Project Overview

This repository contains a comprehensive machine learning pipeline designed to predict the likelihood of a patient having an ischemic or hemorrhagic stroke based on a diverse set of health metrics and lifestyle factors.

The project addresses the critical challenge of class imbalance (low occurrence of stroke cases) by employing the SMOTE (Synthetic Minority Over-sampling Technique) algorithm. It systematically evaluates four distinct classification models—Logistic Regression, Random Forest, Neural Network (MLP), and Support Vector Machine (SVM)—and culminates in a Stacking Ensemble model to achieve optimal predictive performance.

 Dataset and Preprocessing

The model is trained on the healthcare-dataset-stroke-data.csv (or similar) dataset.

Key Preprocessing Steps:

Missing Value Imputation: Missing values in the bmi column are filled with the mean of the column.

Feature Encoding: Categorical features (e.g., work_type, smoking_status) are converted into numerical format using OneHotEncoder.

Data Scaling: Numerical features are standardized using StandardScaler to ensure equal contribution from all variables.

Imbalance Handling: The training data is balanced using the SMOTE technique to mitigate bias towards the majority class.

Modeling and Evaluation

Four base models and one ensemble model were implemented and benchmarked using a suite of metrics critical for imbalanced data problems, including Accuracy, F1-Score, ROC-AUC, and Precision-Recall AUC (PR-AUC).

Model

Key Techniques

Logistic Regression

Baseline performance, class_weight="balanced"

Random Forest

Feature importance analysis, Hyperparameter tuning

Neural Network (MLP)

Multi-layer Perceptron, Early stopping for regularization

Support Vector Machine (SVM)

probability=True, class_weight="balanced"

Stacking Ensemble

Combines predictions from all four base models using a final Logistic Regression meta-classifier for robust generalization.

 Model Interpretability with SHAP

Understanding why a model makes a specific prediction is crucial in a healthcare context. This project integrates SHAP (SHapley Additive exPlanations) to provide local and global interpretability for the Logistic Regression and Neural Network models, allowing insight into which health factors drive the risk prediction for individual patients.

How to Run

This project requires Python and standard machine learning libraries.

Prerequisites

Python 3.8+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn, shap

Installation

pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn shap


Execution

Ensure you have stroke_data.csv in your project directory.

Run the main script:

python stroke.py


The script will output model performance metrics, classification reports, and display visualizations for confusion matrices, ROC curves, and SHAP plots.
