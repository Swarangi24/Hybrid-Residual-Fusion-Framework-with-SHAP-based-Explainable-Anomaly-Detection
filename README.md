Overview

This project implements a hybrid anomaly detection framework for time-series energy consumption data by combining machine learning, deep learning, residual analysis, anomaly fusion, and explainable AI (XAI).

The system is designed to accurately detect anomalies while also providing clear, human-interpretable explanations for why each anomaly occurred, enabling trust, transparency, and real-world deployment readiness.

Key Objectives

Detect anomalous energy consumption patterns in smart meter data

Combine strengths of Random Forest and LSTM models using residual fusion

Improve anomaly robustness using multiple detectors

Explain anomalies using SHAP for model interpretability

Generate user-friendly root-cause explanations for detected anomalies

System Architecture

Feature Engineering

Temporal features (hour, day, week, month)

Rolling statistics and lag-based features

Appliance-level and consumption-derived features

Predictive Models

Random Forest Regressor (captures non-linear feature interactions)

LSTM Network (captures temporal dependencies)

Residual Generation

Compute residuals between actual and predicted energy consumption

Residuals serve as the primary anomaly signal

Anomaly Detection

Moving Average (MAD-based thresholding)

Isolation Forest

One-Class SVM

Hybrid Fusion

Weighted fusion of anomaly scores from multiple detectors

Optimized weights to improve detection reliability

Explainable AI (XAI)

TreeSHAP for Random Forest explanations

Surrogate Random Forest trained on LSTM residuals

Combined SHAP scoring to rank root causes

Human-Readable Reporting

Contextual explanations (time, baseline behavior, deviation)

Feature-level contribution analysis

Appliance and temporal impact summaries

Technologies Used

Programming Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn

Deep Learning: TensorFlow / Keras (LSTM)

Explainability: SHAP (TreeSHAP)

Anomaly Detection: Isolation Forest, One-Class SVM

Visualization: Matplotlib, SHAP plots

Project Structure
.
├── data/
│   └── processed_energy_data.csv
│
├── models/
│   ├── random_forest_model.joblib
│   ├── lstm_model.h5
│   └── surrogate_rf_lstm_resid.joblib
│
├── fusion/
│   ├── fusion_final_results.csv
│   └── fusion_final_meta.json
│
├── explainability_reports/
│   ├── root_cause_reports.json
│   └── plots/
│
├── notebooks/
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   └── explainability_analysis.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── anomaly_detection.py
│   ├── fusion_logic.py
│   └── explainability.py
│
├── requirements.txt
└── README.md

How It Works
Step 1: Feature Engineering

Raw smart meter data is transformed into meaningful time-series features to capture usage patterns and trends.

Step 2: Prediction & Residuals

Both Random Forest and LSTM models predict energy consumption.
Residuals (actual − predicted) represent abnormal deviations.

Step 3: Anomaly Detection

Multiple detectors independently analyze residuals to identify abnormal points.

Step 4: Hybrid Fusion

Anomaly scores are combined using optimized weights to reduce false positives and increase detection confidence.

Step 5: Explainability

SHAP explains Random Forest predictions directly

A surrogate Random Forest explains LSTM residual behavior

SHAP values are combined with observed feature deviations

Step 6: Root-Cause Reporting

Each anomaly is accompanied by:

Top contributing features

Temporal context

Appliance-level impact

Human-readable explanation text

Results & Performance

High predictive accuracy (R² ≈ 0.95)

Over 20,000 high-confidence anomalies detected

Robust detection through multi-detector fusion

Clear root-cause explanations for each anomaly

Research-grade explainability suitable for production use

Use Cases

Smart home energy monitoring

Faulty appliance detection

Energy efficiency optimization

Explainable anomaly detection for IoT systems

Research and academic applications in XAI
