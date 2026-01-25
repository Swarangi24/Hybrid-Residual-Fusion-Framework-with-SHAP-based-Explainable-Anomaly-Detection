# Hybrid Residual Fusion Framework with SHAP-based Explainable Anomaly Detection

## Overview
This project presents a **hybrid anomaly detection framework** for time-series energy consumption data that combines **machine learning, deep learning, residual analysis, multi-detector fusion, and explainable AI (XAI)**.  
The system is designed to not only detect anomalies accurately but also explain **why** they occur, enabling transparency, trust, and production readiness.

---

## Key Features
- Hybrid modeling using **Random Forest** and **LSTM**
- Residual-based anomaly detection for improved sensitivity
- Multi-detector approach (Moving Average, Isolation Forest, One-Class SVM)
- Weighted anomaly score fusion for robustness
- **SHAP-based explainability** using TreeSHAP and surrogate modeling
- Human-readable root-cause explanations for each anomaly

---

## System Architecture
1. Feature Engineering  
2. Random Forest & LSTM Prediction  
3. Residual Generation  
4. Anomaly Detection  
5. Hybrid Residual Fusion  
6. SHAP-based Explainability  
7. Root-Cause Reporting  

---

## Technologies Used
- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Deep Learning:** TensorFlow / Keras (LSTM)  
- **Explainable AI:** SHAP (TreeSHAP)  
- **Anomaly Detection:** Isolation Forest, One-Class SVM  
- **Visualization:** Matplotlib  

---

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



---

## Workflow

### 1. Feature Engineering
Transforms raw smart meter data into temporal, rolling, and lag-based features.

### 2. Prediction & Residuals
Random Forest and LSTM models predict energy consumption.  
Residuals (actual − predicted) represent abnormal deviations.

### 3. Anomaly Detection
Residuals are analyzed using:
- Moving Average thresholding
- Isolation Forest
- One-Class SVM

### 4. Hybrid Fusion
Anomaly scores are fused using optimized weights to reduce false positives and improve reliability.

### 5. Explainable AI (XAI)
- TreeSHAP explains Random Forest predictions  
- Surrogate Random Forest explains LSTM residual behavior  
- Combined SHAP scores rank anomaly root causes

### 6. Root-Cause Reporting
Each anomaly includes:
- Top contributing features
- Temporal context
- Appliance-level impact
- Human-readable explanations

---

## Results
- High predictive performance (R² ≈ 0.95)
- Detection of over **20,000 high-confidence anomalies**
- Robust anomaly detection via score fusion
- Clear, interpretable explanations for each anomaly

---

## Use Cases
- Smart home energy monitoring
- Faulty appliance detection
- Energy efficiency optimization
- Explainable IoT anomaly detection
- Academic and research applications

---


## Project Structure
