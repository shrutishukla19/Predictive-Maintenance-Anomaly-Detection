# Predictive Maintenance using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Highlights

- End-to-end Predictive Maintenance solution
- Complete Data Science Lifecycle implementation
- Domain-driven Feature Engineering
- Isolation Forest and Local Outlier Factor for anomaly detection
- Random Forest classifier for supervised prediction
- Hyperparameter tuning and model comparison
- Custom Scikit-learn Transformers
- Production-style Scikit-learn Pipeline
- Model serialization using Joblib
- Inference directly on raw machine data
- Business-oriented interpretation of predictions

---

# Project Overview

This project develops an end-to-end Machine Learning solution for Predictive Maintenance using the AI4I 2020 Predictive Maintenance Dataset.

The objective is to predict whether an industrial machine is likely to fail based on its operating conditions before an actual breakdown occurs.

The project demonstrates the complete Data Science lifecycle including:

- Business Understanding
- Data Understanding
- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Anomaly Detection
- Supervised Machine Learning
- Model Evaluation
- Production-style ML Pipeline
- Model Serialization
- Inference on Raw Data

Unlike many notebook-based projects, the final solution uses a reusable Scikit-learn Pipeline with custom transformers, ensuring identical preprocessing during both training and inference.

---

# Business Problem

Unexpected machine failures can result in:

- Unplanned production downtime
- Increased maintenance costs
- Reduced operational efficiency
- Safety risks
- Lower equipment availability

Traditional preventive maintenance replaces components at fixed intervals, regardless of their actual condition. This often leads to unnecessary maintenance or unexpected failures.

Predictive Maintenance enables organizations to identify machines at risk of failure in advance, allowing maintenance teams to perform targeted repairs and minimize operational disruption.

---

# Business Impact

A successful predictive maintenance system can help organizations:

- Reduce unplanned downtime
- Optimize maintenance schedules
- Lower maintenance costs
- Increase equipment lifespan
- Improve production efficiency
- Improve asset reliability
- Support data-driven maintenance decisions

---

# Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

## Features

- Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

### Target Variable

Machine Failure

- 0 → Normal
- 1 → Failure

Additional failure-specific columns available:

- TWF
- HDF
- PWF
- OSF
- RNF

These failure-type columns were excluded during training because they are only known after a machine fails and would introduce target leakage.

---

# Project Workflow

```
Business Understanding
        │
        ▼
Data Understanding
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ▼
Model Development
    ├── Isolation Forest
    ├── Local Outlier Factor
    └── Random Forest
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Pipeline Serialization
        │
        ▼
Inference on Raw Data
```

---

# Exploratory Data Analysis

EDA was performed to understand:

- Data quality
- Missing values
- Duplicate records
- Feature distributions
- Class imbalance
- Feature relationships
- Correlation between variables
- Failure distribution

Insights obtained during EDA guided feature engineering and model selection.

---

# Feature Engineering

To better capture machine operating conditions, the following domain-inspired features were engineered.

| Feature | Description |
|----------|-------------|
| Temperature_Difference | Process Temperature − Air Temperature |
| Mechanical_Load | Torque × Rotational Speed |
| Wear_Stress | Tool Wear × Torque |
| Wear_Heat | Tool Wear × Process Temperature |

These engineered features represent mechanical load, thermal stress, and wear conditions that may contribute to machine failure.

---

# Why Multiple Models?

Machine failure prediction can be approached using either supervised or unsupervised learning depending on label availability.

To compare different approaches, three models were implemented.

## Isolation Forest

- Unsupervised anomaly detection
- Suitable when failure labels are unavailable
- Detects abnormal operating behaviour

## Local Outlier Factor (LOF)

- Density-based anomaly detection
- Detects local deviations in data density
- Implemented using Novelty Detection

## Random Forest

- Supervised classification
- Uses historical failure labels
- Handles nonlinear relationships effectively
- Selected as the final model based on overall performance

Implementing multiple approaches provides a better understanding of anomaly detection versus supervised classification.

---

# Hyperparameter Tuning

Isolation Forest was manually tuned using different values of:

- Number of Trees
- Maximum Samples
- Maximum Features
- Bootstrap Sampling
- Contamination Rate

Random Forest was also evaluated with different numbers of trees before selecting the final model.

Models were compared using:

- Precision
- Recall
- F1-score

---

# Model Performance

| Model | Precision | Recall | F1 Score |
|---------|----------:|-------:|---------:|
| Isolation Forest | 0.26 | 0.40 | 0.32 |
| Local Outlier Factor | 0.16 | 0.25 | 0.19 |
| Random Forest | 0.95 | 0.82 | 0.88 |

Random Forest achieved the best balance between precision and recall and was selected as the final model.

---

# Engineering Decisions

Several engineering decisions were made to improve maintainability and reproducibility.

### Custom Scikit-learn Transformers

Reusable transformers were created for:

- Feature Engineering
- Column Dropping

This avoids duplication of preprocessing logic across notebooks.

---

### End-to-End ML Pipeline

A Scikit-learn Pipeline was implemented to automate preprocessing.

```
Raw Machine Data
        │
        ▼
FeatureEngineeringTransformer
        │
        ▼
ColumnDropTransformer
        │
        ▼
OneHotEncoder
        │
        ▼
RandomForestClassifier
        │
        ▼
Prediction
```

The pipeline guarantees identical preprocessing during training and inference.

---

### Model Serialization

The complete pipeline was serialized using Joblib.

The saved pipeline contains:

- Feature Engineering
- Column Dropping
- Encoding
- Trained Random Forest Model

This enables prediction directly from raw machine data without requiring manual preprocessing.

---

# Design Decisions

Several design decisions were made to improve robustness and maintainability.

- Feature engineering was integrated into the pipeline rather than performed manually.
- Custom Scikit-learn transformers were implemented to avoid code duplication.
- Identifier columns were excluded because they do not contribute predictive information.
- Failure-specific columns were removed to prevent target leakage.
- Both supervised and unsupervised approaches were evaluated to compare different predictive strategies.
- The complete pipeline was serialized using Joblib for consistent inference.

---

# Results

The Random Forest model achieved the best overall performance.

| Metric | Value |
|---------|-------|
| Precision | 0.95 |
| Recall | 0.82 |
| F1 Score | 0.88 |

The serialized pipeline successfully predicts machine failures directly from raw machine data without requiring manual preprocessing.

---

# Repository Structure

```
Predictive-Maintenance-Anomaly-Detection/

│
├── data/
│   ├── raw/
│   ├── interim/
│
├── models/
│   ├── random_forest_pipeline.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Inference.ipynb
│
├── reports/
│   ├── inference_predictions.csv
│
├── src/
│   ├── __init__.py
│   └── transformers.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-github-username>/Predictive-Maintenance-Anomaly-Detection.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the notebooks in the following order:

1. 01_EDA.ipynb
2. 02_Model_Training.ipynb
3. 03_Inference.ipynb

---

# Inference Workflow

The trained pipeline accepts completely raw machine data.

```
Load Pipeline
        │
        ▼
Load Raw Machine Data
        │
        ▼
Automatic Feature Engineering
        │
        ▼
Automatic Column Removal
        │
        ▼
Automatic Encoding
        │
        ▼
Failure Prediction
        │
        ▼
Failure Probability
```

No manual preprocessing is required during inference.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

# Skills Demonstrated

- Business Problem Framing
- Exploratory Data Analysis
- Feature Engineering
- Anomaly Detection
- Supervised Machine Learning
- Hyperparameter Tuning
- Class Imbalance Handling
- Custom Scikit-learn Transformers
- Machine Learning Pipelines
- Model Serialization
- Production-style Inference
- Model Evaluation
- ML Engineering Best Practices

---

# Future Improvements

Potential enhancements include:

- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- Optuna Hyperparameter Optimization
- Model Monitoring
- FastAPI Deployment
- Docker
- Kubernetes
- Azure Machine Learning Deployment
- CI/CD Pipeline

---

# License

This project is intended for educational and portfolio purposes.

---

# Acknowledgements

- AI4I 2020 Predictive Maintenance Dataset
- UCI Machine Learning Repository

---

# Author

**Shruti Shukla**

Machine Learning | Data Science | Predictive Maintenance | Anomaly Detection
