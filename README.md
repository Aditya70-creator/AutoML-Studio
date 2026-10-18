# 🧪 AutoML-Studio — Leak-Free Tabular AutoML & Model Registry

![Status](https://img.shields.io/badge/Status-In_Development-A855F7?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-AutoML_%26_MLOps_Platform-00E7FF?style=for-the-badge)
![Stack](https://img.shields.io/badge/Stack-Scikit--Learn_%7C_Optuna_%7C_MLflow_%7C_Streamlit-3B82F6?style=for-the-badge)

> **AutoML-Studio** is a modular tabular AutoML platform designed to automate dataset profiling, task inference, leakage-aware preprocessing, Bayesian hyperparameter optimization, model evaluation, explainability, and experiment tracking through MLflow.

---

## 🏗️ System Architecture

```mermaid id="a7mx3z"
flowchart LR
    CSV["Raw Tabular Dataset"] --> Audit["Data Quality Audit<br/>& Task Inference"]
    Audit --> Pipe["Leak-Free Preprocessing<br/>Scikit-learn Pipeline"]
    Pipe --> Opt["Optuna Hyperparameter<br/>Optimization"]
    Opt --> Gate["Model Evaluation<br/>& Quality Gate"]
    Gate --> SHAP["SHAP Explainability<br/>& Model Report"]
    SHAP --> Reg[("MLflow Tracking<br/>& Model Registry")]

    style CSV fill:#070b14,stroke:#00E7FF,stroke-width:2px,color:#ffffff
    style Audit fill:#0f172a,stroke:#3B82F6,stroke-width:2px,color:#ffffff
    style Pipe fill:#0f172a,stroke:#00E7FF,stroke-width:2px,color:#ffffff
    style Opt fill:#0f172a,stroke:#8B5CF6,stroke-width:2px,color:#ffffff
    style Gate fill:#0f172a,stroke:#10B981,stroke-width:2px,color:#ffffff
    style SHAP fill:#0f172a,stroke:#c084fc,stroke-width:2px,color:#ffffff
    style Reg fill:#070b14,stroke:#f59e0b,stroke-width:2px,color:#ffffff
```

## 🎯 Project Objectives

- Automate dataset profiling and identify potential data quality issues.
- Infer whether a dataset represents a classification or regression task.
- Build leakage-aware preprocessing and model evaluation pipelines.
- Optimize model hyperparameters with Optuna.
- Compare multiple model families using suitable evaluation metrics.
- Generate explainability reports and reproducible experiment artifacts.
- Track model versions and support controlled registration through MLflow.

## 🔧 Core ML & Platform Engineering Focus

### 1. Automated Data Quality & Schema Audit

Profiles datasets to identify:

- High-missingness and constant columns.
- Duplicate rows and duplicate features.
- Potential target leakage and suspicious identifiers.
- Unexpected data types and high-cardinality categorical features.
- Invalid values and inconsistent target labels.

These checks flag potential problems for review; automated profiling alone cannot prove that a feature causes leakage.

### 2. Leakage-Aware Preprocessing

Uses `scikit-learn` `Pipeline` and `ColumnTransformer` to encapsulate preprocessing steps, including:

- Missing-value imputation.
- Numerical scaling where appropriate.
- Categorical encoding.
- Feature selection.
- Optional target encoding with fold-aware handling.

All learned preprocessing transformations must be fitted using training data only. Cross-validation should evaluate the complete pipeline, not preprocessing performed on the full dataset beforehand.

### 3. Bayesian Hyperparameter Optimization

Uses **Optuna** to search model hyperparameters through its optimization framework, with the Tree-structured Parzen Estimator (TPE) sampler as an option.

Candidate models may include:

- Logistic Regression and Ridge baselines.
- Random Forest.
- Gradient-boosted tree models such as XGBoost and LightGBM.

The evaluation strategy adapts to the task: stratified K-fold cross-validation for suitable classification datasets and appropriate K-fold or time-aware validation for regression or temporal data.

### 4. Model Evaluation & Quality Gates

Evaluates candidate models against configurable quality criteria before registration.

| Task | Example metrics |
|---|---|
| Classification | F1-score, Precision, Recall, ROC-AUC, PR-AUC |
| Regression | MAE, RMSE, R² |
| Operational quality | Training time, inference latency, artifact size |

Quality gates can reject models that fail required thresholds or perform worse than an established baseline.

### 5. Explainability & Model Reporting

Uses SHAP where supported to explain model predictions and feature contributions. Generates model reports containing evaluation metrics, confusion matrices where applicable, feature importance visualizations, and relevant training configuration.

### 6. Experiment Tracking & Model Registry

Integrates MLflow to record parameters, metrics, artifacts, and model versions. Registration and promotion should occur only after configured validation checks pass.

Model registration does not automatically guarantee production readiness; deployment should also consider reproducibility, input validation, latency, and monitoring requirements.

## 🧰 Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Hyperparameter Optimization:** Optuna
- **Gradient Boosting:** XGBoost, LightGBM
- **Explainability:** SHAP
- **Experiment Tracking & Registry:** MLflow
- **Interactive Dashboard:** Streamlit
- **Testing:** Pytest

## 📂 Repository Structure

```text id="l6ax3m"
AutoML-Studio/
├── src/
│   ├── profiler/
│   │   └── data_audit.py          # Dataset profiling and task inference
│   ├── pipelines/
│   │   └── preprocessing.py       # Leakage-aware preprocessing
│   ├── optimizer/
│   │   └── search.py              # Optuna tuning and cross-validation
│   ├── evaluation/
│   │   └── quality_gate.py        # Metrics and model acceptance checks
│   └── registry/
│       └── mlflow_manager.py      # Experiment tracking and registration
├── app/
│   └── dashboard.py               # Streamlit interface
├── tests/
├── requirements.txt
└── README.md
```

*This is a suggested repository layout. Adjust module names to match the actual implementation.*

## 🗺️ Engineering Roadmap

- [x] Core orchestrator architecture and fold-safe pipeline design
- [x] Automated dataset profiler and task inference module
- [ ] Optuna hyperparameter tuning across linear and tree-based models
- [ ] Automated cross-validation and model comparison reports
- [ ] SHAP explanation and model card generation
- [ ] Configurable quality gates and MLflow experiment tracking
- [ ] Interactive Streamlit dashboard
- [ ] Validated model registration and promotion workflow
- [ ] Reproducible inference artifacts and automated tests

*Mark a milestone complete only when the corresponding implementation has been tested and is available in the repository.*

## 🔬 Evaluation & Reproducibility

For meaningful comparisons:

1. Separate training, validation, and final test data appropriately.
2. Keep preprocessing inside the cross-validation pipeline.
3. Select evaluation metrics based on the task and class distribution.
4. Use consistent splits and documented random seeds where applicable.
5. Tune models using training/validation data and reserve the final test set for final evaluation.
6. Record dataset versions, preprocessing configuration, hyperparameters, and metrics in MLflow.
7. Compare tuned models against simple baselines to verify that optimization adds value.

## ⚠️ Project Status

AutoML-Studio is under active development. The roadmap distinguishes implemented foundations from planned optimization, explainability, dashboard, and registry integrations.

The actual feature set, supported model families, and quality-gate rules should be documented as they become available.

---

*AutoML-Studio — Reproducible AutoML with leakage-aware evaluation and practical MLOps.*
