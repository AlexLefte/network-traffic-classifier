# Network Traffic Classifier

This repository contains a machine learning–based framework for **network traffic classification**, with a focus on **non-VPN traffic** and **VoIP detection**. The project explores how flow filtering strategies, temporal aggregation, and model-specific preprocessing influence classification performance.

The work evaluates multiple classifiers under different experimental settings and provides insights into the trade-offs between data quality, temporal robustness, and model complexity.

---

## Project Overview

The goal of this project is to classify network traffic flows into application-level categories using **flow-level statistical features**. Special emphasis is placed on:

- Early traffic classification
- Impact of bidirectional flow constraints
- Sensitivity to flow duration
- Model-specific preprocessing strategies

The framework supports comparative evaluation of classical ML models and neural networks under controlled experimental setups.

---

## Key Features

- Flow-based traffic classification
- Support for multiple classifiers:
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Random Forest (RF)
- Configurable flow filtering strategies
- Temporal aggregation analysis
- Optional preprocessing:
  - Synthetic oversampling (SMOTE)
  - Dimensionality reduction (PCA)
- Feature importance analysis using Random Forest (Gini index)

---

## Repository Structure

```text
.
├── scripts/
│   ├── feature_extraction.py
│   ├── train_models.py
│   ├── evaluation.py
│   └── utils.py
├── check_features.ipynb
├── data_curation.ipynb
├── data_visualization.ipynb
├── LICENSE
├── README.md
```

## Data Processing and Model Execution

This section describes how to prepare the data and run the classification models using cross-validation.

---

## Data Processing

Raw network traffic is transformed into flow-level statistical features using the provided notebooks.

### Step 1: Dataset Curation
```bash
jupyter notebook data_curation.ipynb
```
* Applies bidirectional flow filtering

* Enforces minimum packet constraints per direction

* Limits maximum flow duration (e.g., 15s, 120s, 600s)

* Outputs cleaned, labeled flow datasets

### Step 2: Data Visualization
```bash
jupyter notebook data_visualization.ipynb
```
* Analyzes class distribution

* Visualizes temporal feature behavior

* Evaluates impact of flow duration and filtering

### Step 3: Feature Validation
```bash
jupyter notebook check_features.ipynb
```
* Verifies extracted features

* Inspects feature distributions
---
## Running the Models
All models are evaluated using cross-validation for robust performance estimation.

### MLP (Neural Network)
```bash
python scripts/methods/mlp_cross_val.py --csv_file path/to/data.csv --smote
```
- Requires `--csv_file`  
- Optional `--smote` flag enables oversampling  
- Supports PCA dimensionality reduction  
- Best performance achieved with PCA + SMOTE  
- Sensitive to overfitting for long flow durations  

---

### Random Forest (RF)
```bash
python scripts/methods/rf_cross_val.py --csv_file path/to/data.csv --smote
```
- Requires `--csv_file`  
- Optional `--smote` flag (generally not recommended)  
- Does not benefit from PCA  
- Feature importance computed via Gini index decrease  
- Robust to feature scaling  

---

### Support Vector Machine (SVM)
```bash
python scripts/methods/svm_cross_val.py --csv_file path/to/data.csv --smote
```
- Requires `--csv_file`  
- Optional `--smote` flag (often degrades performance)  
- Performs best without PCA  
- Stable decision boundaries on clean, strictly filtered data  
---

## Output & Saved Results

For each experiment and cross-validation run, the scripts automatically log results to CSV files stored in the `results/` directory.

### Saved Metrics
After each hyperparameter configuration, a temporary DataFrame is created and appended to a CSV file containing:

- Simulation index (`SIM`)
- Model-specific hyperparameters (e.g. hidden layers, dropout)
- Preprocessing settings:
  - SMOTE usage
  - PCA dimensionality
- Training metrics:
  - Accuracy
  - F1-score
  - Precision
  - Recall
- Validation metrics:
  - Accuracy
  - F1-score
  - Precision
  - Recall

Each experiment is appended **incrementally**, ensuring that partial results are preserved even if execution is interrupted.

---