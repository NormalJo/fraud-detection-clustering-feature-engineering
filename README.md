# Fraud Detection with Clustering-Based Feature Engineering

## Overview
This project detects fraudulent credit card transactions using **traditional feature engineering**, **unsupervised clustering**, and **supervised machine learning**.  
The key idea is to use clustering to capture latent transaction behavior and turn that information into features that improve fraud classification on a highly imbalanced dataset.

## Problem Statement
Credit card fraud detection is a **binary classification** problem where fraud is extremely rare.

- Fraud rate: ~0.58%
- Predicting “non-fraud” for every transaction would give >99% accuracy
- Accuracy alone is misleading in highly imbalanced settings

The goal is to **maximize fraud detection (recall)** while keeping **false positives manageable (precision)**.

---

Raw Transaction Data
        ↓
Feature Engineering
(time features, spending ratios, etc.)
        ↓
Dimensionality Reduction (PCA)
        ↓
Clustering (K-Means / Agglomerative)
        ↓
Cluster-Based Features
        ↓
Supervised Models
(Logistic Regression, Random Forest, XGBoost)
        ↓
Fraud Prediction

---

## Dataset
- Source: Kaggle — *Synthetic Financial Datasets for Fraud Detection*
- Size: ~1.3M transactions
- Target variable: `is_fraud` (1 = fraud, 0 = legitimate)

Feature types include:
- Transaction details (amount, category, merchant)
- Customer attributes (age, gender, job, city population)
- Spatiotemporal information (timestamp, location, derived time features)

---

## Key Idea
Instead of relying only on raw transaction variables, this project uses **unsupervised clustering as a feature engineering step**.

Workflow:
1. Engineer behavioral and temporal features
2. Reduce dimensionality using PCA
3. Cluster transactions to identify behavioral groups
4. Use cluster-derived information to improve supervised fraud classification

> Clustering is used to generate features and behavioral insight — not as the final prediction objective.

---

## Feature Engineering

### Traditional Features
- Time since last transaction per card
- Average transaction amount per card
- Amount ratio relative to typical spending
- Transaction hour
- Grouped job titles into broader job sectors
- One-hot encoding for categorical variables
- Label encoding for binary gender

### Transformations
- Removed redundant variables after correlation analysis
- Cyclical encoding of time (`hour_sin`, `hour_cos`)
- Log transformation of skewed monetary features
- Feature standardization prior to PCA and clustering

---

## Dimensionality Reduction
To reduce redundancy and prepare data for clustering:

- **LASSO regression** to identify predictive variables
- **Principal Component Analysis (PCA)** on scaled features
  - Retained ~85% of total variance
  - First two components used for visualization and clustering

---

## Clustering (Unsupervised Learning)

### Algorithms Used
- K-Means
- Agglomerative Hierarchical Clustering

### Cluster Selection
- Elbow method
- Silhouette score

Final selections:
- K-Means: 5 clusters
- Agglomerative: 4 clusters

### Insights
Clustering revealed distinct behavioral groups. Fraud was concentrated in clusters characterized by:
- Unusually large transactions relative to typical spending
- Irregular spending patterns
- Higher activity in urban environments

---

## Supervised Modeling

### Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- XGBoost

### Handling Class Imbalance
- Reduced dataset to 20% using stratified sampling (compute constraints)
- 80/20 stratified train-test split
- Primary strategy: **SMOTE oversampling**
- Undersampling tested for comparison but performed poorly

### Evaluation Metrics
Because accuracy is misleading, evaluation focused on:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## Results (Test Set — Oversampling)

| Model | Precision | Recall | F1-score | ROC-AUC |
|------|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.058 | 0.773 | 0.107 | 0.858 |
| Random Forest | 0.878 | 0.717 | 0.789 | 0.984 |
| XGBoost | 0.873 | 0.800 | 0.835 | 0.992 |

### Final Model: Tuned XGBoost
After hyperparameter tuning, XGBoost achieved the best balance between precision and recall:

- Precision: 0.894
- Recall: 0.817
- F1-score: 0.854
- ROC-AUC: 0.994
- Reduced both false positives and false negatives

---

## Limitations
- Dataset is synthetic and may not reflect all real-world fraud patterns
- Only 20% of the data used due to computational constraints
- Lacks real fraud signals (device ID, IP address, account balance)
- SMOTE generates synthetic fraud samples

---

## Future Improvements
- Try SMOTE-ENN or SMOTE-Tomek
- Experiment with LightGBM or CatBoost
- Tune decision thresholds dynamically
- Simulate real-time fraud detection
- Train on the full dataset with additional compute

---

## Tech Stack
Python • pandas • NumPy • scikit-learn • imbalanced-learn • XGBoost • Jupyter Notebook

---

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
