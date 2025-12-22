# Fraud Detection with Clustering-Based Feature Engineering

## Overview
This project detects fraudulent credit card transactions using **traditional feature engineering**, **unsupervised clustering**, and **supervised machine learning**. The key idea is to use clustering to capture latent transaction behavior and turn that information into features that improve fraud classification on a highly imbalanced dataset. 

## Problem Statement
Credit card fraud detection is a **binary classification** problem where fraud is extremely rare.

- Fraud rate: **~0.58%**
- A model that predicts “non-fraud” for every transaction would achieve >99% accuracy, so accuracy alone is misleading.
- The goal is to catch fraud (**high recall**) while keeping false alarms manageable (**precision**).

## Dataset
- Source: Kaggle — *Synthetic Financial Datasets for Fraud Detection*
- Size: ~1.3M transactions
- Target: `is_fraud` (1 = fraud, 0 = legitimate)
- Feature types:
  - Transaction details (amount, category, merchant)
  - Customer attributes (age, gender, job, city population)
  - Spatiotemporal info (timestamp, lat/long, derived time features) :contentReference[oaicite:1]{index=1}

## Key Idea
Instead of relying only on raw transaction variables, this project uses **unsupervised clustering as a feature engineering step**:

1. Reduce/shape behavioral features (scaling + transformations)
2. Use PCA to represent transactions in a lower-dimensional behavioral space
3. Cluster transactions (K-Means / Agglomerative) to identify behavioral groups
4. Use cluster membership and behavior patterns to support supervised fraud detection

> Clustering is used to generate features and behavioral insight — not as the final prediction objective. :contentReference[oaicite:2]{index=2}

---

## Feature Engineering

### Traditional Features
- Time since last transaction per card (`time_since_last_trans`)
- Average transaction amount per card (`avg_trans_amt`)
- Amount ratio relative to typical spending (`amt_ratio`)
- Transaction hour features (`hour`)
- Grouped job titles into broader **job sectors**
- One-hot encoding for categorical variables (category, job sector, state)
- Label encoding for binary gender :contentReference[oaicite:3]{index=3}

### Transformations
- Removed redundancy after correlation analysis (e.g., `amt` removed due to high correlation with `amt_ratio`)
- Cyclical encoding for hour (`hour_sin`, `hour_cos`) to preserve 24-hour periodicity
- Log transform for skewed ratio feature (`log_amt_ratio`)
- Standardized features before PCA/clustering :contentReference[oaicite:4]{index=4}

---

## Dimensionality Reduction
To reduce redundancy and prepare data for clustering:

- **LASSO regression** to identify predictive variables (notably `log_amt_ratio` and time-based signals)
- **PCA** applied to scaled numerical features
  - Retained **~85.1%** of variance using 4 components
  - First two components used for visualization and clustering :contentReference[oaicite:5]{index=5}

---

## Clustering (Unsupervised Learning)

### Algorithms
- **K-Means**
- **Agglomerative Hierarchical Clustering** :contentReference[oaicite:6]{index=6}

### Choosing Number of Clusters
- Elbow method + Silhouette coefficient
- Final selections:
  - **K-Means: k = 5**
  - **Agglomerative: k = 4** :contentReference[oaicite:7]{index=7}

### Cluster Insights
Clustering revealed distinct behavioral groups. Fraud was concentrated in specific clusters associated with:
- higher `log_amt_ratio` (unusually large purchases vs typical spending)
- irregular patterns (timing/transaction gaps)
- differences by urban context (`log_city_pop`) :contentReference[oaicite:8]{index=8}

---

## Supervised Modeling

### Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- XGBoost :contentReference[oaicite:9]{index=9}

### Handling Class Imbalance
Due to extreme imbalance (~0.58% fraud):
- Reduced dataset to 20% (stratified) for compute (~260k records)
- Train/test split: 80/20 stratified
- Primary strategy: **SMOTE oversampling**
- Undersampling tested for comparison (performed poorly in practice) :contentReference[oaicite:10]{index=10}

### Evaluation Metrics
Because accuracy is misleading, evaluation used:
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix :contentReference[oaicite:11]{index=11}

---

## Results (Test Set — Oversampling)

| Model | Precision | Recall | F1-score | ROC-AUC |
|------|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.0576 | 0.7733 | 0.1072 | 0.8582 |
| Random Forest | 0.8776 | 0.7167 | 0.7890 | 0.9841 |
| XGBoost | 0.8727 | 0.8000 | 0.8348 | 0.9924 |

### Final Model: Tuned XGBoost (SMOTE Oversampling)
After hyperparameter tuning (RandomizedSearchCV), XGBoost improved:

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Precision | 0.8727 | 0.8942 | +0.0215 |
| Recall | 0.8000 | 0.8167 | +0.0167 |
| F1-score | 0.8348 | 0.8537 | +0.0189 |
| ROC-AUC | 0.9924 | 0.9939 | +0.0015 |
| False Positives | 35 | 29 | -6 |
| False Negatives | 60 | 55 | -5 | :contentReference[oaicite:12]{index=12}

---

## Limitations
- Dataset is **synthetic**, so patterns may differ from real-world fraud
- Only **20%** of full data used due to computational constraints
- Missing real fraud signals such as device ID, IP consistency, account balance, prior fraud flags
- SMOTE generates synthetic minority examples that may not perfectly match real fraud behavior :contentReference[oaicite:13]{index=13}

---

## Future Improvements
- Try SMOTE-ENN / SMOTE-Tomek
- Evaluate LightGBM / CatBoost
- Tune decision thresholds dynamically
- Test in a simulated real-time / streaming environment
- Train on full dataset with more compute :contentReference[oaicite:14]{index=14}

---

## Tech Stack
Python • pandas • NumPy • scikit-learn • imbalanced-learn • XGBoost • Jupyter

---

## How to Run (Suggested)
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
