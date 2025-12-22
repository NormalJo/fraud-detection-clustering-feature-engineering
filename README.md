# Fraud Detection with Clustering-Based Feature Engineering

## Problem
Fraud detection is challenging due to class imbalance and heterogeneous transaction behavior.
Traditional models struggle to capture latent patterns in transaction data.

## Key Idea
Before training a supervised model, I applied unsupervised clustering to identify distinct transaction behavior groups.
Cluster assignments and statistics were then used as engineered features to improve fraud classification.

## Approach
1. Exploratory data analysis to understand transaction distributions
2. Applied clustering (e.g., KMeans) to group transactions by behavioral similarity
3. Engineered cluster-based features (cluster labels, distance to centroid, cluster fraud rates)
4. Trained a supervised classification model using both original and engineered features
5. Evaluated performance with precision, recall, and ROC-AUC

## Results
- Achieved 82% accuracy
- Improved recall on fraudulent transactions compared to a baseline without clustering features

## Why This Matters
Clustering helped expose latent transaction patterns that were not captured by raw features alone, improving the model’s ability to identify fraud.

## Limitations
- Dataset is simulated and may not represent real-world fraud complexity
- Clustering quality depends on feature scaling and choice of K

