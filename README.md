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

## Dataset
- Source: Kaggle — *Synthetic Financial Datasets for Fraud Detection*
- Size: ~1.3M transactions
- Target variable: `is_fraud`_
