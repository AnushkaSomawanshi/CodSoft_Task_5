# CODSOFT Internship Projects

This repository includes the work I completed as part of the CODSOFT Data Science Internship.  
Each task demonstrates a real-world machine learning use case using Python, ranging from regression to classification problems.

This README describes **Task 5: Credit Card Fraud Detection**.

---

# Task 5: Credit Card Fraud Detection

## Project Overview

This project focuses on developing a machine learning model to detect fraudulent credit card transactions.  
Since fraudulent transactions are extremely rare compared to genuine ones, the problem involves working with highly imbalanced data.  
The model must learn to accurately identify fraudulent activity without being biased toward the majority class (genuine transactions).

The goal is to create a reliable classification model that can help financial institutions flag suspicious transactions in real-time.

---

## What I Did

- Loaded and explored a preprocessed dataset containing transaction features  
- Performed data cleaning and normalization  
- Handled **class imbalance** using oversampling techniques such as **SMOTE**  
- Split the dataset into training and test sets using stratified sampling  
- Trained classification models:
  - Logistic Regression  
  - Random Forest Classifier  
- Evaluated model performance using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Used confusion matrix and classification reports for deeper insight  
- Fine-tuned the model to reduce false negatives and improve fraud detection

---

## What I Observed

- The dataset was heavily imbalanced with fewer than 1% fraudulent transactions  
- Random Forest outperformed Logistic Regression, especially in detecting minority class  
- Using SMOTE (Synthetic Minority Oversampling Technique) significantly improved recall without hurting precision  
- Precision and recall trade-off was key: minimizing false negatives was more important than false positives  
- Most frauds had distinct feature patterns (high variance in certain principal components)

---

## Dataset Description

The dataset contains anonymized credit card transaction data:

- `Time`: Seconds elapsed between this transaction and the first transaction  
- `V1` to `V28`: Result of PCA transformation on original features  
- `Amount`: Transaction amount  
- `Class`: Target variable — 0 for genuine, 1 for fraud

---

## Key Steps in the Project

### 1. Data Preprocessing

- Loaded CSV data into pandas  
- Scaled the `Amount` feature using StandardScaler  
- Checked for null values and verified data distribution  
- Used SMOTE to balance the dataset

### 2. Model Training

- Split data using `train_test_split` with stratification  
- Trained two models:
  - Logistic Regression (baseline)  
  - Random Forest Classifier (ensemble model)

### 3. Evaluation

- Evaluated using:
  - Confusion Matrix  
  - Precision, Recall, F1-score  
- Compared model performance before and after SMOTE  
- Plotted ROC curve and classification report

---

## Sample Code for Prediction

```python
# Example input for prediction
sample = [[-1.35980713, -0.07278117, 2.53634674, ..., 149.62]]  # Replace with actual feature vector

# Predict whether the transaction is fraudulent
prediction = model.predict(sample)
print("Fraudulent Transaction" if prediction[0] == 1 else "Genuine Transaction")
