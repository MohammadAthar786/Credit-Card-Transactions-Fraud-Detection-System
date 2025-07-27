# 💳 Credit Card Transaction Fraud Detection System

## 📌 Project Overview

The goal of this project is to build a fraud detection system using machine learning that can identify fraudulent credit card transactions with high accuracy. A Streamlit-based interactive dashboard is developed to visualize predictions and model performance and model is deployed on streamlit colud community server

---

## 📊 Dataset & Data Analysis

### 🔹 Dataset Description:

The dataset consists of realistic credit card transaction records with the following key columns:

- `transaction_id`: Unique ID for each transaction
- `transaction_amount`: Amount spent
- `merchant`: Merchant where transaction occurred
- `device_type`: Mobile/Desktop device
- `card_present`: Whether the physical card was present
- `customer_location`: Geographic location
- `is_fraudulent`: Target column (1 for fraud, 0 for legitimate)

### 🔹 Exploratory Data Analysis (EDA):

- A pie chart shows the **distribution of fraudulent vs. non-fraudulent transactions**.

- Filters like **merchant**, **customer location**, **device type**, and **amount range** are provided in the sidebar to explore specific transaction subsets.
- Top 10 features influencing fraud were visualized based on feature importance

---

## ⚙️ Preprocessing & Feature Engineering

- Handled missing values and encoded categorical variables.\
- Used SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.\
- Scaled numerical features using StandardScaler/MinMaxScaler as needed.\

---

## 🧠 Model Training

- Model Used: `RandomForestClassifier` (or XGBoost if otherwise)\
- Trained on the balanced dataset using stratified sampling.\
- Saved model using `joblib` as `fraud_model_smote.pkl`.

---

## 🤖 Model Performance

### ✅ Evaluation Metrics:

- **Accuracy**: `0.963`
- **Precision**: `0.967`
- **Recall**: `0.962`
- **F1 Score**: `0.964`

### 🧩 Confusion Matrix:

|                  | Predicted Legit | Predicted Fraud |
| ---------------- | --------------- | --------------- |
| **Actual Legit** | TN: ✓           | FP: ✗           |
| **Actual Fraud** | FN: ✗           | TP: ✓           |

- Low false positives and false negatives indicate strong fraud detection performance.

### 🔍 Feature Importance (Top Contributors):

1. `transaction_amount`
2. `merchant`
3. `device_type`
4. `card_present`
5. `customer_location`

These features had the highest impact on the model's predictions.

---

## 🖥️ Streamlit Dashboard Features

- Real-time visualization of fraud predictions.\
- Filters for customer location, merchant, device, and amount.\
- Visual metrics, confusion matrix, and feature impact charts.\
- Preview of raw transaction data.

---

## 🚀 Future Work

- Deploy on cloud (e.g., AWS, Heroku)\
- Add alert system for high-risk transactions\
- Implement model explainability (SHAP values)

---

## 📇 Contact Information

**👤 Developer**: Mohammad Athar \
**📞 Phone**: +91 9634797852 \
**📧 Email**: mohdatharjamal954@gmail.com\
**🌐 GitHub**: https://github.com/MohammadAthar786/Credit-Card-Transactions-Fraud-Detection-System

---
