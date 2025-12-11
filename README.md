# Customer Churn Prediction with Logistic Regression & Neural Networks  
### Handling Class Imbalance with SMOTE • Model Interpretability • Plotly Visualizations

This repository contains a complete end-to-end customer churn modeling workflow using **Python**, **scikit-learn**, **TensorFlow/Keras**, and **SMOTE** for imbalance handling. The project demonstrates how to clean and encode categorical data, engineer features, build baseline and advanced models, and interpret the key drivers of churn in a telecommunications dataset.

---

## ⭐ Project Overview  
Customer churn is a major cost driver in subscription-based businesses. In telecom, customers often leave early due to service quality, pricing, or contract structure. This project predicts which customers are most likely to churn and examines which features contribute most strongly to churn risk.

The workflow includes:
- Exploratory data analysis (EDA) with **Plotly**
- Cleaning inconsistent values (e.g., *No internet service*, *No phone service*)
- Numerical encoding + one-hot encoding of categorical variables
- Scaling continuous variables (tenure, monthly charges, total charges)
- Train/test splitting with stratification
- **SMOTE** oversampling to balance the minority churn class
- Baseline **Logistic Regression** model
- **Feedforward Neural Network** using TensorFlow/Keras
- Confusion matrices, F1-scores, and classification reports
- Significance testing for logistic regression coefficients

Notebook reference: :contentReference[oaicite:1]{index=1}

---

## 🔧 Tech Stack  
- Python  
- Pandas, NumPy  
- scikit-learn (LogisticRegression, train/test split, metrics)  
- **imbalanced-learn (SMOTE)**  
- TensorFlow / Keras  
- Plotly (interactive visualizations)  
- Statsmodels (coefficient significance testing)

---

## 📊 Dataset  
**7,000+ telecom customer records** with:  
- Demographics: gender, senior citizen, partner, dependents  
- Services: phone, fiber optic, online security, streaming, tech support  
- Contract & billing: contract type, payment method, paperless billing  
- Charges: monthly charges, total charges  
- Target variable: **Churn (Yes/No)**

Rows with missing or invalid values (e.g., `" "`) are cleaned and removed during preprocessing.

---

## 🧠 Modeling Approaches

### **1. Logistic Regression (Baseline Model)**  
After preprocessing and SMOTE balancing:
- Achieves strong interpretability  
- Produces churn probability scores  
- Generates a clean coefficient + p-value table to identify statistically significant predictors  
- Uses Plotly to visualize coefficient magnitudes  

Significant churn drivers often include:
- Short tenure  
- Month-to-month contracts  
- Higher monthly charges  
- Lack of tech support or online security

(See notebook for the exact results.)  

---

### **2. Neural Network (TensorFlow)**  
A simple feed-forward neural network using the same SMOTE-balanced data:  
- 1 input layer  
- Multiple hidden layers with ReLU  
- Sigmoid output for binary classification  
- Trained on scaled features  

Performance is compared with logistic regression using:
- Accuracy  
- Precision/recall  
- F1-scores  
- Confusion matrices  

---

## 📈 Key Findings  
From EDA and model outputs:
- Many customers who churn do so **in their first month**, implying onboarding issues.  
- Customers paying **$70–$110/month** churn more frequently.  
- Month-to-month contract customers show significantly elevated churn risk.  
- Logistic Regression provides transparent insights into which features drive behavior.  
- SMOTE improves recall on the churn class without heavily degrading precision.


