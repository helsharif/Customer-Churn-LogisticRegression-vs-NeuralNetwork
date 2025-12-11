# Customer Churn Prediction & Retention Optimization  
### SparkML • TensorFlow • MLflow • Databricks • Customer Analytics

This repository contains an end-to-end customer churn modeling pipeline developed for a telecommunications provider serving 7,000+ home phone and internet customers. The project demonstrates how exploratory data analysis, machine learning, and production-grade pipelines can be combined to deliver actionable retention insights.

---

## ⭐ Project Overview  
Churn is one of the most critical KPIs for subscription-based businesses, and preventing customer loss requires early, data-driven intervention. This project builds multiple models—including Logistic Regression, Neural Networks, and a scalable SparkML + MLflow workflow—to predict churn and identify the customer segments most at risk.

The work includes:
- Exploratory data analysis  
- Feature engineering & class imbalance handling  
- Baseline & advanced machine learning models  
- Databricks SparkML pipeline  
- MLflow experiment tracking  
- Threshold optimization to improve recall on the churn class  

---

## 🔧 Tech Stack  
- **Python**, Pandas, NumPy  
- **TensorFlow / Keras** (Neural Networks)  
- **SparkML on Databricks**  
- **MLflow** (experiment tracking + model registry)  
- Plotly for visualization  

---

## 📊 Dataset  
Over **7,000 customer records** including demographics, service details, billing, contract type, and churn labels.

Key predictors include:  
- contract type (month-to-month, annual)  
- monthly charges & total charges  
- internet service (DSL, fiber, none)  
- tenure  
- technical support & streaming services  
- payment method and billing preferences  

---

## 🧠 Modeling Approaches  

### **1. Logistic Regression (Interpretable Baseline)**  
- Provides explainable feature coefficients  
- ~77% accuracy on the holdout test set  
- F1 score (churn) in the low 0.60s  

### **2. Neural Network (TensorFlow/Keras)**  
- Three hidden layers + sigmoid output  
- ~79% accuracy  
- Slight improvement in macro F1 over the baseline  

### **3. SparkML Pipeline + MLflow (Production-Ready)**  
Implemented in Databricks to enable large-scale training and iteration.  
Includes:
- automated feature indexing & vector assembly  
- distributed training  
- MLflow experiment tracking (runs, hyperparameters, metrics, artifacts)  
- threshold tuning to increase churn recall from ~50% → ~78%  

---

## 🚀 Deployment & MLOps  
The Databricks implementation demonstrates how a churn model can move from a prototype into a scalable environment:

- Distributed feature engineering & model training  
- MLflow tracking + model registry  
- Precision–recall curve visualization  
- Decision threshold optimization  
- Reusable template for other customer analytics applications  

---

## 💡 Key Business Insights  
- Customers on **month-to-month contracts** with **$70–$110/month charges** show the highest churn risk  
- Many churn events occur during the **first month of service** → need for stronger onboarding  
- Logistic Regression gives transparent insight into churn drivers (billing method, tenure, contract terms)  
- Threshold tuning significantly improves recall on the churn class without excessive retention cost  

