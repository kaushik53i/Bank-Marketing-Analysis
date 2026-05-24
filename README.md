# 🏦 Bank Marketing Analysis System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📌 Project Overview

The **Bank Marketing Analysis System** is an advanced Machine Learning based project designed to predict whether a customer will subscribe to a term deposit using banking and marketing campaign data.

This project combines:

- 📊 Data Preprocessing
- 🔄 SMOTE Balancing
- 🤖 Multiple Machine Learning Models
- 🏆 TOPSIS Ranking
- 📐 Wilcoxon Statistical Testing
- 🔍 Explainable AI using SHAP & LIME

to create a complete intelligent banking prediction system.

---

# 🎯 Project Objectives

- Predict customer term deposit subscriptions
- Handle imbalanced banking datasets
- Compare multiple Machine Learning models
- Rank models scientifically
- Validate model performance statistically
- Explain predictions using Explainable AI

---

# 📊 Dataset Information

| Property | Details |
|---|---|
| Dataset Name | Bank Marketing Dataset |
| Source | UCI Machine Learning Repository |
| Records | 41,188 |
| Features | 20+ |
| Target | Customer Subscription (Yes / No) |

---

# ⚙️ Technologies Used

## 🔹 Programming Language
- Python

## 🔹 Libraries & Frameworks

- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn
- SHAP
- LIME
- SciPy
- OpenPyXL

---

# 🧠 Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Bagging Classifier
- Stacking Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- XGBoost

---

# 🔬 Advanced Techniques Used

## 🔄 SMOTE
Balances the imbalanced dataset using synthetic minority sample generation.

## 🏆 TOPSIS
Ranks Machine Learning models using multi-criteria decision-making techniques.

## 📐 Wilcoxon Signed-Rank Test
Performs statistical significance testing between model performances.

## 🔍 SHAP & LIME
Provides Explainable AI for feature importance and prediction interpretation.

---

# 📏 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Matthews Correlation Coefficient (MCC)
- Cohen’s Kappa Score

---

# ⚡ Project Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
SMOTE Balancing
     ↓
Machine Learning Model Training
     ↓
Performance Evaluation
     ↓
TOPSIS Ranking
     ↓
Wilcoxon Statistical Testing
     ↓
SHAP & LIME Explainability
     ↓
Final Best Model
```

---

# 📁 Project Structure

```text
Bank_Marketing_Project/
│
├── dataset/
│   ├── bank-additional-full.csv
│   └── balanced_bank_data.csv
│
├── scripts/
│   ├── 01_Smote_Preprocessing.py
│   ├── 02_Model_Training.py
│   ├── 03_TOPSIS_Ranking.py
│   ├── 04_Wilcoxon_Test.py
│   ├── 05_SHAP_LIME.py
│   ├── config.py
│   ├── utils.py
│   └── plots.py
│
├── outputs/
│   ├── model_results.csv
│   ├── topsis_results.csv
│   ├── wilcoxon_results.csv
│   │
│   └── graphs/
│       ├── class_distribution.png
│       ├── balanced_dataset.png
│       ├── accuracy_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── shap_summary.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation Guide

## 🔹 Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Bank_Marketing_Project.git
```

---

## 🔹 Move Into Project Folder

```bash
cd Bank_Marketing_Project
```

---

## 🔹 Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ How To Run The Project

## 🔹 Step 1 — Run SMOTE Preprocessing

```bash
python scripts/01_Smote_Preprocessing.py
```

---

## 🔹 Step 2 — Run Machine Learning Models

```bash
python scripts/02_Model_Training.py
```

---

## 🔹 Step 3 — Run TOPSIS Ranking

```bash
python scripts/03_TOPSIS_Ranking.py
```

---

## 🔹 Step 4 — Run Wilcoxon Statistical Testing

```bash
python scripts/04_Wilcoxon_Test.py
```

---

## 🔹 Step 5 — Run SHAP & LIME Explainability

```bash
python scripts/05_SHAP_LIME.py
```

---

# 📊 Expected Outputs

- Balanced Dataset
- Model Evaluation Results
- TOPSIS Ranking Results
- Statistical Test Results
- SHAP Summary Graphs
- LIME Prediction Explanations
- Performance Visualization Graphs

---

# 🌟 Key Features

✅ Advanced Machine Learning Workflow  
✅ Research-Oriented Implementation  
✅ Statistical Validation  
✅ Explainable AI Integration  
✅ Modular Script-Based Structure  
✅ Professional GitHub Repository Design  
✅ VS Code Friendly Project Structure  

---

# 👥 Project Team Members

| Name | Role |
|---|---|
| Member 1 Name | Team Leader / Machine Learning Developer |
| Member 2 Name | Data Preprocessing & Analysis |
| Member 3 Name | Model Training & Evaluation |
| Member 4 Name | Research & Documentation |
| Member 5 Name | Visualization & Testing |



---

# 📚 Future Improvements

- Streamlit Web Application
- Real-time Banking Dashboard
- Hyperparameter Optimization
- Deep Learning Integration
- Cloud Deployment
- Automated ML Pipeline

---

# 📜 License

This project is licensed under the MIT License.

---

# 🙌 Acknowledgement

Special thanks to:

- UCI Machine Learning Repository
- Open Source Machine Learning Community
- Project Team Members

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---
