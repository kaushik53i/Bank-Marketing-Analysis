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
- 📈 ROC Curve & Confusion Matrix Analysis
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
- Generate automated evaluation visualizations

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

## 📈 ROC Curve & Confusion Matrix
Advanced evaluation visualizations for performance analysis.

---

# 📏 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

# ⚡ Complete Project Workflow

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
ROC Curve & Confusion Matrix
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

# 📁 Final Project Structure

```text
Bank-Marketing-Analysis/
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
│   ├── plots.py
│   └── evaluation_plots.py
│
├── outputs/
│   ├── model_results.csv
│   ├── topsis_results.csv
│   ├── wilcoxon_results.csv
│   ├── feature_importance.csv
│   ├── lime_explanation.html
│   │
│   └── graphs/
│       ├── class_distribution_before_smote.png
│       ├── balanced_dataset.png
│       ├── accuracy_comparison.png
│       ├── shap_summary.png
│       ├── roc_curve_*.png
│       └── confusion_matrix_*.png
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation Guide

## 🔹 Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Bank-Marketing-Analysis.git
```

---

## 🔹 Move Into Project Folder

```bash
cd Bank-Marketing-Analysis
```

---

## 🔹 Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ How To Run The Project

## 🔹 Run Complete Automated Pipeline

```bash
python main.py
```

---

# ▶️ Run Individual Scripts

## 🔹 Step 1 — SMOTE Preprocessing

```bash
python scripts/01_Smote_Preprocessing.py
```

---

## 🔹 Step 2 — Model Training

```bash
python scripts/02_Model_Training.py
```

---

## 🔹 Step 3 — TOPSIS Ranking

```bash
python scripts/03_TOPSIS_Ranking.py
```

---

## 🔹 Step 4 — Wilcoxon Statistical Testing

```bash
python scripts/04_Wilcoxon_Test.py
```

---

## 🔹 Step 5 — SHAP & LIME Explainability

```bash
python scripts/05_SHAP_LIME.py
```

---

# 📊 Generated Outputs

## 📄 CSV Files

- Balanced Dataset
- Model Evaluation Results
- TOPSIS Ranking Results
- Wilcoxon Statistical Results
- Feature Importance Results

## 📈 Visualization Graphs

- Class Distribution Graph
- Balanced Dataset Graph
- Accuracy Comparison Graph
- ROC Curves
- Confusion Matrices
- SHAP Summary Plot

## 🔍 Explainable AI Outputs

- SHAP Feature Analysis
- LIME HTML Explanations

---

# 🌟 Key Features

✅ End-to-End Machine Learning Pipeline  
✅ Automated Script Execution  
✅ Research-Oriented Workflow  
✅ Statistical Validation  
✅ Explainable AI Integration  
✅ Modular Script-Based Architecture  
✅ Advanced Visualization System  
✅ Professional GitHub Repository Design  
✅ VS Code Friendly Project Structure  
✅ Fully Automated Output Generation  

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
- Docker Support
- Automated Hyperparameter Tuning

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