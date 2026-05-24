# =========================================================
# BANK MARKETING PROJECT - MODEL TRAINING
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

# Data handling
import pandas as pd
import numpy as np

# System utilities
import os

# Train-test split
from sklearn.model_selection import train_test_split

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# XGBoost
from xgboost import XGBClassifier

# Visualization
import matplotlib.pyplot as plt


# =========================================================
# STEP 2 - CREATE OUTPUT FOLDERS
# =========================================================

os.makedirs("outputs", exist_ok=True)

os.makedirs("outputs/graphs", exist_ok=True)

print("\n✅ Output folders ready")


# =========================================================
# STEP 3 - LOAD BALANCED DATASET
# =========================================================

df = pd.read_csv(
    "dataset/balanced_bank_data.csv"
)

print("\n✅ Balanced Dataset Loaded Successfully")


# =========================================================
# STEP 4 - DATASET INFORMATION
# =========================================================

print("\n==============================")
print("📊 DATASET SHAPE")
print("==============================")

print(df.shape)

print("\n==============================")
print("📄 FIRST 5 ROWS")
print("==============================")

print(df.head())


# =========================================================
# STEP 5 - SEPARATE FEATURES AND TARGET
# =========================================================

# Features
X = df.drop("y", axis=1)

# Target
y = df["y"]

print("\n✅ Features and Target Separated")


# =========================================================
# STEP 6 - TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

print("\n✅ Train-Test Split Completed")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# =========================================================
# STEP 7 - CREATE MACHINE LEARNING MODELS
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "Bagging": BaggingClassifier(),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC(probability=True),

    "MLP": MLPClassifier(max_iter=500),

    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

print("\n✅ Machine Learning Models Initialized")


# =========================================================
# STEP 8 - TRAIN AND EVALUATE MODELS
# =========================================================

results = []

print("\n==============================")
print("🤖 MODEL TRAINING STARTED")
print("==============================")


for name, model in models.items():

    print(f"\n🚀 Training {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Probability prediction
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    # Store results
    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })

    # Print results
    print(f"✅ {name} Completed")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC-AUC   : {roc_auc:.4f}")


# =========================================================
# STEP 9 - CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

print("\n==============================")
print("📊 MODEL RESULTS")
print("==============================")

print(results_df)


# =========================================================
# STEP 10 - SORT BEST MODELS
# =========================================================

results_df = results_df.sort_values(

    by="ROC-AUC",

    ascending=False
)

print("\n==============================")
print("🏆 SORTED MODEL RESULTS")
print("==============================")

print(results_df)


# =========================================================
# STEP 11 - SAVE RESULTS
# =========================================================

results_df.to_csv(

    "outputs/model_results.csv",

    index=False
)

print("\n✅ Model Results Saved Successfully")

print("\n📁 File Saved:")
print("outputs/model_results.csv")


# =========================================================
# STEP 12 - VISUALIZATION
# =========================================================

plt.figure(figsize=(12,6))

plt.bar(

    results_df["Model"],

    results_df["Accuracy"]
)

plt.xticks(rotation=45)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")


# Save graph
plt.savefig("outputs/graphs/accuracy_comparison.png")
plt.close()
print("\n✅ Accuracy Graph Saved")




# =========================================================
# STEP 13 - BEST MODEL
# =========================================================

best_model = results_df.iloc[0]

print("\n==============================")
print("🏆 BEST MODEL")
print("==============================")

print(best_model)


# =========================================================
# STEP 14 - FINAL MESSAGE
# =========================================================

print("\n========================================")
print("🎉 MODEL TRAINING COMPLETED")
print("========================================")

print("""
Next Step:
Run -> 03_TOPSIS_Ranking.py

Generated Files:
1. outputs/model_results.csv
2. outputs/graphs/accuracy_comparison.png
""")