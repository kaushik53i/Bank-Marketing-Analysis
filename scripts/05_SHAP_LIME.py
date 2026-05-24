# =========================================================
# BANK MARKETING PROJECT - SHAP & LIME
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import os

# Train Test Split
from sklearn.model_selection import train_test_split

# XGBoost Model
from xgboost import XGBClassifier

# SHAP
import shap

# LIME
from lime.lime_tabular import LimeTabularExplainer

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
# STEP 4 - DISPLAY DATASET INFO
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

X = df.drop("y", axis=1)

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


# =========================================================
# STEP 7 - TRAIN BEST MODEL
# =========================================================

model = XGBClassifier(

    eval_metric='logloss'
)

print("\n🚀 Training XGBoost Model...")

model.fit(X_train, y_train)

print("✅ XGBoost Model Trained Successfully")


# =========================================================
# STEP 8 - SHAP EXPLAINER
# =========================================================

print("\n==============================")
print("🔍 GENERATING SHAP VALUES")
print("==============================")

# Create SHAP explainer
explainer = shap.Explainer(model)

# Generate SHAP values
shap_values = explainer(X_test)

print("✅ SHAP Values Generated")


# =========================================================
# STEP 9 - SHAP SUMMARY PLOT
# =========================================================

print("\n📊 Creating SHAP Summary Plot...")

shap.summary_plot(

    shap_values,

    X_test,

    show=False
)

# Save graph
plt.savefig(

    "outputs/graphs/shap_summary.png",

    bbox_inches='tight'
)

print("✅ SHAP Summary Plot Saved")

plt.close()


# =========================================================
# STEP 10 - FEATURE IMPORTANCE
# =========================================================

print("\n==============================")
print("🏆 FEATURE IMPORTANCE")
print("==============================")

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": np.abs(
        shap_values.values
    ).mean(axis=0)
})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False
)

print(feature_importance.head(10))


# =========================================================
# STEP 11 - LIME EXPLAINER
# =========================================================

print("\n==============================")
print("🔍 GENERATING LIME EXPLANATION")
print("==============================")

lime_explainer = LimeTabularExplainer(

    training_data=np.array(X_train),

    feature_names=X.columns.tolist(),

    class_names=['No', 'Yes'],

    mode='classification'
)

print("✅ LIME Explainer Created")


# =========================================================
# STEP 12 - EXPLAIN SINGLE PREDICTION
# =========================================================

exp = lime_explainer.explain_instance(

    X_test.iloc[0].values,

    model.predict_proba
)

print("\n✅ LIME Explanation Generated")


# =========================================================
# STEP 13 - SAVE LIME OUTPUT
# =========================================================

exp.save_to_file(

    "outputs/lime_explanation.html"
)

print("\n✅ LIME Explanation Saved")

print("\n📁 File Saved:")
print("outputs/lime_explanation.html")


# =========================================================
# STEP 14 - SAVE FEATURE IMPORTANCE
# =========================================================

feature_importance.to_csv(

    "outputs/feature_importance.csv",

    index=False
)

print("\n✅ Feature Importance Saved")

print("\n📁 File Saved:")
print("outputs/feature_importance.csv")


# =========================================================
# STEP 15 - FINAL MESSAGE
# =========================================================

print("\n========================================")
print("🎉 SHAP & LIME ANALYSIS COMPLETED")
print("========================================")

print("""
Generated Files:

1. outputs/graphs/shap_summary.png
2. outputs/lime_explanation.html
3. outputs/feature_importance.csv

Project Pipeline Completed Successfully.
""")