# =========================================================
# BANK MARKETING PROJECT - SHAP & LIME
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

# Data handling
import pandas as pd
import numpy as np

# System utilities
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
# IMPORT CUSTOM MODULES
# =========================================================

from config import *

from utils import *

from plots import *


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # STEP TITLE
    # =====================================================

    print_step(5, "SHAP & LIME ANALYSIS")


    # =====================================================
    # CREATE OUTPUT FOLDERS
    # =====================================================

    os.makedirs("outputs", exist_ok=True)

    os.makedirs("outputs/graphs", exist_ok=True)

    print_success("Output folders ready")


    # =====================================================
    # STEP 2 - LOAD BALANCED DATASET
    # =====================================================

    try:

        df = pd.read_csv(
            BALANCED_DATASET_PATH
        )

        print_success(
            "Balanced Dataset Loaded Successfully"
        )

    except FileNotFoundError:

        print_error(
            "Balanced Dataset File Not Found"
        )

        return


    # =====================================================
    # STEP 3 - DISPLAY DATASET INFO
    # =====================================================

    print_heading("📊 DATASET SHAPE")

    print_shape(df)


    print_heading("📄 FIRST 5 ROWS")

    print_head(df)


    # =====================================================
    # STEP 4 - SEPARATE FEATURES & TARGET
    # =====================================================

    X = df.drop("y", axis=1)

    y = df["y"]

    print_success(
        "Features and Target Separated"
    )


    # =====================================================
    # STEP 5 - TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE
    )

    print_success(
        "Train-Test Split Completed"
    )


    # =====================================================
    # STEP 6 - TRAIN BEST MODEL
    # =====================================================

    model = XGBClassifier(

        eval_metric='logloss'
    )

    print("\n🚀 Training XGBoost Model...")

    model.fit(

        X_train,
        y_train
    )

    print_success(
        "XGBoost Model Trained Successfully"
    )


    # =====================================================
    # STEP 7 - SHAP EXPLAINER
    # =====================================================

    print_heading(
        "🔍 GENERATING SHAP VALUES"
    )

    # Create SHAP explainer
    explainer = shap.Explainer(model)

    # Generate SHAP values
    shap_values = explainer(X_test)

    print_success(
        "SHAP Values Generated"
    )


    # =====================================================
    # STEP 8 - SHAP SUMMARY PLOT
    # =====================================================

    print("\n📊 Creating SHAP Summary Plot...")

    shap.summary_plot(

        shap_values,

        X_test,

        show=False
    )

    # Save graph automatically
    save_shap_summary()

    print_success(
        "SHAP Summary Plot Saved"
    )


    # =====================================================
    # STEP 9 - FEATURE IMPORTANCE
    # =====================================================

    print_heading(
        "🏆 FEATURE IMPORTANCE"
    )

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


    # =====================================================
    # STEP 10 - LIME EXPLAINER
    # =====================================================

    print_heading(
        "🔍 GENERATING LIME EXPLANATION"
    )

    lime_explainer = LimeTabularExplainer(

        training_data=np.array(X_train),

        feature_names=X.columns.tolist(),

        class_names=['No', 'Yes'],

        mode='classification'
    )

    print_success(
        "LIME Explainer Created"
    )


    # =====================================================
    # STEP 11 - EXPLAIN SINGLE PREDICTION
    # =====================================================

    exp = lime_explainer.explain_instance(

        X_test.iloc[0].values,

        model.predict_proba
    )

    print_success(
        "LIME Explanation Generated"
    )


    # =====================================================
    # STEP 12 - SAVE LIME OUTPUT
    # =====================================================

    exp.save_to_file(

        LIME_OUTPUT_PATH
    )

    print_success(
        "LIME Explanation Saved"
    )

    print("\n📁 File Saved:")

    print(LIME_OUTPUT_PATH)


    # =====================================================
    # STEP 13 - SAVE FEATURE IMPORTANCE
    # =====================================================

    feature_importance.to_csv(

        FEATURE_IMPORTANCE_PATH,

        index=False
    )

    print_success(
        "Feature Importance Saved"
    )

    print("\n📁 File Saved:")

    print(FEATURE_IMPORTANCE_PATH)


    # =====================================================
    # STEP 14 - FINAL MESSAGE
    # =====================================================

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


# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    main()