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
    roc_auc_score
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


# =========================================================
# IMPORT CUSTOM MODULES
# =========================================================

from config import *

from utils import *

from plots import *

from evaluation_plots import *


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # STEP TITLE
    # =====================================================

    print_step(2, "MODEL TRAINING")


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
    # STEP 3 - DATASET INFORMATION
    # =====================================================

    print_heading("📊 DATASET SHAPE")

    print_shape(df)


    print_heading("📄 FIRST 5 ROWS")

    print_head(df)


    # =====================================================
    # STEP 4 - SEPARATE FEATURES & TARGET
    # =====================================================

    # Features
    X = df.drop("y", axis=1)

    # Target
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

    print("\n📊 Training Data Shape:")

    print(X_train.shape)

    print("\n📊 Testing Data Shape:")

    print(X_test.shape)


    # =====================================================
    # STEP 6 - CREATE MACHINE LEARNING MODELS
    # =====================================================

    models = {

        "Logistic Regression":

            LogisticRegression(),


        "Decision Tree":

            DecisionTreeClassifier(),


        "Random Forest":

            RandomForestClassifier(),


        "Bagging":

            BaggingClassifier(),


        "KNN":

            KNeighborsClassifier(),


        "SVM":

            SVC(probability=True),


        "MLP":

            MLPClassifier(
                max_iter=MAX_ITER
            ),


        "XGBoost":

            XGBClassifier(
                eval_metric='logloss'
            )
    }

    print_success(
        "Machine Learning Models Initialized"
    )


    # =====================================================
    # STEP 7 - TRAIN & EVALUATE MODELS
    # =====================================================

    results = []

    print_heading(
        "🤖 MODEL TRAINING STARTED"
    )


    for name, model in models.items():

        print(f"\n🚀 Training {name}...")


        # =================================================
        # TRAIN MODEL
        # =================================================

        model.fit(

            X_train,
            y_train
        )


        # =================================================
        # PREDICTIONS
        # =================================================

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]


        # =================================================
        # CALCULATE METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )


        # =================================================
        # STORE RESULTS
        # =================================================

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC-AUC": roc_auc
        })


        # =================================================
        # PRINT RESULTS
        # =================================================

        print(f"✅ {name} Completed")

        print(f"Accuracy  : {accuracy:.4f}")

        print(f"Precision : {precision:.4f}")

        print(f"Recall    : {recall:.4f}")

        print(f"F1 Score  : {f1:.4f}")

        print(f"ROC-AUC   : {roc_auc:.4f}")


        # =================================================
        # SAVE ROC CURVE
        # =================================================

        plot_roc_curve(

            y_test,
            y_prob,
            name
        )


        # =================================================
        # SAVE CONFUSION MATRIX
        # =================================================

        plot_confusion_matrix(

            y_test,
            y_pred,
            name
        )


    # =====================================================
    # STEP 8 - CREATE RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(results)


    # =====================================================
    # SORT BEST MODELS
    # =====================================================

    results_df = results_df.sort_values(

        by="ROC-AUC",

        ascending=False
    )


    print_heading(
        "🏆 FINAL MODEL RESULTS"
    )

    print(results_df)


    # =====================================================
    # STEP 9 - SAVE RESULTS
    # =====================================================

    results_df.to_csv(

        MODEL_RESULTS_PATH,

        index=False
    )

    print_success(
        "Model Results Saved Successfully"
    )

    print("\n📁 File Saved:")

    print(MODEL_RESULTS_PATH)


    # =====================================================
    # STEP 10 - SAVE ACCURACY GRAPH
    # =====================================================

    plot_model_accuracy(results_df)

    print_success(
        "Accuracy Comparison Graph Saved"
    )


    # =====================================================
    # STEP 11 - BEST MODEL
    # =====================================================

    best_model = results_df.iloc[0]

    print_heading(
        "🥇 BEST MODEL"
    )

    print(best_model)


    # =====================================================
    # STEP 12 - FINAL MESSAGE
    # =====================================================

    print("\n========================================")

    print("🎉 MODEL TRAINING COMPLETED")

    print("========================================")

    print("""
Generated Outputs:

1. outputs/model_results.csv

2. outputs/graphs/accuracy_comparison.png

3. ROC Curves for all models

4. Confusion Matrices for all models


Next Step:
Run -> 03_TOPSIS_Ranking.py
""")


# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    main()