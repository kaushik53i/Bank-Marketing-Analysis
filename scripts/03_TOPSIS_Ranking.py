# =========================================================
# BANK MARKETING PROJECT - TOPSIS RANKING
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

# Data handling
import pandas as pd
import numpy as np

# System utilities
import os


# =========================================================
# IMPORT CUSTOM MODULES
# =========================================================

from config import *

from utils import *


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # STEP TITLE
    # =====================================================

    print_step(3, "TOPSIS RANKING")


    # =====================================================
    # CREATE OUTPUT FOLDERS
    # =====================================================

    os.makedirs("outputs", exist_ok=True)

    print_success("Output folders ready")


    # =====================================================
    # STEP 2 - LOAD MODEL RESULTS
    # =====================================================

    try:

        df = pd.read_csv(
            MODEL_RESULTS_PATH
        )

        print_success(
            "Model Results Loaded Successfully"
        )

    except FileNotFoundError:

        print_error(
            "Model Results File Not Found"
        )

        return


    # =====================================================
    # STEP 3 - DISPLAY DATASET
    # =====================================================

    print_heading("📊 MODEL RESULTS")

    print(df)


    # =====================================================
    # STEP 4 - SEPARATE NUMERICAL DATA
    # =====================================================

    # Remove model names
    data = df.drop("Model", axis=1)

    print_success(
        "Numerical Metrics Extracted"
    )


    # =====================================================
    # STEP 5 - NORMALIZE DATA
    # =====================================================

    normalized_data = data / np.sqrt(
        (data ** 2).sum()
    )

    print_success(
        "Data Normalization Completed"
    )


    # =====================================================
    # STEP 6 - ASSIGN WEIGHTS
    # =====================================================

    # Equal importance to all metrics
    weights = np.array(
        TOPSIS_WEIGHTS
    )

    # Normalize weights
    weights = weights / weights.sum()

    print_success(
        "Weights Assigned"
    )

    print("\n📊 Weights:")

    print(weights)


    # =====================================================
    # STEP 7 - CREATE WEIGHTED MATRIX
    # =====================================================

    weighted_data = normalized_data * weights

    print_success(
        "Weighted Matrix Created"
    )


    # =====================================================
    # STEP 8 - DETERMINE IDEAL BEST & WORST
    # =====================================================

    ideal_best = weighted_data.max()

    ideal_worst = weighted_data.min()

    print_success(
        "Ideal Best and Worst Calculated"
    )


    # =====================================================
    # STEP 9 - CALCULATE DISTANCES
    # =====================================================

    distance_best = np.sqrt(
        ((weighted_data - ideal_best) ** 2).sum(axis=1)
    )

    distance_worst = np.sqrt(
        ((weighted_data - ideal_worst) ** 2).sum(axis=1)
    )

    print_success(
        "Distance Calculation Completed"
    )


    # =====================================================
    # STEP 10 - CALCULATE TOPSIS SCORE
    # =====================================================

    topsis_score = distance_worst / (

        distance_best + distance_worst
    )

    df["TOPSIS Score"] = topsis_score

    print_success(
        "TOPSIS Scores Generated"
    )


    # =====================================================
    # STEP 11 - GENERATE RANKINGS
    # =====================================================

    df["Rank"] = df["TOPSIS Score"].rank(

        ascending=False
    )

    # Sort models by score
    df = df.sort_values(

        by="TOPSIS Score",

        ascending=False
    )

    print_heading(
        "🏆 FINAL MODEL RANKINGS"
    )

    print(df)


    # =====================================================
    # STEP 12 - SAVE RESULTS
    # =====================================================

    df.to_csv(

        TOPSIS_RESULTS_PATH,

        index=False
    )

    print_success(
        "TOPSIS Results Saved Successfully"
    )

    print("\n📁 File Saved:")

    print(TOPSIS_RESULTS_PATH)


    # =====================================================
    # STEP 13 - DISPLAY BEST MODEL
    # =====================================================

    best_model = df.iloc[0]

    print_heading(
        "🥇 BEST MODEL"
    )

    print(best_model)


    # =====================================================
    # STEP 14 - FINAL MESSAGE
    # =====================================================

    print("\n========================================")

    print("🎉 TOPSIS RANKING COMPLETED")

    print("========================================")

    print("""
Generated File:

1. outputs/topsis_results.csv


Next Step:
Run -> 04_Wilcoxon_Test.py
""")


# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    main()