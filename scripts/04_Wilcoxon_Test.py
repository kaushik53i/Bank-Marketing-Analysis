# =========================================================
# BANK MARKETING PROJECT - WILCOXON TEST
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

# Data handling
import pandas as pd
import numpy as np

# System utilities
import os

# Statistical Testing
from scipy.stats import wilcoxon


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

    print_step(4, "WILCOXON TEST")


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
    # STEP 3 - DISPLAY DATA
    # =====================================================

    print_heading("📊 MODEL RESULTS")

    print(df)


    # =====================================================
    # STEP 4 - PREPARE MODEL SCORES
    # =====================================================

    # Using ROC-AUC scores for comparison

    models = {}

    for index, row in df.iterrows():

        model_name = row["Model"]

        # Create small repeated sample
        # (for demonstration/testing purpose)

        models[model_name] = [

            row["ROC-AUC"],

            row["ROC-AUC"] - 0.01,

            row["ROC-AUC"] + 0.01,

            row["ROC-AUC"] - 0.02,

            row["ROC-AUC"] + 0.02
        ]

    print_success(
        "Model Scores Prepared"
    )


    # =====================================================
    # STEP 5 - PERFORM WILCOXON TEST
    # =====================================================

    results = []

    model_names = list(models.keys())

    print_heading(
        "📐 WILCOXON TEST STARTED"
    )


    for i in range(len(model_names)):

        for j in range(i + 1, len(model_names)):

            model_1 = model_names[i]

            model_2 = model_names[j]

            print(f"\n🔍 Comparing:")

            print(f"{model_1}  VS  {model_2}")


            # =============================================
            # PERFORM WILCOXON TEST
            # =============================================

            stat, p_value = wilcoxon(

                models[model_1],

                models[model_2]
            )


            # =============================================
            # CHECK SIGNIFICANCE
            # =============================================

            significant = p_value < 0.05


            # =============================================
            # STORE RESULTS
            # =============================================

            results.append({

                "Model 1": model_1,

                "Model 2": model_2,

                "Statistic": stat,

                "P-Value": p_value,

                "Significant": significant
            })


            # =============================================
            # PRINT RESULTS
            # =============================================

            print(f"Statistic  : {stat:.4f}")

            print(f"P-Value    : {p_value:.4f}")

            print(f"Significant: {significant}")


    # =====================================================
    # STEP 6 - CREATE RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(results)

    print_heading(
        "📊 WILCOXON TEST RESULTS"
    )

    print(results_df)


    # =====================================================
    # STEP 7 - SAVE RESULTS
    # =====================================================

    results_df.to_csv(

        WILCOXON_RESULTS_PATH,

        index=False
    )

    print_success(
        "Wilcoxon Results Saved Successfully"
    )

    print("\n📁 File Saved:")

    print(WILCOXON_RESULTS_PATH)


    # =====================================================
    # STEP 8 - FINAL MESSAGE
    # =====================================================

    print("\n========================================")

    print("🎉 WILCOXON TEST COMPLETED")

    print("========================================")

    print("""
Generated File:

1. outputs/wilcoxon_results.csv


Next Step:
Run -> 05_SHAP_LIME.py
""")


# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    main()