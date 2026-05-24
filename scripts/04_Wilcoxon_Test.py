# =========================================================
# BANK MARKETING PROJECT - WILCOXON TEST
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import os

# Statistical Testing
from scipy.stats import wilcoxon


# =========================================================
# STEP 2 - CREATE OUTPUT FOLDERS
# =========================================================

os.makedirs("outputs", exist_ok=True)

print("\n✅ Output folders ready")


# =========================================================
# STEP 3 - LOAD MODEL RESULTS
# =========================================================

df = pd.read_csv(
    "outputs/model_results.csv"
)

print("\n✅ Model Results Loaded Successfully")


# =========================================================
# STEP 4 - DISPLAY DATA
# =========================================================

print("\n==============================")
print("📊 MODEL RESULTS")
print("==============================")

print(df)


# =========================================================
# STEP 5 - PREPARE MODEL SCORES
# =========================================================

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

print("\n✅ Model Scores Prepared")


# =========================================================
# STEP 6 - PERFORM WILCOXON TEST
# =========================================================

results = []

model_names = list(models.keys())

print("\n==============================")
print("📐 WILCOXON TEST STARTED")
print("==============================")


for i in range(len(model_names)):

    for j in range(i + 1, len(model_names)):

        model_1 = model_names[i]

        model_2 = model_names[j]

        print(f"\n🔍 Comparing:")
        print(f"{model_1}  VS  {model_2}")

        # Perform Wilcoxon Test
        stat, p_value = wilcoxon(

            models[model_1],

            models[model_2]
        )

        # Check significance
        significant = p_value < 0.05

        # Store results
        results.append({

            "Model 1": model_1,

            "Model 2": model_2,

            "Statistic": stat,

            "P-Value": p_value,

            "Significant": significant
        })

        print(f"Statistic  : {stat:.4f}")

        print(f"P-Value    : {p_value:.4f}")

        print(f"Significant: {significant}")


# =========================================================
# STEP 7 - CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

print("\n==============================")
print("📊 WILCOXON TEST RESULTS")
print("==============================")

print(results_df)


# =========================================================
# STEP 8 - SAVE RESULTS
# =========================================================

results_df.to_csv(

    "outputs/wilcoxon_results.csv",

    index=False
)

print("\n✅ Wilcoxon Results Saved Successfully")

print("\n📁 File Saved:")
print("outputs/wilcoxon_results.csv")


# =========================================================
# STEP 9 - FINAL MESSAGE
# =========================================================

print("\n========================================")
print("🎉 WILCOXON TEST COMPLETED")
print("========================================")

print("""
Next Step:
Run -> 05_SHAP_LIME.py

Generated File:
outputs/wilcoxon_results.csv
""")