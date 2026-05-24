# =========================================================
# BANK MARKETING PROJECT - TOPSIS RANKING
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import os

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
# STEP 4 - DISPLAY DATASET
# =========================================================

print("\n==============================")
print("📊 MODEL RESULTS")
print("==============================")

print(df)


# =========================================================
# STEP 5 - SEPARATE NUMERICAL DATA
# =========================================================

# Remove model names
data = df.drop("Model", axis=1)

print("\n✅ Numerical Metrics Extracted")


# =========================================================
# STEP 6 - NORMALIZE DATA
# =========================================================

normalized_data = data / np.sqrt(
    (data ** 2).sum()
)

print("\n✅ Data Normalization Completed")


# =========================================================
# STEP 7 - ASSIGN WEIGHTS
# =========================================================

# Equal importance to all metrics
weights = np.array([1, 1, 1, 1, 1])

# Normalize weights
weights = weights / weights.sum()

print("\n✅ Weights Assigned")

print("\nWeights:")
print(weights)


# =========================================================
# STEP 8 - CREATE WEIGHTED MATRIX
# =========================================================

weighted_data = normalized_data * weights

print("\n✅ Weighted Matrix Created")


# =========================================================
# STEP 9 - DETERMINE IDEAL BEST & WORST
# =========================================================

ideal_best = weighted_data.max()

ideal_worst = weighted_data.min()

print("\n✅ Ideal Best and Worst Calculated")


# =========================================================
# STEP 10 - CALCULATE DISTANCES
# =========================================================

distance_best = np.sqrt(
    ((weighted_data - ideal_best) ** 2).sum(axis=1)
)

distance_worst = np.sqrt(
    ((weighted_data - ideal_worst) ** 2).sum(axis=1)
)

print("\n✅ Distance Calculation Completed")


# =========================================================
# STEP 11 - CALCULATE TOPSIS SCORE
# =========================================================

topsis_score = distance_worst / (
    distance_best + distance_worst
)

df["TOPSIS Score"] = topsis_score

print("\n✅ TOPSIS Scores Generated")


# =========================================================
# STEP 12 - GENERATE RANKINGS
# =========================================================

df["Rank"] = df["TOPSIS Score"].rank(
    ascending=False
)

# Sort models by score
df = df.sort_values(
    by="TOPSIS Score",
    ascending=False
)

print("\n==============================")
print("🏆 FINAL MODEL RANKINGS")
print("==============================")

print(df)


# =========================================================
# STEP 13 - SAVE RESULTS
# =========================================================

df.to_csv(
    "outputs/topsis_results.csv",
    index=False
)

print("\n✅ TOPSIS Results Saved Successfully")

print("\n📁 File Saved:")
print("outputs/topsis_results.csv")


# =========================================================
# STEP 14 - DISPLAY BEST MODEL
# =========================================================

best_model = df.iloc[0]

print("\n==============================")
print("🥇 BEST MODEL")
print("==============================")

print(best_model)


# =========================================================
# STEP 15 - FINAL MESSAGE
# =========================================================

print("\n========================================")
print("🎉 TOPSIS RANKING COMPLETED")
print("========================================")

print("""
Next Step:
Run -> 04_Wilcoxon_Test.py

Generated File:
outputs/topsis_results.csv
""")