# =========================================================
# PROJECT CONFIGURATION FILE
# =========================================================

# =========================================================
# DATASET PATHS
# =========================================================

RAW_DATASET_PATH = "dataset/bank-additional-full.csv"

BALANCED_DATASET_PATH = "dataset/balanced_bank_data.csv"


# =========================================================
# OUTPUT PATHS
# =========================================================

MODEL_RESULTS_PATH = "outputs/model_results.csv"

TOPSIS_RESULTS_PATH = "outputs/topsis_results.csv"

WILCOXON_RESULTS_PATH = "outputs/wilcoxon_results.csv"

FEATURE_IMPORTANCE_PATH = "outputs/feature_importance.csv"

LIME_OUTPUT_PATH = "outputs/lime_explanation.html"


# =========================================================
# GRAPH PATHS
# =========================================================

CLASS_DISTRIBUTION_GRAPH = (
    "outputs/graphs/class_distribution_before_smote.png"
)

BALANCED_DATASET_GRAPH = (
    "outputs/graphs/balanced_dataset.png"
)

ACCURACY_GRAPH = (
    "outputs/graphs/accuracy_comparison.png"
)

SHAP_SUMMARY_GRAPH = (
    "outputs/graphs/shap_summary.png"
)


# =========================================================
# MACHINE LEARNING SETTINGS
# =========================================================

TEST_SIZE = 0.2

RANDOM_STATE = 42

MAX_ITER = 500


# =========================================================
# TOPSIS SETTINGS
# =========================================================

TOPSIS_WEIGHTS = [1, 1, 1, 1, 1]