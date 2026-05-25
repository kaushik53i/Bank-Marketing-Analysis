# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================

import matplotlib.pyplot as plt
import os

# Create graph folder automatically
os.makedirs("outputs/graphs", exist_ok=True)


# =========================================================
# CLASS DISTRIBUTION GRAPH
# =========================================================

def plot_class_distribution(data):

    data.value_counts().plot(kind='bar')

    plt.title("Class Distribution Before SMOTE")

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.savefig(
        "outputs/graphs/class_distribution_before_smote.png"
    )

    plt.close()


# =========================================================
# BALANCED DATASET GRAPH
# =========================================================

def plot_balanced_dataset(data):

    data.value_counts().plot(kind='bar')

    plt.title("Balanced Dataset After SMOTE")

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.savefig(
        "outputs/graphs/balanced_dataset.png"
    )

    plt.close()


# =========================================================
# MODEL ACCURACY GRAPH
# =========================================================

def plot_model_accuracy(results_df):

    plt.figure(figsize=(12, 6))

    plt.bar(

        results_df["Model"],

        results_df["Accuracy"]
    )

    plt.xticks(rotation=45)

    plt.title("Model Accuracy Comparison")

    plt.xlabel("Models")

    plt.ylabel("Accuracy")

    plt.savefig(
        "outputs/graphs/accuracy_comparison.png"
    )

    plt.close()


# =========================================================
# SAVE SHAP SUMMARY
# =========================================================

def save_shap_summary():

    plt.savefig(

        "outputs/graphs/shap_summary.png",

        bbox_inches='tight'
    )

    plt.close()