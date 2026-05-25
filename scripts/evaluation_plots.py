# =========================================================
# ADVANCED EVALUATION VISUALIZATIONS
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import os


# =========================================================
# CREATE GRAPH FOLDER
# =========================================================

os.makedirs("outputs/graphs", exist_ok=True)


# =========================================================
# ROC CURVE FUNCTION
# =========================================================

def plot_roc_curve(

    y_test,

    y_prob,

    model_name
):

    # Calculate ROC values
    fpr, tpr, _ = roc_curve(

        y_test,

        y_prob
    )

    roc_auc = auc(fpr, tpr)

    # Create graph
    plt.figure(figsize=(8, 6))

    plt.plot(

        fpr,

        tpr,

        label=f"AUC = {roc_auc:.4f}"
    )

    # Random line
    plt.plot(

        [0, 1],

        [0, 1],

        linestyle='--'
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve - {model_name}")

    plt.legend(loc="lower right")

    # Save graph
    plt.savefig(

        f"outputs/graphs/roc_curve_{model_name}.png"
    )

    plt.close()

    print(
        f"\n✅ ROC Curve Saved for {model_name}"
    )


# =========================================================
# CONFUSION MATRIX FUNCTION
# =========================================================

def plot_confusion_matrix(

    y_test,

    y_pred,

    model_name
):

    cm = confusion_matrix(

        y_test,

        y_pred
    )

    # Display matrix
    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    # Save graph
    plt.savefig(

        f"outputs/graphs/confusion_matrix_{model_name}.png"
    )

    plt.close()

    print(
        f"\n✅ Confusion Matrix Saved for {model_name}"
    )