# =========================================================
# BANK MARKETING PROJECT - SMOTE PREPROCESSING
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Train-test split
from sklearn.model_selection import train_test_split

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

# SMOTE for balancing dataset
from imblearn.over_sampling import SMOTE

# System utilities
import os


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

    print_step(1, "SMOTE PREPROCESSING")


    # =====================================================
    # CREATE OUTPUT FOLDERS
    # =====================================================

    os.makedirs("outputs", exist_ok=True)

    os.makedirs("outputs/graphs", exist_ok=True)

    print_success("Output folders ready")


    # =====================================================
    # STEP 2 - LOAD DATASET
    # =====================================================

    # IMPORTANT:
    # Dataset uses SEMICOLON (;) separator

    try:

        df = pd.read_csv(

            RAW_DATASET_PATH,

            sep=';'
        )

        print_success(
            "Dataset Loaded Successfully"
        )

    except FileNotFoundError:

        print_error(
            "Dataset File Not Found"
        )

        return


    # =====================================================
    # STEP 3 - BASIC DATASET INFORMATION
    # =====================================================

    # Shape of dataset
    print_heading("📊 DATASET SHAPE")

    print_shape(df)


    # Column names
    print_heading("📌 COLUMN NAMES")

    print(df.columns)


    # First 5 rows
    print_heading("📄 FIRST 5 ROWS")

    print_head(df)


    # Missing values
    print_heading("❓ MISSING VALUES")

    print(df.isnull().sum())


    # =====================================================
    # STEP 4 - CHECK TARGET CLASS DISTRIBUTION
    # =====================================================

    print_heading("🎯 TARGET CLASS DISTRIBUTION")

    print(df['y'].value_counts())


    # =====================================================
    # VISUALIZE CLASS DISTRIBUTION
    # =====================================================

    plot_class_distribution(df['y'])

    print_success(
        "Class Distribution Graph Saved"
    )


    # =====================================================
    # STEP 5 - ENCODE CATEGORICAL DATA
    # =====================================================

    # Machine Learning models cannot understand text
    # So we convert text into numbers

    # Create separate encoder for each column
    label_encoders = {}

    # Encode all categorical columns
    for column in df.select_dtypes(
        include='object'
    ).columns:

        le = LabelEncoder()

        df[column] = le.fit_transform(
            df[column]
        )

        label_encoders[column] = le

    print_success(
        "Categorical Data Encoded Successfully"
    )


    # =====================================================
    # STEP 6 - SEPARATE FEATURES AND TARGET
    # =====================================================

    # Features (Input data)
    X = df.drop('y', axis=1)

    # Target (Output data)
    y = df['y']

    print_success(
        "Features and Target Separated"
    )


    # =====================================================
    # STEP 7 - TRAIN TEST SPLIT
    # =====================================================

    # 80% Training Data
    # 20% Testing Data

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
    # STEP 8 - APPLY SMOTE
    # =====================================================

    # Create SMOTE object
    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    # Apply SMOTE only on training data
    X_train_smote, y_train_smote = smote.fit_resample(

        X_train,
        y_train
    )

    print_success(
        "SMOTE Applied Successfully"
    )


    # =====================================================
    # STEP 9 - CHECK CLASS DISTRIBUTION
    # =====================================================

    print_heading("📊 BEFORE SMOTE")

    print(y_train.value_counts())


    print_heading("📊 AFTER SMOTE")

    print(y_train_smote.value_counts())


    # =====================================================
    # STEP 10 - VISUALIZE BALANCED DATASET
    # =====================================================

    plot_balanced_dataset(y_train_smote)

    print_success(
        "Balanced Dataset Graph Saved"
    )


    # =====================================================
    # STEP 11 - SAVE BALANCED DATASET
    # =====================================================

    # Convert into DataFrame
    balanced_df = pd.DataFrame(

        X_train_smote,

        columns=X.columns
    )

    # Add target column
    balanced_df['y'] = y_train_smote

    # Save CSV file
    balanced_df.to_csv(

        BALANCED_DATASET_PATH,

        index=False
    )

    print_success(
        "Balanced Dataset Saved Successfully"
    )

    print("\n📁 File Saved:")

    print(BALANCED_DATASET_PATH)


    # =====================================================
    # STEP 12 - FINAL MESSAGE
    # =====================================================

    print("\n========================================")

    print("🎉 SMOTE PREPROCESSING COMPLETED")

    print("========================================")

    print("""
Generated Outputs:

1. dataset/balanced_bank_data.csv

2. outputs/graphs/class_distribution_before_smote.png

3. outputs/graphs/balanced_dataset.png


Next Step:
Run -> 02_Model_Training.py
""")


# =========================================================
# RUN MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    main()