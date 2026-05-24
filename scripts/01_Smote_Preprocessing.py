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


# =========================================================
# STEP 2 - LOAD DATASET
# =========================================================

# IMPORTANT:
# Dataset uses SEMICOLON (;) separator
# So we must use sep=';'

df = pd.read_csv(
    "dataset/bank-additional-full.csv",
    sep=';'
)

print("\n✅ Dataset Loaded Successfully")


# =========================================================
# STEP 3 - BASIC DATASET INFORMATION
# =========================================================

# Shape of dataset
print("\n==============================")
print("📊 DATASET SHAPE")
print("==============================")

print(df.shape)

# Column names
print("\n==============================")
print("📌 COLUMN NAMES")
print("==============================")

print(df.columns)

# First 5 rows
print("\n==============================")
print("📄 FIRST 5 ROWS")
print("==============================")

print(df.head())

# Missing values
print("\n==============================")
print("❓ MISSING VALUES")
print("==============================")

print(df.isnull().sum())


# =========================================================
# STEP 4 - CHECK TARGET CLASS DISTRIBUTION
# =========================================================

print("\n==============================")
print("🎯 TARGET CLASS DISTRIBUTION")
print("==============================")

print(df['y'].value_counts())

# Visualization
df['y'].value_counts().plot(
    kind='bar'
)

plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig("outputs/graphs/class_distribution_before_smote.png")
plt.close()


# =========================================================
# STEP 5 - ENCODE CATEGORICAL DATA
# =========================================================

# Machine Learning models cannot understand text
# So we convert text into numbers

from sklearn.preprocessing import LabelEncoder

# Create separate encoder for each column
label_encoders = {}

# Encode all categorical columns
for column in df.select_dtypes(include='object').columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    label_encoders[column] = le

print("\n✅ Categorical Data Encoded Successfully")


# =========================================================
# STEP 6 - SEPARATE FEATURES AND TARGET
# =========================================================

# Features (Input data)
X = df.drop('y', axis=1)

# Target (Output data)
y = df['y']

print("\n✅ Features and Target Separated")


# =========================================================
# STEP 7 - TRAIN TEST SPLIT
# =========================================================

# 80% Training Data
# 20% Testing Data

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

print("\n✅ Train-Test Split Completed")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# =========================================================
# STEP 8 - APPLY SMOTE
# =========================================================

# Create SMOTE object
smote = SMOTE(random_state=42)

# Apply SMOTE only on training data
X_train_smote, y_train_smote = smote.fit_resample(

    X_train,
    y_train
)

print("\n✅ SMOTE Applied Successfully")


# =========================================================
# STEP 9 - CHECK CLASS DISTRIBUTION
# =========================================================

print("\n==============================")
print("📊 BEFORE SMOTE")
print("==============================")

print(y_train.value_counts())

print("\n==============================")
print("📊 AFTER SMOTE")
print("==============================")

print(y_train_smote.value_counts())


# =========================================================
# STEP 10 - VISUALIZE BALANCED DATASET
# =========================================================

pd.Series(y_train_smote).value_counts().plot(
    kind='bar'
)

plt.title("Balanced Dataset After SMOTE")

plt.xlabel("Class")

plt.ylabel("Count")

plt.savefig("outputs/graphs/balanced_dataset.png")
plt.close()

# =========================================================
# STEP 11 - SAVE BALANCED DATASET
# =========================================================

# Convert into DataFrame
balanced_df = pd.DataFrame(

    X_train_smote,
    columns=X.columns
)

# Add target column
balanced_df['y'] = y_train_smote

# Save CSV file
balanced_df.to_csv(

    "dataset/balanced_bank_data.csv",

    index=False
)

print("\n✅ Balanced Dataset Saved Successfully")

print("\n📁 File Name:")
print("dataset/balanced_bank_data.csv")


# =========================================================
# STEP 12 - FINAL MESSAGE
# =========================================================

print("\n========================================")
print("🎉 SMOTE PREPROCESSING COMPLETED")
print("========================================")

print("""
Next Step:
Run -> 02_Model_Training.py

Generated File:
dataset/balanced_bank_data.csv
""")