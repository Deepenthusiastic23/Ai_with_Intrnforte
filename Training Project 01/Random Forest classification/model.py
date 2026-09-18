import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Churn"].value_counts())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# TotalCharges sometimes contains blank spaces.
# Convert it into numeric values.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Drop customerID because it is only an identifier.
df = df.drop(columns=["customerID"])


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


print("\n========== FEATURES ==========")
print(X.head())

print("\n========== TARGET ==========")
print(y.head())


# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\n========== CATEGORICAL FEATURES ==========")
print(categorical_features)

print("\n========== NUMERICAL FEATURES ==========")
print(numerical_features)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 7. PREPROCESSING
# ============================================================

# Numerical columns:
# Fill missing values using median.

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Categorical columns:
# Fill missing values using most frequent value.
# Then convert categories into numerical values.

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine both preprocessing strategies.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 9. CREATE COMPLETE ML PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            random_forest
        )
    ]
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

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


print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ============================================================
# 14. SAVE MODEL
# ============================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_churn.pkl"
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n========== MODEL SAVED ==========")

print(
    f"Model saved successfully at: {MODEL_PATH}"
)