"""Feature encoding and train/test split for the census income dataset."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "Income"


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Label-encode categoricals, drop Fnlwgt, and binarize the target."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].apply(lambda v: 1 if v == ">50K" else 0)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le

    X = X.drop(columns=["Fnlwgt"], errors="ignore")
    return X, y, encoders


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split plus a fitted StandardScaler for linear models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
