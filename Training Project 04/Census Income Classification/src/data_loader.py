"""Load and clean the Census Income dataset."""

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/census_income.csv"


def load_raw_data(source: str = DATA_URL) -> pd.DataFrame:
    """Load the raw census income CSV from a URL or local path."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, handle missing values marked as '?'."""
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()

    df.replace("?", np.nan, inplace=True)
    for c in ["Workclass", "Occupation", "Native_country"]:
        if c in df.columns:
            df[c].fillna(df[c].mode()[0], inplace=True)
    return df


def load_clean_data(source: str = DATA_URL) -> pd.DataFrame:
    return clean_data(load_raw_data(source))
