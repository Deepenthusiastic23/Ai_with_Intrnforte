import pandas as pd

from src.preprocess import encode_features


def test_encode_features_binarizes_target():
    df = pd.DataFrame({
        "Age": [25, 40],
        "Workclass": ["Private", "Self-emp"],
        "Fnlwgt": [1, 2],
        "Income": ["<=50K", ">50K"],
    })
    X, y, encoders = encode_features(df)

    assert "Fnlwgt" not in X.columns
    assert list(y) == [0, 1]
    assert "Workclass" in encoders
