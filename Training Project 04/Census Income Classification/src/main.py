"""
Census Income Project — main pipeline.

Run with:  python -m src.main
"""

import warnings

from src.data_loader import load_clean_data
from src.eda import run_all as run_eda
from src.evaluate import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
from src.preprocess import encode_features, split_and_scale
from src.train import get_models, train_and_compare

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs/figures"


def main():
    print("Loading data...")
    df = load_clean_data()
    print("Shape:", df.shape)

    print("Running EDA...")
    run_eda(df, OUTPUT_DIR)

    print("Encoding features...")
    X, y, _ = encode_features(df)
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, _ = split_and_scale(X, y)

    print("Training and comparing models...")
    models = get_models()
    results = train_and_compare(models, X_train, y_train, X_test, y_test, X_train_s, X_test_s)
    print(results.to_string(index=False))
    results.to_csv("outputs/model_comparison.csv", index=False)

    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    if best_name == "Logistic Regression":
        preds, probs = best_model.predict(X_test_s), best_model.predict_proba(X_test_s)[:, 1]
    else:
        preds, probs = best_model.predict(X_test), best_model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, preds, best_name, OUTPUT_DIR)
    plot_roc_curve(y_test, probs, best_name, OUTPUT_DIR)
    fi = plot_feature_importance(best_model, X.columns, best_name, OUTPUT_DIR)
    if fi is not None:
        print("\nTop features:\n", fi.head(10))

    print("\nDone. See outputs/ for plots and model_comparison.csv")


if __name__ == "__main__":
    main()
