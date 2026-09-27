"""Detailed evaluation plots for the best-performing model."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def plot_confusion_matrix(y_test, preds, model_name: str, out_dir: str) -> None:
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=120)
    plt.close()


def plot_roc_curve(y_test, probs, model_name: str, out_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, probs):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, model_name: str, out_dir: str) -> pd.Series | None:
    if not hasattr(model, "feature_importances_"):
        return None
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=fi.values[:10], y=fi.index[:10])
    plt.title(f"Top 10 Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=120)
    plt.close()
    return fi
