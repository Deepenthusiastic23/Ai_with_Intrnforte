"""Exploratory data analysis plots for the census income dataset."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_overview(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x="Income", data=df, ax=axes[0])
    axes[0].set_title("Target Class Balance")
    sns.histplot(df["Age"], bins=30, kde=True, ax=axes[1])
    axes[1].set_title("Age Distribution")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_overview.png", dpi=120)
    plt.close()


def plot_hours_vs_income(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Income", y="Hours_per_week", data=df)
    plt.title("Hours per Week vs Income")
    plt.savefig(f"{out_dir}/hours_vs_income.png", dpi=120)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png", dpi=120)
    plt.close()


def run_all(df: pd.DataFrame, out_dir: str) -> None:
    plot_overview(df, out_dir)
    plot_hours_vs_income(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
