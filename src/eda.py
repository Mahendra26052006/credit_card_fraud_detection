"""Exploratory data analysis and figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, PCA_FEATURES, TARGET_COLUMN
from src.utils import get_logger, save_json

logger = get_logger("fraud.eda")


def run_eda(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120

    summary = _dataset_summary(df)
    _plot_class_imbalance(df, figures_dir / "class_imbalance.png")
    _plot_amount_distribution(df, figures_dir / "amount_distribution.png")
    _plot_amount_by_class(df, figures_dir / "amount_fraud_vs_legit.png")
    _plot_correlation(df, figures_dir / "correlation_heatmap.png")
    _plot_time(df, figures_dir / "time_distribution.png")
    _plot_feature_distributions(df, figures_dir / "feature_distributions.png")
    _plot_boxplots(df, figures_dir / "boxplots_fraud_vs_legit.png")
    _plot_fraud_patterns(df, figures_dir / "fraud_patterns.png")
    _plot_hour_fraud_rate(df, figures_dir / "hour_fraud_rate.png")
    save_json(summary, figures_dir.parent / "metrics" / "eda_summary.json")
    logger.info("EDA complete. Figures saved to %s", figures_dir)
    return summary


def _dataset_summary(df: pd.DataFrame) -> Dict:
    n = len(df)
    n_fraud = int((df[TARGET_COLUMN] == 1).sum())
    n_legit = n - n_fraud
    numeric = df.select_dtypes(include=[np.number])
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))).sum()
    return {
        "n_rows": int(n),
        "n_columns": int(df.shape[1]),
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_percentage": float(100.0 * n_fraud / n) if n else 0.0,
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "amount_describe": df["Amount"].describe().to_dict() if "Amount" in df else {},
        "time_describe": df["Time"].describe().to_dict() if "Time" in df else {},
        "iqr_outlier_counts": {k: int(v) for k, v in outlier_counts.items()},
        "notes": {
            "outliers": (
                "IQR outlier counts are reported for transparency. They are NOT "
                "removed: fraud often lives in the tails."
            ),
            "accuracy_warning": (
                "A majority-class classifier would be ~99.83% accurate and catch zero fraud."
            ),
        },
    }


def _plot_class_imbalance(df: pd.DataFrame, path: Path) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#4C78A8", "#E45756"]
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=colors)
    ax.set_ylabel("Transactions")
    ax.set_title("Extreme class imbalance")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_amount_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log1p(df["Amount"]), bins=80, color="#4C78A8", alpha=0.9)
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Count")
    ax.set_title("Transaction amount distribution (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_amount_by_class(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color, name in [(0, "#4C78A8", "Legitimate"), (1, "#E45756", "Fraud")]:
        subset = np.log1p(df.loc[df[TARGET_COLUMN] == label, "Amount"])
        ax.hist(subset, bins=60, alpha=0.55, color=color, label=name, density=True)
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Density")
    ax.set_title("Amount distribution: fraud vs legitimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame, path: Path) -> None:
    cols = ["Time", "Amount", TARGET_COLUMN] + list(PCA_FEATURES[:12])
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Correlation heatmap (subset of features + target)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_time(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = (df["Time"] / 3600.0) % 24
    ax.hist(hours, bins=24, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Hour of day (from Time)")
    ax.set_ylabel("Transactions")
    ax.set_title("Time-based transaction distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_feature_distributions(df: pd.DataFrame, path: Path) -> None:
    cols = list(PCA_FEATURES[:8])
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col], bins=40, color="#72B7B2")
        ax.set_title(col)
    fig.suptitle("PCA feature distributions (V1–V8)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_boxplots(df: pd.DataFrame, path: Path) -> None:
    cols = ["V4", "V10", "V12", "V14", "V17", "Amount"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), cols):
        data = df[[col, TARGET_COLUMN]].copy()
        if col == "Amount":
            data[col] = np.log1p(data[col])
        sns.boxplot(
            data=data,
            x=TARGET_COLUMN,
            y=col,
            hue=TARGET_COLUMN,
            ax=ax,
            palette=["#4C78A8", "#E45756"],
            legend=False,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_xlabel("")
    fig.suptitle("Fraud vs legitimate box plots (selected features)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_fraud_patterns(df: pd.DataFrame, path: Path) -> None:
    sample_legit = df[df[TARGET_COLUMN] == 0].sample(n=min(8000, (df[TARGET_COLUMN] == 0).sum()), random_state=42)
    fraud = df[df[TARGET_COLUMN] == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample_legit["Time"] / 3600, np.log1p(sample_legit["Amount"]), s=6, alpha=0.15, label="Legitimate", c="#4C78A8")
    ax.scatter(fraud["Time"] / 3600, np.log1p(fraud["Amount"]), s=18, alpha=0.8, label="Fraud", c="#E45756")
    ax.set_xlabel("Hours from first transaction")
    ax.set_ylabel("log1p(Amount)")
    ax.set_title("Fraud transaction patterns over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_hour_fraud_rate(df: pd.DataFrame, path: Path) -> None:
    tmp = df.copy()
    tmp["Hour"] = ((tmp["Time"] / 3600.0) % 24).astype(int)
    rates = tmp.groupby("Hour")[TARGET_COLUMN].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(rates.index, rates.values, color="#E45756", alpha=0.85)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by hour of day")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
