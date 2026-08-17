"""Imbalanced-classification metrics, plots, and comparison tables."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, METRICS_DIR
from src.utils import get_logger, save_json

logger = get_logger("fraud.evaluate")


def compute_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: Optional[Iterable] = None,
    prefix: str = "",
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics: Dict[str, float] = {}

    tn, fp, fn, tp = _confusion_parts(y_true, y_pred)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) else 0.0
    metrics["support_fraud"] = float(int((y_true == 1).sum()))
    metrics["support_legit"] = float(int((y_true == 0).sum()))
    metrics["tp"] = float(tp)
    metrics["fp"] = float(fp)
    metrics["tn"] = float(tn)
    metrics["fn"] = float(fn)

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        metrics["roc_auc"] = _safe_roc_auc(y_true, proba)
        metrics["pr_auc"] = _safe_pr_auc(y_true, proba)
        metrics["brier"] = float(brier_score_loss(y_true, proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["brier"] = float("nan")

    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    return metrics


def _confusion_parts(y_true, y_pred) -> tuple:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _safe_roc_auc(y_true, proba) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def _safe_pr_auc(y_true, proba) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, proba))


def expected_cost(
    y_true,
    y_pred,
    fraud_miss_cost: float,
    false_alert_cost: float,
) -> float:
    tn, fp, fn, tp = _confusion_parts(y_true, y_pred)
    return float(fn * fraud_miss_cost + fp * false_alert_cost)


def plot_confusion_matrix(y_true, y_pred, title: str, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pr_curves(curves: Dict[str, tuple], path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, (y_true, proba) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = _safe_pr_auc(y_true, proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves (primary ranking metric)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc_curves(curves: Dict[str, tuple], path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, (y_true, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = _safe_roc_auc(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, path) -> None:
    plot_df = df.copy()
    metrics = [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"] if c in plot_df.columns]
    melted = plot_df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation model comparison")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_comparison_table(rows: List[Dict[str, Any]], path=None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    preferred = [
        "model",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "brier",
        "train_seconds",
        "inference_seconds",
        "n_features",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    out = path or (METRICS_DIR / "model_comparison.csv")
    df.to_csv(out, index=False)
    save_json({"rows": df.to_dict(orient="records")}, METRICS_DIR / "model_comparison.json")
    return df


def pr_auc_why() -> str:
    return (
        "Accuracy is dominated by the 99.83% legitimate majority: a model that "
        "never predicts fraud is ~99.8% accurate and useless. ROC-AUC can look "
        "optimistic because true negatives are easy. PR-AUC focuses on the rare "
        "positive class (precision vs recall) and is the primary ranking metric."
    )
