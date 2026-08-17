"""False-positive / false-negative analysis."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COLUMN
from src.utils import get_logger, save_json

logger = get_logger("fraud.errors")


def run_error_analysis(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    raw_frame: pd.DataFrame,
    prefix: str = "test",
) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    high_conf_wrong = ((y_pred != y_true) & (np.maximum(y_proba, 1 - y_proba) >= 0.90)).sum()
    low_conf = (np.abs(y_proba - 0.5) < 0.10).sum()

    report = {
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        "n_true_positives": int(tp_mask.sum()),
        "n_true_negatives": int(tn_mask.sum()),
        "high_confidence_errors": int(high_conf_wrong),
        "low_confidence_predictions": int(low_conf),
        "fn_mean_probability": float(y_proba[fn_mask].mean()) if fn_mask.any() else None,
        "fp_mean_probability": float(y_proba[fp_mask].mean()) if fp_mask.any() else None,
        "fn_mean_amount": _mean_amount(raw_frame, fn_mask),
        "fp_mean_amount": _mean_amount(raw_frame, fp_mask),
        "tp_mean_amount": _mean_amount(raw_frame, tp_mask),
        "interpretation": _interpret(fp_mask, fn_mask, y_proba, raw_frame),
    }
    save_json(report, FIGURES_DIR.parent / "metrics" / f"{prefix}_error_analysis.json")
    _plot_fp_fn(raw_frame, y_proba, fp_mask, fn_mask, tp_mask, FIGURES_DIR / f"{prefix}_error_analysis.png")
    logger.info(
        "Error analysis: FP=%s FN=%s high-conf errors=%s",
        report["n_false_positives"],
        report["n_false_negatives"],
        report["high_confidence_errors"],
    )
    return report


def _mean_amount(raw_frame: pd.DataFrame, mask) -> float | None:
    if "Amount" not in raw_frame.columns or not np.any(mask):
        return None
    return float(raw_frame.loc[mask, "Amount"].mean())


def _interpret(fp_mask, fn_mask, y_proba, raw_frame) -> str:
    parts = []
    if fn_mask.any() and "Amount" in raw_frame.columns:
        fn_amt = raw_frame.loc[fn_mask, "Amount"]
        parts.append(
            f"Missed frauds (FN={int(fn_mask.sum())}) have mean amount {fn_amt.mean():.2f} "
            f"and mean predicted probability {y_proba[fn_mask].mean():.3f}. "
            "These are typically frauds whose PCA signature is closer to the legitimate bulk."
        )
    if fp_mask.any() and "Amount" in raw_frame.columns:
        fp_amt = raw_frame.loc[fp_mask, "Amount"]
        parts.append(
            f"False alerts (FP={int(fp_mask.sum())}) have mean amount {fp_amt.mean():.2f} "
            f"and mean probability {y_proba[fp_mask].mean():.3f}. "
            "In production these would enter a human review queue rather than an auto-block."
        )
    if not parts:
        parts.append("No FP/FN cases available in this split.")
    return " ".join(parts)


def _plot_fp_fn(raw_frame, y_proba, fp_mask, fn_mask, tp_mask, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(y_proba[fp_mask], bins=20, color="#F58518", alpha=0.85, label="FP")
    axes[0].hist(y_proba[fn_mask], bins=20, color="#E45756", alpha=0.85, label="FN")
    axes[0].hist(y_proba[tp_mask], bins=20, color="#54A24B", alpha=0.5, label="TP")
    axes[0].set_xlabel("Predicted fraud probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error probability profiles")
    axes[0].legend()

    if "Amount" in raw_frame.columns:
        data = pd.DataFrame(
            {
                "log_amount": np.log1p(raw_frame["Amount"]),
                "error_type": np.where(
                    fp_mask, "FP", np.where(fn_mask, "FN", np.where(tp_mask, "TP", "TN"))
                ),
            }
        )
        subset = data[data["error_type"].isin(["FP", "FN", "TP"])]
        if not subset.empty:
            sns.boxplot(data=subset, x="error_type", y="log_amount", ax=axes[1])
            axes[1].set_title("Amount by error type")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
