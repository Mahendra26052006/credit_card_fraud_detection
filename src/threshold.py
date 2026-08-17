"""Decision-threshold optimization and cost-sensitive selection.

The default probability cutoff of 0.5 is almost never optimal under 0.17% fraud
prevalence. Thresholds are tuned on validation data only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_COST,
    FIGURES_DIR,
    MIN_RECALL_CONSTRAINT,
    THRESHOLD_GRID,
)
from src.evaluate import compute_metrics, expected_cost
from src.utils import get_logger, save_json

logger = get_logger("fraud.threshold")


def evaluate_thresholds(
    y_true,
    y_proba,
    thresholds: Iterable[float] = THRESHOLD_GRID,
    fraud_miss_cost: float = DEFAULT_COST.fraud_miss_cost,
    false_alert_cost: float = DEFAULT_COST.false_alert_cost,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rows: List[dict] = []
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        metrics = compute_metrics(y_true, pred, y_proba)
        metrics["threshold"] = float(thr)
        metrics["expected_cost"] = expected_cost(
            y_true, pred, fraud_miss_cost, false_alert_cost
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_operating_threshold(
    sweep: pd.DataFrame,
    policy: str = "min_cost_recall_constrained",
    min_recall: float = MIN_RECALL_CONSTRAINT,
) -> Dict:
    """Choose a threshold from a validation sweep.

    Policies
    --------
    min_cost:
        Minimize FN * fraud_miss_cost + FP * false_alert_cost.
    max_f1:
        Maximize F1.
    high_recall:
        Among rows with recall >= min_recall, maximize precision.
        If none qualify, take the highest-recall row.
    min_cost_recall_constrained (default):
        Among rows with recall >= min_recall, minimize expected cost.
        Falls back to min_cost if the constraint is infeasible.
    """
    if sweep.empty:
        raise ValueError("Empty threshold sweep")

    policy = policy.lower()
    feasible = sweep[sweep["recall"] >= min_recall]

    if policy == "max_f1":
        row = sweep.loc[sweep["f1"].idxmax()]
        reason = "Selected to maximize validation F1."
    elif policy == "high_recall":
        pool = feasible if not feasible.empty else sweep
        row = pool.loc[pool["precision"].idxmax()] if not feasible.empty else pool.loc[pool["recall"].idxmax()]
        reason = (
            f"Selected for recall >= {min_recall:.2f} with best precision."
            if not feasible.empty
            else "No threshold met the recall constraint; chose highest recall."
        )
    elif policy == "min_cost":
        row = sweep.loc[sweep["expected_cost"].idxmin()]
        reason = (
            "Selected to minimize validation expected cost "
            "(FN is much more expensive than FP)."
        )
    else:
        pool = feasible if not feasible.empty else sweep
        row = pool.loc[pool["expected_cost"].idxmin()]
        if feasible.empty:
            reason = (
                f"No threshold reached recall {min_recall:.2f}; "
                "fell back to minimum expected cost."
            )
        else:
            reason = (
                f"Selected to minimize expected cost among thresholds with "
                f"recall >= {min_recall:.2f}. Missed fraud (FN) is priced at a "
                "much higher cost than a false alert (FP), matching a banking "
                "review workflow."
            )

    result = row.to_dict()
    result["policy"] = policy
    result["reason"] = reason
    logger.info(
        "Operating threshold=%.3f policy=%s recall=%.3f precision=%.3f cost=%.1f",
        result["threshold"],
        policy,
        result["recall"],
        result["precision"],
        result["expected_cost"],
    )
    return result


def plot_threshold_tradeoff(sweep: pd.DataFrame, selected: float, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(sweep["threshold"], sweep["precision"], label="Precision")
    axes[0].plot(sweep["threshold"], sweep["recall"], label="Recall")
    axes[0].plot(sweep["threshold"], sweep["f1"], label="F1")
    axes[0].axvline(selected, color="black", linestyle="--", label=f"Selected {selected:.2f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / Recall / F1 vs threshold")
    axes[0].legend(fontsize=8)

    axes[1].plot(sweep["threshold"], sweep["expected_cost"], color="firebrick")
    axes[1].axvline(selected, color="black", linestyle="--")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Expected cost")
    axes[1].set_title("Validation expected cost vs threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def risk_level(
    probability: float,
    threshold: float,
    low_fraction: float = 0.5,
    high_fraction: float = 1.0,
) -> str:
    low_cut = threshold * low_fraction
    high_cut = threshold * high_fraction
    if probability >= high_cut:
        return "HIGH"
    if probability >= low_cut:
        return "MEDIUM"
    return "LOW"


def persist_threshold_artifacts(sweep: pd.DataFrame, selected: Dict, prefix: str = "val") -> None:
    sweep.to_csv(FIGURES_DIR.parent / "metrics" / f"{prefix}_threshold_sweep.csv", index=False)
    save_json(selected, FIGURES_DIR.parent / "metrics" / f"{prefix}_selected_threshold.json")
