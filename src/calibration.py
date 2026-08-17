"""Calibration curves and CalibratedClassifierCV comparison."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import FIGURES_DIR, RANDOM_SEED
from src.utils import get_logger

logger = get_logger("fraud.calibration")


def plot_calibration(y_true, y_proba, path, title: str = "Calibration curve") -> float:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    brier = float(brier_score_loss(y_true, y_proba))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=f"Model (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Observed fraud fraction")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return brier


def maybe_calibrate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    enabled: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """Compare raw vs isotonic-calibrated probabilities on validation data.

    Calibration is fit on training data with internal CV so the validation
    set remains a fair comparison. If calibration does not improve Brier
    score, the raw model is kept.
    """
    raw_proba = model.predict_proba(X_val)[:, 1]
    raw_brier = float(brier_score_loss(y_val, raw_proba))
    plot_calibration(y_val, raw_proba, FIGURES_DIR / "calibration_raw.png", "Raw probabilities")

    if not enabled:
        return model, {"raw_brier": raw_brier, "calibrated_brier": raw_brier, "used_calibration": False}

    logger.info("Fitting CalibratedClassifierCV (isotonic, cv=3)")
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv=3)
    try:
        calibrated.fit(X_train, y_train)
        cal_proba = calibrated.predict_proba(X_val)[:, 1]
        cal_brier = float(brier_score_loss(y_val, cal_proba))
        plot_calibration(
            y_val, cal_proba, FIGURES_DIR / "calibration_isotonic.png", "Isotonic-calibrated probabilities"
        )
    except Exception as exc:
        logger.warning("Calibration failed (%s); keeping raw model", exc)
        return model, {"raw_brier": raw_brier, "calibrated_brier": raw_brier, "used_calibration": False}

    use_cal = cal_brier < raw_brier
    logger.info("Brier raw=%.5f calibrated=%.5f use_calibrated=%s", raw_brier, cal_brier, use_cal)
    return (calibrated if use_cal else model), {
        "raw_brier": raw_brier,
        "calibrated_brier": cal_brier,
        "used_calibration": bool(use_cal),
    }
