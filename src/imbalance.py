"""Class-imbalance strategies. Resampling is applied ONLY to training data."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import RANDOM_SEED, RESAMPLE_STRATEGY
from src.utils import get_logger

logger = get_logger("fraud.imbalance")


def class_counts(y: pd.Series) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    mapping = {int(v): int(c) for v, c in zip(values, counts)}
    return {"n_legit": mapping.get(0, 0), "n_fraud": mapping.get(1, 0)}


def scale_pos_weight(y: pd.Series) -> float:
    counts = class_counts(y)
    if counts["n_fraud"] == 0:
        return 1.0
    return float(counts["n_legit"] / counts["n_fraud"])


def resample_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    random_state: int = RANDOM_SEED,
    sampling_strategy: float = RESAMPLE_STRATEGY,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Resample training features/labels.

    Parameters
    ----------
    strategy:
        class_weight | random_under | random_over | smote | smote_tomek
    sampling_strategy:
        Desired minority/majority ratio after resampling. Using 1.0 (full
        balance) on ~200k rows creates a very large synthetic set; 0.2 is a
        practical compromise that still lifts minority representation ~100x.
    """
    before = class_counts(y)
    name = strategy.lower().strip()
    sampler = _build_sampler(name, random_state, sampling_strategy)
    if sampler is None:
        after = before
        report = _report(name, before, after, applied=False)
        logger.info("Imbalance strategy=%s (no resampling) %s", name, after)
        return X, y, report

    logger.info("Applying %s on training data %s", name, before)
    X_work, y_work = X, y
    if name in {"smote_tomek", "smotetomek"} and len(X) > 80_000:
        n_maj = min(40_000, int((np.asarray(y) == 0).sum()))
        logger.info("SMOTE-Tomek pre-undersample of majority to %s for laptop runtime", n_maj)
        pre = RandomUnderSampler(sampling_strategy={0: n_maj}, random_state=random_state)
        X_work, y_work = pre.fit_resample(X, y)
        if not isinstance(X_work, pd.DataFrame):
            X_work = pd.DataFrame(X_work, columns=X.columns)
        y_work = pd.Series(np.asarray(y_work), name=getattr(y, "name", "Class"))
    X_res, y_res = sampler.fit_resample(X_work, y_work)
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(np.asarray(y_res), name=getattr(y, "name", "Class"))
    after = class_counts(y_res)
    report = _report(name, before, after, applied=True)
    logger.info("After %s: %s", name, after)
    return X_res, y_res, report


def _build_sampler(name: str, random_state: int, sampling_strategy: float):
    if name in {"class_weight", "none", "no_resample"}:
        return None
    if name in {"random_under", "undersample", "rus"}:
        return RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    if name in {"random_over", "oversample", "ros"}:
        return RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    if name == "smote":
        return SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5,
        )
    if name in {"smote_tomek", "smotetomek"}:
        return SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    raise ValueError(f"Unknown imbalance strategy: {name}")


def _report(name: str, before: dict, after: dict, applied: bool) -> dict:
    return {
        "strategy": name,
        "applied": applied,
        "before": before,
        "after": after,
        "note": (
            "Resampling is training-only. Validation and test distributions "
            "remain the real-world ~0.17% fraud prior."
        ),
    }
