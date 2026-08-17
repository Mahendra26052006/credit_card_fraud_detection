"""SHAP explainability for the production tree model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, RANDOM_SEED, SHAP_BACKGROUND_SAMPLES, SHAP_EXPLAIN_SAMPLES
from src.utils import get_logger

logger = get_logger("fraud.shap")


def _import_shap():
    import shap

    return shap


def build_explainer(model, X_background: pd.DataFrame):
    shap = _import_shap()
    background = _sample_frame(X_background, SHAP_BACKGROUND_SAMPLES)
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("Using SHAP TreeExplainer")
        return explainer
    except Exception as exc:
        logger.info("TreeExplainer fallback (%s); using Explainer", exc)
        return shap.Explainer(model, background)


def shap_values_for(explainer, X: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    return arr


def plot_global_shap(model, X: pd.DataFrame, prefix: str = "shap") -> Dict[str, str]:
    shap = _import_shap()
    sample = _sample_frame(X, SHAP_EXPLAIN_SAMPLES)
    explainer = build_explainer(model, sample)
    shap_values = shap_values_for(explainer, sample)

    paths = {}
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, show=False, plot_type="bar")
    bar_path = FIGURES_DIR / f"{prefix}_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close()
    paths["bar"] = str(bar_path)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, show=False)
    beeswarm_path = FIGURES_DIR / f"{prefix}_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=140, bbox_inches="tight")
    plt.close()
    paths["beeswarm"] = str(beeswarm_path)
    logger.info("Saved SHAP global plots")
    return paths


def explain_instance(
    model,
    explainer,
    X_row: pd.DataFrame,
    top_k: int = 8,
) -> Dict[str, Any]:
    if len(X_row) != 1:
        X_row = X_row.iloc[[0]]
    values = shap_values_for(explainer, X_row)[0]
    names = list(X_row.columns)
    pairs = sorted(zip(names, values), key=lambda kv: abs(kv[1]), reverse=True)
    top = pairs[:top_k]
    positive = [(n, float(v)) for n, v in pairs if v > 0][:top_k]
    negative = [(n, float(v)) for n, v in pairs if v < 0][:top_k]
    return {
        "top_features": [{"feature": n, "shap_value": float(v)} for n, v in top],
        "positive_contributors": [{"feature": n, "shap_value": v} for n, v in positive],
        "negative_contributors": [{"feature": n, "shap_value": v} for n, v in negative],
        "shap_values": {n: float(v) for n, v in zip(names, values)},
    }


def top_feature_summary(explanation: Dict[str, Any], n: int = 5) -> List[str]:
    lines = []
    for item in explanation.get("top_features", [])[:n]:
        direction = "increases" if item["shap_value"] > 0 else "decreases"
        lines.append(f"{item['feature']} → {direction} fraud probability")
    return lines


def _sample_frame(X: pd.DataFrame, n: int, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=random_state)
