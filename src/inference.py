"""Production inference: load artifacts and score transactions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BASE_FEATURES,
    BEST_MODEL_PATH,
    DEFAULT_RISK,
    METADATA_PATH,
    PREPROCESSOR_PATH,
)
from src.preprocessing import transform_features, validate_transaction_frame
from src.threshold import risk_level
from src.utils import get_logger, load_json

logger = get_logger("fraud.inference")

TransactionLike = Union[Dict[str, Any], pd.Series, pd.DataFrame, List[Dict[str, Any]]]


class ArtifactError(RuntimeError):
    """Raised when production artifacts are missing or incompatible."""


def load_artifacts(
    model_path: Path = BEST_MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    metadata_path: Path = METADATA_PATH,
) -> Dict[str, Any]:
    if not Path(model_path).exists():
        raise ArtifactError(
            f"No saved model at {model_path}. Run `python train.py` first."
        )
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = load_json(metadata_path)
    return {"model": model, "preprocessor": preprocessor, "metadata": metadata}


@lru_cache(maxsize=1)
def _cached_artifacts() -> Dict[str, Any]:
    return load_artifacts()


def clear_artifact_cache() -> None:
    _cached_artifacts.cache_clear()


def predict_transaction(
    transaction: TransactionLike,
    artifacts: Optional[Dict[str, Any]] = None,
    return_shap: bool = True,
) -> Dict[str, Any]:
    """Score a single transaction or a one-row frame.

    Returns fraud probability, label, risk level, threshold, and top features.
    """
    frame = _to_frame(transaction)
    results = predict_dataframe(frame, artifacts=artifacts, return_shap=return_shap)
    record = results.iloc[0].to_dict()
    top = record.get("top_features")
    if isinstance(top, str):
        try:
            import json

            top = json.loads(top)
        except Exception:
            top = []
    return {
        "fraud_probability": float(record["fraud_probability"]),
        "prediction": str(record["prediction"]),
        "risk_level": str(record["risk_level"]),
        "threshold": float(record["threshold"]),
        "top_features": top,
        "model_confidence": float(record["model_confidence"]),
    }


def predict_dataframe(
    df: pd.DataFrame,
    artifacts: Optional[Dict[str, Any]] = None,
    return_shap: bool = False,
    shap_limit: int = 50,
) -> pd.DataFrame:
    artifacts = artifacts or _cached_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    metadata = artifacts["metadata"]
    threshold = float(metadata["threshold"])
    low_fraction = float(metadata.get("risk_low_fraction", DEFAULT_RISK.low_fraction))
    high_fraction = float(metadata.get("risk_high_fraction", DEFAULT_RISK.high_fraction))

    validated, issues = validate_transaction_frame(df, required=list(BASE_FEATURES))
    if issues:
        logger.warning("Validation issues: %s", issues)

    X = transform_features(preprocessor, validated[list(BASE_FEATURES)])
    proba = _predict_proba(model, X, metadata.get("model_type", "supervised"))
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = proba
    out["prediction"] = np.where(pred == 1, "FRAUD", "LEGITIMATE")
    out["risk_level"] = [
        risk_level(p, threshold, low_fraction, high_fraction) for p in proba
    ]
    out["threshold"] = threshold
    out["model_confidence"] = np.where(pred == 1, proba, 1.0 - proba)

    if return_shap:
        explanations = _batch_shap(model, X, metadata, limit=shap_limit)
        padded = explanations + [[] for _ in range(len(out) - len(explanations))]
        out["top_features"] = padded
    return out


def _predict_proba(model, X: pd.DataFrame, model_type: str) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return _minmax01(scores)
    if hasattr(model, "score_samples"):
        scores = -np.asarray(model.score_samples(X), dtype=float)
        return _minmax01(scores)
    pred = np.asarray(model.predict(X), dtype=float)
    return pred


def _minmax01(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _batch_shap(model, X: pd.DataFrame, metadata: dict, limit: int) -> List[list]:
    try:
        from src.explainability import build_explainer, explain_instance

        n = min(limit, len(X))
        explainer = build_explainer(model, X.iloc[: min(len(X), 200)])
        rows = []
        for i in range(n):
            explanation = explain_instance(model, explainer, X.iloc[[i]])
            rows.append(explanation["top_features"][:5])
        return rows
    except Exception as exc:
        logger.warning("SHAP explanations unavailable: %s", exc)
        return [[] for _ in range(min(limit, len(X)))]


def _to_frame(transaction: TransactionLike) -> pd.DataFrame:
    if isinstance(transaction, pd.DataFrame):
        return transaction
    if isinstance(transaction, pd.Series):
        return transaction.to_frame().T
    if isinstance(transaction, list):
        return pd.DataFrame(transaction)
    if isinstance(transaction, dict):
        return pd.DataFrame([transaction])
    raise TypeError(f"Unsupported transaction type: {type(transaction)}")
