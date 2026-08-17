"""Preprocessing pipeline fitted only on training data."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PCA_FEATURES
from src.feature_engineering import TransactionFeatureEngineer
from src.utils import get_logger

logger = get_logger("fraud.preprocess")

SCALE_FEATURES = ["Time", "Amount", "Amount_log", "Hour"]
PASSTHROUGH_FEATURES = list(PCA_FEATURES) + ["Hour_sin", "Hour_cos"]


def build_preprocessor() -> Pipeline:
    """sklearn Pipeline: feature engineering → impute/scale selected columns.

    V1–V28 are PCA components from the original study and are left unscaled.
    Time/Amount-derived columns are standardized using training statistics only.
    """
    engineer = TransactionFeatureEngineer()
    column_ops = ColumnTransformer(
        transformers=[
            (
                "scale",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                SCALE_FEATURES,
            ),
            (
                "pass",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                PASSTHROUGH_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_ops.set_output(transform="pandas")
    pipeline = Pipeline(
        steps=[
            ("engineer", engineer),
            ("columns", column_ops),
        ]
    )
    return pipeline


def fit_preprocessor(pipeline: Pipeline, X_train: pd.DataFrame) -> Pipeline:
    logger.info("Fitting preprocessor on training data only (%s rows)", len(X_train))
    pipeline.fit(X_train)
    return pipeline


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    transformed = pipeline.transform(X)
    if not isinstance(transformed, pd.DataFrame):
        names = get_feature_names(pipeline)
        transformed = pd.DataFrame(transformed, columns=names, index=getattr(X, "index", None))
    return transformed


def get_feature_names(pipeline: Pipeline) -> List[str]:
    try:
        return list(pipeline.get_feature_names_out())
    except Exception:
        return SCALE_FEATURES + PASSTHROUGH_FEATURES


def validate_transaction_frame(
    df: pd.DataFrame,
    required: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce types and report problems without mutating caller input."""
    required = list(required or ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)])
    issues: List[str] = []
    out = df.copy()
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        n_bad = int(out[col].isna().sum())
        if n_bad:
            issues.append(f"{col}: {n_bad} non-numeric/missing values")
    if "Amount" in out.columns and (out["Amount"] < 0).any():
        issues.append("Negative Amount values present")
    if "Time" in out.columns and (out["Time"] < 0).any():
        issues.append("Negative Time values present")
    return out, issues
