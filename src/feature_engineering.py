"""Leakage-safe feature engineering transformers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import BASE_FEATURES, PCA_FEATURES


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add a small set of justified transaction features.

    Features
    --------
    Amount_log : log1p(Amount)
        Amount is right-skewed; log compression helps linear models and
        reduces the pull of a few very large transactions.
    Hour : (Time / 3600) mod 24
        Time is seconds elapsed from the first transaction. Hour-of-day is a
        more interpretable circadian signal for fraud bursts.
    Hour_sin, Hour_cos : cyclical encoding of Hour
        Hour 23 and Hour 0 are adjacent; a raw integer would treat them as far
        apart. Sine/cosine encoding preserves that wrap-around.

    No global aggregations (means, frequencies) are computed here so the
    transformer cannot leak holdout statistics.
    """

    def __init__(self, amount_col: str = "Amount", time_col: str = "Time") -> None:
        self.amount_col = amount_col
        self.time_col = time_col
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = self._as_frame(X)
        self.feature_names_in_ = list(X.columns)
        sample = self._transform(X)
        self.feature_names_out_ = list(sample.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_out_ is None:
            raise RuntimeError("Transformer is not fitted")
        return self._transform(self._as_frame(X))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.feature_names_out_ is None:
            raise RuntimeError("Transformer is not fitted")
        return np.array(self.feature_names_out_, dtype=object)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        amount = pd.to_numeric(out[self.amount_col], errors="coerce").fillna(0.0).clip(lower=0)
        time_seconds = pd.to_numeric(out[self.time_col], errors="coerce").fillna(0.0)
        out["Amount_log"] = np.log1p(amount)
        hour = (time_seconds / 3600.0) % 24.0
        out["Hour"] = hour
        out["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        return out

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, pd.Series):
            return X.to_frame().T
        return pd.DataFrame(X, columns=list(BASE_FEATURES)[: np.asarray(X).shape[1]])


def expected_raw_columns() -> List[str]:
    return list(BASE_FEATURES)


def pca_feature_names() -> List[str]:
    return list(PCA_FEATURES)
