"""Shared pytest fixtures. Uses a tiny synthetic dataset — never the Kaggle CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import BASE_FEATURES, PCA_FEATURES, TARGET_COLUMN


def make_synthetic_frame(n: int = 240, n_fraud: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172_800, n)
    data["Amount"] = np.abs(rng.normal(80, 40, n))
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    for feat in ["V4", "V10", "V12", "V14", "V17"]:
        data[feat][:n_fraud] += 2.8
    data[TARGET_COLUMN] = y
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture
def raw_transaction() -> dict:
    tx = {name: 0.0 for name in BASE_FEATURES}
    tx["Time"] = 1000.0
    tx["Amount"] = 88.12
    tx["V14"] = -3.2
    tx["V10"] = -2.1
    return tx
