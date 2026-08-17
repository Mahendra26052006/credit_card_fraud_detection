import numpy as np

from src.threshold import evaluate_thresholds, risk_level, select_operating_threshold


def test_evaluate_thresholds_returns_expected_columns():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9])
    sweep = evaluate_thresholds(y, p, thresholds=[0.3, 0.5, 0.8])
    assert {"threshold", "precision", "recall", "f1", "expected_cost"}.issubset(sweep.columns)
    assert len(sweep) == 3


def test_select_operating_threshold_min_cost():
    y = np.array([0] * 50 + [1] * 10)
    p = np.concatenate([np.linspace(0.01, 0.4, 50), np.linspace(0.55, 0.99, 10)])
    sweep = evaluate_thresholds(y, p)
    selected = select_operating_threshold(sweep, policy="min_cost")
    assert 0.0 < selected["threshold"] < 1.0
    assert "reason" in selected


def test_risk_level_bands():
    assert risk_level(0.05, threshold=0.40) == "LOW"
    assert risk_level(0.25, threshold=0.40) == "MEDIUM"
    assert risk_level(0.80, threshold=0.40) == "HIGH"
