"""Model factories for baselines, boosting, and anomaly detectors."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from src.config import N_JOBS, RANDOM_SEED
from src.utils import get_logger

logger = get_logger("fraud.models")


def build_model(
    name: str,
    *,
    n_estimators: int = 200,
    random_state: int = RANDOM_SEED,
    scale_pos_weight: float = 1.0,
    use_class_weight: bool = True,
    n_jobs: int = N_JOBS,
) -> Any:
    key = name.lower().strip()
    builders = {
        "logistic_regression": _logistic_regression,
        "decision_tree": _decision_tree,
        "random_forest": _random_forest,
        "xgboost": _xgboost,
        "lightgbm": _lightgbm,
        "catboost": _catboost,
        "hist_gradient_boosting": _hist_gb,
        "isolation_forest": _isolation_forest,
        "autoencoder": _autoencoder,
    }
    if key not in builders:
        raise ValueError(f"Unknown model: {name}")
    return builders[key](
        n_estimators=n_estimators,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        use_class_weight=use_class_weight,
        n_jobs=n_jobs,
    )


def _logistic_regression(**kwargs) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced" if kwargs["use_class_weight"] else None,
        max_iter=1000,
        solver="lbfgs",
        random_state=kwargs["random_state"],
    )


def _decision_tree(**kwargs) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        class_weight="balanced" if kwargs["use_class_weight"] else None,
        max_depth=8,
        min_samples_leaf=20,
        random_state=kwargs["random_state"],
    )


def _random_forest(**kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=min(kwargs["n_estimators"], 200),
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced" if kwargs["use_class_weight"] else None,
        n_jobs=kwargs["n_jobs"],
        random_state=kwargs["random_state"],
    )


def _xgboost(**kwargs):
    from xgboost import XGBClassifier

    params: Dict[str, Any] = dict(
        n_estimators=kwargs["n_estimators"],
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        n_jobs=kwargs["n_jobs"],
        random_state=kwargs["random_state"],
        tree_method="hist",
        verbosity=0,
    )
    if kwargs["use_class_weight"]:
        params["scale_pos_weight"] = kwargs["scale_pos_weight"]
    try:
        import xgboost as xgb

        if hasattr(xgb, "build_info"):
            # GPU is optional; hist on CPU is the laptop default.
            pass
    except Exception:
        pass
    return XGBClassifier(**params)


def _lightgbm(**kwargs):
    from lightgbm import LGBMClassifier

    params: Dict[str, Any] = dict(
        n_estimators=kwargs["n_estimators"],
        num_leaves=15,
        max_depth=5,
        min_child_samples=50,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=kwargs["n_jobs"],
        random_state=kwargs["random_state"],
        verbosity=-1,
    )
    # Full n_neg/n_pos scale_pos_weight (~600) collapses LightGBM ranking
    # quality on this dataset. Leave the booster unweighted; class_weight
    # models (LR/RF) and XGBoost/CatBoost still receive imbalance handling.
    return LGBMClassifier(**params)


def _catboost(**kwargs):
    from catboost import CatBoostClassifier

    params: Dict[str, Any] = dict(
        iterations=kwargs["n_estimators"],
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=kwargs["random_state"],
        verbose=False,
        thread_count=-1 if kwargs["n_jobs"] == -1 else kwargs["n_jobs"],
        allow_writing_files=False,
    )
    if kwargs["use_class_weight"]:
        params["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**params)


def _hist_gb(**kwargs) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=kwargs["n_estimators"],
        learning_rate=0.08,
        max_depth=6,
        class_weight="balanced" if kwargs["use_class_weight"] else None,
        random_state=kwargs["random_state"],
    )


def _isolation_forest(**kwargs) -> IsolationForest:
    return IsolationForest(
        n_estimators=min(kwargs["n_estimators"], 200),
        contamination=0.0017,
        n_jobs=kwargs["n_jobs"],
        random_state=kwargs["random_state"],
    )


def _autoencoder(**kwargs) -> "AutoencoderAnomalyDetector":
    return AutoencoderAnomalyDetector(random_state=kwargs["random_state"])


class AutoencoderAnomalyDetector:
    """Lightweight MLP reconstruction-error anomaly detector.

    Trained only on legitimate transactions. High reconstruction error is
    treated as an anomaly score. This avoids a heavy TensorFlow/PyTorch
    dependency while still demonstrating the autoencoder idea.
    """

    def __init__(
        self,
        hidden_layer_sizes=(16, 8, 16),
        random_state: int = RANDOM_SEED,
        max_iter: int = 40,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.threshold_: Optional[float] = None

    def fit(self, X, y=None):
        self.model.fit(X, X)
        return self

    def reconstruction_error(self, X):
        import numpy as np

        recon = self.model.predict(X)
        return np.mean(np.square(np.asarray(X) - recon), axis=1)

    def decision_function(self, X):
        # Higher = more anomalous (fraud-like).
        return self.reconstruction_error(X)

    def score_samples(self, X):
        # sklearn-style: higher = more normal.
        return -self.reconstruction_error(X)

    def predict(self, X):
        import numpy as np

        scores = self.reconstruction_error(X)
        if self.threshold_ is None:
            self.threshold_ = float(np.quantile(scores, 0.99))
        return (scores >= self.threshold_).astype(int)
