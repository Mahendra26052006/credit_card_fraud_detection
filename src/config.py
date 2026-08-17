"""Project-wide configuration for the credit-card fraud detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED: int = 42
TARGET_COLUMN: str = "Class"
PCA_FEATURES: Tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))
BASE_FEATURES: Tuple[str, ...] = ("Time", "Amount") + PCA_FEATURES
ENGINEERED_FEATURES: Tuple[str, ...] = (
    "Amount_log",
    "Hour",
    "Hour_sin",
    "Hour_cos",
)

TRAIN_SIZE: float = 0.70
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15

# Resampling target: minority / majority after sampling (training data only).
RESAMPLE_STRATEGY: float = 0.20

# Business cost framework (relative units).
FRAUD_MISS_COST: float = 500.0  # false negative: missed fraud / chargeback
FALSE_ALERT_COST: float = 5.0  # false positive: analyst review / customer friction

THRESHOLD_GRID: Tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)

# Risk bands relative to the chosen operating threshold.
RISK_LOW_FRACTION: float = 0.50  # p < 0.5 * threshold → LOW
RISK_HIGH_FRACTION: float = 1.00  # p >= threshold → HIGH

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_CONFIG_PATH = MODELS_DIR / "config.json"

TENSORFLOW_DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
EXPECTED_N_ROWS: int = 284_807
EXPECTED_N_FRAUD: int = 492

PRIMARY_METRIC: str = "pr_auc"
SECONDARY_METRIC: str = "f1"

N_JOBS: int = -1
CV_FOLDS: int = 3
OPTUNA_TRIALS: int = 12
OPTUNA_TIMEOUT_SECONDS: int = 600
SHAP_BACKGROUND_SAMPLES: int = 500
SHAP_EXPLAIN_SAMPLES: int = 400
ANOMALY_FIT_SAMPLES: int = 40_000
AUTOENCODER_FIT_SAMPLES: int = 30_000

# High-recall constraint used during threshold search.
MIN_RECALL_CONSTRAINT: float = 0.80


@dataclass
class TrainRuntimeConfig:
    """Runtime knobs that trade completeness for wall-clock time."""

    fast: bool = False
    skip_eda: bool = False
    skip_hpo: bool = False
    skip_shap: bool = False
    skip_autoencoder: bool = False
    optuna_trials: int = OPTUNA_TRIALS
    cv_folds: int = CV_FOLDS
    n_estimators: int = 200
    random_seed: int = RANDOM_SEED

    def apply_fast_mode(self) -> None:
        if not self.fast:
            return
        self.optuna_trials = 6
        self.cv_folds = 3
        self.n_estimators = 80


@dataclass
class CostConfig:
    fraud_miss_cost: float = FRAUD_MISS_COST
    false_alert_cost: float = FALSE_ALERT_COST


@dataclass
class RiskConfig:
    low_fraction: float = RISK_LOW_FRACTION
    high_fraction: float = RISK_HIGH_FRACTION


DEFAULT_COST = CostConfig()
DEFAULT_RISK = RiskConfig()

IMBALANCE_STRATEGIES: List[str] = [
    "class_weight",
    "random_under",
    "random_over",
    "smote",
    "smote_tomek",
]

BASELINE_MODELS: List[str] = ["logistic_regression", "decision_tree", "random_forest"]
ADVANCED_MODELS: List[str] = ["xgboost", "lightgbm", "catboost", "hist_gradient_boosting"]
ANOMALY_MODELS: List[str] = ["isolation_forest", "autoencoder"]

PRODUCTION_CANDIDATES: List[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "isolation_forest",
]
