"""Dataset download, validation, and loading.

The original dataset is the ULB / Worldline Credit Card Fraud Detection set
published on Kaggle (`mlg-ulb/creditcardfraud`). Redistribution of the raw CSV
on GitHub is avoided; this module downloads it locally when needed.

Download order:
1. Use an already-present local CSV.
2. TensorFlow public mirror of the same file (no Kaggle credentials required).
3. KaggleHub / Kaggle CLI if credentials are available.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import (
    BASE_FEATURES,
    DEFAULT_DATA_PATH,
    EXPECTED_N_FRAUD,
    EXPECTED_N_ROWS,
    KAGGLE_DATASET,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TENSORFLOW_DATASET_URL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from src.utils import ensure_directories, get_logger, save_json

logger = get_logger("fraud.data")

REQUIRED_COLUMNS = list(BASE_FEATURES) + [TARGET_COLUMN]


class DatasetError(Exception):
    """Raised when the dataset cannot be loaded or fails validation."""


def download_dataset(
    destination: Path = DEFAULT_DATA_PATH,
    source_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download or copy the credit-card fraud CSV into data/raw/."""
    ensure_directories()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source_path is not None:
        source_path = Path(source_path)
        if not source_path.exists():
            raise DatasetError(f"Local dataset not found: {source_path}")
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
            logger.info("Copied local dataset to %s", destination)
        return destination

    if destination.exists() and not force:
        logger.info("Dataset already present at %s", destination)
        return destination

    logger.info("Downloading dataset from TensorFlow public mirror...")
    try:
        _download_url(TENSORFLOW_DATASET_URL, destination)
        logger.info("Saved dataset to %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
        return destination
    except Exception as exc:
        logger.warning("TensorFlow mirror failed (%s); trying Kaggle", exc)

    if _try_kagglehub(destination):
        return destination
    if _try_kaggle_cli(destination):
        return destination
    raise DatasetError(
        "Could not download creditcard.csv. Place it at data/raw/creditcard.csv "
        "or pass --source /path/to/creditcard.csv"
    )


def _download_url(url: str, destination: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        tmp = destination.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(destination)


def _try_kagglehub(destination: Path) -> bool:
    try:
        import kagglehub

        logger.info("Attempting KaggleHub download for %s", KAGGLE_DATASET)
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csv_files = list(path.rglob("creditcard.csv"))
        if not csv_files:
            return False
        shutil.copy2(csv_files[0], destination)
        logger.info("Copied KaggleHub dataset to %s", destination)
        return True
    except Exception as exc:
        logger.info("KaggleHub download skipped: %s", exc)
        return False


def _try_kaggle_cli(destination: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)
        csv_files = list(RAW_DATA_DIR.rglob("creditcard.csv"))
        if not csv_files:
            return False
        if csv_files[0].resolve() != destination.resolve():
            shutil.copy2(csv_files[0], destination)
        logger.info("Downloaded dataset via Kaggle API")
        return True
    except Exception as exc:
        logger.info("Kaggle CLI download skipped: %s", exc)
        return False


def load_raw_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        csv_path = download_dataset(destination=DEFAULT_DATA_PATH if path is None else csv_path)
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    return df


def validate_dataset(df: pd.DataFrame, strict_shape: bool = False) -> dict:
    """Validate schema, types, and basic integrity. Returns a report dict."""
    report: dict = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "issues": [],
    }

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError(f"Dataset missing required columns: {missing_cols}")

    if TARGET_COLUMN in df.columns:
        classes = set(pd.unique(df[TARGET_COLUMN]))
        if not classes.issubset({0, 1}):
            raise DatasetError(f"Target must be binary {{0, 1}}, found {classes}")
        report["n_fraud"] = int((df[TARGET_COLUMN] == 1).sum())
        report["n_legit"] = int((df[TARGET_COLUMN] == 0).sum())
        report["fraud_rate"] = float(report["n_fraud"] / max(report["n_rows"], 1))

    numeric = df[list(BASE_FEATURES)]
    if not all(pd.api.types.is_numeric_dtype(numeric[c]) for c in BASE_FEATURES):
        raise DatasetError("All transaction features must be numeric")

    n_inf = int(np_isinf(numeric).sum().sum())
    if n_inf:
        report["issues"].append(f"Found {n_inf} infinite values")

    if (df["Amount"] < 0).any():
        report["issues"].append("Negative Amount values detected")
    if (df["Time"] < 0).any():
        report["issues"].append("Negative Time values detected")

    if strict_shape:
        if report["n_rows"] != EXPECTED_N_ROWS:
            report["issues"].append(
                f"Row count {report['n_rows']} != expected {EXPECTED_N_ROWS}"
            )
        if report.get("n_fraud") != EXPECTED_N_FRAUD:
            report["issues"].append(
                f"Fraud count {report.get('n_fraud')} != expected {EXPECTED_N_FRAUD}"
            )

    logger.info(
        "Validation complete: rows=%s fraud=%s issues=%s",
        report["n_rows"],
        report.get("n_fraud"),
        report["issues"] or "none",
    )
    return report


def np_isinf(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.isin([float("inf"), float("-inf")])


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Data-quality cleaning that does not learn parameters from the data.

    Decisions (documented in the returned report and preprocessing_decisions):
    - Drop exact duplicate rows: identical records add no new information and
      can slightly inflate frequency-based metrics. This is a row-identity
      operation, not a distributional statistic, so it is safe before splitting.
    - Replace infinities with NaN, then drop rows with NaN in required columns.
    - Do NOT drop statistical outliers: fraud is often an outlier.
    """
    decisions = {
        "drop_exact_duplicates": True,
        "drop_non_finite_rows": True,
        "drop_statistical_outliers": False,
        "reason_keep_outliers": (
            "Fraudulent transactions are frequently extreme in PCA space or amount. "
            "Removing outliers would discard the signal we need to detect."
        ),
        "n_rows_before": int(len(df)),
        "n_duplicates": int(df.duplicated().sum()),
    }
    cleaned = df.copy()
    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned.drop_duplicates()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    decisions["n_rows_after"] = int(len(cleaned))
    decisions["n_rows_removed"] = decisions["n_rows_before"] - decisions["n_rows_after"]
    logger.info("Cleaning removed %s rows", decisions["n_rows_removed"])
    return cleaned, decisions


def stratified_split(
    df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe 70/15/15 stratified split. Test is held out until the end."""
    if abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) > 1e-9:
        raise ValueError("Train/val/test fractions must sum to 1")

    y = df[TARGET_COLUMN]
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_SIZE),
        stratify=y,
        random_state=random_state,
    )
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df[TARGET_COLUMN],
        random_state=random_state,
    )
    logger.info(
        "Split sizes | train=%s (fraud=%s) val=%s (fraud=%s) test=%s (fraud=%s)",
        len(train_df),
        int(train_df[TARGET_COLUMN].sum()),
        len(val_df),
        int(val_df[TARGET_COLUMN].sum()),
        len(test_df),
        int(test_df[TARGET_COLUMN].sum()),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def persist_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
        val_df.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
        test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)
    except Exception:
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    save_json(
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_fraud": int(train_df[TARGET_COLUMN].sum()),
            "val_fraud": int(val_df[TARGET_COLUMN].sum()),
            "test_fraud": int(test_df[TARGET_COLUMN].sum()),
        },
        PROCESSED_DATA_DIR / "split_summary.json",
    )


def features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y
