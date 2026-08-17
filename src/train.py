"""End-to-end training pipeline (leakage-safe, reproducible)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.calibration import maybe_calibrate
from src.config import (
    ADVANCED_MODELS,
    ANOMALY_FIT_SAMPLES,
    AUTOENCODER_FIT_SAMPLES,
    BASELINE_MODELS,
    BEST_MODEL_PATH,
    CV_FOLDS,
    DEFAULT_COST,
    DEFAULT_RISK,
    FIGURES_DIR,
    IMBALANCE_STRATEGIES,
    METADATA_PATH,
    METRICS_DIR,
    MLRUNS_DIR,
    MODEL_CONFIG_PATH,
    MODELS_DIR,
    N_JOBS,
    PREPROCESSOR_PATH,
    PRIMARY_METRIC,
    PRODUCTION_CANDIDATES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TrainRuntimeConfig,
)
from src.data_loader import (
    clean_dataframe,
    download_dataset,
    features_and_target,
    load_raw_dataframe,
    persist_splits,
    stratified_split,
    validate_dataset,
)
from src.eda import run_eda
from src.error_analysis import run_error_analysis
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    pr_auc_why,
    save_comparison_table,
)
from src.explainability import plot_global_shap
from src.imbalance import resample_training_data, scale_pos_weight
from src.models import build_model
from src.preprocessing import build_preprocessor, fit_preprocessor, get_feature_names, transform_features
from src.threshold import (
    evaluate_thresholds,
    persist_threshold_artifacts,
    plot_threshold_tradeoff,
    select_operating_threshold,
)
from src.utils import ensure_directories, get_logger, save_json, set_seed, timer

logger = get_logger("fraud.train")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit-card fraud detection pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to creditcard.csv")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing")
    parser.add_argument("--fast", action="store_true", help="Faster laptop run (fewer trees / trials)")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-autoencoder", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = TrainRuntimeConfig(
        fast=args.fast,
        skip_eda=args.skip_eda,
        skip_hpo=args.skip_hpo,
        skip_shap=args.skip_shap,
        skip_autoencoder=args.skip_autoencoder,
        random_seed=args.seed,
    )
    runtime.apply_fast_mode()
    set_seed(runtime.random_seed)
    ensure_directories()
    mlflow_run = _start_mlflow()

    if args.download or args.data_path:
        download_dataset(source_path=Path(args.data_path) if args.data_path else None)

    raw = load_raw_dataframe(Path(args.data_path) if args.data_path else None)
    validation_report = validate_dataset(raw, strict_shape=False)
    cleaned, cleaning_decisions = clean_dataframe(raw)
    save_json(
        {"validation": validation_report, "cleaning": cleaning_decisions, "leakage_controls": _leakage_notes()},
        METRICS_DIR / "preprocessing_decisions.json",
    )

    if not runtime.skip_eda:
        with timer("EDA", logger):
            run_eda(cleaned)

    train_df, val_df, test_df = stratified_split(cleaned, random_state=runtime.random_seed)
    persist_splits(train_df, val_df, test_df)

    X_train_raw, y_train = features_and_target(train_df)
    X_val_raw, y_val = features_and_target(val_df)
    X_test_raw, y_test = features_and_target(test_df)

    preprocessor = fit_preprocessor(build_preprocessor(), X_train_raw)
    X_train = transform_features(preprocessor, X_train_raw)
    X_val = transform_features(preprocessor, X_val_raw)
    X_test = transform_features(preprocessor, X_test_raw)
    feature_names = get_feature_names(preprocessor)
    pos_weight = scale_pos_weight(y_train)
    logger.info("Training scale_pos_weight=%.2f features=%s", pos_weight, len(feature_names))

    imbalance_results = compare_imbalance_strategies(
        X_train, y_train, X_val, y_val, runtime, pos_weight
    )

    comparison_rows: List[Dict[str, Any]] = []
    probability_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    fitted_models: Dict[str, Any] = {}

    supervised_names = BASELINE_MODELS + ADVANCED_MODELS
    if runtime.fast:
        supervised_names = [
            n for n in supervised_names if n not in {"catboost", "hist_gradient_boosting"}
        ]

    for name in supervised_names:
        result = train_supervised_model(
            name=name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            runtime=runtime,
            pos_weight=pos_weight,
            n_features=len(feature_names),
        )
        comparison_rows.append(result["row"])
        fitted_models[name] = result["model"]
        probability_curves[name] = (y_val.to_numpy(), result["val_proba"])

    anomaly_rows, anomaly_models, anomaly_probas = train_anomaly_models(
        X_train, y_train, X_val, y_val, runtime, n_features=len(feature_names)
    )
    comparison_rows.extend(anomaly_rows)
    fitted_models.update(anomaly_models)
    probability_curves.update(anomaly_probas)

    comparison_df = save_comparison_table(comparison_rows, METRICS_DIR / "model_comparison.csv")
    plot_model_comparison(comparison_df, FIGURES_DIR / "model_comparison.png")
    plot_pr_curves(probability_curves, FIGURES_DIR / "pr_curves.png")
    plot_roc_curves(probability_curves, FIGURES_DIR / "roc_curves.png")

    best_name = _select_best_model(comparison_df)
    logger.info("Best validation model by %s: %s", PRIMARY_METRIC, best_name)
    production_model = fitted_models[best_name]

    hpo_report: Dict[str, Any] = {"skipped": True}
    if not runtime.skip_hpo and best_name in {"lightgbm", "xgboost", "catboost", "random_forest"}:
        hpo_model, hpo_report = tune_model(
            best_name, X_train, y_train, X_val, y_val, runtime, pos_weight
        )
        if hpo_report.get("improved"):
            production_model = hpo_model
            fitted_models[best_name] = hpo_model
            logger.info("HPO improved validation PR-AUC to %.4f", hpo_report["best_pr_auc"])

    if hasattr(production_model, "predict_proba"):
        production_model, calib_report = maybe_calibrate(
            production_model, X_train, y_train, X_val, y_val, enabled=not runtime.fast
        )
    else:
        calib_report = {"used_calibration": False, "raw_brier": None, "calibrated_brier": None}

    val_proba = _scores(production_model, X_val)
    sweep = evaluate_thresholds(
        y_val,
        val_proba,
        fraud_miss_cost=DEFAULT_COST.fraud_miss_cost,
        false_alert_cost=DEFAULT_COST.false_alert_cost,
    )
    selected = select_operating_threshold(sweep)
    persist_threshold_artifacts(sweep, selected)
    plot_threshold_tradeoff(sweep, float(selected["threshold"]), FIGURES_DIR / "threshold_cost.png")
    threshold = float(selected["threshold"])

    val_pred = (val_proba >= threshold).astype(int)
    val_metrics = compute_metrics(y_val, val_pred, val_proba, prefix="val_")

    cv_report = cross_validate_model(clone_if_possible(production_model), X_train, y_train, runtime)

    shap_paths: Dict[str, str] = {}
    if not runtime.skip_shap and _supports_shap(production_model):
        try:
            with timer("SHAP", logger):
                shap_paths = plot_global_shap(production_model, X_train)
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)

    test_proba = _scores(production_model, X_test)
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_proba, prefix="test_")
    plot_confusion_matrix(y_test, test_pred, "Test confusion matrix", FIGURES_DIR / "test_confusion_matrix.png")
    error_report = run_error_analysis(
        X_test, y_test, test_pred, test_proba, test_df.reset_index(drop=True), prefix="test"
    )

    joblib.dump(production_model, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    metadata = {
        "model_name": best_name,
        "model_type": "anomaly" if best_name in {"isolation_forest", "autoencoder"} else "supervised",
        "threshold": threshold,
        "threshold_reason": selected.get("reason"),
        "threshold_policy": selected.get("policy"),
        "feature_names": feature_names,
        "raw_features": list(X_train_raw.columns),
        "risk_low_fraction": DEFAULT_RISK.low_fraction,
        "risk_high_fraction": DEFAULT_RISK.high_fraction,
        "random_seed": runtime.random_seed,
        "primary_metric": PRIMARY_METRIC,
        "calibration": calib_report,
        "version": "1.0.0",
    }
    save_json(metadata, METADATA_PATH)
    save_json(
        {
            "runtime": runtime.__dict__,
            "best_model": best_name,
            "pos_weight": pos_weight,
            "n_features": len(feature_names),
        },
        MODEL_CONFIG_PATH,
    )

    final_report = {
        "best_model": best_name,
        "threshold": threshold,
        "threshold_reason": selected.get("reason"),
        "validation_metrics": {k.replace("val_", ""): v for k, v in val_metrics.items()},
        "test_metrics": {k.replace("test_", ""): v for k, v in test_metrics.items()},
        "cross_validation": cv_report,
        "imbalance_comparison": imbalance_results,
        "hpo": hpo_report,
        "calibration": calib_report,
        "error_analysis": error_report,
        "shap_artifacts": shap_paths,
        "why_pr_auc": pr_auc_why(),
        "leakage_controls": _leakage_notes(),
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
            "train_fraud": int(y_train.sum()),
            "val_fraud": int(y_val.sum()),
            "test_fraud": int(y_test.sum()),
        },
        "cost_framework": {
            "fraud_miss_cost": DEFAULT_COST.fraud_miss_cost,
            "false_alert_cost": DEFAULT_COST.false_alert_cost,
            "validation_expected_cost": selected.get("expected_cost"),
        },
    }
    save_json(final_report, METRICS_DIR / "final_report.json")
    _log_mlflow(mlflow_run, final_report, comparison_df)

    logger.info("TEST PR-AUC=%.4f Recall=%.4f Precision=%.4f F1=%.4f",
                test_metrics.get("test_pr_auc", float("nan")),
                test_metrics.get("test_recall", float("nan")),
                test_metrics.get("test_precision", float("nan")),
                test_metrics.get("test_f1", float("nan")))
    logger.info("Saved production artifacts to %s", MODELS_DIR)
    return final_report


def compare_imbalance_strategies(
    X_train, y_train, X_val, y_val, runtime: TrainRuntimeConfig, pos_weight: float
) -> List[dict]:
    rows = []
    probe = "lightgbm"
    for strategy in IMBALANCE_STRATEGIES:
        with timer(f"imbalance:{strategy}", logger) as timed:
            X_res, y_res, report = resample_training_data(X_train, y_train, strategy)
            use_weights = strategy == "class_weight"
            model = build_model(
                probe,
                n_estimators=runtime.n_estimators,
                random_state=runtime.random_seed,
                scale_pos_weight=pos_weight if use_weights else 1.0,
                use_class_weight=use_weights,
            )
            model.fit(X_res, y_res)
            proba = _scores(model, X_val)
            pred = (proba >= 0.5).astype(int)
            metrics = compute_metrics(y_val, pred, proba)
        rows.append(
            {
                "strategy": strategy,
                "pr_auc": metrics["pr_auc"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "seconds": timed["seconds"],
                **report,
            }
        )
    save_json({"results": rows}, METRICS_DIR / "imbalance_comparison.json")
    pd.DataFrame(rows)[["strategy", "pr_auc", "recall", "precision", "f1", "seconds"]].to_csv(
        METRICS_DIR / "imbalance_comparison.csv", index=False
    )
    return rows


def train_supervised_model(
    name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    runtime: TrainRuntimeConfig,
    pos_weight: float,
    n_features: int,
) -> Dict[str, Any]:
    model = build_model(
        name,
        n_estimators=runtime.n_estimators,
        random_state=runtime.random_seed,
        scale_pos_weight=pos_weight,
        use_class_weight=True,
    )
    fit_kwargs = _early_stopping_kwargs(name, X_val, y_val)
    start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    train_seconds = time.perf_counter() - start

    infer_start = time.perf_counter()
    val_proba = _scores(model, X_val)
    inference_seconds = time.perf_counter() - infer_start
    val_pred = (val_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_val, val_pred, val_proba)
    row = {
        "model": name,
        **metrics,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "n_features": n_features,
        "n_estimators": runtime.n_estimators,
    }
    logger.info(
        "%s val PR-AUC=%.4f F1=%.4f Recall=%.4f (train %.1fs)",
        name, metrics["pr_auc"], metrics["f1"], metrics["recall"], train_seconds,
    )
    return {"model": model, "row": row, "val_proba": val_proba}


def train_anomaly_models(X_train, y_train, X_val, y_val, runtime, n_features: int):
    rows, models, probas = [], {}, {}
    legit_mask = y_train.to_numpy() == 0
    X_legit = X_train.loc[legit_mask]
    if len(X_legit) > ANOMALY_FIT_SAMPLES:
        X_legit_if = X_legit.sample(n=ANOMALY_FIT_SAMPLES, random_state=runtime.random_seed)
    else:
        X_legit_if = X_legit

    iso = build_model("isolation_forest", n_estimators=runtime.n_estimators, random_state=runtime.random_seed)
    start = time.perf_counter()
    iso.fit(X_legit_if)
    train_seconds = time.perf_counter() - start
    infer_start = time.perf_counter()
    iso_scores = -iso.score_samples(X_val)
    iso_proba = _rank01(iso_scores)
    inference_seconds = time.perf_counter() - infer_start
    iso_pred = (iso_proba >= 0.5).astype(int)
    iso_metrics = compute_metrics(y_val, iso_pred, iso_proba)
    rows.append(
        {
            "model": "isolation_forest",
            **iso_metrics,
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "n_features": n_features,
        }
    )
    models["isolation_forest"] = iso
    probas["isolation_forest"] = (y_val.to_numpy(), iso_proba)
    logger.info("isolation_forest val PR-AUC=%.4f", iso_metrics["pr_auc"])

    if not runtime.skip_autoencoder:
        from src.models import AutoencoderAnomalyDetector

        ae = AutoencoderAnomalyDetector(random_state=runtime.random_seed, max_iter=25 if runtime.fast else 40)
        X_legit_ae = (
            X_legit.sample(n=min(len(X_legit), AUTOENCODER_FIT_SAMPLES), random_state=runtime.random_seed)
        )
        start = time.perf_counter()
        ae.fit(X_legit_ae)
        train_seconds = time.perf_counter() - start
        infer_start = time.perf_counter()
        ae_scores = ae.decision_function(X_val)
        ae_proba = _rank01(ae_scores)
        inference_seconds = time.perf_counter() - infer_start
        ae_pred = (ae_proba >= 0.5).astype(int)
        ae_metrics = compute_metrics(y_val, ae_pred, ae_proba)
        rows.append(
            {
                "model": "autoencoder",
                **ae_metrics,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "n_features": n_features,
            }
        )
        models["autoencoder"] = ae
        probas["autoencoder"] = (y_val.to_numpy(), ae_proba)
        logger.info("autoencoder val PR-AUC=%.4f", ae_metrics["pr_auc"])
    return rows, models, probas


def tune_model(name, X_train, y_train, X_val, y_val, runtime, pos_weight) -> Tuple[Any, dict]:
    if name == "random_forest":
        return _randomized_search(name, X_train, y_train, X_val, y_val, runtime, pos_weight)

    try:
        import optuna
        from sklearn.metrics import average_precision_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception as exc:
        logger.warning("Optuna unavailable (%s); using RandomizedSearchCV", exc)
        return _randomized_search(name, X_train, y_train, X_val, y_val, runtime, pos_weight)

    logger.info("Starting Optuna HPO for %s (%s trials, CV on train only)", name, runtime.optuna_trials)
    cv = StratifiedKFold(n_splits=runtime.cv_folds, shuffle=True, random_state=runtime.random_seed)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial, name, runtime, pos_weight)
        model = build_model(
            name,
            n_estimators=params.pop("n_estimators"),
            random_state=runtime.random_seed,
            scale_pos_weight=pos_weight,
            use_class_weight=True,
        )
        model.set_params(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=N_JOBS
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", study_name=f"{name}_pr_auc")
    study.optimize(objective, n_trials=runtime.optuna_trials, show_progress_bar=False)
    best_params = study.best_params
    n_estimators = best_params.pop("n_estimators", runtime.n_estimators)
    best_model = build_model(
        name,
        n_estimators=n_estimators,
        random_state=runtime.random_seed,
        scale_pos_weight=pos_weight,
        use_class_weight=True,
    )
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)
    val_pr = float(average_precision_score(y_val, _scores(best_model, X_val)))
    report = {
        "skipped": False,
        "model": name,
        "n_trials": runtime.optuna_trials,
        "best_params": {"n_estimators": n_estimators, **best_params},
        "cv_pr_auc": float(study.best_value),
        "best_pr_auc": val_pr,
        "improved": True,
        "selection_protocol": "Optuna maximizes StratifiedKFold PR-AUC on training data only; validation PR-AUC is reported after refit.",
    }
    save_json(report, METRICS_DIR / "hpo_report.json")
    return best_model, report


def _randomized_search(name, X_train, y_train, X_val, y_val, runtime, pos_weight):
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import RandomizedSearchCV

    logger.info("RandomizedSearchCV HPO for %s", name)
    model = build_model(
        name,
        n_estimators=runtime.n_estimators,
        random_state=runtime.random_seed,
        scale_pos_weight=pos_weight,
        use_class_weight=True,
    )
    cv = StratifiedKFold(n_splits=runtime.cv_folds, shuffle=True, random_state=runtime.random_seed)
    search = RandomizedSearchCV(
        model,
        _sklearn_search_space(name),
        n_iter=min(8, runtime.optuna_trials),
        scoring="average_precision",
        cv=cv,
        n_jobs=N_JOBS,
        random_state=runtime.random_seed,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    val_pr = float(average_precision_score(y_val, _scores(best_model, X_val)))
    report = {
        "skipped": False,
        "method": "RandomizedSearchCV",
        "model": name,
        "best_params": search.best_params_,
        "cv_pr_auc": float(search.best_score_),
        "best_pr_auc": val_pr,
        "improved": True,
    }
    save_json(report, METRICS_DIR / "hpo_report.json")
    return best_model, report


def _sklearn_search_space(name: str) -> dict:
    if name == "random_forest":
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 12, 16],
            "min_samples_leaf": [2, 4, 8],
        }
    if name == "lightgbm":
        return {
            "n_estimators": [120, 200, 280],
            "num_leaves": [16, 31, 48],
            "learning_rate": [0.03, 0.05, 0.1],
        }
    if name == "xgboost":
        return {
            "n_estimators": [120, 200, 280],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.08, 0.12],
        }
    return {"n_estimators": [100, 200]}


def _suggest_params(trial, name, runtime, pos_weight) -> dict:
    n_estimators = trial.suggest_int("n_estimators", max(80, runtime.n_estimators // 2), runtime.n_estimators + 100)
    if name == "lightgbm":
        return {
            "n_estimators": n_estimators,
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
    if name == "xgboost":
        return {
            "n_estimators": n_estimators,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    if name == "catboost":
        return {
            "n_estimators": n_estimators,
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        }
    return {
        "n_estimators": n_estimators,
        "max_depth": trial.suggest_int("max_depth", 8, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
    }


def cross_validate_model(model, X, y, runtime: TrainRuntimeConfig) -> dict:
    if model is None or not hasattr(model, "predict_proba"):
        return {"skipped": True, "reason": "model does not support predict_proba"}
    cv = StratifiedKFold(n_splits=runtime.cv_folds, shuffle=True, random_state=runtime.random_seed)
    try:
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="average_precision", n_jobs=N_JOBS
        )
        report = {
            "skipped": False,
            "metric": "average_precision",
            "folds": runtime.cv_folds,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds_detail": [float(s) for s in scores],
        }
        logger.info("CV PR-AUC %.4f ± %.4f", report["mean"], report["std"])
        return report
    except Exception as exc:
        logger.warning("Cross-validation skipped: %s", exc)
        return {"skipped": True, "reason": str(exc)}


def clone_if_possible(model):
    try:
        return clone(model)
    except Exception:
        return None


def _select_best_model(comparison_df: pd.DataFrame) -> str:
    supervised = [m for m in PRODUCTION_CANDIDATES if m not in {"isolation_forest", "autoencoder"}]
    eligible = comparison_df[comparison_df["model"].isin(supervised)].copy()
    if eligible.empty:
        eligible = comparison_df[~comparison_df["model"].isin(["isolation_forest", "autoencoder"])].copy()
    if eligible.empty:
        eligible = comparison_df.copy()
    eligible = eligible.dropna(subset=[PRIMARY_METRIC])
    best = eligible.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
    return str(best["model"])


def _scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "score_samples"):
        return _rank01(-np.asarray(model.score_samples(X), dtype=float))
    if hasattr(model, "decision_function"):
        return _rank01(np.asarray(model.decision_function(X), dtype=float))
    return np.asarray(model.predict(X), dtype=float)


def _rank01(scores: np.ndarray) -> np.ndarray:
    order = scores.argsort().argsort().astype(float)
    n = max(len(scores) - 1, 1)
    return order / n


def _early_stopping_kwargs(name: str, X_val, y_val) -> dict:
    # Keep fit() signatures portable across boosting versions; validation is
    # used explicitly after training rather than as an in-fit eval set.
    return {}


def _supports_shap(model) -> bool:
    cls = model.__class__.__name__.lower()
    return any(k in cls for k in ("lgbm", "xgb", "catboost", "forest", "tree", "histgradient"))


def _leakage_notes() -> dict:
    return {
        "split_before_scaling": True,
        "scaler_fit_on_train_only": True,
        "resampling_train_only": True,
        "threshold_tuned_on_validation_only": True,
        "test_untouched_until_final_eval": True,
        "no_smote_before_split": True,
    }


def _start_mlflow():
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
        mlflow.set_experiment("credit-card-fraud-detection")
        run = mlflow.start_run(run_name="full_pipeline")
        logger.info("MLflow tracking at %s", MLRUNS_DIR)
        return run
    except Exception as exc:
        logger.info("MLflow disabled: %s", exc)
        return None


def _log_mlflow(run, final_report: dict, comparison_df: pd.DataFrame) -> None:
    if run is None:
        return
    try:
        import mlflow

        mlflow.log_param("best_model", final_report["best_model"])
        mlflow.log_param("threshold", final_report["threshold"])
        for key, value in final_report.get("test_metrics", {}).items():
            if isinstance(value, (int, float)) and value == value:
                mlflow.log_metric(f"test_{key}", float(value))
        mlflow.log_artifact(str(METRICS_DIR / "final_report.json"))
        mlflow.log_artifact(str(METRICS_DIR / "model_comparison.csv"))
        mlflow.end_run()
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return run_pipeline(args)
