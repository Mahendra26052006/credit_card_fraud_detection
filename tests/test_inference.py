import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.inference import predict_dataframe, predict_transaction
from src.preprocessing import build_preprocessor, fit_preprocessor, transform_features


def _tiny_artifacts(synthetic_df):
    X_raw = synthetic_df.drop(columns=["Class"])
    y = synthetic_df["Class"]
    preprocessor = fit_preprocessor(build_preprocessor(), X_raw)
    X = transform_features(preprocessor, X_raw)
    model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=0)
    model.fit(X, y)
    metadata = {
        "model_name": "logistic_regression",
        "model_type": "supervised",
        "threshold": 0.5,
        "feature_names": list(X.columns),
        "risk_low_fraction": 0.5,
        "risk_high_fraction": 1.0,
        "version": "test",
    }
    return {"model": model, "preprocessor": preprocessor, "metadata": metadata}


def test_predict_transaction_schema(synthetic_df, raw_transaction):
    artifacts = _tiny_artifacts(synthetic_df)
    result = predict_transaction(raw_transaction, artifacts=artifacts, return_shap=False)
    assert set(result) >= {
        "fraud_probability",
        "prediction",
        "risk_level",
        "threshold",
        "top_features",
        "model_confidence",
    }
    assert result["prediction"] in {"FRAUD", "LEGITIMATE"}
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_dataframe_batch(synthetic_df):
    artifacts = _tiny_artifacts(synthetic_df)
    out = predict_dataframe(
        synthetic_df.drop(columns=["Class"]),
        artifacts=artifacts,
        return_shap=False,
    )
    assert len(out) == len(synthetic_df)
    assert {"fraud_probability", "prediction", "risk_level"}.issubset(out.columns)


def test_model_roundtrip(tmp_path, synthetic_df, raw_transaction):
    artifacts = _tiny_artifacts(synthetic_df)
    path = tmp_path / "best_model.pkl"
    joblib.dump(artifacts["model"], path)
    loaded = joblib.load(path)
    artifacts["model"] = loaded
    result = predict_transaction(raw_transaction, artifacts=artifacts, return_shap=False)
    assert result["prediction"] in {"FRAUD", "LEGITIMATE"}
