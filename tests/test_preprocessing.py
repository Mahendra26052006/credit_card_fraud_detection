import pandas as pd

from src.feature_engineering import TransactionFeatureEngineer
from src.preprocessing import build_preprocessor, fit_preprocessor, transform_features, validate_transaction_frame


def test_feature_engineer_adds_justified_columns(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    engineer = TransactionFeatureEngineer().fit(X)
    out = engineer.transform(X)
    for col in ["Amount_log", "Hour", "Hour_sin", "Hour_cos"]:
        assert col in out.columns
    assert (out["Hour"] >= 0).all() and (out["Hour"] < 24).all()
    assert pd.Series(out["Amount_log"]).notna().all()


def test_preprocessor_fit_on_train_only_and_transform_val(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    train = X.iloc[:180]
    val = X.iloc[180:]
    pipe = fit_preprocessor(build_preprocessor(), train)
    Xt = transform_features(pipe, val)
    assert len(Xt) == len(val)
    assert Xt.isna().sum().sum() == 0
    assert Xt.shape[1] >= X.shape[1]


def test_validate_transaction_frame_flags_negatives(raw_transaction):
    df = pd.DataFrame([raw_transaction])
    df.loc[0, "Amount"] = -1
    _, issues = validate_transaction_frame(df)
    assert any("Amount" in issue for issue in issues)


def test_validate_transaction_frame_missing_column(raw_transaction):
    df = pd.DataFrame([raw_transaction]).drop(columns=["V28"])
    try:
        validate_transaction_frame(df)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "V28" in str(exc)
