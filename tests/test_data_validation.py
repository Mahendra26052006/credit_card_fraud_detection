import pandas as pd

from src.data_loader import DatasetError, clean_dataframe, stratified_split, validate_dataset


def test_validate_dataset_accepts_schema(synthetic_df):
    report = validate_dataset(synthetic_df)
    assert report["n_rows"] == len(synthetic_df)
    assert report["n_fraud"] == 24
    assert isinstance(report["issues"], list)


def test_validate_dataset_rejects_missing_columns(synthetic_df):
    bad = synthetic_df.drop(columns=["V1"])
    try:
        validate_dataset(bad)
        assert False, "expected DatasetError"
    except DatasetError as exc:
        assert "V1" in str(exc)


def test_validate_dataset_rejects_non_binary_target(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "Class"] = 2
    try:
        validate_dataset(bad)
        assert False, "expected DatasetError"
    except DatasetError:
        pass


def test_clean_dataframe_drops_duplicates(synthetic_df):
    doubled = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    cleaned, decisions = clean_dataframe(doubled)
    assert decisions["n_duplicates"] >= 1
    assert len(cleaned) == len(synthetic_df)


def test_stratified_split_preserves_both_classes(synthetic_df):
    train, val, test = stratified_split(synthetic_df, random_state=0)
    assert train["Class"].nunique() == 2
    assert val["Class"].nunique() == 2
    assert test["Class"].nunique() == 2
    assert abs(len(train) / len(synthetic_df) - 0.70) < 0.08
