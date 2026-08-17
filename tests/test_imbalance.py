from src.imbalance import resample_training_data, scale_pos_weight


def test_scale_pos_weight(synthetic_df):
    y = synthetic_df["Class"]
    weight = scale_pos_weight(y)
    n_legit = int((y == 0).sum())
    n_fraud = int((y == 1).sum())
    assert abs(weight - n_legit / n_fraud) < 1e-6


def test_smote_only_changes_training_length(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    y = synthetic_df["Class"]
    X_res, y_res, report = resample_training_data(X, y, "smote", sampling_strategy=0.5)
    assert report["applied"] is True
    assert int(y_res.sum()) >= int(y.sum())
    assert len(X_res) >= len(X)
