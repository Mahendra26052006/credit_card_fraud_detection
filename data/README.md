"""
# Credit Card Fraud Detection — Data

This folder holds the **ULB / Worldline Credit Card Fraud Detection** dataset
(Kaggle: `mlg-ulb/creditcardfraud`).

## Why the CSV is not in Git

The dataset license and Kaggle terms do **not** permit redistributing the raw
file on GitHub. Download it locally instead.

## How to obtain the data

From the project root (with the virtualenv activated):

```bash
python scripts/download_data.py
```

This will try, in order:

1. A `--source` path you pass (any local `creditcard.csv`)
2. KaggleHub / Kaggle API if you have credentials
3. The TensorFlow public mirror of the same file:
   `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

The file is written to:

```text
data/raw/creditcard.csv
```

You can also copy a CSV you already downloaded from Kaggle into `data/raw/`.

## Schema

| Column   | Description                                      |
|----------|--------------------------------------------------|
| Time     | Seconds elapsed between this tx and the first tx |
| V1–V28   | PCA-anonymized features                          |
| Amount   | Transaction amount                               |
| Class    | 0 = legitimate, 1 = fraud                        |

Approximate size: **284,807** rows, **492** frauds (~0.172%).
