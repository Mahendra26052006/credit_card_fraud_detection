"""Insert real metrics from outputs/metrics/final_report.json into README.md."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

README = ROOT / "README.md"
REPORT = ROOT / "outputs" / "metrics" / "final_report.json"
COMPARISON = ROOT / "outputs" / "metrics" / "model_comparison.csv"
IMBALANCE = ROOT / "outputs" / "metrics" / "imbalance_comparison.csv"


def _md_table(path: Path, keep: List[str]) -> str:
    import pandas as pd

    df = pd.read_csv(path)
    cols = [c for c in keep if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def render() -> str:
    report = json.loads(REPORT.read_text())
    test = report.get("test_metrics", {})
    val = report.get("validation_metrics", {})
    cv = report.get("cross_validation", {})
    split = report.get("split_sizes", {})
    hpo = report.get("hpo", {})
    calib = report.get("calibration", {})
    cost = report.get("cost_framework", {})
    err = report.get("error_analysis", {})

    table_md = (
        _md_table(
            COMPARISON,
            ["model", "precision", "recall", "f1", "roc_auc", "pr_auc", "inference_seconds"],
        )
        if COMPARISON.exists()
        else ""
    )
    imb_md = (
        _md_table(IMBALANCE, ["strategy", "pr_auc", "recall", "precision", "f1"])
        if IMBALANCE.exists()
        else ""
    )

    cv_line = "n/a"
    if not cv.get("skipped"):
        cv_line = (
            f"{cv.get('mean', float('nan')):.4f} ± {cv.get('std', float('nan')):.4f} "
            f"(PR-AUC, {cv.get('folds')} folds)"
        )
    hpo_pr = hpo.get("best_pr_auc")
    hpo_pr_s = f"{hpo_pr:.4f}" if isinstance(hpo_pr, float) else str(hpo_pr)

    return f"""**Production model:** `{report.get('best_model')}`  
**Operating threshold:** `{report.get('threshold'):.3f}`  
**Why this threshold:** {report.get('threshold_reason')}

These numbers come from `outputs/metrics/final_report.json` after `python train.py`. They are not hardcoded.

### Held-out test set (untouched until final evaluation)

| Metric | Test |
| --- | ---: |
| PR-AUC | {test.get('pr_auc', float('nan')):.4f} |
| Recall | {test.get('recall', float('nan')):.4f} |
| Precision | {test.get('precision', float('nan')):.4f} |
| F1 | {test.get('f1', float('nan')):.4f} |
| ROC-AUC | {test.get('roc_auc', float('nan')):.4f} |
| Accuracy | {test.get('accuracy', float('nan')):.4f} |
| Brier | {test.get('brier', float('nan')):.4f} |
| Specificity | {test.get('specificity', float('nan')):.4f} |

Confusion matrix on test: TP={int(test.get('tp', 0))}, FP={int(test.get('fp', 0))}, TN={int(test.get('tn', 0))}, FN={int(test.get('fn', 0))}.

Validation (threshold / model choice only): PR-AUC **{val.get('pr_auc', float('nan')):.4f}**, F1 **{val.get('f1', float('nan')):.4f}**, recall **{val.get('recall', float('nan')):.4f}**, precision **{val.get('precision', float('nan')):.4f}**.

Cross-validation on train: **{cv_line}**

Split sizes: train {split.get('train')} (fraud {split.get('train_fraud')}), val {split.get('validation')} (fraud {split.get('val_fraud')}), test {split.get('test')} (fraud {split.get('test_fraud')}).

HPO: Optuna / RandomizedSearchCV on **training folds only**; post-HPO validation PR-AUC **{hpo_pr_s}**. Isotonic calibration was evaluated and **not** used (`used_calibration={calib.get('used_calibration')}`): raw Brier {calib.get('raw_brier')}, calibrated Brier {calib.get('calibrated_brier')}.

Cost framework: FN={cost.get('fraud_miss_cost')}, FP={cost.get('false_alert_cost')}, validation expected cost **{cost.get('validation_expected_cost')}**.

Test error analysis: FP={err.get('n_false_positives')}, FN={err.get('n_false_negatives')}. {err.get('interpretation', '')}

The comparison table below is **validation performance at probability 0.5**, used only to rank models. Production uses the cost-sensitive threshold above; the test table is the honest final score.

### Validation comparison (threshold = 0.5)

{table_md or '_model_comparison.csv not found_'}

### Imbalance strategies (LightGBM probe, training-only resampling)

{imb_md or '_imbalance_comparison.csv not found_'}
"""


def main() -> None:
    if not REPORT.exists():
        raise SystemExit("Run python train.py first")
    text = README.read_text()
    start = "<!-- METRICS:START -->"
    end = "<!-- METRICS:END -->"
    if start not in text or end not in text:
        raise SystemExit("README metrics markers missing")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    README.write_text(before + start + "\n" + render() + end + after)
    print("Updated README.md with metrics from", REPORT)


if __name__ == "__main__":
    main()
