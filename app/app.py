"""Streamlit fraud-detection dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BASE_FEATURES, FIGURES_DIR, METRICS_DIR, PCA_FEATURES
from src.inference import ArtifactError, load_artifacts, predict_dataframe, predict_transaction

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def _init_history():
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame()


def sidebar_status(artifacts) -> None:
    meta = artifacts["metadata"]
    st.sidebar.title("Fraud Sentinel")
    st.sidebar.markdown("Production inference for European card transactions (PCA-anonymized).")
    st.sidebar.metric("Model", meta.get("model_name", "unknown"))
    st.sidebar.metric("Threshold", f"{meta.get('threshold', 0):.3f}")
    st.sidebar.caption(meta.get("threshold_reason", ""))
    report_path = METRICS_DIR / "final_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        test = report.get("test_metrics", {})
        st.sidebar.markdown("### Held-out test metrics")
        cols = st.sidebar.columns(2)
        cols[0].metric("PR-AUC", f"{test.get('pr_auc', float('nan')):.3f}")
        cols[1].metric("Recall", f"{test.get('recall', float('nan')):.3f}")
        cols = st.sidebar.columns(2)
        cols[0].metric("Precision", f"{test.get('precision', float('nan')):.3f}")
        cols[1].metric("F1", f"{test.get('f1', float('nan')):.3f}")


def dashboard_from_frame(df: pd.DataFrame) -> None:
    n = len(df)
    n_fraud = int((df["prediction"] == "FRAUD").sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total transactions", f"{n:,}")
    c2.metric("Fraud detected", f"{n_fraud:,}")
    c3.metric("Fraud percentage", f"{(100 * n_fraud / max(n, 1)):.2f}%")
    c4.metric("Avg fraud probability", f"{df['fraud_probability'].mean():.3f}")
    c5.metric("High-risk", int((df["risk_level"] == "HIGH").sum()))

    left, right = st.columns(2)
    with left:
        fig = px.pie(df, names="prediction", title="Fraud vs legitimate")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(df, x="risk_level", color="risk_level", title="Risk distribution")
        st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(df, x="fraud_probability", nbins=30, title="Fraud probability distribution")
    st.plotly_chart(fig, use_container_width=True)


def single_transaction_form(artifacts) -> None:
    st.subheader("Score a single transaction")
    st.caption("Enter PCA features V1–V28 plus Time (seconds) and Amount.")
    cols = st.columns(4)
    values = {}
    values["Time"] = cols[0].number_input("Time", min_value=0.0, value=0.0, step=1.0)
    values["Amount"] = cols[1].number_input("Amount", min_value=0.0, value=88.0, step=0.01)
    for i, feat in enumerate(PCA_FEATURES):
        col = cols[i % 4]
        values[feat] = col.number_input(feat, value=0.0, format="%.6f")

    if st.button("Predict transaction", type="primary"):
        result = predict_transaction(values, artifacts=artifacts, return_shap=True)
        _render_prediction(result)
        hist_row = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **result,
                    "top_features": json.dumps(result.get("top_features", [])),
                }
            ]
        )
        st.session_state.history = pd.concat([st.session_state.history, hist_row], ignore_index=True)


def _render_prediction(result: dict) -> None:
    pred = result["prediction"]
    color = "#E45756" if pred == "FRAUD" else "#54A24B"
    st.markdown(
        f"<div style='padding:1rem;border-radius:12px;background:{color}22;border:1px solid {color};'>"
        f"<h3 style='margin:0;color:{color}'>{pred} · {result['risk_level']} risk</h3>"
        f"<p style='margin:0.4rem 0 0'>Fraud probability: <b>{result['fraud_probability']:.4f}</b> "
        f"(threshold {result['threshold']:.3f}) · confidence {result['model_confidence']:.3f}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(result["fraud_probability"], 0.0), 1.0))
    top = result.get("top_features") or []
    if top:
        st.markdown("#### Top contributing features (SHAP)")
        st.dataframe(pd.DataFrame(top), use_container_width=True)


def batch_tab(artifacts) -> None:
    st.subheader("Batch scoring")
    st.caption("Upload a CSV with columns Time, Amount, V1–V28. Optional Class column is ignored.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return
    df = pd.read_csv(uploaded)
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    with st.spinner("Scoring transactions..."):
        scored = predict_dataframe(df, artifacts=artifacts, return_shap=True, shap_limit=25)
    dashboard_from_frame(scored)
    st.dataframe(scored.head(100), use_container_width=True)
    csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv, file_name="fraud_predictions.csv", mime="text/csv")


def history_tab() -> None:
    st.subheader("Session prediction history")
    hist = st.session_state.history
    if hist.empty:
        st.info("No scored transactions in this session yet.")
        return
    dashboard_from_frame(hist)
    st.dataframe(hist, use_container_width=True)


def explainability_tab() -> None:
    st.subheader("Global SHAP explainability")
    bar = FIGURES_DIR / "shap_bar.png"
    bees = FIGURES_DIR / "shap_beeswarm.png"
    if bar.exists():
        st.image(str(bar), caption="Global SHAP bar plot")
    if bees.exists():
        st.image(str(bees), caption="SHAP beeswarm")
    if not bar.exists() and not bees.exists():
        st.info("Train the pipeline (`python train.py`) to generate SHAP plots.")
    cm = FIGURES_DIR / "test_confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), caption="Held-out test confusion matrix")


def main() -> None:
    _init_history()
    st.title("Credit Card Fraud Detection")
    st.markdown(
        "Leakage-safe ML pipeline with class-imbalance handling, threshold optimization, "
        "and SHAP explanations. Test metrics below come from `outputs/metrics/final_report.json` "
        "after training — they are never hardcoded."
    )
    try:
        artifacts = get_artifacts()
    except ArtifactError as exc:
        st.error(str(exc))
        st.stop()

    sidebar_status(artifacts)
    tab1, tab2, tab3, tab4 = st.tabs(["Single transaction", "Batch CSV", "History", "Explainability"])
    with tab1:
        single_transaction_form(artifacts)
    with tab2:
        batch_tab(artifacts)
    with tab3:
        history_tab()
    with tab4:
        explainability_tab()


if __name__ == "__main__":
    main()
