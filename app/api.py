"""FastAPI inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METADATA_PATH, PCA_FEATURES
from src.inference import ArtifactError, clear_artifact_cache, load_artifacts, predict_dataframe, predict_transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
    except ArtifactError:
        pass
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Score card transactions for fraud probability, risk, and explanations.",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    Time: float = Field(..., ge=0)
    Amount: float = Field(..., ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    model_config = {"extra": "forbid"}


class PredictResponse(BaseModel):
    fraud_probability: float
    prediction: str
    risk_level: str
    threshold: float
    top_features: Optional[List[Dict[str, Any]]] = None
    model_confidence: float


@app.get("/health")
def health() -> Dict[str, str]:
    try:
        load_artifacts()
        return {"status": "ok"}
    except ArtifactError as exc:
        return {"status": "model_not_loaded", "detail": str(exc)}


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    try:
        artifacts = load_artifacts()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    meta = artifacts["metadata"]
    return {
        "model_name": meta.get("model_name"),
        "version": meta.get("version"),
        "threshold": meta.get("threshold"),
        "threshold_reason": meta.get("threshold_reason"),
        "feature_count": len(meta.get("feature_names", [])),
        "pca_features": list(PCA_FEATURES),
        "metadata_path": str(METADATA_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: Transaction) -> PredictResponse:
    try:
        artifacts = load_artifacts()
        result = predict_transaction(transaction.model_dump(), artifacts=artifacts, return_shap=True)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result)


@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]) -> Dict[str, Any]:
    try:
        artifacts = load_artifacts()
        frame = predict_dataframe(
            pd_frame(transactions), artifacts=artifacts, return_shap=False
        )
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    records = frame[["fraud_probability", "prediction", "risk_level"]].to_dict(orient="records")
    n_fraud = int((frame["prediction"] == "FRAUD").sum())
    return {
        "n_transactions": int(len(frame)),
        "n_fraud": n_fraud,
        "fraud_rate": float(n_fraud / max(len(frame), 1)),
        "predictions": records,
    }


def pd_frame(transactions: List[Transaction]):
    import pandas as pd

    return pd.DataFrame([t.model_dump() for t in transactions])


@app.post("/reload")
def reload_model() -> Dict[str, str]:
    clear_artifact_cache()
    load_artifacts()
    return {"status": "reloaded"}
