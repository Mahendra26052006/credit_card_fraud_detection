from fastapi.testclient import TestClient

from app.api import app
from src.inference import clear_artifact_cache
from tests.test_inference import _tiny_artifacts


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint_with_injected_artifacts(synthetic_df, raw_transaction, monkeypatch):
    artifacts = _tiny_artifacts(synthetic_df)

    def _fake_load():
        return artifacts

    monkeypatch.setattr("app.api.load_artifacts", _fake_load)
    monkeypatch.setattr("src.inference.load_artifacts", _fake_load)
    clear_artifact_cache()

    client = TestClient(app)
    response = client.post("/predict", json=raw_transaction)
    assert response.status_code == 200
    payload = response.json()
    assert "fraud_probability" in payload
    assert payload["prediction"] in {"FRAUD", "LEGITIMATE"}
    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
