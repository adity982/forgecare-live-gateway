import time
import base64
from io import BytesIO
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from PIL import Image

from forgecare.ops_api import app
from forgecare.federation import VISUAL_FEATURE_COUNT
from forgecare.config import SERVICE_KEY
from forgecare.db import connect, row


def test_background_experiments_and_federation():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "job-site", "site_name": "Jobs", "username": "jobadmin", "password": "super-secure-password"}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.post("/experiments/run", headers=headers, json={"site_id": "job-site"}).status_code == 200
        assert client.post("/federation/run", headers=headers, json={"site_id": "job-site"}).status_code == 200
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            experiments = client.get("/experiments?site_id=job-site", headers=headers).json()
            rounds = client.get("/federation?site_id=job-site", headers=headers).json()
            if experiments and all(x["status"] != "running" for x in experiments) and rounds[0]["status"] != "running":
                break
            time.sleep(.2)
        assert len(experiments) == 7
        assert all(x["status"] in ("complete", "insufficient_data") for x in experiments), experiments
        assert next(x for x in experiments if x["task"] == "rag_retrieval")["status"] == "insufficient_data"
        assert rounds[0]["status"] == "complete", rounds


def test_real_round_protocol_rejects_bad_update_and_requires_validation():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "pilot", "site_name": "Pilot", "username": "reviewadmin", "password": "super-secure-password"}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=headers, json={"id": "iphone-14-plus", "site_id": "pilot", "name": "iPhone 14 Plus", "family": "iphone-exterior", "model": "iPhone 14 Plus", "criticality": 3})
        created = client.post("/federation/rounds", headers=headers, json={"site_id": "pilot", "family": "iphone-exterior", "model": "iPhone 14 Plus"})
        assert created.status_code == 200, created.text
        round_id = created.json()["id"]
        schema = created.json()["schema_hash"]
        update = {"site_id": "pilot", "client_id": "client-a", "schema_hash": schema,
                  "sample_count": 30, "weights": [0.1] * (VISUAL_FEATURE_COUNT + 1)}
        assert client.post(f"/federation/rounds/{round_id}/updates", headers=headers, json={**update, "schema_hash": "wrong"}).status_code == 422
        assert client.post(f"/federation/rounds/{round_id}/updates", headers=headers, json=update).status_code == 200
        assert client.post(f"/federation/rounds/{round_id}/updates", headers=headers, json=update).status_code == 409
        assert client.post(f"/federation/rounds/{round_id}/aggregate", headers=headers).status_code == 409
        assert client.post(f"/federation/rounds/{round_id}/updates", headers=headers, json={**update, "client_id": "client-b", "sample_count": 10, "weights": [0.0] * (VISUAL_FEATURE_COUNT + 1)}).status_code == 200
        combined = client.post(f"/federation/rounds/{round_id}/aggregate", headers=headers)
        assert combined.status_code == 200, combined.text
        assert abs(combined.json()["weights"][0] - .075) < 1e-9
        assert client.get("/models?site_id=pilot", headers=headers).json() == []
        validation = {"site_id": "pilot", "client_id": "client-a", "sample_count": 10, "positive_count": 3,
                      "recall": .8, "false_alert_rate": .1, "latency_ms": 2, "local_recall": .7, "local_false_alert_rate": .15}
        assert client.post(f"/federation/rounds/{round_id}/validations", headers=headers, json=validation).json()["status"] == "awaiting_validation"
        final = client.post(f"/federation/rounds/{round_id}/validations", headers=headers, json={**validation, "client_id": "client-b"})
        assert final.status_code == 200, final.text
        assert final.json()["status"] == "candidate_review"
        candidate_id = final.json()["candidate_model_id"]
        assert client.post(f"/models/{candidate_id}/approve", headers=headers).status_code == 200
        client.post("/assets", headers=headers, json={"id": "realme-60", "site_id": "pilot", "name": "realme narzo 60", "family": "realme-exterior", "model": "realme narzo 60 5G", "criticality": 3})
        client.post("/cameras", headers=headers, json={"id": "realme-camera", "site_id": "pilot", "asset_id": "realme-60", "name": "Second camera", "source": "1", "sample_id": "realme-physical-1"})
        frame = BytesIO()
        Image.new("RGB", (128, 128), "white").save(frame, format="JPEG")
        event = {"camera_id": "realme-camera", "asset_id": "realme-60", "site_id": "pilot", "session_id": "test-session", "sample_id": "realme-physical-1",
                 "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok", "jpeg_b64": base64.b64encode(frame.getvalue()).decode()}
        capture = client.post("/internal/captures", headers={"X-Service-Key": SERVICE_KEY}, json=event)
        assert capture.status_code == 200, capture.text
        with connect() as db:
            observation = row(db, "SELECT * FROM observations WHERE capture_id=?", (capture.json()["id"],))
        assert observation["model_id"] is None
        assert observation["predicted_class"] is None
