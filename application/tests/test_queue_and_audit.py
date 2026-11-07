"""Regression checks for camera status, independent reviewed work, and readable audit."""
import base64
from datetime import datetime, timedelta, timezone
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import requests
from streamlit.testing.v1 import AppTest
from pathlib import Path

from forgecare.audit_view import business_events
from forgecare.config import SERVICE_KEY
from forgecare.db import connect
from forgecare.ops_api import app


def _jpeg() -> str:
    output = BytesIO()
    Image.new("RGB", (128, 128), "#92999a").save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode()


def test_each_reviewed_capture_gets_its_own_work_order_even_with_approved_asset_case():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "demo", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        internal = {"X-Service-Key": SERVICE_KEY}
        client.post("/assets", headers=auth, json={"id": "iphone", "site_id": "demo", "name": "Phone",
            "family": "phone", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "samsung", "site_id": "demo", "asset_id": "iphone",
            "name": "Samsung camera", "source": "0", "sample_id": "iphone-physical-01"})
        captures = []
        for index in range(2):
            capture = client.post("/internal/captures", headers=internal, json={
                "camera_id": "samsung", "asset_id": "iphone", "site_id": "demo",
                "session_id": "session-one", "sample_id": "iphone-physical-01",
                "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok", "jpeg_b64": _jpeg()})
            assert capture.status_code == 200, capture.text
            capture_id = capture.json()["id"]
            captures.append(capture_id)
            assert client.post(f"/captures/{capture_id}/label", headers=auth, json={
                "grade": "minor", "note": f"Reviewed angle {index}", "scratch_count": 1}).status_code == 200
        ready = client.get("/cases/review-ready?site_id=demo", headers=auth).json()
        assert {x["capture_id"] for x in ready} == set(captures)
        first = client.post("/cases", headers=auth, json={"capture_id": captures[0]}).json()
        assert client.post(f"/cases/{first['id']}/approve", headers=auth,
            json={"decision_note": "Inspect the phone exterior"}).status_code == 200
        second = client.post("/cases", headers=auth, json={"capture_id": captures[1]}).json()
        assert second["id"] != first["id"] and not second.get("duplicate_linked")
        assert client.post("/cases", headers=auth, json={"capture_id": captures[1]}).json()["duplicate_linked"]
        assert client.get("/cases/review-ready?site_id=demo", headers=auth).json() == []
        assert len(client.get("/cases?site_id=demo", headers=auth).json()) == 2


def test_camera_with_old_frame_and_disconnect_error_is_offline_not_live_or_stale():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "demo", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone", "site_id": "demo", "name": "Phone",
            "family": "phone", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "samsung", "site_id": "demo", "asset_id": "iphone",
            "name": "Samsung camera", "source": "0", "sample_id": "phone-1"})
        with connect() as db:
            db.execute("UPDATE cameras SET last_frame_at=?,last_error=? WHERE id='samsung'",
                ((datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(), "Camera disconnected"))
        camera = client.get("/cameras?site_id=demo", headers=auth).json()[0]
        assert camera["status"] == "offline"
        with connect() as db:
            db.execute("UPDATE cameras SET last_error=NULL WHERE id='samsung'")
        assert client.get("/cameras?site_id=demo", headers=auth).json()[0]["status"] == "no_signal"
        assert client.post("/cameras/samsung/stop", headers=auth).status_code == 200
        assert client.get("/cameras?site_id=demo", headers=auth).json()[0]["status"] == "stopped"


def test_audit_projection_keeps_business_details_without_exposing_json():
    events = [{"at": "2026-11-06T00:33:59+00:00", "actor_id": 1, "action": "label",
        "object_type": "capture", "object_id": "photo-123", "details_json":
        '{"grade":"significant","scratch_count":2,"dent_count":null}'},
        {"at": "2026-11-06T00:34:09+00:00", "actor_id": None, "action": "capture_rejected",
         "object_type": "camera_event", "object_id": "unaccepted", "details_json": '{"status_code":409}'}]
    result = business_events(events, {1: "technician"})
    assert result[0]["Activity"] == "Technician review saved"
    assert result[0]["Who"] == "technician"
    assert "Scratches: 2" in result[0]["Business detail"]
    assert "dents: not assessed" in result[0]["Business detail"]
    assert "details_json" not in result[0]
    assert result[1]["Activity"] == "Camera submission rejected"
    assert "409" in result[1]["Business detail"]


def test_saving_clear_review_in_dashboard_opens_queue_case(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "demo", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone", "site_id": "demo", "name": "Phone",
            "family": "phone", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "samsung", "site_id": "demo", "asset_id": "iphone",
            "name": "Samsung camera", "source": "0", "sample_id": "phone-1"})
        captured = client.post("/internal/captures", headers={"X-Service-Key": SERVICE_KEY}, json={
            "camera_id": "samsung", "asset_id": "iphone", "site_id": "demo", "session_id": "session-1",
            "sample_id": "phone-1", "captured_at": datetime.now(timezone.utc).isoformat(),
            "quality": "ok", "jpeg_b64": _jpeg()})
        assert captured.status_code == 200

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Capture review").run()
        assert not ui.exception, [item.message for item in ui.exception]
        [item for item in ui.selectbox if item.label == "Visible condition"][0].set_value("minor")
        [item for item in ui.text_area if item.label == "Technician's corrected review"][0].set_value("One visible edge mark")
        [item for item in ui.button if item.label == "Save reviewed label"][0].click().run()
        assert not ui.exception, [item.message for item in ui.exception]
        cases = client.get("/cases?site_id=demo", headers=auth).json()
        assert len(cases) == 1
        ui.radio[0].set_value("Inspection queue").run()
        assert not ui.exception, [item.message for item in ui.exception]
        assert any("One visible edge mark" in str(item.value) for item in ui.markdown)


def test_federation_readiness_explains_conflicting_views_of_one_phone():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "demo", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone", "site_id": "demo", "name": "Phone",
            "family": "phone", "model": "iPhone 14 Plus", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "samsung", "site_id": "demo", "asset_id": "iphone",
            "name": "Samsung camera", "source": "0", "sample_id": "phone-1"})
        for grade in ("none", "significant"):
            capture = client.post("/internal/captures", headers={"X-Service-Key": SERVICE_KEY}, json={
                "camera_id": "samsung", "asset_id": "iphone", "site_id": "demo", "session_id": "session-1",
                "sample_id": "phone-1", "captured_at": datetime.now(timezone.utc).isoformat(),
                "quality": "ok", "jpeg_b64": _jpeg()})
            assert capture.status_code == 200
            assert client.post(f"/captures/{capture.json()['id']}/label", headers=auth,
                json={"grade": grade, "note": "Different visible area"}).status_code == 200
        state = client.get("/federation/readiness?site_id=demo&asset_id=iphone", headers=auth).json()
        assert state["captures"] == 2 and state["reviewed_usable_photos"] == 2
        assert state["eligible_training_photos"] == 0 and state["conflicting_samples"] == 1
        assert not state["ready_for_scheduled_training"]


def test_distinct_physical_phones_can_be_registered_and_mapped_to_their_model():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "demo", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        for asset_id in ("iphone", "realme"):
            assert client.post("/assets", headers=auth, json={"id": asset_id, "site_id": "demo",
                "name": asset_id, "family": "phone", "model": asset_id, "criticality": 3}).status_code == 200
        phone = client.post("/samples", headers=auth, json={"id": "iphone-unit-02", "site_id": "demo",
            "asset_id": "iphone", "display_name": "Second actual iPhone"})
        assert phone.status_code == 200, phone.text
        assert client.post("/samples", headers=auth, json={"id": "iphone-unit-02", "site_id": "demo",
            "asset_id": "iphone", "display_name": "Duplicate"}).status_code == 409
        assert client.post("/cameras", headers=auth, json={"id": "camera", "site_id": "demo",
            "asset_id": "iphone", "name": "Camera", "source": "0", "sample_id": "iphone-unit-02"}).status_code == 200
        assert client.put("/cameras/camera", headers=auth, json={"asset_id": "realme", "source": "0",
            "sample_id": "iphone-unit-02", "roi": [0, 0, 1, 1]}).status_code == 409
        registered = client.get("/samples?site_id=demo&asset_id=iphone", headers=auth).json()
        assert len(registered) == 1 and registered[0]["display_name"] == "Second actual iPhone"
