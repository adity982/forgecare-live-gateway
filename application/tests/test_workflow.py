import base64
import json
from datetime import datetime, timezone
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from forgecare.config import SERVICE_KEY
from forgecare.edge import inspect_frame, run_camera
from forgecare.ops_api import app
from forgecare.ops_api import CAMERA_PROCESSES
from forgecare.federation import aggregate
from forgecare.priority import fit_capacity
from forgecare.vision import grouped_splits

import numpy as np
import pytest


def jpeg() -> str:
    image = Image.new("RGB", (128, 128), "#a1a6a4")
    output = BytesIO()
    image.save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode()


def test_black_frame_is_dark_and_usable_capture_filter_survives_later_bad_frames():
    _, quality, _, brightness = inspect_frame(np.zeros((64, 64, 3), dtype=np.uint8), [0, 0, 1, 1])
    assert quality == "dark" and brightness == 0
    with TestClient(app) as client:
        setup = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo", "username": "admin", "password": "super-secure-password"})
        auth = {"Authorization": f"Bearer {setup.json()['token']}"}
        internal = {"X-Service-Key": SERVICE_KEY}
        assert client.post("/assets", headers=auth, json={"id": "iphone-1", "site_id": "site-a", "name": "Exterior", "family": "iphone-exterior", "criticality": 3}).status_code == 200
        assert client.post("/cameras", headers=auth, json={"id": "camera-1", "site_id": "site-a", "asset_id": "iphone-1", "name": "Camera", "source": "0", "sample_id": "phone-1"}).status_code == 200
        event = {"camera_id": "camera-1", "asset_id": "iphone-1", "site_id": "site-a", "session_id": "session-1", "sample_id": "phone-1",
                 "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok", "jpeg_b64": jpeg()}
        good = client.post("/internal/captures", headers=internal, json=event).json()["id"]
        bad = client.post("/internal/captures", headers=internal, json={**event, "quality": "dark", "captured_at": datetime.now(timezone.utc).isoformat()}).json()["id"]
        captures = client.get("/captures?site_id=site-a&quality=ok", headers=auth).json()
        assert [capture["id"] for capture in captures] == [good]
        assert client.post(f"/captures/{bad}/label", headers=auth, json={"grade": "minor", "note": "Not visible"}).status_code == 422
        assert client.post(f"/captures/{bad}/label", headers=auth, json={"grade": "unclear", "note": "Recapture required"}).status_code == 200


def test_near_black_stream_exits_without_saving_frames(monkeypatch):
    class BlackCamera:
        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((64, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    from forgecare import edge
    calls = []
    monkeypatch.setattr(edge.cv2, "VideoCapture", lambda _: BlackCamera())
    monkeypatch.setattr(edge.requests, "post", lambda url, **kwargs: calls.append((url, kwargs)))
    run_camera({"id": "camera-1", "source": "0", "roi_json": "[0,0,1,1]"}, interval=0)
    assert len(calls) == 1
    assert calls[0][0].endswith("/internal/camera-status")
    assert "near black" in calls[0][1]["params"]["error"]


def test_multi_photo_draft_and_technician_correction_with_stopped_worker(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo", "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        internal = {"X-Service-Key": SERVICE_KEY}
        client.post("/assets", headers=auth, json={"id": "iphone-1", "site_id": "site-a", "name": "Exterior", "family": "iphone-exterior", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "camera-1", "site_id": "site-a", "asset_id": "iphone-1", "name": "Camera", "source": "0", "sample_id": "phone-1"})
        event = {"camera_id": "camera-1", "asset_id": "iphone-1", "site_id": "site-a", "session_id": "session-1", "sample_id": "phone-1",
                 "quality": "ok", "jpeg_b64": jpeg()}
        first = client.post("/internal/captures", headers=internal, json={**event, "captured_at": datetime.now(timezone.utc).isoformat()}).json()["id"]
        second = client.post("/internal/captures", headers=internal, json={**event, "captured_at": datetime.now(timezone.utc).isoformat()}).json()["id"]
        other_session = client.post("/internal/captures", headers=internal, json={**event, "session_id": "session-2", "captured_at": datetime.now(timezone.utc).isoformat()}).json()["id"]
        candidates = client.get(f"/captures/{second}/draft-candidates", headers=auth)
        assert candidates.status_code == 200
        assert {item["id"] for item in candidates.json()} == {first, second}
        assert client.post(f"/captures/{second}/draft", headers=auth, json={"photos": [
            {"capture_id": second, "view": "front"}, {"capture_id": other_session, "view": "back"}]}).status_code == 422
        assert client.post(f"/captures/{second}/draft", headers=auth, json={"photos": [
            {"capture_id": second, "view": "front"}, {"capture_id": second, "view": "back"}]}).status_code == 422
        draft_response = client.post(f"/captures/{second}/draft", headers=auth, json={"photos": [
            {"capture_id": second, "view": "front"}, {"capture_id": first, "view": "back"}]})
        assert draft_response.status_code == 200, draft_response.text
        draft = draft_response.json()
        assert set(json.loads(draft["source_ids_json"])) == {first, second}
        assert json.loads(draft["views_json"]) == {second: "front", first: "back"}
        assert "Technician-selected views: back, front" in draft["draft_text"]
        assert other_session not in draft["source_ids_json"]
        assert "Scratch and dent counts are unavailable" in draft["draft_text"]
        assert client.get(f"/captures/{second}/draft", headers=auth).json()["draft_text"] == draft["draft_text"]
        from forgecare import ops_api
        seen = []
        def fake_vision(photos, subject):
            seen.append((photos, subject))
            return {"summary": "A line is visible; its cause is uncertain.",
                    "observations": ["Front: one visible line"], "scratch_count": 1,
                    "dent_count": None, "count_basis": "Front photo only",
                    "uncertainty": "Dent count not established", "next_view": "Edge close-up"}
        monkeypatch.setattr(ops_api, "describe_selected_photos", fake_vision)
        ai_response = client.post(f"/captures/{second}/vision-review", headers=auth)
        assert ai_response.status_code == 200, ai_response.text
        assert ai_response.json()["ai_text"].startswith("AI visual draft")
        assert ai_response.json()["ai_scratch_count"] == 1
        assert ai_response.json()["ai_dent_count"] is None
        assert set(json.loads(ai_response.json()["ai_image_ids_json"])) == {first, second}
        assert len(seen[0][0]) == 2
        reviewed = client.post(f"/captures/{second}/label", headers=auth, json={
            "grade": "minor", "note": "Technician sees one visible scratch; no dent confirmed", "scratch_count": 1, "dent_count": 0})
        assert reviewed.status_code == 200, reviewed.text
        saved = next(x for x in client.get("/captures?site_id=site-a", headers=auth).json() if x["id"] == second)
        assert saved["reviewer_note"].startswith("Technician sees")
        assert saved["scratch_count"] == 1 and saved["dent_count"] == 0
        case_id = client.post("/cases", headers=auth, json={"capture_id": second}).json()["id"]
        case = client.get(f"/cases/{case_id}", headers=auth).json()
        assert case["capture"]["scratch_count"] == 1
        assert case["review_draft"]["anchor_capture_id"] == second
        assert client.post(f"/captures/{second}/label", headers=auth, json={"grade": "unclear", "scratch_count": 1}).status_code == 422
        assert client.post("/cameras/camera-1/stop", headers=auth).status_code == 200
        assert client.get("/cameras?site_id=site-a", headers=auth).json()[0]["status"] == "stopped"
        assert client.post("/internal/captures", headers=internal, json={**event, "captured_at": datetime.now(timezone.utc).isoformat()}).status_code == 409


def test_case_lifecycle_and_site_scope():
    with TestClient(app) as client:
        setup = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo", "username": "admin", "password": "super-secure-password"})
        assert setup.status_code == 200, setup.text
        auth = {"Authorization": f"Bearer {setup.json()['token']}"}
        assert client.post("/assets", headers=auth, json={"id": "iphone-1", "site_id": "site-a", "name": "Exterior", "family": "iphone-exterior", "criticality": 3}).status_code == 200
        assert client.post("/cameras", headers=auth, json={"id": "camera-1", "site_id": "site-a", "asset_id": "iphone-1", "name": "Laptop", "source": "0", "sample_id": "phone-1"}).status_code == 200
        assert client.post("/cameras", headers=auth, json={"id": "replay", "site_id": "site-a", "asset_id": "iphone-1", "name": "Replay", "source": "https://example.com/recording.mp4", "sample_id": "phone-1"}).status_code == 422
        event = {"camera_id": "camera-1", "asset_id": "iphone-1", "site_id": "site-a", "session_id": "session-1", "sample_id": "phone-1",
                 "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok", "jpeg_b64": jpeg()}
        internal = {"X-Service-Key": SERVICE_KEY}
        assert client.post("/internal/captures", headers=internal, json={**event, "captured_at": "2020-01-01T00:00:00+00:00"}).status_code == 422
        assert client.post("/internal/captures", headers=internal, json={**event, "sample_id": "other-phone"}).status_code == 422
        assert any(x["action"] == "capture_rejected" for x in client.get("/audit/global", headers=auth).json())
        capture = client.post("/internal/captures", headers=internal, json=event)
        assert capture.status_code == 200, capture.text
        capture_id = capture.json()["id"]
        assert client.post(f"/captures/{capture_id}/label", headers=auth, json={"grade": "minor", "note": "Visible mark"}).status_code == 200
        reviewed = client.get("/captures?site_id=site-a", headers=auth).json()
        assert reviewed[0]["grade"] == "minor" and reviewed[0]["reviewer_note"] == "Visible mark"
        case = client.post("/cases", headers=auth, json={"capture_id": capture_id})
        assert case.status_code == 200, case.text
        case_id = case.json()["id"]
        assert isinstance(case.json()["priority"], float)
        work_order = client.get(f"/cases/{case_id}", headers=auth).json()
        assert work_order["asset"]["id"] == "iphone-1"
        assert work_order["capture"]["reviewed_grade"] == "minor"
        assert work_order["capture"]["reviewer_note"] == "Visible mark"
        assert work_order["capture"]["reviewer_name"] == "admin"
        assert client.post("/cases", headers=auth, json={"capture_id": capture_id}).json()["duplicate_linked"]
        assert len(client.get("/cases/selected?site_id=site-a&capacity_minutes=15", headers=auth).json()) == 1
        admin_id = client.get("/auth/me", headers=auth).json()["id"]
        assignment = client.post(f"/cases/{case_id}/assign", headers=auth, json={"user_id": admin_id})
        assert assignment.status_code == 200, assignment.text
        assigned_order = client.get(f"/cases/{case_id}", headers=auth).json()
        assert assigned_order["status"] == "assigned" and assigned_order["assignee_name"]["username"] == "admin"
        assert client.post(f"/cases/{case_id}/approve", headers=auth, json={"decision_note": "Inspect visible exterior"}).status_code == 200
        assert client.post(f"/cases/{case_id}/close", headers=auth, json={"actual_finding": "Photo-only review; no hands-on inspection", "actual_minutes": 0, "note": "Demo", "inspection_performed": False}).status_code == 200
        recorded = client.get(f"/cases/{case_id}", headers=auth).json()
        assert recorded["status"] == "closed"
        assert recorded["decision_note"] == "Inspect visible exterior"
        assert recorded["closure_note"] == "Demo"
        assert recorded["inspection_performed"] == 0
        assert client.get("/assets?site_id=site-a", headers=auth).json()[0]["last_inspection_at"] is None
        same_capture = client.post("/cases", headers=auth, json={"capture_id": capture_id}).json()
        assert same_capture["id"] == case_id and same_capture["duplicate_linked"]
        second = client.post("/internal/captures", headers=internal, json={**event, "captured_at": datetime.now(timezone.utc).isoformat()})
        assert client.post("/cases", headers=auth, json={"capture_id": second.json()["id"]}).status_code == 422
        assert client.post(f"/captures/{second.json()['id']}/label", headers=auth,
                           json={"grade": "minor", "note": "Second view checked"}).status_code == 200
        second_case = client.post("/cases", headers=auth, json={"capture_id": second.json()["id"]}).json()["id"]
        assert client.post(f"/cases/{second_case}/approve", headers=auth, json={"decision_note": "Hands-on inspection"}).status_code == 200
        assert client.post(f"/cases/{second_case}/close", headers=auth, json={"actual_finding": "Inspected exterior", "actual_minutes": 7, "inspection_performed": True}).status_code == 200
        assert client.get("/assets?site_id=site-a", headers=auth).json()[0]["last_inspection_at"] is not None
        user = client.post("/users", headers=auth, json={"username": "tech", "password": "another-secure-password", "role": "technician", "site_id": "site-a"})
        assert user.status_code == 200
        assert client.post("/sites", headers=auth, json={"id": "site-b", "name": "Other"}).status_code == 200
        token = client.post("/auth/login", json={"username": "tech", "password": "another-secure-password"}).json()["token"]
        assert client.get("/assets?site_id=site-b", headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_laptop_camera_can_switch_subjects_only_while_stopped():
    with TestClient(app) as client:
        setup = client.post("/setup/bootstrap", json={"site_id": "demo", "site_name": "Demo", "username": "admin", "password": "initial-secure-password"})
        auth = {"Authorization": f"Bearer {setup.json()['token']}"}
        for asset_id in ("realme", "iphone"):
            assert client.post("/assets", headers=auth, json={"id": asset_id, "site_id": "demo", "name": asset_id,
                    "family": asset_id, "criticality": 3}).status_code == 200
        assert client.post("/cameras", headers=auth, json={"id": "laptop", "site_id": "demo", "asset_id": "realme",
                "name": "Built-in webcam", "source": "0", "sample_id": "realme-physical-01"}).status_code == 200
        mapping = {"asset_id": "iphone", "source": "0", "sample_id": "iphone-physical-01", "roi": [0, 0, 1, 1]}

        class RunningWorker:
            def poll(self):
                return None

        CAMERA_PROCESSES["laptop"] = RunningWorker()
        try:
            assert client.put("/cameras/laptop", headers=auth, json=mapping).status_code == 409
        finally:
            CAMERA_PROCESSES.pop("laptop", None)
        assert client.put("/cameras/laptop", headers=auth, json={**mapping, "source": "saved.mp4"}).status_code == 422
        changed = client.put("/cameras/laptop", headers=auth, json=mapping)
        assert changed.status_code == 200, changed.text
        camera = client.get("/cameras?site_id=demo", headers=auth).json()[0]
        assert (camera["asset_id"], camera["sample_id"], camera["status"]) == ("iphone", "iphone-physical-01", "offline")
        event = {"camera_id": "laptop", "asset_id": "realme", "site_id": "demo", "session_id": "session-1",
                 "sample_id": "realme-physical-01", "captured_at": datetime.now(timezone.utc).isoformat(),
                 "quality": "ok", "jpeg_b64": jpeg()}
        internal = {"X-Service-Key": SERVICE_KEY}
        assert client.post("/internal/captures", headers=internal, json=event).status_code == 422
        assert client.post("/internal/captures", headers=internal, json={**event, "asset_id": "iphone",
                "sample_id": "iphone-physical-01"}).status_code == 200

        assert client.post("/auth/change-password", headers=auth, json={"current_password": "wrong-password",
                "new_password": "replacement-secure-password"}).status_code == 403
        assert client.post("/auth/change-password", headers=auth, json={"current_password": "initial-secure-password",
                "new_password": "replacement-secure-password"}).status_code == 200
        assert client.get("/auth/me", headers=auth).status_code == 401
        assert client.post("/auth/login", json={"username": "admin", "password": "initial-secure-password"}).status_code == 401
        assert client.post("/auth/login", json={"username": "admin", "password": "replacement-secure-password"}).status_code == 200


def test_fedavg_rejects_bad_updates():
    start = np.zeros(3)
    one = {"round_id": "r1", "schema_hash": "s1", "client_id": "a", "count": 2, "weights": np.ones(3)}
    two = {"round_id": "r1", "schema_hash": "s1", "client_id": "b", "count": 1, "weights": np.zeros(3)}
    assert np.allclose(aggregate([one, two], round_id="r1", schema_hash="s1", start=start), np.ones(3) * 2 / 3)
    with pytest.raises(ValueError):
        aggregate([one, {**two, "weights": np.array([np.nan, 0, 0])}], round_id="r1", schema_hash="s1", start=start)
    with pytest.raises(ValueError):
        aggregate([{**one, "round_id": "old"}], round_id="r1", schema_hash="s1", start=start)


def test_visual_split_keeps_physical_samples_apart():
    groups = np.repeat(np.array([f"normal-{i}" for i in range(6)] + [f"review-{i}" for i in range(6)]), 3)
    labels = np.repeat(np.array([0] * 6 + [1] * 6), 3)
    train, validation, test = grouped_splits(labels, groups)
    assert set(groups[train]).isdisjoint(groups[validation])
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(groups[validation]).isdisjoint(groups[test])
