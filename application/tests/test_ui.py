"""Smoke the actual Streamlit script through its authenticated navigation paths."""
import base64
from datetime import datetime, timezone
from io import BytesIO
import requests
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from forgecare.config import SERVICE_KEY
from forgecare.ops_api import app
from forgecare.sample_queue import sample_capacity_plan


def test_sample_capacity_scenario_uses_real_planner():
    selected, used = sample_capacity_plan(60)
    assert [item["id"] for item in selected] == ["SAMPLE-101", "SAMPLE-102", "SAMPLE-104"]
    assert used == 60
    assert sample_capacity_plan(0) == ([], 0)


def test_dashboard_pages_execute(monkeypatch):
    with TestClient(app) as client:
        if client.get("/setup/status").json()["needed"]:
            token = client.post("/setup/bootstrap", json={"site_id": "ui-site", "site_name": "UI Test", "username": "uiadmin", "password": "super-secure-password"}).json()["token"]
        else:
            token = client.post("/auth/login", json={"username": "uiadmin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        for asset_id in ("realme-narzo-60", "iphone-14-plus"):
            response = client.post("/assets", headers=auth, json={"id": asset_id, "site_id": "ui-site",
                    "name": asset_id, "family": "phone-exterior", "criticality": 3})
            assert response.status_code == 200, response.text
        response = client.post("/cameras", headers=auth, json={"id": "camera-laptop", "site_id": "ui-site",
                "asset_id": "realme-narzo-60", "name": "Lenovo webcam", "source": "0",
                "sample_id": "realme-physical-01"})
        assert response.status_code == 200, response.text

        original = requests.request

        def local_request(method, url, **kwargs):
            assert url.startswith("http://127.0.0.1:8000")
            kwargs.pop("timeout", None)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        assert not ui.exception, [x.message for x in ui.exception]
        for page in ("Live cameras", "Capture review", "Inspection queue", "Assets & setup",
                     "Knowledge assistant", "Models & experiments", "Federation", "Reports", "Administration", "Audit"):
            ui.radio[0].set_value(page).run()
            assert not ui.exception, (page, [x.message for x in ui.exception])
            if page == "Inspection queue":
                assert any(item.label == "Explore with sample cases" and item.value for item in ui.toggle)
                assert any("SAMPLE-101" in str(frame.value) for frame in ui.dataframe)
            if page == "Assets & setup":
                presets = [item for item in ui.selectbox if item.label == "Camera device"]
                assert len(presets) == 1 and presets[0].value == "Samsung Galaxy S24 FE stream"
            if page == "Models & experiments":
                assert any(item.label == "Minutes to check one flagged part" for item in ui.number_input)
                rendered = " ".join(str(item.value) for item in ui.warning)
                assert "119" in rendered and "23" in rendered and "21" in rendered
        monkeypatch.setattr(requests, "request", original)


def test_agent_without_approved_source_is_not_shown_as_success(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "ui-site", "site_name": "UI Test",
            "username": "uiadmin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone-1", "site_id": "ui-site",
            "name": "iPhone", "family": "iphone-exterior", "model": "iPhone 14 Plus", "criticality": 3})

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        class AgentResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "abstained_no_applicable_document", "draft": None,
                        "evidence": [], "trace": []}

        monkeypatch.setattr(requests, "request", local_request)
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: AgentResponse())
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Knowledge assistant").run()
        [button for button in ui.button if button.label == "Run bounded agent"][0].click().run()
        assert not ui.exception, [item.message for item in ui.exception]
        assert any("The answer to the question was not found based on the references searched. Please find the given resources." in str(item.value) for item in ui.warning)
        assert not any("Agent run finished" in str(item.value) for item in ui.success)


def test_populated_work_order_shows_review_and_worker_fields(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "ui-site", "site_name": "UI Test", "username": "uiadmin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone-1", "site_id": "ui-site", "name": "Phone exterior",
                    "family": "iphone-exterior", "model": "iPhone 14 Plus", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "camera-1", "site_id": "ui-site", "asset_id": "iphone-1",
                    "name": "Camera", "source": "0", "sample_id": "phone-1"})
        output = BytesIO()
        Image.new("RGB", (128, 128), "#a1a6a4").save(output, format="JPEG")
        capture = client.post("/internal/captures", headers={"X-Service-Key": SERVICE_KEY}, json={
            "camera_id": "camera-1", "asset_id": "iphone-1", "site_id": "ui-site", "session_id": "session-1",
            "sample_id": "phone-1", "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok",
            "jpeg_b64": base64.b64encode(output.getvalue()).decode()}).json()["id"]
        client.post(f"/captures/{capture}/label", headers=auth, json={"grade": "minor", "note": "Visible edge scratch"})
        draft = client.post(f"/captures/{capture}/draft", headers=auth)
        assert draft.status_code == 200
        case = client.post("/cases", headers=auth, json={"capture_id": capture}).json()["id"]
        client.post(f"/cases/{case}/approve", headers=auth, json={"decision_note": "Check the screen and edges"})

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Capture review").run()
        assert not ui.exception, [x.message for x in ui.exception]
        assert any(item.label == "Confirmed visible scratches" for item in ui.number_input)
        assert any(item.label == "Technician's corrected review" for item in ui.text_area)
        damage_buttons = [item for item in ui.button if item.label == "Analyze selected photos for visible damage"]
        if damage_buttons:
            damage_buttons[0].click().run()
            assert not ui.exception, [x.message for x in ui.exception]
            assert any("Local detector ran" in str(item.value) for item in ui.info)
        ui.radio[0].set_value("Inspection queue").run()
        assert not ui.exception, [x.message for x in ui.exception]
        assert any(item.label == "What did you actually find?" for item in ui.text_area)
        assert any(item.label == "I physically inspected this phone" for item in ui.checkbox)
        assert any(item.value == "Finish this inspection" for item in ui.subheader)
        rendered = " ".join(str(item.value) for item in ui.markdown)
        assert "Visible edge scratch" in rendered
        assert "Check the screen and edges" in rendered


def test_damage_analysis_prefills_editable_review_without_saving_label(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "ui-site", "site_name": "UI Test",
            "username": "uiadmin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone-1", "site_id": "ui-site",
            "name": "iPhone", "family": "iphone-exterior", "model": "iPhone 14 Plus", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "camera-1", "site_id": "ui-site",
            "asset_id": "iphone-1", "name": "Camera", "source": "0", "sample_id": "phone-1"})
        image = BytesIO()
        Image.new("RGB", (128, 128), "#a1a6a4").save(image, format="JPEG")
        capture = client.post("/internal/captures", headers={"X-Service-Key": SERVICE_KEY}, json={
            "camera_id": "camera-1", "asset_id": "iphone-1", "site_id": "ui-site", "session_id": "session-1",
            "sample_id": "phone-1", "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok",
            "jpeg_b64": base64.b64encode(image.getvalue()).decode()}).json()["id"]
        assert client.post(f"/captures/{capture}/draft", headers=auth).status_code == 200

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Capture review").run()
        [button for button in ui.button if button.label == "Analyze selected photos for visible damage"][0].click().run()
        assert not ui.exception, [item.message for item in ui.exception]
        note = [item for item in ui.text_area if item.label == "Technician's corrected review"][0]
        assert "Model-assisted draft" in note.value
        assert any(item.label == "Confirmed visible scratches" and item.value == 0 for item in ui.number_input)
        assert any("I checked every selected photo" in item.label and not item.value for item in ui.checkbox)
        from forgecare import ops_api
        monkeypatch.setattr(ops_api, "describe_selected_photos", lambda photos, subject: {
            "summary": "Two scratches and one dent appear in the close-up.",
            "observations": ["Front: two lines", "Edge: one indentation"],
            "scratch_count": 2, "dent_count": 1, "count_basis": "Different marks in front and edge views",
            "uncertainty": "Verify under direct light", "next_view": "Back close-up"})
        [button for button in ui.button if button.label == "Generate AI review from selected photos"][0].click().run()
        assert not ui.exception, [item.message for item in ui.exception]
        assert "AI visual draft" in [item for item in ui.text_area if item.label == "Technician's corrected review"][0].value
        assert any(item.label == "Confirmed visible scratches" and item.value == 2 for item in ui.number_input)
        assert any(item.label == "Confirmed visible dents" and item.value == 1 for item in ui.number_input)
        [button for button in ui.button if button.label == "Save reviewed label"][0].click().run()
        assert any("Check the photos" in str(item.value) for item in ui.error)
        assert client.get(f"/captures?site_id=ui-site", headers=auth).json()[0]["grade"] is None
