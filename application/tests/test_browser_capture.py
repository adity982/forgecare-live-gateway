"""Browser captures use the viewing device and join the existing review workflow."""
import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import requests
from pathlib import Path
from streamlit.testing.v1 import AppTest

from forgecare.ops_api import app
from forgecare.config import SERVICE_KEY
from datetime import datetime, timezone


def photo() -> str:
    rng = np.random.default_rng(17)
    pixels = rng.integers(70, 185, size=(240, 320, 3), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=88)
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_browser_photo_is_scored_and_available_for_review():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone-14-plus", "site_id": "site-a",
            "name": "iPhone", "family": "iphone-exterior", "criticality": 3})
        registered = client.post("/cameras", headers=auth, json={"id": "camera-samsung", "site_id": "site-a",
            "asset_id": "iphone-14-plus", "name": "Samsung browser", "source": "browser",
            "sample_id": "iphone-physical-01"})
        assert registered.status_code == 200, registered.text
        assert client.get("/cameras?site_id=site-a", headers=auth).json()[0]["status"] == "browser_ready"
        assert client.post("/cameras/camera-samsung/start", headers=auth).status_code == 422
        body = {"session_id": "browser-demo-01", "jpeg_b64": photo()}
        first = client.post("/cameras/camera-samsung/browser-captures", headers=auth, json=body)
        second = client.post("/cameras/camera-samsung/browser-captures", headers=auth, json=body)
        assert first.status_code == second.status_code == 200
        assert first.json()["quality"] == "ok"
        captures = client.get("/captures?site_id=site-a&quality=ok", headers=auth).json()
        assert [item["id"] for item in captures] == [second.json()["id"], first.json()["id"]]
        assert all(item["sample_id"] == "iphone-physical-01" for item in captures)
        candidates = client.get(f"/captures/{second.json()['id']}/draft-candidates", headers=auth).json()
        assert {item["id"] for item in candidates} == {first.json()["id"], second.json()["id"]}
        assert client.get(f"/captures/{first.json()['id']}/image", headers=auth).content.startswith(b"\xff\xd8")


def test_non_browser_source_cannot_accept_browser_photo():
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone-14-plus", "site_id": "site-a",
            "name": "iPhone", "family": "iphone-exterior", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "server-webcam", "site_id": "site-a",
            "asset_id": "iphone-14-plus", "name": "Server webcam", "source": "0",
            "sample_id": "iphone-physical-01"})
        response = client.post("/cameras/server-webcam/browser-captures", headers=auth,
            json={"session_id": "browser-demo-01", "jpeg_b64": photo()})
        assert response.status_code == 422


def test_browser_camera_page_renders_for_authenticated_operator(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "iphone-14-plus", "site_id": "site-a",
            "name": "iPhone", "family": "iphone-exterior", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "camera-samsung", "site_id": "site-a",
            "asset_id": "iphone-14-plus", "name": "Samsung browser", "source": "browser",
            "sample_id": "iphone-physical-01"})
        saved = client.post("/cameras/camera-samsung/browser-captures", headers=auth,
            json={"session_id": "browser-earlier-photo", "jpeg_b64": photo()}).json()["id"]
        viewed = []

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            viewed.append(url)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Live cameras").run()
        assert not ui.exception, [item.message for item in ui.exception]
        assert any(item.label == "Photo destination" for item in ui.selectbox)
        assert not any(f"/captures/{saved}/image" in url for url in viewed)


def test_local_camera_page_offers_direct_start_without_photo_widget(monkeypatch):
    import forgecare.camera_setup as setup
    monkeypatch.setattr(setup, "is_local_view", lambda _url: True)
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "realme-narzo-60", "site_id": "site-a",
            "name": "realme", "family": "realme-exterior", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "camera-laptop", "site_id": "site-a",
            "asset_id": "realme-narzo-60", "name": "Lenovo webcam", "source": "browser",
            "sample_id": "realme-physical-01"})

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Live cameras").run()
        assert not ui.exception, [item.message for item in ui.exception]
        assert any(item.label == "Camera to start" for item in ui.selectbox)
        assert any(item.label == "Start selected camera" for item in ui.button)
        assert not ui.camera_input


def test_old_server_photo_is_not_shown_as_current_browser_camera(monkeypatch):
    with TestClient(app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/assets", headers=auth, json={"id": "realme-narzo-60", "site_id": "site-a",
            "name": "realme", "family": "realme-exterior", "criticality": 3})
        client.post("/cameras", headers=auth, json={"id": "camera-laptop", "site_id": "site-a",
            "asset_id": "realme-narzo-60", "name": "Lenovo webcam", "source": "0",
            "sample_id": "realme-physical-01"})
        old = client.post("/internal/captures", headers={"X-Service-Key": SERVICE_KEY}, json={
            "camera_id": "camera-laptop", "asset_id": "realme-narzo-60", "site_id": "site-a",
            "session_id": "old-home-worker", "sample_id": "realme-physical-01",
            "captured_at": datetime.now(timezone.utc).isoformat(), "quality": "ok", "jpeg_b64": photo()})
        assert old.status_code == 200, old.text
        client.put("/cameras/camera-laptop", headers=auth, json={"asset_id": "realme-narzo-60",
            "source": "browser", "sample_id": "realme-physical-01", "roi": [0, 0, 1, 1]})
        viewed = []

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            viewed.append(url)
            return client.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
        ui.session_state["token"] = token
        ui.run()
        ui.radio[0].set_value("Live cameras").run()
        assert not ui.exception, [item.message for item in ui.exception]
        assert not any(f"/captures/{old.json()['id']}/image" in url for url in viewed)
