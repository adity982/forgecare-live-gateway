"""A private phone stream can be sent through a narrowly paired public relay."""
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from forgecare.ops_api import app as operations
from forgecare.relay_api import app as relay
from forgecare.config import DATA
from forgecare.camera_setup import ip_webcam_direct_source, ip_webcam_relay_source, is_local_view


def test_samsung_address_form_keeps_http_and_explains_https():
    assert ip_webcam_direct_source("http://10.73.3.83:8080") == "http://10.73.3.83:8080/video"
    assert ip_webcam_direct_source("10.73.3.83:8080/video") == "http://10.73.3.83:8080/video"
    assert ip_webcam_relay_source("http://10.73.3.83:8080") == "relay:http://10.73.3.83:8080/video"
    assert ip_webcam_relay_source("10.73.3.83:8080") == "relay:http://10.73.3.83:8080/video"
    try:
        ip_webcam_relay_source("https://10.73.3.83:8080")
    except ValueError as exc:
        assert "HTTP" in str(exc) and "HTTPS" in str(exc)
    else:
        raise AssertionError("HTTPS must prompt for the HTTP camera address")
    assert is_local_view("http://127.0.0.1:8501/")
    assert is_local_view("http://localhost:8501/")
    assert not is_local_view("https://forgecare-ai-live.onrender.com/")


def jpeg() -> str:
    image = np.random.default_rng(9).integers(65, 185, size=(160, 220, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_paired_ip_webcam_relay_enters_capture_review():
    with TestClient(operations) as ops, TestClient(relay) as bridge:
        token = ops.post("/setup/bootstrap", json={"site_id": "demo-site", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        ops.post("/assets", headers=auth, json={"id": "iphone-14-plus", "site_id": "demo-site",
            "name": "iPhone", "family": "iphone-exterior", "criticality": 3})
        created = ops.post("/cameras", headers=auth, json={"id": "camera-samsung", "site_id": "demo-site",
            "asset_id": "iphone-14-plus", "name": "Samsung", "source": "relay:http://10.10.1.42:8080/video",
            "sample_id": "iphone-physical-01"})
        assert created.status_code == 200, created.text
        assert ops.get("/cameras?site_id=demo-site", headers=auth).json()[0]["status"] == "relay_waiting"
        assert ops.post("/cameras/camera-samsung/start", headers=auth).status_code == 422
        assert bridge.get("/relay/camera-samsung/config").status_code == 401
        pair = ops.post("/cameras/camera-samsung/relay-pair", headers=auth).json()
        code = {"X-Relay-Code": pair["pairing_code"]}
        config = bridge.get("/relay/camera-samsung/config", headers=code)
        assert config.status_code == 200
        assert config.json()["video_url"] == "http://10.10.1.42:8080/video"
        result = bridge.post("/relay/camera-samsung/captures", headers=code,
            json={"session_id": "relay-test-01", "jpeg_b64": jpeg()})
        assert result.status_code == 200, result.text
        assert result.json()["quality"] == "ok"
        captures = ops.get("/captures?site_id=demo-site", headers=auth).json()
        assert captures[0]["id"] == result.json()["id"]
        assert captures[0]["sample_id"] == "iphone-physical-01"
        assert ops.get("/cameras?site_id=demo-site", headers=auth).json()[0]["status"] == "live"
        bridge.post("/relay/camera-samsung/status", headers=code,
                    json={"error": "Local camera bridge stopped"})
        assert ops.get("/cameras?site_id=demo-site", headers=auth).json()[0]["status"] == "relay_waiting"
        assert bridge.post("/relay/camera-samsung/captures", headers={"X-Relay-Code": "wrong"},
            json={"session_id": "relay-test-01", "jpeg_b64": jpeg()}).status_code == 401
        updated = ops.put("/cameras/camera-samsung", headers=auth, json={"asset_id": "iphone-14-plus",
            "source": "relay:http://10.10.1.43:8080/video", "sample_id": "iphone-physical-01",
            "roi": [0, 0, 1, 1]})
        assert updated.status_code == 200
        assert bridge.get("/relay/camera-samsung/config", headers=code).status_code == 401


def test_relay_requires_numeric_private_http_address():
    with TestClient(operations) as ops:
        token = ops.post("/setup/bootstrap", json={"site_id": "demo-site", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        ops.post("/assets", headers=auth, json={"id": "iphone-14-plus", "site_id": "demo-site",
            "name": "iPhone", "family": "iphone-exterior", "criticality": 3})
        for source in ("relay:http://example.com:8080/video", "relay:https://10.10.1.42:8080/video",
                       "relay:http://127.0.0.1:8080/video"):
            response = ops.post("/cameras", headers=auth, json={"id": "camera-samsung", "site_id": "demo-site",
                "asset_id": "iphone-14-plus", "name": "Samsung", "source": source,
                "sample_id": "iphone-physical-01"})
            assert response.status_code == 422


def test_samsung_ip_setup_renders_on_hosted_page(monkeypatch):
    with TestClient(operations) as ops:
        token = ops.post("/setup/bootstrap", json={"site_id": "demo-site", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        auth = {"Authorization": f"Bearer {token}"}
        ops.post("/assets", headers=auth, json={"id": "iphone-14-plus", "site_id": "demo-site",
            "name": "iPhone", "family": "iphone-exterior", "criticality": 3})
        ops.post("/cameras", headers=auth, json={"id": "camera-samsung", "site_id": "demo-site",
            "asset_id": "iphone-14-plus", "name": "Samsung", "source": "relay:http://10.10.1.42:8080/video",
            "sample_id": "iphone-physical-01"})
        endpoint = DATA / "relay-public-url.txt"
        endpoint.write_text("https://example.trycloudflare.com", encoding="utf-8")

        def local_request(method, url, **kwargs):
            kwargs.pop("timeout", None)
            return ops.request(method, url.removeprefix("http://127.0.0.1:8000"), **kwargs)

        monkeypatch.setattr(requests, "request", local_request)
        try:
            ui = AppTest.from_file(Path(__file__).parents[1] / "forgecare" / "ui.py", default_timeout=20)
            ui.session_state["token"] = token
            ui.run()
            ui.radio[0].set_value("Assets & setup").run()
            assert not ui.exception, [item.message for item in ui.exception]
            assert any(item.label == "Samsung IP Webcam IPv4 address and port" for item in ui.text_input)
        finally:
            endpoint.unlink(missing_ok=True)
