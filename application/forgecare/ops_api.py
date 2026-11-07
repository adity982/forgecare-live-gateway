from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import re
import secrets
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import cv2
import joblib
import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import DATA, KNOW_URL, SERVICE_KEY
from .db import audit, check_password, connect, create_session, hash_password, migrate, now, row, rows, user_from_token
from .priority import band, fit_capacity, priority
from .vision import predict, train_visual
from .experiments import run_suite
from .federation import VISUAL_FEATURE_COUNT, aggregate, run_federation, visual_schema_hash
from .phone_federation_schedule import select_rows as phone_training_rows
from .edge import camera_source, inspect_frame
from .groq_provider import GroqUnavailable, VISION_MODEL, describe_selected_photos

@asynccontextmanager
async def lifespan(_app: FastAPI):
    migrate()
    yield


app = FastAPI(title="ForgeCare Operations", version="0.1.0", lifespan=lifespan)
CAMERA_PROCESSES: dict[str, subprocess.Popen] = {}


def valid_camera_source(value: str) -> None:
    if value == "browser":
        return
    if value.startswith("relay:"):
        stream = value.removeprefix("relay:")
        parsed = urlparse(stream)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError:
            raise ValueError("Enter the Samsung's numeric private Wi-Fi address")
        if (parsed.scheme != "http" or not address.is_private or address.is_loopback
                or not parsed.port or parsed.username or parsed.password):
            raise ValueError("Relay source must be http://PRIVATE_PHONE_IP:PORT/video")
        camera_source(stream)
        return
    camera_source(value)


@app.middleware("http")
async def record_rejected_events(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/internal/captures" and response.status_code >= 400:
        with connect() as db:
            audit(db, None, None, "capture_rejected", "camera_event", "unaccepted",
                  json.dumps({"status_code": response.status_code}))
    return response


class Bootstrap(BaseModel):
    site_id: str = Field(pattern=r"^[A-Za-z0-9_-]{2,40}$")
    site_name: str
    username: str = Field(min_length=3)
    password: str = Field(min_length=10)


class Login(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=10)


class SiteIn(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{2,40}$")
    name: str


class UserIn(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=10)
    role: Literal["technician", "supervisor", "reviewer", "admin"]
    site_id: str | None = None


class AssetIn(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{2,64}$")
    site_id: str
    name: str
    family: str
    model: str | None = None
    criticality: int = Field(ge=1, le=5)
    inspection_interval_days: int = Field(default=30, ge=1)


class PhysicalPhoneIn(BaseModel):
    id: str = Field(min_length=2, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    site_id: str
    asset_id: str
    display_name: str = Field(min_length=2, max_length=120)


class CameraIn(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{2,64}$")
    site_id: str
    asset_id: str
    name: str
    source: str
    sample_id: str = Field(min_length=2, max_length=80)
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


class CameraUpdate(BaseModel):
    asset_id: str
    source: str
    sample_id: str = Field(min_length=2, max_length=80)
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


class CaptureIn(BaseModel):
    camera_id: str
    asset_id: str
    site_id: str
    session_id: str
    sample_id: str
    captured_at: str
    roi_version: str = "1"
    quality: Literal["ok", "blurred", "dark", "bright", "invalid"]
    blur: float | None = None
    brightness: float | None = None
    jpeg_b64: str


class BrowserCaptureIn(BaseModel):
    session_id: str = Field(min_length=2, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    jpeg_b64: str = Field(max_length=12_000_000)


class SampleIn(BaseModel):
    sample_id: str = Field(min_length=2, max_length=80)


class LabelIn(BaseModel):
    grade: Literal["none", "minor", "significant", "unclear"]
    note: str = ""
    scratch_count: int | None = Field(default=None, ge=0, le=100)
    dent_count: int | None = Field(default=None, ge=0, le=100)


class DraftPhoto(BaseModel):
    capture_id: str
    view: Literal["front", "back", "left", "right", "top", "bottom", "detail", "unassigned"] = "unassigned"


class DraftIn(BaseModel):
    photos: list[DraftPhoto] = Field(default_factory=list, max_length=5)


class RuleIn(BaseModel):
    site_id: str
    criticality_weight: float = Field(ge=0, le=1)
    repeat_weight: float = Field(ge=0, le=1)
    overdue_weight: float = Field(ge=0, le=1)
    default_minutes: int = Field(ge=1, le=480)


class CaseIn(BaseModel):
    capture_id: str
    note: str = ""


class AssignIn(BaseModel):
    user_id: int
    override_reason: str | None = None


class ApproveIn(BaseModel):
    decision_note: str
    override_reason: str | None = None


class CloseIn(BaseModel):
    actual_finding: str = Field(min_length=1)
    actual_minutes: int = Field(ge=0, le=1440)
    note: str = ""
    inspection_performed: bool = False


class AskIn(BaseModel):
    site_id: str
    asset_id: str
    question: str
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    scope: Literal["auto", "phone", "manufacturing", "automotive"] = "auto"


class AgentRequest(BaseModel):
    site_id: str
    asset_id: str
    case_id: str | None = None
    request: str = Field(min_length=3, max_length=1000)


class TrainIn(BaseModel):
    site_id: str
    family: str | None = None
    model: str | None = None


class FedUpdateIn(BaseModel):
    site_id: str
    client_id: str = Field(min_length=2, max_length=80)
    schema_hash: str
    sample_count: int = Field(ge=1)
    weights: list[float] = Field(min_length=VISUAL_FEATURE_COUNT + 1, max_length=VISUAL_FEATURE_COUNT + 1)


class FedValidationIn(BaseModel):
    site_id: str
    client_id: str = Field(min_length=2, max_length=80)
    sample_count: int = Field(ge=5)
    positive_count: int = Field(ge=2)
    recall: float = Field(ge=0, le=1)
    false_alert_rate: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    local_recall: float | None = Field(default=None, ge=0, le=1)
    local_false_alert_rate: float | None = Field(default=None, ge=0, le=1)


class DocumentIn(BaseModel):
    site_id: str
    family: str
    model: str = "*"
    url: str
    revision: str = "web-current"


def current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    token = (authorization or "").removeprefix("Bearer ")
    with connect() as db:
        user = user_from_token(db, token) if token else None
    if not user or not user["active"]:
        raise HTTPException(401, "Sign in required")
    return user


def check_site(user: dict, site_id: str) -> None:
    if user["role"] != "admin" and user["site_id"] != site_id:
        raise HTTPException(403, "Site access denied")


def check_role(user: dict, *roles: str) -> None:
    if user["role"] not in roles:
        raise HTTPException(403, "Role cannot perform this action")


def internal(x_service_key: Annotated[str | None, Header()] = None) -> None:
    if not x_service_key or not secrets.compare_digest(x_service_key, SERVICE_KEY):
        raise HTTPException(403, "Invalid service key")


def active_rules(db, site: str) -> dict:
    rules = row(db, "SELECT * FROM rules WHERE site_id=? ORDER BY id DESC LIMIT 1", (site,))
    if not rules:
        db.execute("INSERT INTO rules(site_id,criticality_weight,repeat_weight,overdue_weight,default_minutes,created_at) VALUES(?,?,?,?,?,?)",
                   (site, .4, .35, .25, 15, now()))
        rules = row(db, "SELECT * FROM rules WHERE site_id=? ORDER BY id DESC LIMIT 1", (site,))
    return rules


@app.get("/health")
def health():
    return {"status": "ok", "time": now()}


@app.get("/setup/status")
def setup_status():
    with connect() as db:
        return {"needed": db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0}


@app.post("/setup/bootstrap")
def bootstrap(data: Bootstrap):
    with connect() as db:
        if db.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            raise HTTPException(409, "Setup already completed")
        db.execute("INSERT INTO sites VALUES(?,?,?)", (data.site_id, data.site_name, now()))
        db.execute("INSERT INTO users(username,password_hash,role,site_id) VALUES(?,?,?,?)",
                   (data.username, hash_password(data.password), "admin", data.site_id))
        user = row(db, "SELECT id FROM users WHERE username=?", (data.username,))
        audit(db, user["id"], data.site_id, "bootstrap", "site", data.site_id)
        return {"token": create_session(db, user["id"])}


@app.post("/auth/login")
def login(data: Login):
    with connect() as db:
        user = row(db, "SELECT * FROM users WHERE username=? AND active=1", (data.username,))
        if not user or not check_password(data.password, user["password_hash"]):
            raise HTTPException(401, "Invalid credentials")
        return {"token": create_session(db, user["id"]), "role": user["role"], "site_id": user["site_id"]}


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return {k: user[k] for k in ("id", "username", "role", "site_id")}


@app.post("/auth/change-password")
def change_password(data: PasswordChange, user: dict = Depends(current_user)):
    with connect() as db:
        account = row(db, "SELECT password_hash FROM users WHERE id=?", (user["id"],))
        if not account or not check_password(data.current_password, account["password_hash"]):
            raise HTTPException(403, "Current password is incorrect")
        db.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(data.new_password), user["id"]))
        db.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
        audit(db, user["id"], user["site_id"], "password_change", "user", str(user["id"]))
    return {"status": "changed", "sign_in_again": True}


@app.post("/sites")
def add_site(data: SiteIn, user: dict = Depends(current_user)):
    check_role(user, "admin")
    with connect() as db:
        db.execute("INSERT INTO sites VALUES(?,?,?)", (data.id, data.name, now()))
        audit(db, user["id"], data.id, "create", "site", data.id)
    return data


@app.get("/sites")
def list_sites(user: dict = Depends(current_user)):
    with connect() as db:
        return rows(db, "SELECT * FROM sites ORDER BY name" if user["role"] == "admin" else "SELECT * FROM sites WHERE id=?", () if user["role"] == "admin" else (user["site_id"],))


@app.post("/users")
def add_user(data: UserIn, user: dict = Depends(current_user)):
    check_role(user, "admin")
    with connect() as db:
        if not row(db, "SELECT id FROM sites WHERE id=?", (data.site_id,)):
            raise HTTPException(404, "Site not found")
        cur = db.execute("INSERT INTO users(username,password_hash,role,site_id) VALUES(?,?,?,?)",
                         (data.username, hash_password(data.password), data.role, data.site_id))
        audit(db, user["id"], data.site_id, "create", "user", str(cur.lastrowid))
        return {"id": cur.lastrowid, "username": data.username, "role": data.role}


@app.get("/users")
def list_users(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return rows(db, "SELECT id,username,role,site_id,active FROM users WHERE site_id=? ORDER BY username", (site_id,))


@app.post("/assets")
def add_asset(data: AssetIn, user: dict = Depends(current_user)):
    check_role(user, "admin", "supervisor")
    check_site(user, data.site_id)
    with connect() as db:
        db.execute("INSERT INTO assets(id,site_id,name,family,model,criticality,inspection_interval_days) VALUES(?,?,?,?,?,?,?)",
                   (data.id, data.site_id, data.name, data.family, data.model, data.criticality, data.inspection_interval_days))
        audit(db, user["id"], data.site_id, "create", "asset", data.id)
    return data


@app.get("/assets")
def list_assets(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return rows(db, "SELECT * FROM assets WHERE site_id=? ORDER BY name", (site_id,))


def ensure_physical_phone(db, site_id: str, asset_id: str, sample_id: str) -> None:
    existing = row(db, "SELECT asset_id FROM physical_samples WHERE site_id=? AND id=?", (site_id, sample_id))
    if existing and existing["asset_id"] != asset_id:
        raise HTTPException(409, "This physical phone ID is already registered under a different model")
    if not existing:
        db.execute("INSERT INTO physical_samples(id,site_id,asset_id,display_name,created_at) VALUES(?,?,?,?,?)",
                   (sample_id, site_id, asset_id, sample_id, now()))


@app.post("/samples")
def add_physical_phone(data: PhysicalPhoneIn, user: dict = Depends(current_user)):
    check_role(user, "admin", "supervisor")
    check_site(user, data.site_id)
    with connect() as db:
        if not row(db, "SELECT id FROM assets WHERE site_id=? AND id=?", (data.site_id, data.asset_id)):
            raise HTTPException(404, "Phone model not found at this site")
        if row(db, "SELECT id FROM physical_samples WHERE site_id=? AND id=?", (data.site_id, data.id)):
            raise HTTPException(409, "Physical phone ID is already registered")
        db.execute("INSERT INTO physical_samples(id,site_id,asset_id,display_name,created_at) VALUES(?,?,?,?,?)",
                   (data.id, data.site_id, data.asset_id, data.display_name, now()))
        audit(db, user["id"], data.site_id, "create", "physical_phone", data.id,
              json.dumps({"model_asset_id": data.asset_id, "display_name": data.display_name}))
    return data


@app.get("/samples")
def list_physical_phones(site_id: str, asset_id: str | None = None, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        if asset_id:
            return rows(db, "SELECT * FROM physical_samples WHERE site_id=? AND asset_id=? ORDER BY created_at DESC", (site_id, asset_id))
        return rows(db, "SELECT * FROM physical_samples WHERE site_id=? ORDER BY created_at DESC", (site_id,))


@app.post("/cameras")
def add_camera(data: CameraIn, user: dict = Depends(current_user)):
    check_role(user, "admin")
    check_site(user, data.site_id)
    if not all(0 <= x <= 1 for x in data.roi) or data.roi[0] >= data.roi[2] or data.roi[1] >= data.roi[3]:
        raise HTTPException(422, "ROI must be normalized x1,y1,x2,y2")
    try:
        valid_camera_source(data.source)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    with connect() as db:
        asset = row(db, "SELECT * FROM assets WHERE id=? AND site_id=?", (data.asset_id, data.site_id))
        if not asset:
            raise HTTPException(404, "Asset not found at site")
        ensure_physical_phone(db, data.site_id, data.asset_id, data.sample_id)
        db.execute("INSERT INTO cameras(id,site_id,asset_id,name,source,roi_json,sample_id) VALUES(?,?,?,?,?,?,?)",
                   (data.id, data.site_id, data.asset_id, data.name, data.source, json.dumps(data.roi), data.sample_id))
        audit(db, user["id"], data.site_id, "create", "camera", data.id)
    return data


@app.put("/cameras/{camera_id}")
def update_camera(camera_id: str, data: CameraUpdate, user: dict = Depends(current_user)):
    check_role(user, "admin")
    if not all(0 <= x <= 1 for x in data.roi) or data.roi[0] >= data.roi[2] or data.roi[1] >= data.roi[3]:
        raise HTTPException(422, "ROI must be normalized x1,y1,x2,y2")
    try:
        valid_camera_source(data.source)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=?", (camera_id,))
        if not camera:
            raise HTTPException(404, "Camera not found")
        check_site(user, camera["site_id"])
        process = CAMERA_PROCESSES.get(camera_id)
        if process and process.poll() is None:
            raise HTTPException(409, "Stop this camera worker before changing its configuration")
        if not row(db, "SELECT id FROM assets WHERE id=? AND site_id=?", (data.asset_id, camera["site_id"])):
            raise HTTPException(404, "Asset not found at camera site")
        ensure_physical_phone(db, camera["site_id"], data.asset_id, data.sample_id)
        db.execute("""UPDATE cameras SET asset_id=?,source=?,sample_id=?,roi_json=?,
                      stopped_at=NULL,last_frame_at=NULL,last_error=NULL,last_error_at=NULL,last_quality=NULL WHERE id=?""",
                   (data.asset_id, data.source, data.sample_id, json.dumps(data.roi), camera_id))
        db.execute("DELETE FROM relay_pairs WHERE camera_id=?", (camera_id,))
        audit(db, user["id"], camera["site_id"], "configure", "camera", camera_id,
              json.dumps({"asset_id": data.asset_id, "sample_id": data.sample_id, "source": data.source}))
    return {"camera_id": camera_id, "asset_id": data.asset_id, "source": data.source,
            "sample_id": data.sample_id, "roi": data.roi}


@app.post("/cameras/{camera_id}/sample")
def set_camera_sample(camera_id: str, data: SampleIn, user: dict = Depends(current_user)):
    check_role(user, "supervisor", "admin")
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=?", (camera_id,))
        if not camera:
            raise HTTPException(404, "Camera not found")
        check_site(user, camera["site_id"])
        process = CAMERA_PROCESSES.get(camera_id)
        if process and process.poll() is None:
            raise HTTPException(409, "Stop this camera worker before changing the physical sample")
        ensure_physical_phone(db, camera["site_id"], camera["asset_id"], data.sample_id)
        db.execute("UPDATE cameras SET sample_id=? WHERE id=?", (data.sample_id, camera_id))
        audit(db, user["id"], camera["site_id"], "sample_change", "camera", camera_id, json.dumps({"sample_id": data.sample_id}))
    return {"camera_id": camera_id, "sample_id": data.sample_id}


@app.get("/cameras")
def list_cameras(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        data = rows(db, "SELECT * FROM cameras WHERE site_id=? ORDER BY name", (site_id,))
    for camera in data:
        if camera["source"] == "browser":
            camera["status"] = "browser_ready"
            continue
        if camera["source"].startswith("relay:"):
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(camera["last_frame_at"])).total_seconds()
            except (TypeError, ValueError):
                age = float("inf")
            camera["status"] = "live" if age <= 15 and not camera.get("last_error") else "relay_waiting"
            continue
        if camera.get("stopped_at"):
            camera["status"] = "stopped"
            continue
        process = CAMERA_PROCESSES.get(camera["id"])
        if process and process.poll() is not None:
            camera["status"] = "offline"
            continue
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(camera["last_frame_at"])).total_seconds()
            camera["status"] = "live" if age <= 10 else "offline" if camera.get("last_error") else "no_signal"
        except (TypeError, ValueError):
            camera["status"] = "offline"
    return data


@app.post("/cameras/{camera_id}/relay-pair")
def pair_relay(camera_id: str, user: dict = Depends(current_user)):
    check_role(user, "admin")
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=? AND enabled=1", (camera_id,))
        if not camera:
            raise HTTPException(404, "Camera not found")
        check_site(user, camera["site_id"])
        if not camera["source"].startswith("relay:"):
            raise HTTPException(422, "Set an IP Webcam relay source before pairing this laptop")
        code = secrets.token_urlsafe(24)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        db.execute("DELETE FROM relay_pairs WHERE camera_id=? OR expires_at<?", (camera_id, now()))
        db.execute("INSERT INTO relay_pairs(token_hash,camera_id,created_by,expires_at,created_at) VALUES(?,?,?,?,?)",
                   (hashlib.sha256(code.encode()).hexdigest(), camera_id, user["id"], expires_at, now()))
        audit(db, user["id"], camera["site_id"], "relay_pair", "camera", camera_id)
    return {"camera_id": camera_id, "pairing_code": code, "expires_at": expires_at}


@app.get("/internal/cameras", dependencies=[Depends(internal)])
def edge_cameras():
    with connect() as db:
        return rows(db, "SELECT * FROM cameras WHERE enabled=1")


@app.post("/cameras/{camera_id}/start")
def start_camera(camera_id: str, user: dict = Depends(current_user)):
    check_role(user, "admin")
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=?", (camera_id,))
    if not camera:
        raise HTTPException(404, "Camera not found")
    check_site(user, camera["site_id"])
    if camera["source"] == "browser" or camera["source"].startswith("relay:"):
        raise HTTPException(422, "This camera is captured on the viewing device or its local relay, not by the server worker")
    running = CAMERA_PROCESSES.get(camera_id)
    if running and running.poll() is None:
        return {"status": "already_running", "pid": running.pid}
    log = (DATA / f"camera-{camera_id}.log").open("a", encoding="utf-8")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    process = subprocess.Popen([sys.executable, "-m", "forgecare.edge", "--camera", camera_id],
                               cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT, **kwargs)
    log.close()
    CAMERA_PROCESSES[camera_id] = process
    with connect() as db:
        db.execute("UPDATE cameras SET stopped_at=NULL,last_frame_at=NULL,last_error='Connecting to camera',last_error_at=?,last_quality=NULL WHERE id=?", (now(), camera_id))
        audit(db, user["id"], camera["site_id"], "start", "camera", camera_id)
    return {"status": "started", "pid": process.pid}


@app.post("/cameras/{camera_id}/stop")
def stop_camera(camera_id: str, user: dict = Depends(current_user)):
    check_role(user, "admin")
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=?", (camera_id,))
    if not camera:
        raise HTTPException(404, "Camera not found")
    check_site(user, camera["site_id"])
    process = CAMERA_PROCESSES.pop(camera_id, None)
    if process and process.poll() is None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False, timeout=10)
        else:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    with connect() as db:
        db.execute("UPDATE cameras SET stopped_at=?,last_frame_at=NULL,last_error='Worker stopped',last_error_at=? WHERE id=?", (now(), now(), camera_id))
        audit(db, user["id"], camera["site_id"], "stop", "camera", camera_id)
    return {"status": "stopped"}


@app.post("/internal/camera-status", dependencies=[Depends(internal)])
def camera_status(camera_id: str, error: str = ""):
    with connect() as db:
        db.execute("UPDATE cameras SET last_error=?,last_error_at=? WHERE id=?", (error[:500], now(), camera_id))
    return {"ok": True}


@app.post("/internal/captures", dependencies=[Depends(internal)])
def ingest_capture(data: CaptureIn):
    try:
        when = datetime.fromisoformat(data.captured_at)
        age = abs((datetime.now(timezone.utc) - when).total_seconds())
    except (ValueError, TypeError):
        raise HTTPException(422, "Invalid timestamp")
    if when.tzinfo is None or age > 10:
        raise HTTPException(422, "Capture is stale or has no timezone")
    try:
        content = base64.b64decode(data.jpeg_b64, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(422, "Invalid image data")
    if len(content) > 2_000_000 or not content.startswith(b"\xff\xd8"):
        raise HTTPException(422, "Expected JPEG under 2 MB")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[0] < 64 or decoded.shape[1] < 64:
        raise HTTPException(422, "JPEG must decode to a frame of at least 64 by 64 pixels")
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=? AND enabled=1", (data.camera_id,))
        if not camera or camera["asset_id"] != data.asset_id or camera["site_id"] != data.site_id:
            raise HTTPException(422, "Camera/asset/site mapping invalid")
        if camera["sample_id"] and camera["sample_id"] != data.sample_id:
            raise HTTPException(422, "Physical sample ID does not match the registered camera session")
        if camera.get("stopped_at") and when >= datetime.fromisoformat(camera["stopped_at"]):
            raise HTTPException(409, "Camera worker is stopped")
        capture_id = str(uuid.uuid4())
        folder = DATA / "images" / data.site_id
        folder.mkdir(exist_ok=True)
        path = folder / f"{capture_id}.jpg"
        path.write_bytes(content)
        db.execute("INSERT INTO captures VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (capture_id, data.site_id, data.asset_id, data.camera_id, data.session_id, data.sample_id,
                    data.captured_at, str(path), data.quality, data.blur, data.brightness, data.roi_version, now()))
        db.execute("UPDATE cameras SET last_frame_at=?,last_error=NULL,last_error_at=NULL,last_quality=? WHERE id=?",
                   (data.captured_at, data.quality, data.camera_id))
        observation_id = str(uuid.uuid4())
        evidence = "accepted" if data.quality == "ok" else "recapture"
        asset = row(db, "SELECT * FROM assets WHERE id=? AND site_id=?", (data.asset_id, data.site_id))
        deployed = row(db, "SELECT * FROM models WHERE site_id=? AND task='visual' AND status='active' AND family=? AND equipment_model IS ? ORDER BY approved_at DESC LIMIT 1",
                       (data.site_id, asset["family"], asset["model"]))
        predicted_class, score = None, None
        if evidence == "accepted" and deployed and deployed["artifact_path"]:
            try:
                predicted_class, score = predict(path, deployed["artifact_path"])
            except Exception as exc:
                evidence = "model_unavailable"
                audit(db, None, data.site_id, "inference_failed", "model", deployed["id"], json.dumps({"error": str(exc)}))
        db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?)", (observation_id, capture_id, deployed["id"] if deployed else None, predicted_class, score, evidence, now()))
        if evidence == "accepted" and predicted_class == "review_needed":
            existing = row(db, "SELECT id FROM cases WHERE asset_id=? AND status IN ('open','assigned','approved')", (data.asset_id,))
            if not existing:
                asset = row(db, "SELECT * FROM assets WHERE id=?", (data.asset_id,))
                rules = active_rules(db, data.site_id)
                repeats = db.execute("SELECT COUNT(*) FROM labels l JOIN captures c ON l.capture_id=c.id WHERE c.asset_id=? AND l.binary_label=1", (data.asset_id,)).fetchone()[0]
                urgency, parts = priority(asset["criticality"], repeats, asset["inspection_interval_days"], asset["last_inspection_at"],
                                          (rules["criticality_weight"], rules["repeat_weight"], rules["overdue_weight"]))
                case_id = str(uuid.uuid4())
                db.execute("INSERT INTO cases(id,site_id,asset_id,observation_id,status,priority,priority_band,reason_json,rule_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (case_id, data.site_id, data.asset_id, observation_id, "open", urgency, band(urgency), json.dumps(parts), rules["id"], now(), now()))
                audit(db, None, data.site_id, "auto_create", "case", case_id)
        audit(db, None, data.site_id, "capture", "capture", capture_id,
              json.dumps({"quality": data.quality, "asset_id": data.asset_id,
                          "camera_id": data.camera_id, "sample_id": data.sample_id}))
        return {"id": capture_id, "observation_id": observation_id, "evidence_status": evidence}


@app.post("/cameras/{camera_id}/browser-captures")
def capture_from_browser(camera_id: str, data: BrowserCaptureIn, user: dict = Depends(current_user)):
    """Accept a photo from the viewing device instead of opening a server webcam."""
    with connect() as db:
        camera = row(db, "SELECT * FROM cameras WHERE id=? AND enabled=1", (camera_id,))
    if not camera:
        raise HTTPException(404, "Camera not found")
    check_site(user, camera["site_id"])
    if camera["source"] != "browser" and not camera["source"].startswith("relay:"):
        raise HTTPException(422, "Set this camera's source to browser or relay before sending photos from another device")
    try:
        raw = base64.b64decode(data.jpeg_b64, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(422, "Invalid photo data")
    if len(raw) > 8_000_000:
        raise HTTPException(422, "Photo is too large; use a lower camera resolution")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or min(frame.shape[:2]) < 64:
        raise HTTPException(422, "Take a JPEG photo at least 64 by 64 pixels")
    height, width = frame.shape[:2]
    if max(height, width) > 1600:
        scale = 1600 / max(height, width)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    crop, quality, blur, brightness = inspect_frame(frame, json.loads(camera["roi_json"]))
    if min(crop.shape[:2]) < 64:
        raise HTTPException(422, "Camera crop is too small; adjust its region")
    encoded = None
    for jpeg_quality in (80, 65, 50):
        success, candidate = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if success and len(candidate) <= 1_900_000:
            encoded = candidate
            break
    if encoded is None:
        raise HTTPException(422, "Photo cannot be prepared under the capture size limit")
    result = ingest_capture(CaptureIn(
        camera_id=camera_id, asset_id=camera["asset_id"], site_id=camera["site_id"],
        session_id=data.session_id, sample_id=camera["sample_id"] or camera["asset_id"],
        captured_at=now(), quality=quality, blur=blur, brightness=brightness,
        jpeg_b64=base64.b64encode(encoded).decode("ascii"),
    ))
    with connect() as db:
        audit(db, user["id"], camera["site_id"], "client_capture", "capture", result["id"],
              json.dumps({"camera_id": camera_id, "quality": quality}))
    return {**result, "quality": quality, "blur": blur, "brightness": brightness}


@app.get("/captures")
def captures(site_id: str, limit: int = 50, quality: Literal["ok", "blurred", "dark", "bright", "invalid"] | None = None,
             user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        query = "SELECT c.id,c.site_id,c.asset_id,c.camera_id,c.session_id,c.sample_id,c.captured_at,c.quality,c.blur,c.brightness,l.grade,l.note AS reviewer_note,l.scratch_count,l.dent_count,o.id AS observation_id,o.evidence_status,o.predicted_class,o.score AS model_score,o.model_id FROM captures c LEFT JOIN labels l ON l.capture_id=c.id JOIN observations o ON o.capture_id=c.id WHERE c.site_id=?"
        params = [site_id]
        if quality is not None:
            query += " AND c.quality=?"
            params.append(quality)
        query += " ORDER BY c.captured_at DESC LIMIT ?"
        params.append(min(max(limit, 1), 200))
        return rows(db, query, tuple(params))


def draft_candidates(db, anchor: dict) -> list[dict]:
    earliest = (datetime.fromisoformat(anchor["captured_at"]) - timedelta(minutes=5)).isoformat()
    return rows(db, """SELECT c.id,c.captured_at,c.quality,c.blur,c.brightness,
                      o.predicted_class,o.score,o.model_id
        FROM captures c JOIN observations o ON o.capture_id=c.id
        WHERE c.site_id=? AND c.asset_id=? AND c.camera_id=? AND c.session_id=?
          AND c.sample_id=? AND c.quality='ok' AND c.captured_at BETWEEN ? AND ?
        ORDER BY c.captured_at DESC LIMIT 100""",
        (anchor["site_id"], anchor["asset_id"], anchor["camera_id"], anchor["session_id"],
         anchor["sample_id"], earliest, anchor["captured_at"]))


@app.get("/captures/{capture_id}/draft-candidates")
def get_draft_candidates(capture_id: str, user: dict = Depends(current_user)):
    with connect() as db:
        anchor = row(db, "SELECT * FROM captures WHERE id=?", (capture_id,))
        if not anchor:
            raise HTTPException(404, "Capture not found")
        check_site(user, anchor["site_id"])
        if anchor["quality"] != "ok":
            return []
        return draft_candidates(db, anchor)


@app.get("/captures/{capture_id}/draft")
def get_review_draft(capture_id: str, user: dict = Depends(current_user)):
    with connect() as db:
        capture = row(db, "SELECT site_id FROM captures WHERE id=?", (capture_id,))
        if not capture:
            raise HTTPException(404, "Capture not found")
        check_site(user, capture["site_id"])
        return row(db, "SELECT * FROM review_drafts WHERE anchor_capture_id=?", (capture_id,))


@app.post("/captures/{capture_id}/draft")
def create_review_draft(capture_id: str, data: DraftIn = DraftIn(), user: dict = Depends(current_user)):
    check_role(user, "technician", "supervisor", "reviewer", "admin")
    with connect() as db:
        anchor = row(db, "SELECT * FROM captures WHERE id=?", (capture_id,))
        if not anchor:
            raise HTTPException(404, "Capture not found")
        check_site(user, anchor["site_id"])
        if anchor["quality"] != "ok":
            raise HTTPException(422, "Choose a usable image to start a multi-photo review")
        existing = row(db, "SELECT * FROM review_drafts WHERE anchor_capture_id=?", (capture_id,))
        if existing:
            return existing
        candidates = draft_candidates(db, anchor)
        by_id = {item["id"]: item for item in candidates}
        if data.photos:
            source_ids = [photo.capture_id for photo in data.photos]
            if len(source_ids) != len(set(source_ids)) or capture_id not in source_ids or any(item not in by_id for item in source_ids):
                raise HTTPException(422, "Select distinct usable photos from this camera, session and physical sample, including the chosen capture")
            nearby = [by_id[item] for item in source_ids]
            views = {photo.capture_id: photo.view for photo in data.photos}
        else:
            nearby = candidates[:5]
            views = {item["id"]: "unassigned" for item in nearby}
        if not nearby:
            raise HTTPException(422, "No usable photos are available for this review")
        predictions = [item for item in nearby if item["model_id"] and item["predicted_class"]]
        model_id = predictions[0]["model_id"] if predictions else None
        if predictions:
            flagged = sum(item["predicted_class"] == "review_needed" for item in predictions)
            model_text = f"An approved general visual model flagged {flagged} of {len(predictions)} analyzed photos for closer review. It does not distinguish scratches from dents."
        else:
            model_text = "No approved visual defect model analyzed these photos. Scratch and dent counts are unavailable."
        named_views = sorted({view for view in views.values() if view != "unassigned"})
        view_text = f"Technician-selected views: {', '.join(named_views)}." if named_views else "Photo views have not been identified."
        coverage_text = " Ask for another angle if a relevant surface is not visible." if len(named_views) < 2 else " Check that each relevant surface is visible before confirming a finding."
        draft_text = (f"Automated draft from {len(nearby)} usable photo{'s' if len(nearby) != 1 else ''} "
                      f"of the same camera, session and physical sample within five minutes. {view_text}{coverage_text} {model_text} "
                      "A technician must check the actual photos, describe only visible findings, and correct this draft before saving. "
                      "A camera photo cannot establish internal or functional condition.")
        db.execute("""INSERT INTO review_drafts(anchor_capture_id,site_id,camera_id,sample_id,source_ids_json,draft_text,model_id,created_at,views_json)
                      VALUES(?,?,?,?,?,?,?,?,?)""",
                   (capture_id, anchor["site_id"], anchor["camera_id"], anchor["sample_id"],
                    json.dumps([item["id"] for item in nearby]), draft_text, model_id, now(), json.dumps(views)))
        audit(db, user["id"], anchor["site_id"], "draft", "capture", capture_id,
              json.dumps({"source_capture_ids": [item["id"] for item in nearby], "views": views, "model_id": model_id}))
        return row(db, "SELECT * FROM review_drafts WHERE anchor_capture_id=?", (capture_id,))


@app.post("/captures/{capture_id}/vision-review")
def generate_vision_review(capture_id: str, user: dict = Depends(current_user)):
    check_role(user, "technician", "supervisor", "reviewer", "admin")
    with connect() as db:
        draft = row(db, "SELECT * FROM review_drafts WHERE anchor_capture_id=?", (capture_id,))
        if not draft:
            raise HTTPException(404, "Choose photos and build a review draft first")
        check_site(user, draft["site_id"])
        capture = row(db, "SELECT asset_id FROM captures WHERE id=?", (capture_id,))
        asset = row(db, "SELECT name,model FROM assets WHERE id=?", (capture["asset_id"],))
        selected_ids = json.loads(draft["source_ids_json"])
        image_ids = [capture_id] + [item for item in selected_ids if item != capture_id]
        image_ids = image_ids[:3]
        views = json.loads(draft["views_json"] or "{}")
        photos = []
        for image_id in image_ids:
            item = row(db, "SELECT image_path,quality,site_id,camera_id,sample_id FROM captures WHERE id=?", (image_id,))
            if not item or item["quality"] != "ok" or item["site_id"] != draft["site_id"] or item["camera_id"] != draft["camera_id"] or item["sample_id"] != draft["sample_id"]:
                raise HTTPException(422, "Selected photo set is no longer valid")
            photos.append((Path(item["image_path"]), views.get(image_id, "unassigned")))
    try:
        generated = describe_selected_photos(photos, asset["model"] or asset["name"])
    except GroqUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(422, f"Could not read selected photos: {type(exc).__name__}") from exc
    visual_text = "AI visual draft — verify against the photos: " + generated["summary"].strip()
    if generated["observations"]:
        visual_text += "\n\nVisible observations:\n" + "\n".join(f"- {item}" for item in generated["observations"])
    if generated["count_basis"]:
        visual_text += "\n\nCount basis: " + generated["count_basis"]
    if generated["uncertainty"]:
        visual_text += "\n\nUncertainty: " + generated["uncertainty"]
    if generated["next_view"]:
        visual_text += "\n\nNext useful view: " + generated["next_view"]
    with connect() as db:
        db.execute("""UPDATE review_drafts SET ai_text=?,ai_model=?,ai_image_ids_json=?,
                   ai_scratch_count=?,ai_dent_count=?,ai_count_basis=? WHERE anchor_capture_id=?""",
                   (visual_text, VISION_MODEL, json.dumps(image_ids), generated["scratch_count"],
                    generated["dent_count"], generated["count_basis"], capture_id))
        audit(db, user["id"], draft["site_id"], "vision_draft", "capture", capture_id,
              json.dumps({"image_ids": image_ids, "model": VISION_MODEL,
                          "proposed_scratch_count": generated["scratch_count"],
                          "proposed_dent_count": generated["dent_count"]}))
        return row(db, "SELECT * FROM review_drafts WHERE anchor_capture_id=?", (capture_id,))


@app.get("/captures/{capture_id}/image")
def capture_image(capture_id: str, user: dict = Depends(current_user)):
    with connect() as db:
        capture = row(db, "SELECT * FROM captures WHERE id=?", (capture_id,))
    if not capture:
        raise HTTPException(404, "Capture not found")
    check_site(user, capture["site_id"])
    return FileResponse(Path(capture["image_path"]), media_type="image/jpeg")


@app.post("/captures/{capture_id}/label")
def label_capture(capture_id: str, data: LabelIn, user: dict = Depends(current_user)):
    check_role(user, "technician", "supervisor", "reviewer", "admin")
    with connect() as db:
        capture = row(db, "SELECT * FROM captures WHERE id=?", (capture_id,))
        if not capture:
            raise HTTPException(404, "Capture not found")
        check_site(user, capture["site_id"])
        if capture["quality"] != "ok" and data.grade != "unclear":
            raise HTTPException(422, "Poor-quality capture must be marked unclear")
        if data.grade == "unclear" and (data.scratch_count is not None or data.dent_count is not None):
            raise HTTPException(422, "Do not count defects in an unclear image")
        if data.grade == "none" and (data.scratch_count or data.dent_count):
            raise HTTPException(422, "A no-issue label cannot have positive defect counts")
        binary = None if data.grade == "unclear" else (0 if data.grade == "none" else 1)
        db.execute("""INSERT INTO labels(capture_id,grade,binary_label,reviewer_id,reviewed_at,note,scratch_count,dent_count)
                      VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(capture_id) DO UPDATE SET grade=excluded.grade,
                      binary_label=excluded.binary_label,reviewer_id=excluded.reviewer_id,reviewed_at=excluded.reviewed_at,
                      note=excluded.note,scratch_count=excluded.scratch_count,dent_count=excluded.dent_count""",
                   (capture_id, data.grade, binary, user["id"], now(), data.note, data.scratch_count, data.dent_count))
        audit(db, user["id"], capture["site_id"], "label", "capture", capture_id,
              json.dumps({"grade": data.grade, "scratch_count": data.scratch_count,
                          "dent_count": data.dent_count, "review_note": data.note}))
    return {"capture_id": capture_id, "grade": data.grade, "binary_label": binary}


@app.get("/rules")
def get_rules(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return active_rules(db, site_id)


@app.post("/rules")
def set_rules(data: RuleIn, user: dict = Depends(current_user)):
    check_role(user, "supervisor", "admin")
    check_site(user, data.site_id)
    if abs(data.criticality_weight + data.repeat_weight + data.overdue_weight - 1) > 0.000001:
        raise HTTPException(422, "Weights must sum to one")
    with connect() as db:
        cur = db.execute("INSERT INTO rules(site_id,criticality_weight,repeat_weight,overdue_weight,default_minutes,created_at,created_by) VALUES(?,?,?,?,?,?,?)",
                         (data.site_id, data.criticality_weight, data.repeat_weight, data.overdue_weight, data.default_minutes, now(), user["id"]))
        audit(db, user["id"], data.site_id, "create", "rule", str(cur.lastrowid))
    return {"id": cur.lastrowid}


@app.post("/cases")
def create_case(data: CaseIn, user: dict = Depends(current_user)):
    with connect() as db:
        capture = row(db, "SELECT * FROM captures WHERE id=?", (data.capture_id,))
        if not capture:
            raise HTTPException(404, "Capture not found")
        check_site(user, capture["site_id"])
        if capture["quality"] != "ok":
            raise HTTPException(422, "Recapture required before case creation")
        reviewed = row(db, "SELECT grade FROM labels WHERE capture_id=?", (data.capture_id,))
        if not reviewed or reviewed["grade"] == "unclear":
            raise HTTPException(422, "Save a clear technician photo review before opening an inspection case")
        observation = row(db, "SELECT id FROM observations WHERE capture_id=?", (data.capture_id,))
        same_capture = row(db, "SELECT * FROM cases WHERE observation_id=? ORDER BY created_at DESC LIMIT 1", (observation["id"],))
        if same_capture:
            audit(db, user["id"], capture["site_id"], "duplicate_link", "case", same_capture["id"], json.dumps({"capture_id": data.capture_id}))
            return {**same_capture, "duplicate_linked": True}
        asset = row(db, "SELECT * FROM assets WHERE id=?", (capture["asset_id"],))
        rules = active_rules(db, capture["site_id"])
        repeats = db.execute("SELECT COUNT(*) FROM labels l JOIN captures c ON l.capture_id=c.id WHERE c.asset_id=? AND l.binary_label=1", (asset["id"],)).fetchone()[0]
        score, parts = priority(asset["criticality"], repeats, asset["inspection_interval_days"], asset["last_inspection_at"],
                                (rules["criticality_weight"], rules["repeat_weight"], rules["overdue_weight"]))
        case_id = str(uuid.uuid4())
        db.execute("INSERT INTO cases(id,site_id,asset_id,observation_id,status,priority,priority_band,reason_json,rule_id,created_at,updated_at,decision_note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (case_id, capture["site_id"], asset["id"], observation["id"], "open", score, band(score), json.dumps(parts), rules["id"], now(), now(), data.note))
        audit(db, user["id"], capture["site_id"], "create", "case", case_id,
              json.dumps({"capture_id": data.capture_id, "asset_id": asset["id"],
                          "reviewed_grade": reviewed["grade"], "attention": band(score)}))
        return row(db, "SELECT * FROM cases WHERE id=?", (case_id,))


@app.get("/cases")
def list_cases(site_id: str, status: str | None = None, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        if status:
            return rows(db, "SELECT * FROM cases WHERE site_id=? AND status=? ORDER BY priority DESC,created_at", (site_id, status))
        return rows(db, "SELECT * FROM cases WHERE site_id=? ORDER BY priority DESC,created_at", (site_id,))


@app.get("/cases/review-ready")
def reviewed_captures_without_case(site_id: str, user: dict = Depends(current_user)):
    """Saved clear reviews still awaiting a distinct, linked work order."""
    check_site(user, site_id)
    with connect() as db:
        return rows(db, """SELECT c.id AS capture_id,c.asset_id,c.sample_id,c.captured_at,
                      l.reviewed_at,l.grade,l.note AS reviewer_note,l.scratch_count,l.dent_count
                      FROM labels l JOIN captures c ON c.id=l.capture_id
                      JOIN observations o ON o.capture_id=c.id
                      LEFT JOIN cases cs ON cs.observation_id=o.id
                      WHERE c.site_id=? AND c.quality='ok' AND l.grade IN ('none','minor','significant')
                        AND cs.id IS NULL
                      ORDER BY l.reviewed_at DESC LIMIT 100""", (site_id,))


@app.get("/cases/selected")
def selected_cases(site_id: str, capacity_minutes: int = 60, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        rules = active_rules(db, site_id)
        open_cases = rows(db, "SELECT * FROM cases WHERE site_id=? AND status='open'", (site_id,))
        for item in open_cases:
            item["estimated_minutes"] = rules["default_minutes"]
        return fit_capacity(open_cases, capacity_minutes, rules["default_minutes"])


@app.get("/cases/{case_id}")
def get_case(case_id: str, user: dict = Depends(current_user)):
    with connect() as db:
        case = row(db, "SELECT * FROM cases WHERE id=?", (case_id,))
        if not case:
            raise HTTPException(404, "Case not found")
        check_site(user, case["site_id"])
        case["asset"] = row(db, "SELECT id,name,model,criticality,inspection_interval_days,last_inspection_at FROM assets WHERE id=?", (case["asset_id"],))
        case["capture"] = row(db, """SELECT c.*,l.grade AS reviewed_grade,l.note AS reviewer_note,
                              l.reviewed_at,l.scratch_count,l.dent_count,u.username AS reviewer_name
                              FROM captures c JOIN observations o ON o.capture_id=c.id
                              LEFT JOIN labels l ON l.capture_id=c.id
                              LEFT JOIN users u ON u.id=l.reviewer_id WHERE o.id=?""", (case["observation_id"],))
        case["review_draft"] = row(db, "SELECT * FROM review_drafts WHERE anchor_capture_id=?", (case["capture"]["id"],)) if case["capture"] else None
        case["assignee_name"] = row(db, "SELECT username FROM users WHERE id=?", (case["assigned_to"],)) if case["assigned_to"] else None
        return case


@app.post("/cases/{case_id}/assign")
def assign_case(case_id: str, data: AssignIn, user: dict = Depends(current_user)):
    check_role(user, "supervisor", "admin")
    with connect() as db:
        case = row(db, "SELECT * FROM cases WHERE id=?", (case_id,))
        if not case or case["status"] != "open":
            raise HTTPException(409, "Case is not open")
        check_site(user, case["site_id"])
        assignee = row(db, "SELECT * FROM users WHERE id=? AND site_id=? AND active=1", (data.user_id, case["site_id"]))
        if not assignee:
            raise HTTPException(422, "Assignee must belong to site")
        db.execute("UPDATE cases SET assigned_to=?,status='assigned',override_reason=?,updated_at=? WHERE id=?",
                   (data.user_id, data.override_reason, now(), case_id))
        audit(db, user["id"], case["site_id"], "assign", "case", case_id, json.dumps(data.model_dump()))
    return {"status": "assigned"}


@app.post("/cases/{case_id}/approve")
def approve_case(case_id: str, data: ApproveIn, user: dict = Depends(current_user)):
    check_role(user, "supervisor", "admin")
    with connect() as db:
        case = row(db, "SELECT * FROM cases WHERE id=?", (case_id,))
        if not case or case["status"] not in ("open", "assigned"):
            raise HTTPException(409, "Case cannot be approved")
        check_site(user, case["site_id"])
        db.execute("UPDATE cases SET status='approved',decision_note=?,override_reason=?,approved_by=?,approved_at=?,updated_at=? WHERE id=?",
                   (data.decision_note, data.override_reason, user["id"], now(), now(), case_id))
        audit(db, user["id"], case["site_id"], "approve", "case", case_id,
              json.dumps({"decision_note": data.decision_note, "override_reason": data.override_reason}))
    return {"status": "approved"}


@app.post("/cases/{case_id}/close")
def close_case(case_id: str, data: CloseIn, user: dict = Depends(current_user)):
    with connect() as db:
        case = row(db, "SELECT * FROM cases WHERE id=?", (case_id,))
        if not case or case["status"] != "approved":
            raise HTTPException(409, "Approved task required before closure")
        check_site(user, case["site_id"])
        db.execute("UPDATE cases SET status='closed',actual_finding=?,actual_minutes=?,closure_note=?,inspection_performed=?,updated_at=? WHERE id=?",
                   (data.actual_finding, data.actual_minutes, data.note, int(data.inspection_performed), now(), case_id))
        if data.inspection_performed:
            db.execute("UPDATE assets SET last_inspection_at=? WHERE id=?", (now(), case["asset_id"]))
        audit(db, user["id"], case["site_id"], "close", "case", case_id, json.dumps(data.model_dump()))
    return {"status": "closed"}


@app.get("/audit")
def audit_list(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return rows(db, "SELECT * FROM audit WHERE site_id=? ORDER BY id DESC LIMIT 200", (site_id,))


@app.get("/audit/global")
def audit_global(user: dict = Depends(current_user)):
    check_role(user, "admin")
    with connect() as db:
        return rows(db, "SELECT * FROM audit WHERE site_id IS NULL ORDER BY id DESC LIMIT 200")


@app.get("/dashboard")
def dashboard(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return {"assets": db.execute("SELECT COUNT(*) FROM assets WHERE site_id=?", (site_id,)).fetchone()[0],
                "cameras": db.execute("SELECT COUNT(*) FROM cameras WHERE site_id=?", (site_id,)).fetchone()[0],
                "captures": db.execute("SELECT COUNT(*) FROM captures WHERE site_id=?", (site_id,)).fetchone()[0],
                "open_cases": db.execute("SELECT COUNT(*) FROM cases WHERE site_id=? AND status='open'", (site_id,)).fetchone()[0],
                "approved_cases": db.execute("SELECT COUNT(*) FROM cases WHERE site_id=? AND status='approved'", (site_id,)).fetchone()[0],
                "reviewed_labels": db.execute("SELECT COUNT(*) FROM labels l JOIN captures c ON c.id=l.capture_id WHERE c.site_id=?", (site_id,)).fetchone()[0]}


@app.get("/monitoring")
def monitoring(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        recent = rows(db, "SELECT quality FROM captures WHERE site_id=? ORDER BY captured_at DESC LIMIT 100", (site_id,))
        reviewed = rows(db, """SELECT l.binary_label,o.predicted_class,o.model_id FROM labels l
          JOIN captures c ON c.id=l.capture_id JOIN observations o ON o.capture_id=c.id
          WHERE c.site_id=? AND l.binary_label IS NOT NULL AND o.predicted_class IS NOT NULL
          ORDER BY c.captured_at DESC LIMIT 100""", (site_id,))
    midpoint = len(recent) // 2
    newest = recent[:midpoint]
    older = recent[midpoint:]
    ok_rate = lambda group: round(sum(x["quality"] == "ok" for x in group) / len(group), 4) if group else None
    newer_rate, older_rate = ok_rate(newest), ok_rate(older)
    quality_shift = len(newest) >= 20 and len(older) >= 20 and abs(newer_rate - older_rate) >= .2
    same_model = len({x["model_id"] for x in reviewed}) == 1
    performance = None
    if len(reviewed) >= 10 and same_model:
        tp = sum(x["binary_label"] == 1 and x["predicted_class"] == "review_needed" for x in reviewed)
        fn = sum(x["binary_label"] == 1 and x["predicted_class"] != "review_needed" for x in reviewed)
        fp = sum(x["binary_label"] == 0 and x["predicted_class"] == "review_needed" for x in reviewed)
        tn = sum(x["binary_label"] == 0 and x["predicted_class"] != "review_needed" for x in reviewed)
        performance = {"model_id": reviewed[0]["model_id"], "reviewed_count": len(reviewed),
                       "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                       "false_alert_rate": round(fp / (fp + tn), 4) if fp + tn else None}
    return {"recent_captures": len(recent), "quality_ok_newer": newer_rate,
            "quality_ok_older": older_rate, "quality_shift_review_signal": quality_shift,
            "reviewed_visual_performance": performance,
            "note": "Shift is a review signal, not proof of model failure. Mixed model versions are excluded from performance."}


@app.get("/models")
def list_models(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return rows(db, "SELECT * FROM models WHERE site_id=? ORDER BY created_at DESC", (site_id,))


@app.post("/models/train")
def start_training(data: TrainIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    if not data.family or not data.model:
        raise HTTPException(422, "Choose an exact family and model for visual training")
    model_id = str(uuid.uuid4())
    with connect() as db:
        if not row(db, "SELECT id FROM assets WHERE site_id=? AND family=? AND model=?", (data.site_id, data.family, data.model)):
            raise HTTPException(422, "No registered asset matches this family and model")
        db.execute("INSERT INTO models(id,site_id,task,status,family,equipment_model,config_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                   (model_id, data.site_id, "visual", "training", data.family, data.model,
                    json.dumps({"feature_version": "gray16-hist16-v1", "seed": 42}), "{}", now()))
        audit(db, user["id"], data.site_id, "train_start", "model", model_id)
    threading.Thread(target=train_visual, args=(data.site_id, model_id, data.family, data.model), daemon=True).start()
    return {"id": model_id, "status": "training"}


@app.post("/models/{model_id}/approve")
def approve_model(model_id: str, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    with connect() as db:
        model = row(db, "SELECT * FROM models WHERE id=?", (model_id,))
        if not model or model["status"] != "candidate":
            raise HTTPException(409, "Only a candidate that passed its gate can be approved")
        check_site(user, model["site_id"])
        db.execute("UPDATE models SET status='retired' WHERE site_id=? AND task=? AND family IS ? AND equipment_model IS ? AND status='active'",
                   (model["site_id"], model["task"], model["family"], model["equipment_model"]))
        db.execute("UPDATE models SET status='active',approved_by=?,approved_at=? WHERE id=?", (user["id"], now(), model_id))
        audit(db, user["id"], model["site_id"], "approve", "model", model_id)
    return {"id": model_id, "status": "active"}


@app.post("/models/{model_id}/rollback")
def rollback_model(model_id: str, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    with connect() as db:
        model = row(db, "SELECT * FROM models WHERE id=?", (model_id,))
        if not model or model["status"] != "retired" or not model["approved_at"]:
            raise HTTPException(409, "Only a previously approved model can be restored")
        check_site(user, model["site_id"])
        db.execute("UPDATE models SET status='retired' WHERE site_id=? AND task=? AND family IS ? AND equipment_model IS ? AND status='active'",
                   (model["site_id"], model["task"], model["family"], model["equipment_model"]))
        db.execute("UPDATE models SET status='active',approved_by=?,approved_at=? WHERE id=?", (user["id"], now(), model_id))
        audit(db, user["id"], model["site_id"], "rollback", "model", model_id)
    return {"id": model_id, "status": "active"}


@app.get("/documents")
def documents(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    try:
        res = requests.get(f"{KNOW_URL}/documents", params={"site_id": site_id}, headers={"X-Service-Key": SERVICE_KEY}, timeout=10)
        res.raise_for_status()
        return res.json()
    except requests.RequestException as exc:
        raise HTTPException(503, f"Knowledge service unavailable: {exc}")


@app.post("/documents")
def add_document(data: DocumentIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    try:
        res = requests.post(f"{KNOW_URL}/documents", json=data.model_dump(), headers={"X-Service-Key": SERVICE_KEY}, timeout=40)
        res.raise_for_status()
        result = res.json()
    except requests.RequestException as exc:
        raise HTTPException(503, f"Document import failed: {exc}")
    with connect() as db:
        audit(db, user["id"], data.site_id, "index", "document", result["id"], json.dumps({"url": data.url}))
    return result


@app.post("/documents/{document_id}/deactivate")
def deactivate_document(document_id: str, site_id: str, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, site_id)
    available = documents(site_id, user)
    if not any(x["id"] == document_id for x in available):
        raise HTTPException(404, "Document not found at site")
    try:
        response = requests.post(f"{KNOW_URL}/documents/{document_id}/deactivate",
                                 headers={"X-Service-Key": SERVICE_KEY}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(503, f"Document deactivation failed: {exc}")
    with connect() as db:
        audit(db, user["id"], site_id, "deactivate", "document", document_id)
    return response.json()


@app.post("/assistant/ask")
def ask(data: AskIn, user: dict = Depends(current_user)):
    check_site(user, data.site_id)
    with connect() as db:
        asset = row(db, "SELECT * FROM assets WHERE id=? AND site_id=?", (data.asset_id, data.site_id))
        if asset and data.scope in ("auto", "phone"):
            question_family = ("iphone" if re.search(r"\biphone\b", data.question, re.I) else
                               "realme" if re.search(r"\b(realme|narzo)\b", data.question, re.I) else None)
            if question_family and not asset["family"].startswith(question_family):
                candidates = rows(db, "SELECT * FROM assets WHERE site_id=? AND family LIKE ?",
                                  (data.site_id, question_family + "%"))
                if len(candidates) == 1:
                    asset = candidates[0]
    if not asset:
        raise HTTPException(404, "Asset not found")
    try:
        response = requests.post(f"{KNOW_URL}/ask", json={"site_id": data.site_id, "family": asset["family"], "model": asset["model"],
                                 "question": data.question, "history": data.history, "scope": data.scope},
                                 headers={"X-Service-Key": SERVICE_KEY}, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(503, f"Knowledge service unavailable: {exc}")
    with connect() as db:
        audit(db, user["id"], data.site_id, "ask", "asset", asset["id"], json.dumps({"question": data.question[:300], "scope": data.scope}))
    result = response.json()
    result["asset_id_used"] = asset["id"]
    result["asset_model_used"] = asset["model"] or asset["name"]
    return result


@app.get("/experiments")
def list_experiments(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return rows(db, "SELECT * FROM experiments WHERE site_id=? ORDER BY started_at DESC", (site_id,))


@app.post("/experiments/run")
def start_experiments(data: TrainIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    tasks = {"classification": "sklearn_digits_research", "regression": "synthetic_learning_fixture",
             "unsupervised": "sklearn_digits_research", "time_series": "synthetic_learning_fixture",
             "deep_learning": "sklearn_digits_and_synthetic", "optimization": "sklearn_digits_research",
             "rag_retrieval": "human_reviewed_site_gold_questions"}
    ids = {task: str(uuid.uuid4()) for task in tasks}
    with connect() as db:
        for task, dataset in tasks.items():
            db.execute("INSERT INTO experiments(id,site_id,task,status,dataset,config_json,results_json,started_at) VALUES(?,?,?,?,?,?,?,?)",
                       (ids[task], data.site_id, task, "running", dataset, json.dumps({"seed": 42}), "{}", now()))
        audit(db, user["id"], data.site_id, "experiment_start", "suite", ids["classification"])
    threading.Thread(target=run_suite, args=(data.site_id, ids), daemon=True).start()
    return {"status": "running", "ids": ids}


@app.get("/federation")
def list_federation(site_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        return rows(db, "SELECT * FROM federation_rounds WHERE site_id=? ORDER BY created_at DESC", (site_id,))


@app.get("/federation/readiness")
def federation_readiness(site_id: str, asset_id: str, user: dict = Depends(current_user)):
    check_site(user, site_id)
    with connect() as db:
        asset = row(db, "SELECT id,name,model FROM assets WHERE id=? AND site_id=?", (asset_id, site_id))
        if not asset:
            raise HTTPException(404, "Asset not found at this site")
        total, usable = db.execute("SELECT COUNT(*),SUM(quality='ok') FROM captures WHERE asset_id=? AND site_id=?",
                                   (asset_id, site_id)).fetchone()
    _, counts = phone_training_rows(site_id, asset_id)
    ready = (counts["usable_labeled_photos"] >= 24 and
             counts["distinct_acceptable_phones"] >= 6 and counts["distinct_damaged_phones"] >= 6)
    return {"asset": asset["model"] or asset["name"], "captures": total,
            "usable_captures": usable or 0, "reviewed_usable_photos": counts["reviewed_photos"],
            "eligible_training_photos": counts["usable_labeled_photos"],
            "reviewed_acceptable_samples": counts["distinct_acceptable_phones"],
            "reviewed_defective_samples": counts["distinct_damaged_phones"],
            "conflicting_samples": counts["conflicting_phone_ids"],
            "ready_for_two_clients": ready,
            "ready_for_scheduled_training": ready,
            "minimum_reviewed_photos": 24,
            "minimum_distinct_phones_per_class": 6,
            "minimum_per_client_per_class": 3}


@app.post("/federation/run")
def start_federation(data: TrainIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    round_id = str(uuid.uuid4())
    with connect() as db:
        db.execute("INSERT INTO federation_rounds(id,site_id,schema_hash,status,participating_json,results_json,created_at) VALUES(?,?,?,?,?,?,?)",
                   (round_id, data.site_id, "pending", "running", "[]", "{}", now()))
        audit(db, user["id"], data.site_id, "fedavg_start", "federation_round", round_id)
    threading.Thread(target=run_federation, args=(data.site_id, round_id), daemon=True).start()
    return {"id": round_id, "status": "running"}


@app.post("/federation/rounds")
def create_real_round(data: TrainIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    if not data.family or not data.model:
        raise HTTPException(422, "Choose an exact family and model for this federated round")
    round_id = str(uuid.uuid4())
    schema_hash = visual_schema_hash()
    with connect() as db:
        if not row(db, "SELECT id FROM assets WHERE site_id=? AND family=? AND model=?", (data.site_id, data.family, data.model)):
            raise HTTPException(422, "No registered asset matches this family and model")
        db.execute("INSERT INTO federation_rounds(id,site_id,schema_hash,status,family,equipment_model,participating_json,results_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                   (round_id, data.site_id, schema_hash, "collecting", data.family, data.model, "[]",
                    json.dumps({"feature_version": "gray16-hist16-v1", "initial_weights": [0.0] * (VISUAL_FEATURE_COUNT + 1),
                                "dataset": "client_local_reviewed_images", "gate": {"recall_min": .7, "false_alert_rate_max": .2, "latency_ms_max": 500}}), now()))
        audit(db, user["id"], data.site_id, "fed_round_create", "federation_round", round_id)
    return {"id": round_id, "status": "collecting", "schema_hash": schema_hash,
            "initial_weights": [0.0] * (VISUAL_FEATURE_COUNT + 1)}


@app.get("/federation/rounds/{round_id}")
def get_real_round(round_id: str, user: dict = Depends(current_user)):
    with connect() as db:
        round_data = row(db, "SELECT * FROM federation_rounds WHERE id=?", (round_id,))
        if not round_data:
            raise HTTPException(404, "Round not found")
        check_site(user, round_data["site_id"])
        round_data["updates"] = rows(db, "SELECT client_id,sample_count,submitted_at FROM federation_updates WHERE round_id=?", (round_id,))
        round_data["validations"] = rows(db, "SELECT * FROM federation_validations WHERE round_id=?", (round_id,))
    return round_data


@app.post("/federation/rounds/{round_id}/updates")
def submit_real_update(round_id: str, data: FedUpdateIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    weights = np.asarray(data.weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise HTTPException(422, "Non-finite weights rejected")
    with connect() as db:
        round_data = row(db, "SELECT * FROM federation_rounds WHERE id=? AND site_id=?", (round_id, data.site_id))
        if not round_data or round_data["status"] != "collecting":
            raise HTTPException(409, "Round is not collecting updates")
        if data.schema_hash != round_data["schema_hash"]:
            raise HTTPException(422, "Feature schema mismatch")
        if row(db, "SELECT client_id FROM federation_updates WHERE round_id=? AND client_id=?", (round_id, data.client_id)):
            raise HTTPException(409, "Duplicate client update")
        db.execute("INSERT INTO federation_updates VALUES(?,?,?,?,?,?)",
                   (round_id, data.client_id, data.schema_hash, data.sample_count, json.dumps(data.weights), now()))
        audit(db, user["id"], data.site_id, "fed_update", "federation_round", round_id,
              json.dumps({"client_id": data.client_id, "sample_count": data.sample_count}))
    return {"status": "accepted", "client_id": data.client_id}


@app.post("/federation/rounds/{round_id}/aggregate")
def aggregate_real_round(round_id: str, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    with connect() as db:
        round_data = row(db, "SELECT * FROM federation_rounds WHERE id=?", (round_id,))
        if not round_data or round_data["status"] != "collecting":
            raise HTTPException(409, "Round is not collecting updates")
        check_site(user, round_data["site_id"])
        updates = rows(db, "SELECT * FROM federation_updates WHERE round_id=?", (round_id,))
        if len(updates) < 2:
            raise HTTPException(409, "At least two distinct local clients are required")
        payloads = [{"round_id": round_id, "schema_hash": x["schema_hash"], "client_id": x["client_id"],
                     "count": x["sample_count"], "weights": json.loads(x["weights_json"])} for x in updates]
        combined = aggregate(payloads, round_id=round_id, schema_hash=round_data["schema_hash"],
                             start=np.zeros(VISUAL_FEATURE_COUNT + 1))
        result = json.loads(round_data["results_json"])
        result["aggregated_weights"] = combined.tolist()
        result["client_counts"] = {x["client_id"]: x["sample_count"] for x in updates}
        db.execute("UPDATE federation_rounds SET status='awaiting_validation',participating_json=?,results_json=? WHERE id=?",
                   (json.dumps([x["client_id"] for x in updates]), json.dumps(result), round_id))
        audit(db, user["id"], round_data["site_id"], "fed_aggregate", "federation_round", round_id)
    return {"id": round_id, "status": "awaiting_validation", "weights": combined.tolist()}


@app.post("/federation/rounds/{round_id}/validations")
def validate_real_round(round_id: str, data: FedValidationIn, user: dict = Depends(current_user)):
    check_role(user, "reviewer", "admin")
    check_site(user, data.site_id)
    with connect() as db:
        round_data = row(db, "SELECT * FROM federation_rounds WHERE id=? AND site_id=?", (round_id, data.site_id))
        if not round_data or round_data["status"] != "awaiting_validation":
            raise HTTPException(409, "Round is not awaiting validation")
        update = row(db, "SELECT * FROM federation_updates WHERE round_id=? AND client_id=?", (round_id, data.client_id))
        if not update:
            raise HTTPException(422, "Client has not submitted a training update")
        if row(db, "SELECT client_id FROM federation_validations WHERE round_id=? AND client_id=?", (round_id, data.client_id)):
            raise HTTPException(409, "Duplicate validation")
        db.execute("INSERT INTO federation_validations(round_id,client_id,sample_count,positive_count,recall,false_alert_rate,latency_ms,local_recall,local_false_alert_rate,submitted_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (round_id, data.client_id, data.sample_count, data.positive_count, data.recall,
                    data.false_alert_rate, data.latency_ms, data.local_recall, data.local_false_alert_rate, now()))
        audit(db, user["id"], data.site_id, "fed_validate", "federation_round", round_id, json.dumps(data.model_dump()))
        validations = rows(db, "SELECT * FROM federation_validations WHERE round_id=?", (round_id,))
        count = db.execute("SELECT COUNT(*) FROM federation_updates WHERE round_id=?", (round_id,)).fetchone()[0]
        if len(validations) < count:
            return {"status": "awaiting_validation", "received": len(validations), "expected": count}
        passed = all(x["recall"] >= .7 and x["false_alert_rate"] <= .2 and x["latency_ms"] <= 500 for x in validations)
        result = json.loads(round_data["results_json"])
        result["validations"] = validations
        result["gate_passed"] = passed
        model_id = None
        if passed:
            model_id = str(uuid.uuid4())
            artifact = DATA / "models" / f"{model_id}.joblib"
            joblib.dump({"weights": result["aggregated_weights"], "threshold": .5,
                         "feature_version": "gray16-hist16-v1"}, artifact)
            db.execute("INSERT INTO models(id,site_id,task,status,family,equipment_model,artifact_path,dataset_hash,config_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (model_id, data.site_id, "visual", "candidate", round_data["family"], round_data["equipment_model"], str(artifact), round_data["schema_hash"],
                        json.dumps({"method": "FedAvg", "round_id": round_id}), json.dumps(result), now()))
        db.execute("UPDATE federation_rounds SET status=?,results_json=?,candidate_model_id=? WHERE id=?",
                   ("candidate_review" if passed else "complete_rejected", json.dumps(result), model_id, round_id))
        audit(db, user["id"], data.site_id, "fed_gate", "federation_round", round_id,
              json.dumps({"passed": passed, "candidate_model_id": model_id}))
    return {"status": "candidate_review" if passed else "complete_rejected", "candidate_model_id": model_id}


@app.post("/assistant/prepare")
def prepare_task(data: AgentRequest, user: dict = Depends(current_user)):
    check_site(user, data.site_id)
    with connect() as db:
        asset = row(db, "SELECT * FROM assets WHERE id=? AND site_id=?", (data.asset_id, data.site_id))
        if not asset:
            raise HTTPException(404, "Asset not found")
        case = row(db, "SELECT * FROM cases WHERE id=? AND site_id=? AND asset_id=?",
                   (data.case_id, data.site_id, data.asset_id)) if data.case_id else None
        if data.case_id and not case:
            raise HTTPException(404, "Case not found")
        history = rows(db, "SELECT actual_finding,actual_minutes,updated_at FROM cases WHERE asset_id=? AND status='closed' ORDER BY updated_at DESC LIMIT 10", (data.asset_id,))
        queue = rows(db, "SELECT id,asset_id,priority,status FROM cases WHERE site_id=? AND status IN ('open','assigned') ORDER BY priority DESC LIMIT 10", (data.site_id,))
    context = {"site_id": data.site_id, "asset": asset, "case": case, "history": history, "queue": queue}
    try:
        response = requests.post(f"{KNOW_URL}/agent/prepare", json={"context": context, "request": data.request},
                                 headers={"X-Service-Key": SERVICE_KEY}, timeout=240)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(503, f"Agent service unavailable: {exc}")
    result = response.json()
    with connect() as db:
        audit(db, user["id"], data.site_id, "agent_prepare", "asset", data.asset_id,
              json.dumps({"case_id": data.case_id, "status": result.get("status"), "trace": result.get("trace", [])}))
    return result
