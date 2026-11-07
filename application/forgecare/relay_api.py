"""Narrow, paired HTTPS entrypoint for a same-LAN IP Webcam bridge."""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .db import connect, migrate, now, row
from .ops_api import BrowserCaptureIn, capture_from_browser


@asynccontextmanager
async def lifespan(_app: FastAPI):
    migrate()
    yield


app = FastAPI(title="ForgeCare Camera Relay", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


def paired_camera(camera_id: str, code: str | None):
    if not code or len(code) > 100:
        raise HTTPException(401, "Pairing code required")
    digest = hashlib.sha256(code.encode()).hexdigest()
    with connect() as db:
        pair = row(db, """SELECT p.camera_id,p.created_by,p.expires_at,c.site_id,c.source,c.enabled,
                         u.username,u.role,u.active,u.site_id AS user_site
                         FROM relay_pairs p JOIN cameras c ON c.id=p.camera_id
                         JOIN users u ON u.id=p.created_by
                         WHERE p.token_hash=? AND p.camera_id=?""", (digest, camera_id))
    if (not pair or pair["expires_at"] <= now() or not pair["enabled"] or not pair["active"]
            or not pair["source"].startswith("relay:")):
        raise HTTPException(401, "Pairing code expired or camera mapping changed")
    if pair["role"] != "admin" or (pair["user_site"] and pair["user_site"] != pair["site_id"]):
        raise HTTPException(403, "Camera is outside this account's site")
    return pair


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/relay/{camera_id}/config")
def camera_config(camera_id: str, x_relay_code: str | None = Header(default=None)):
    camera = paired_camera(camera_id, x_relay_code)
    return {"camera_id": camera_id, "video_url": camera["source"].removeprefix("relay:"),
            "expires_at": camera["expires_at"]}


class RelayStatusIn(BaseModel):
    error: str = Field(default="", max_length=240)


@app.post("/relay/{camera_id}/status")
def relay_status(camera_id: str, data: RelayStatusIn, x_relay_code: str | None = Header(default=None)):
    paired_camera(camera_id, x_relay_code)
    with connect() as db:
        db.execute("UPDATE cameras SET last_error=?,last_error_at=? WHERE id=?",
                   (data.error or None, now() if data.error else None, camera_id))
    return {"ok": True}


@app.post("/relay/{camera_id}/captures")
def relay_capture(camera_id: str, data: BrowserCaptureIn, x_relay_code: str | None = Header(default=None)):
    pair = paired_camera(camera_id, x_relay_code)
    user = {"id": pair["created_by"], "username": pair["username"], "role": pair["role"],
            "site_id": pair["user_site"]}
    return capture_from_browser(camera_id, data, user)
