"""Real camera capture worker. No file or prerecorded-video source is accepted."""
from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from .config import OPS_URL, SERVICE_KEY


def camera_source(value: str):
    if value.isdecimal():
        return int(value)
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https", "rtsp") and parsed.hostname and not parsed.path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return value
    raise ValueError("Live camera source must be a device index or network MJPEG/RTSP URL")


def inspect_frame(frame: np.ndarray, roi: list[float]) -> tuple[np.ndarray, str, float, float]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    crop = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    if crop.size == 0:
        return frame, "invalid", 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    quality = "dark" if brightness < 35 else "bright" if brightness > 225 else "blurred" if blur < 35 else "ok"
    return crop, quality, blur, brightness


def run_camera(camera: dict, interval: int = 5) -> None:
    source = camera_source(camera["source"])
    headers = {"X-Service-Key": SERVICE_KEY}
    roi = json.loads(camera["roi_json"])
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + camera["id"]
    cap = cv2.VideoCapture(source)
    last_sent = 0.0
    dark_streak = 0
    disconnect_streak = 0
    try:
        while True:
            if not cap.isOpened():
                requests.post(f"{OPS_URL}/internal/camera-status", params={"camera_id": camera["id"], "error": "Camera disconnected"}, headers=headers, timeout=5)
                disconnect_streak += 1
                cap.release()
                if disconnect_streak >= 3:
                    print(f"{camera['id']}: camera disconnected; worker exited", flush=True)
                    break
                time.sleep(3)
                cap = cv2.VideoCapture(source)
                continue
            ok, frame = cap.read()
            if not ok:
                requests.post(f"{OPS_URL}/internal/camera-status", params={"camera_id": camera["id"], "error": "Camera stream stopped or disconnected"}, headers=headers, timeout=5)
                disconnect_streak += 1
                cap.release()
                if disconnect_streak >= 3:
                    print(f"{camera['id']}: camera stream stopped; worker exited", flush=True)
                    break
                time.sleep(1)
                cap = cv2.VideoCapture(source)
                continue
            disconnect_streak = 0
            if time.monotonic() - last_sent < interval:
                time.sleep(.1)
                continue
            crop, quality, blur, brightness = inspect_frame(frame, roi)
            if quality == "dark" and brightness < 8:
                dark_streak += 1
                if dark_streak == 3:
                    requests.post(f"{OPS_URL}/internal/camera-status", params={"camera_id": camera["id"], "error": "Camera view is near black; check the phone stream and lighting"}, headers=headers, timeout=5)
                    print(f"{camera['id']}: near-black stream; worker exited", flush=True)
                    break
                last_sent = time.monotonic()
                continue
            dark_streak = 0
            success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not success:
                continue
            if len(encoded) > 1_900_000:
                crop = cv2.resize(crop, (640, 480))
                _, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
            payload = {"camera_id": camera["id"], "asset_id": camera["asset_id"], "site_id": camera["site_id"],
                       "session_id": session_id, "sample_id": camera.get("sample_id") or camera["asset_id"],
                       "captured_at": datetime.now(timezone.utc).isoformat(), "roi_version": "1", "quality": quality,
                       "blur": blur, "brightness": brightness, "jpeg_b64": base64.b64encode(encoded).decode("ascii")}
            try:
                res = requests.post(f"{OPS_URL}/internal/captures", json=payload, headers=headers, timeout=8)
                res.raise_for_status()
                print(f"{camera['id']}: {quality} capture {res.json()['id']}", flush=True)
            except requests.RequestException as exc:
                print(f"{camera['id']}: upload failed: {exc}", flush=True)
            last_sent = time.monotonic()
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, help="Registered camera ID")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    result = requests.get(f"{OPS_URL}/internal/cameras", headers={"X-Service-Key": SERVICE_KEY}, timeout=10)
    result.raise_for_status()
    camera = next((x for x in result.json() if x["id"] == args.camera), None)
    if camera is None:
        raise SystemExit("Camera is not registered or enabled")
    run_camera(camera, max(2, args.interval))


if __name__ == "__main__":
    main()
