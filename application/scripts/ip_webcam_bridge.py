"""Run on the laptop sharing Wi-Fi with the Samsung IP Webcam.

Uses only Python's standard library. The pairing code is entered interactively
and is never placed in a command line, log, or downloaded script.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


MAX_PHOTO = 8_000_000
LAN_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def json_request(url: str, code: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body,
        headers={"X-Relay-Code": code, "Content-Type": "application/json"},
        method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def camera_photo(video_url: str) -> bytes:
    parsed = urllib.parse.urlparse(video_url)
    address = ipaddress.ip_address(parsed.hostname or "")
    if not address.is_private or address.is_loopback or parsed.scheme != "http":
        raise ValueError("The saved IP Webcam address is not a private HTTP address")
    root = f"{parsed.scheme}://{parsed.netloc}"
    failures = []
    for suffix in ("/shot.jpg", "/photo.jpg"):
        try:
            with LAN_OPENER.open(root + suffix, timeout=6) as response:
                content = response.read(MAX_PHOTO + 1)
            if content.startswith(b"\xff\xd8") and len(content) <= MAX_PHOTO:
                return content
            failures.append(f"{suffix}: not a usable JPEG")
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"{suffix}: {exc}")
    try:
        with LAN_OPENER.open(video_url, timeout=8) as response:
            buffer = bytearray()
            while len(buffer) < MAX_PHOTO + 100_000:
                chunk = response.read(8192)
                if not chunk:
                    break
                buffer.extend(chunk)
                start = buffer.find(b"\xff\xd8")
                if start >= 0:
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end >= 0:
                        photo = bytes(buffer[start:end + 2])
                        if len(photo) <= MAX_PHOTO:
                            return photo
                        break
    except (OSError, urllib.error.URLError) as exc:
        failures.append(f"/video: {exc}")
    raise RuntimeError("Cannot read the Samsung IP Webcam on this laptop. " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Relay Samsung IP Webcam photos to hosted ForgeCare")
    parser.add_argument("--relay", required=True, help="HTTPS relay address shown in ForgeCare")
    parser.add_argument("--camera", default="camera-samsung", help="Registered camera ID")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between photos, minimum 3")
    args = parser.parse_args()
    if not args.relay.startswith("https://"):
        raise SystemExit("The relay address must start with https://")
    relay = args.relay.rstrip("/") + "/relay/" + urllib.parse.quote(args.camera, safe="")
    code = getpass.getpass("Pairing code from Assets & setup: ").strip()
    if not code:
        raise SystemExit("A pairing code is required")
    session_id = "relay-" + uuid.uuid4().hex
    print("Connecting to ForgeCare and the Samsung IP Webcam. Press Ctrl+C to stop.", flush=True)
    last_config = 0.0
    video_url = ""
    try:
        while True:
            started = time.monotonic()
            try:
                if not video_url or started - last_config >= 30:
                    config = json_request(relay + "/config", code)
                    video_url = config["video_url"]
                    last_config = started
                photo = camera_photo(video_url)
                result = json_request(relay + "/captures", code,
                    {"session_id": session_id, "jpeg_b64": base64.b64encode(photo).decode("ascii")})
                print(f"Saved {result['id'][:8]} · image check: {result['quality']}", flush=True)
                json_request(relay + "/status", code, {"error": ""})
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise SystemExit("Pairing code expired or was replaced. Generate a new one in ForgeCare.")
                error = f"Relay returned HTTP {exc.code}"
                print(error, flush=True)
            except Exception as exc:
                error = str(exc)[:240]
                print(error, flush=True)
                try:
                    json_request(relay + "/status", code, {"error": error})
                except Exception:
                    pass
            time.sleep(max(3, args.interval) - min(max(3, args.interval), time.monotonic() - started))
    except KeyboardInterrupt:
        try:
            json_request(relay + "/status", code, {"error": "Local camera bridge stopped"})
        except Exception:
            pass
        print("Stopped.")


if __name__ == "__main__":
    main()
