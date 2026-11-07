"""Download the public four-class phone damage dataset without storing its API key."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


SOURCE = "https://universe.roboflow.com/pharmacy-shelf-medicine-detection/mobile-phone-defect-detection-faraf/dataset/30"
EXPORT = "https://api.roboflow.com/pharmacy-shelf-medicine-detection/mobile-phone-defect-detection-faraf/30/yolov8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Set ROBOFLOW_API_KEY in the current or Windows user environment")
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "roboflow_phone_damage_v30_yolov8.zip"
    if not archive.exists():
        link = None
        for attempt in range(30):
            response = requests.get(EXPORT, params={"api_key": key}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if response.status_code == 200 and payload.get("export", {}).get("link"):
                link = payload["export"]["link"]
                break
            time.sleep(5)
        if not link:
            raise RuntimeError("Roboflow export was not ready after 150 seconds")
        pending = root / "roboflow_phone_damage_v30_yolov8.partial"
        with requests.get(link, stream=True, timeout=120) as response:
            response.raise_for_status()
            with pending.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if not zipfile.is_zipfile(pending):
            raise ValueError("Downloaded file is not a ZIP archive")
        pending.replace(archive)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as source:
        files = [item for item in source.infolist() if not item.is_dir()]
        for item in files:
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {item.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != item.file_size:
                with source.open(item) as input_file, target.open("wb") as output:
                    while chunk := input_file.read(1024 * 1024):
                        output.write(chunk)
    (root / "source.json").write_text(json.dumps({
        "source": SOURCE,
        "license": "CC BY 4.0 according to public Roboflow Universe page",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha,
        "files_extracted": len(files),
        "note": "API key was used only for acquisition and is not stored in this folder",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": sha, "files_extracted": len(files)}))


if __name__ == "__main__":
    main()
