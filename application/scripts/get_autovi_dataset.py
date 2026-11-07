"""Download AutoVI's pipe-clip category for local noncommercial research."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


RECORD = "https://zenodo.org/api/records/10459003"
SOURCE = "https://zenodo.org/records/10459003"
FILE = "pipe_clip.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(RECORD, timeout=30).json()
    entry = next(item for item in metadata["files"] if item["key"] == FILE)
    archive = root / FILE
    if not archive.exists():
        with requests.get(entry["links"]["self"], stream=True, timeout=90) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
    if archive.stat().st_size != entry["size"]:
        raise ValueError("Downloaded AutoVI archive has the wrong size")
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
        "source": SOURCE, "title": metadata["metadata"]["title"],
        "license": "CC BY-NC-SA 4.0; local educational research only",
        "archive_sha256": sha, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_extracted": len(files),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": sha, "files_extracted": len(files)}))


if __name__ == "__main__":
    main()
