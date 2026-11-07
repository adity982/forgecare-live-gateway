"""Download the public casting-impeller dataset for local educational research."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


URL = "https://www.kaggle.com/api/v1/datasets/download/ravirajsinh45/real-life-industrial-dataset-of-casting-product"
PAGE = "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "kaggle_casting_product.zip"
    if not archive.exists():
        with requests.get(URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            if "zip" not in response.headers.get("content-type", ""):
                raise ValueError("Kaggle did not provide the public ZIP archive")
            with archive.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    count = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with source.open(member) as input_file, target.open("wb") as output:
                    while chunk := input_file.read(1024 * 1024):
                        output.write(chunk)
            count += 1
    (root / "source.json").write_text(json.dumps({"source": PAGE, "download_url": URL,
        "license": "CC BY-NC-ND 4.0; local educational research only",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": digest, "extracted_files": count}, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": digest, "files": count}))


if __name__ == "__main__":
    main()
