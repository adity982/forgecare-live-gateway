"""Download the public noncommercial cracked/intact smartphone research dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


REF = "axondata/cracked-and-intact-smartphone-images-dataset"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{REF}"
PAGE = f"https://www.kaggle.com/datasets/{REF}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset-ref", default=REF)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_ref = args.dataset_ref
    archive = root / f"kaggle_{dataset_ref.split('/')[-1]}.zip"
    if not archive.exists():
        with requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}", stream=True, timeout=90) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
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
        "source": f"https://www.kaggle.com/datasets/{dataset_ref}",
        "license": "Check the dataset page; local educational research only",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha, "extracted_files": len(files),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": sha, "files": len(files)}))


if __name__ == "__main__":
    main()
