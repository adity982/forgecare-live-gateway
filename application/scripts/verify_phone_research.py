"""Verify source hashes, split counts, and saved phone-model test predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from forgecare.phone_demo import inspect, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != metrics["images"]:
        raise ValueError("Manifest size differs from metrics")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source bytes found")
    for row in rows:
        if hashlib.sha256(Path(row["source_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source hash changed: {row['source_path']}")
    count = Counter((row["split"], row["label"]) for row in rows)
    for name, labels in metrics["split_counts"].items():
        for label, expected in labels.items():
            if count[name, label] != expected:
                raise ValueError(f"Split count changed: {name}/{label}")
    model = load_model(run)
    confusion = Counter()
    for row in rows:
        if row["split"] != "test":
            continue
        _, flagged = inspect(Path(row["source_path"]), model, metrics["threshold"])
        actual = row["label"] == "cracked"
        confusion["tp" if actual and flagged else "fn" if actual else "fp" if flagged else "tn"] += 1
    if dict(confusion) != metrics["test"]["confusion"]:
        raise ValueError(f"Saved-model confusion differs: {dict(confusion)}")
    print(json.dumps({"verified_source_images": len(rows), "test_images": metrics["test"]["images"],
                      "confusion": dict(confusion), "model": str(run / "phone_crack_screen_scripted.pt")}, indent=2))


if __name__ == "__main__":
    main()
