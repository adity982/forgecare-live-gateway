"""Verify every source image and saved Flower casting run artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import torch


def verify(run: Path) -> dict:
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1300
    assert len({row["source_path"] for row in records}) == 1300
    assert len({row["sha256"] for row in records}) == 1300
    counts = Counter((row["split"], row["label"]) for row in records)
    clients = Counter(row["client_id"] for row in records if row["split"] == "train")
    assert clients == {"client-1": 455, "client-2": 455}, clients
    assert all((row["client_id"] in clients) == (row["split"] == "train") for row in records)
    for row in records:
        path = Path(row["source_path"])
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
        assert cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None, path
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["original_images"] == 1300 and metrics["rounds"] == 3
    assert len(metrics["history"]) == 3
    assert all(row["started_at_utc"] <= row["finished_at_utc"] for row in metrics["history"])
    test = metrics["test"]
    assert test["images"] == 195 and test["normal"] == 78 and test["defective"] == 117
    assert test["confusion"]["tn"] + test["confusion"]["fp"] == 78
    assert test["confusion"]["fn"] + test["confusion"]["tp"] == 117
    saved = torch.load(run / "flower_casting_cnn.pt", map_location="cpu", weights_only=False)
    assert saved["scope"] == "research only"
    model = torch.jit.load(str(run / "flower_casting_cnn_scripted.pt"), map_location="cpu")
    assert torch.isfinite(model(torch.zeros(1, 1, 96, 96))).all()
    return {"verified_images": len(records), "counts": {f"{a}_{b}": n for (a, b), n in counts.items()},
            "clients": dict(clients), "rounds": len(metrics["history"]), "test_auc": test["roc_auc"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
