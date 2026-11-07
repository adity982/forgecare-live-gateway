"""Check the public-image research artifacts without rerunning training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import torch
from PIL import Image


def verify(run: Path) -> dict:
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    assert len(records) == 702, "Expected all 702 VisA capsules images"
    assert len({row["source_path"] for row in records}) == len(records), "Repeated source image"
    assert len({row["sha256"] for row in records}) == len(records), "Repeated image content"
    counts = Counter((row["split"], row["label"]) for row in records)
    clients = Counter(row["client_id"] for row in records if row["split"] == "train")
    assert clients == {"client-1": 210, "client-2": 211}, clients
    assert all((row["client_id"] in clients) == (row["split"] == "train") for row in records)
    assert counts == {("train", "normal"): 361, ("train", "anomaly"): 60,
                      ("validation", "normal"): 120, ("validation", "anomaly"): 20,
                      ("test", "normal"): 121, ("test", "anomaly"): 20}, counts
    for row in records:
        source = Path(row["source_path"])
        assert source.is_file(), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"], source
        with Image.open(source) as image:
            image.verify()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for name in ("patch_baseline", "flower_fedavg"):
        result = metrics[name]
        matrix = result["confusion"]
        assert matrix["tn"] + matrix["fp"] == 121
        assert matrix["fn"] + matrix["tp"] == 20
        assert result["test_images"] == 141
    patch = joblib.load(run / "patch_baseline.joblib")
    assert patch["category"] == "capsules" and patch["model"].n_clusters == 96
    weights = torch.load(run / "flower_fedavg_cnn.pt", map_location="cpu", weights_only=False)
    assert weights["category"] == "capsules"
    assert all(torch.isfinite(value).all() for value in weights["state_dict"].values())
    scripted = torch.jit.load(str(run / "flower_fedavg_cnn_scripted.pt"), map_location="cpu")
    assert torch.isfinite(scripted(torch.zeros(1, 3, 64, 64))).all()
    anomaly = json.loads((run / "anomalib_padim" / "anomalib_metrics.json").read_text(encoding="utf-8"))
    assert anomaly["input_images"] == 220
    checkpoint = run / "anomalib_padim" / "results" / "Padim" / "visa_capsules" / "v0" / "weights" / "lightning" / "model.ckpt"
    assert checkpoint.is_file() and checkpoint.stat().st_size > 50_000_000
    return {"images_verified": len(records), "split_counts": {f"{a}_{b}": n for (a, b), n in counts.items()},
            "flower_clients": dict(clients), "models_loaded": ["patch_baseline", "flower_fedavg"],
            "anomalib_checkpoint_bytes": checkpoint.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
