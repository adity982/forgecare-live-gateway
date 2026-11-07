"""Read-only inference on held-out VisA capsule photos for the research page."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch


def test_photos(run: Path) -> list[dict]:
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["split"] == "test" and Path(row["source_path"]).is_file()]


def load_models(run: Path):
    patch = joblib.load(run / "patch_baseline.joblib")
    flower = torch.jit.load(str(run / "flower_fedavg_cnn_scripted.pt"), map_location="cpu").eval()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return patch, flower, metrics


def inspect(path: Path, patch: dict, flower, metrics: dict) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unreadable image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.cvtColor(cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    features = cv2.resize(hsv, (32, 32), interpolation=cv2.INTER_AREA)
    features = features.reshape(8, 4, 8, 4, 3).transpose(0, 2, 1, 3, 4).reshape(64, 48)
    squared = ((features[:, None, :] - patch["model"].cluster_centers_[None, :, :]) ** 2).sum(axis=-1)
    local = np.sqrt(squared.min(axis=-1)).reshape(8, 8)
    patch_score = float(local.max())
    colored = cv2.applyColorMap(cv2.normalize(cv2.resize(local, (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(rgb, .62, cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), .38, 0)
    tensor = torch.from_numpy(small.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
    with torch.no_grad():
        flower_score = float(torch.sigmoid(flower(tensor))[0])
    return {"original": rgb, "patch_overlay": overlay,
            "patch_score": patch_score, "patch_alert": patch_score >= patch["threshold"],
            "flower_score": flower_score,
            "flower_alert": flower_score >= metrics["flower_fedavg"]["threshold"]}
