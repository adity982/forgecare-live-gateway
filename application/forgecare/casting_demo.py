"""Read-only inference on held-out cast pump impeller photos."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import torch


def test_photos(run: Path) -> list[dict]:
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file)
                if row["split"] == "test" and Path(row["source_path"]).is_file()]


def load_model(run: Path):
    return torch.jit.load(str(run / "flower_casting_cnn_scripted.pt"), map_location="cpu").eval()


def inspect(path: Path, model, threshold: float):
    grayscale = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise ValueError(f"Unreadable casting image: {path}")
    small = cv2.resize(grayscale, (96, 96), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small.copy()).float().reshape(1, 1, 96, 96) / 255
    with torch.no_grad():
        score = float(torch.sigmoid(model(tensor))[0])
    return grayscale, score, score >= threshold
