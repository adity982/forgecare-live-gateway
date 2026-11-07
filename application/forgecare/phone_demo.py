"""Read-only inference for the phone crack-screen research artifact."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_model(run: Path):
    return torch.jit.load(str(run / "phone_crack_screen_scripted.pt"), map_location="cpu").eval()


def test_photos(run: Path):
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source)
                if row["split"] == "test" and Path(row["source_path"]).is_file()]


def inspect(data: bytes | Path, model, threshold: float):
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(source) as original:
        pixels = np.asarray(original.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / \
             torch.tensor([.229, .224, .225])[:, None, None]
    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor[None])).item())
    return probability, probability >= threshold
