"""Fit a local four-class phone-damage detector on the grouped public split."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=416)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    pretrained = args.out / "pretrained_yolo11n.pt"
    if not pretrained.exists():
        model = YOLO("yolo11n.pt")
        shutil.copy2(model.ckpt_path, pretrained)
    model = YOLO(str(pretrained))
    model.train(data=str(args.data.resolve() / "data.yaml"), epochs=args.epochs,
                imgsz=args.image_size, batch=8, workers=0, device="cpu", seed=42,
                deterministic=True, patience=5, project=str(args.out.resolve()),
                name="training", exist_ok=True, verbose=False, plots=True)
    weights = args.out / "training" / "weights" / "best.pt"
    if not weights.is_file():
        raise RuntimeError("Training did not save a best checkpoint")
    shutil.copy2(weights, args.out / "phone_damage_detector.pt")
    summary = {"source": "https://universe.roboflow.com/pharmacy-shelf-medicine-detection/mobile-phone-defect-detection-faraf/dataset/30",
               "prepared_data": str(args.data.resolve()), "started_at_utc": started,
               "finished_at_utc": utc_now(), "architecture": "YOLO11n object detector fine-tuned locally",
               "epochs_requested": args.epochs, "image_size": args.image_size,
               "classes": ["Broken", "Dent", "Major Scratch", "Small Scratch"],
               "pretrained_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest(),
               "checkpoint_sha256": hashlib.sha256((args.out / "phone_damage_detector.pt").read_bytes()).hexdigest(),
               "scope": "research only; do not automatically accept or reject a physical phone",
               "weights": str((args.out / "phone_damage_detector.pt").resolve())}
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
