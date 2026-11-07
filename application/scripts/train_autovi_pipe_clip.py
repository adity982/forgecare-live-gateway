"""Fit PaDiM on Renault AutoVI pipe-clip images and report worker-readable counts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from anomalib.data import AutoVI
from anomalib.engine import Engine
from anomalib.models import Padim
from torchvision.transforms.v2 import Resize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted((args.data / "pipe_clip").rglob("*.png"))
    if not images:
        raise ValueError("AutoVI pipe_clip images are not extracted")
    manifest = [{"source_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in images]
    (args.out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.set_num_threads(2)
    resized = Resize((256, 256))
    data = AutoVI(root=args.data, category="pipe_clip", train_batch_size=8, eval_batch_size=8,
                  num_workers=0, train_augmentations=resized, val_augmentations=resized,
                  test_augmentations=resized, val_split_mode="from_test", val_split_ratio=.5, seed=42)
    model = Padim(backbone="resnet18", n_features=50)
    engine = Engine(default_root_dir=args.out / "results", accelerator="cpu", devices=1,
                    logger=False, enable_progress_bar=False)
    started = datetime.now(timezone.utc).isoformat()
    engine.fit(model=model, datamodule=data)
    evaluation = engine.test(model=model, datamodule=data)
    predictions = engine.predict(model=model, datamodule=data, return_predictions=True)
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for batch in predictions or []:
        actual = batch.gt_label.detach().cpu().numpy().reshape(-1)
        predicted = batch.pred_label.detach().cpu().numpy().reshape(-1)
        for truth, alarm in zip(actual, predicted):
            counts["tp" if truth and alarm else "fn" if truth else "fp" if alarm else "tn"] += 1
    summary = {"dataset": "Renault AutoVI pipe_clip", "source": "https://zenodo.org/records/10459003",
               "license": "CC BY-NC-SA 4.0; local educational research only",
               "method": "Anomalib PaDiM ResNet18, 50 feature dimensions, 256px input, normal-only training",
               "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_png_files": len(manifest), "test_confusion": counts,
               "anomalib_metrics": evaluation,
               "limitations": "Research only. Validation and test share the original AutoVI test directory through a fixed seeded split. No deployment on the user's camera or actual factory line; defects were intentionally staged and corrected by Renault experts."}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
