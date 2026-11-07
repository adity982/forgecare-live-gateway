"""Run actual Anomalib PaDiM on a documented VisA capsules research subset."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    folders = args.out / "input"
    selected = []
    for split, label, target, limit in (("train", "normal", "normal_train", 100),
                                        ("validation", "normal", "normal_test", 40),
                                        ("validation", "anomaly", "abnormal_test", 20),
                                        ("test", "normal", "normal_test", 40),
                                        ("test", "anomaly", "abnormal_test", 20)):
        subset = [record for record in records if record["split"] == split and record["label"] == label][:limit]
        for record in subset:
            source = Path(record["source_path"])
            destination = folders / target / f"{split}-{source.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(source, destination)
            selected.append({"source_path": str(source), "split": split, "label": label,
                             "anomalib_path": str(destination)})
    (args.out / "anomalib_input_manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    torch.set_num_threads(2)
    data = Folder(name="visa_capsules", root=folders, normal_dir="normal_train",
                  normal_test_dir="normal_test", abnormal_dir="abnormal_test",
                  train_batch_size=8, eval_batch_size=8,
                  num_workers=0, val_split_mode="from_test", val_split_ratio=.5, seed=42)
    model = Padim(backbone="resnet18", n_features=50)
    engine = Engine(default_root_dir=args.out / "results",
                    accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
    engine.fit(model=model, datamodule=data)
    evaluation = engine.test(model=model, datamodule=data)
    (args.out / "anomalib_metrics.json").write_text(json.dumps({
        "dataset": "Amazon VisA capsules", "library": "Anomalib 2.7.0", "model": "PaDiM ResNet18",
        "topology": "single-laptop real-image anomaly detection", "input_images": len(selected),
        "test_result": evaluation}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(evaluation, indent=2, default=str))


if __name__ == "__main__":
    main()
