"""Evaluate box-level damage findings and per-photo counts on held-out phone groups."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


NAMES = ["Broken", "Dent", "Major Scratch", "Small Scratch"]


def boxes_from_label(path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        klass, xc, yc, bw, bh = map(float, line.split()[:5])
        boxes.append({"class_id": int(klass), "xyxy": [(xc - bw / 2) * width, (yc - bh / 2) * height,
                                                   (xc + bw / 2) * width, (yc + bh / 2) * height]})
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersect / (area_a + area_b - intersect) if area_a + area_b > intersect else 0.0


def run_predictions(model: YOLO, data: Path, split: str, image_size: int) -> list[dict]:
    photos = sorted((data / split / "images").glob("*.jpg"))
    rows = []
    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None:
            raise ValueError(f"Cannot decode {photo}")
        height, width = image.shape[:2]
        gt = boxes_from_label(data / split / "labels" / (photo.stem + ".txt"), width, height)
        result = model.predict(source=str(photo), conf=.05, iou=.5, imgsz=image_size,
                               device="cpu", verbose=False)[0]
        pred = [{"class_id": int(box.cls[0]), "confidence": float(box.conf[0]),
                 "xyxy": [float(value) for value in box.xyxy[0].tolist()]}
                for box in result.boxes]
        rows.append({"image_path": str(photo), "ground_truth": gt, "predictions": pred})
    return rows


def summarize(rows: list[dict], threshold: float) -> dict:
    by_class = {}
    for klass, name in enumerate(NAMES):
        tp = fp = fn = exact = exact_positive = positive_photos = negative_photos = false_alert_photos = 0
        total_gt = total_pred = 0
        for row in rows:
            truth = [box for box in row["ground_truth"] if box["class_id"] == klass]
            guesses = sorted([box for box in row["predictions"] if box["class_id"] == klass and box["confidence"] >= threshold],
                             key=lambda box: -box["confidence"])
            total_gt += len(truth)
            total_pred += len(guesses)
            exact += len(truth) == len(guesses)
            exact_positive += bool(truth) and len(truth) == len(guesses)
            positive_photos += bool(truth)
            negative_photos += not truth
            false_alert_photos += bool(guesses) and not truth
            matched = set()
            for guess in guesses:
                candidates = [(iou(guess["xyxy"], label["xyxy"]), index) for index, label in enumerate(truth)
                              if index not in matched]
                best = max(candidates, default=(0, -1))
                if best[0] >= .5:
                    tp += 1
                    matched.add(best[1])
                else:
                    fp += 1
            fn += len(truth) - len(matched)
        by_class[name] = {"annotated_regions": total_gt, "suggested_regions": total_pred,
                          "localized_correctly": tp, "false_regions": fp, "missed_regions": fn,
                          "photos_with_class": positive_photos, "photos_without_class": negative_photos,
                          "false_alert_photos": false_alert_photos,
                          "exact_count_photos": exact, "exact_positive_count_photos": exact_positive,
                          "total_photos": len(rows),
                          "precision": tp / (tp + fp) if tp + fp else 0,
                          "recall": tp / (tp + fn) if tp + fn else 0}
    f1s = [2 * value["precision"] * value["recall"] / (value["precision"] + value["recall"])
           if value["precision"] + value["recall"] else 0 for value in by_class.values()]
    return {"photos": len(rows), "threshold": threshold, "classes": by_class,
            "macro_box_f1": float(np.mean(f1s))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=416)
    args = parser.parse_args()
    model = YOLO(str(args.run / "phone_damage_detector.pt"))
    validation = run_predictions(model, args.data, "valid", args.image_size)
    candidates = [round(value, 2) for value in np.arange(.1, .81, .05)]
    threshold = max(candidates, key=lambda value: summarize(validation, value)["macro_box_f1"])
    held_out = run_predictions(model, args.data, "test", args.image_size)
    evaluated = {"source": "https://universe.roboflow.com/pharmacy-shelf-medicine-detection/mobile-phone-defect-detection-faraf/dataset/30",
                 "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "confidence_selection": "Validation macro box F1, IoU >= 0.5; held-out test was not used to choose confidence",
                 "validation": summarize(validation, threshold), "test": summarize(held_out, threshold),
                 "limitations": "Clip-grouped source split, but no physical-phone IDs. Some source annotations conflicted and were excluded. Held-out classes have few instances, especially Dent. Counts are candidate regions per photo, not verified physical-phone totals. No iPhone 14 Plus or realme test labels."}
    (args.run / "evaluation.json").write_text(json.dumps(evaluated, indent=2), encoding="utf-8")
    with (args.run / "held_out_predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in held_out:
            output.write(json.dumps(row) + "\n")
    print(json.dumps(evaluated, indent=2), flush=True)


if __name__ == "__main__":
    main()
