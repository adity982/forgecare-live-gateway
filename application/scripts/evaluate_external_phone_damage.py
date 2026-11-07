"""Check the Broken alert on a separate Kaggle cracked/intact screen sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(data: Path, prepared: Path, run: Path) -> dict:
    settings = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    threshold = settings["validation"]["threshold"]
    known_hashes = {hashlib.sha256(path.read_bytes()).hexdigest()
                    for split in ("train", "valid", "test") for path in (prepared / split / "images").glob("*.jpg")}
    model = YOLO(str(run / "phone_damage_detector.pt"))
    result = {"source": "https://www.kaggle.com/datasets/axondata/cracked-and-intact-smartphone-images-dataset",
              "meaning": "External screen-level check of the Broken class, not a box localization or dent/scratch validation",
              "threshold_from_validation": threshold, "groups": {}}
    for folder, label in (("Image_brokenphones", "cracked screen"), ("Image_phones", "intact screen")):
        checked = alerts = unreadable = exact_overlap = 0
        for path in sorted((data / folder).iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            exact_overlap += hashlib.sha256(path.read_bytes()).hexdigest() in known_hashes
            try:
                prediction = model.predict(source=str(path), conf=max(.05, threshold), iou=.5,
                                           imgsz=416, device="cpu", verbose=False)[0]
            except Exception:
                unreadable += 1
                continue
            checked += 1
            alerts += any(int(box.cls[0]) == 0 for box in prediction.boxes)
        result["groups"][label] = {"checked_photos": checked, "broken_alert_photos": alerts,
                                    "unreadable": unreadable, "exact_hash_overlap_with_training_study": exact_overlap}
    result["caution"] = "An intact screen may still have exterior marks; only the Broken alert is compared with screen status. No iPhone 14 Plus or realme-specific labels."
    (run / "external_screen_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.prepared, args.run), indent=2))
