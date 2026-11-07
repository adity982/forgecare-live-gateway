"""Create a clip-grouped, deduplicated YOLO split from the public phone dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


CLASS_NAMES = ["Broken", "Dent", "Major Scratch", "Small Scratch"]
SEED = 42


def source_group(path: Path) -> tuple[str, str]:
    base = re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)
    clip = re.sub(r"[-_]\d{3,5}(_jpg)?$", "", base) if "_mp4-" in base else base
    return base, clip


def load(source: Path) -> list[dict]:
    records = []
    for original_split in ("train", "valid", "test"):
        for image in sorted((source / original_split / "images").glob("*.jpg")):
            label = source / original_split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing annotation: {image}")
            lines = tuple(sorted(line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()))
            classes = sorted({int(line.split()[0]) for line in lines})
            if any(index not in range(4) for index in classes):
                raise ValueError(f"Unexpected class in {label}")
            base, group = source_group(image)
            records.append({"image": image, "label": label, "source_split": original_split,
                            "base": base, "group": group, "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                            "annotation": lines, "classes": classes})
    return records


def prepare(source: Path, out: Path) -> dict:
    records = load(source)
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    clean = []
    exclusions = Counter()
    for copies in by_hash.values():
        if len({record["annotation"] for record in copies}) != 1:
            for record in copies:
                record["decision"] = "excluded_conflicting_annotation"
            exclusions["conflicting_annotation_files"] += len(copies)
            exclusions["conflicting_image_hashes"] += 1
            continue
        chosen = sorted(copies, key=lambda row: ({"test": 0, "valid": 1, "train": 2}[row["source_split"]], str(row["image"])))[0]
        chosen["decision"] = "eligible"
        clean.append(chosen)
        for record in copies:
            if record is not chosen:
                record["decision"] = "excluded_duplicate_bytes"
                exclusions["duplicate_image_files"] += 1
    by_group = defaultdict(list)
    for record in clean:
        by_group[record["group"]].append(record)
    # Stratify at the clip/original-photo level. Give rare Dent groups priority.
    priorities = (1, 2, 3, 0)
    keys = sorted(by_group)
    group_class = [next((klass for klass in priorities if any(klass in row["classes"] for row in by_group[key])), 0)
                   for key in keys]
    train_keys, rest_keys = train_test_split(keys, test_size=.30, stratify=group_class, random_state=SEED)
    class_for = dict(zip(keys, group_class))
    valid_keys, test_keys = train_test_split(rest_keys, test_size=.5,
                                             stratify=[class_for[key] for key in rest_keys], random_state=SEED)
    assignments = {key: split for split, selected in (("train", train_keys), ("valid", valid_keys), ("test", test_keys))
                   for key in selected}
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    eval_seen = set()
    included = Counter()
    for record in sorted(clean, key=lambda row: ({"test": 0, "valid": 1, "train": 2}[row["source_split"]], str(row["image"]))):
        split = assignments[record["group"]]
        if split != "train" and record["base"] in eval_seen:
            record["decision"] = "excluded_repeated_eval_photo"
            exclusions["repeated_eval_photo_files"] += 1
            continue
        if split != "train":
            eval_seen.add(record["base"])
        stem = hashlib.sha256(str(record["image"]).encode()).hexdigest()[:20]
        target_image = out / split / "images" / f"{stem}.jpg"
        target_label = out / split / "labels" / f"{stem}.txt"
        shutil.copy2(record["image"], target_image)
        shutil.copy2(record["label"], target_label)
        record["decision"] = f"included_{split}"
        record["prepared_path"] = str(target_image)
        included[split] += 1
    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_path", "sha256", "original_split", "original_photo", "source_group", "classes", "decision", "prepared_path"])
        for record in records:
            writer.writerow([str(record["image"]), record["sha256"], record["source_split"], record["base"],
                             record["group"], ";".join(CLASS_NAMES[index] for index in record["classes"]),
                             record["decision"], record.get("prepared_path", "")])
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 4\nnames: {CLASS_NAMES!r}\n", encoding="utf-8")
    summary = {"source": "https://universe.roboflow.com/pharmacy-shelf-medicine-detection/mobile-phone-defect-detection-faraf/dataset/30",
               "license": "CC BY 4.0", "seed": SEED, "source_files": len(records), "unique_image_hashes": len(by_hash),
               "clean_unique_images": len(clean), "source_groups": len(by_group), "included": dict(included),
               "included_groups": dict(Counter(assignments.values())), "excluded": dict(exclusions),
               "split_policy": "Same original photo and all frames of a named video clip stay in one split. Exact duplicate bytes with conflicting annotations are excluded. Validation/test retain one image per original photo. No physical phone IDs are available."}
    (out / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))
