"""Verify provenance, split isolation, and saved detector evaluation arithmetic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.evaluate_phone_damage_detector import summarize


def verify(data: Path, run: Path) -> dict:
    preparation = json.loads((data / "preparation.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    with (data / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    assert len(manifest) == preparation["source_files"]
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    included = Counter()
    for item in manifest:
        path = item["prepared_path"]
        if not path:
            continue
        split = item["decision"].removeprefix("included_")
        assert split in ("train", "valid", "test")
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == item["sha256"]
        assert Path(path).with_suffix(".txt").exists() is False  # labels are in a separate sibling folder
        assert (data / split / "labels" / (Path(path).stem + ".txt")).is_file()
        groups[item["source_group"]].add(split)
        hashes[item["sha256"]].add(split)
        included[split] += 1
    assert all(len(values) == 1 for values in groups.values())
    assert all(len(values) == 1 for values in hashes.values())
    assert dict(included) == preparation["included"]
    assert (run / "phone_damage_detector.pt").is_file()
    predictions = [json.loads(line) for line in (run / "held_out_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    actual = summarize(predictions, evaluation["test"]["threshold"])
    assert actual == evaluation["test"]
    assert len(predictions) == included["test"]
    return {"status": "passed", "source_files_audited": len(manifest),
            "included": dict(included), "held_out_predictions_recomputed": len(predictions),
            "model_sha256": hashlib.sha256((run / "phone_damage_detector.pt").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.run), indent=2))
