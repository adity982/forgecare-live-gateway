"""Audit the saved AutoVI Flower split, source bytes, and held-out predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


def verify(run: Path) -> dict:
    with (run / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == summary["source_images"] == 519
    assert summary["excluded_exact_duplicate_images"] == 13
    assert len({row["source_path"] for row in rows}) == len(rows)
    assert len({row["sha256"] for row in rows}) == len(rows)
    source_root = Path(rows[0]["source_path"]).parents[3] / "pipe_clip"
    all_source = [*source_root.glob("train/good/*.png"), *source_root.glob("test/good/*.png"),
                  *source_root.glob("test/operator/*.png"), *source_root.glob("test/unclipped/*.png")]
    assert len(all_source) == 532
    assert {hashlib.sha256(path.read_bytes()).hexdigest() for path in all_source} == {row["sha256"] for row in rows}
    groups = {}
    for row in rows:
        path = Path(row["source_path"])
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
        assert cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None, path
        prior = groups.setdefault(row["group"], row["split"])
        assert prior == row["split"], row["group"]
        assert (row["client_id"] in {"client-1", "client-2"}) == (row["split"] == "train")
    client_counts = Counter(row["client_id"] for row in rows if row["split"] == "train")
    assert client_counts == {"client-1": 179, "client-2": 179}
    assert len(summary["history"]) == 3
    assert all(item["client_examples"] == [179, 179] and
               item["started_at_utc"] <= item["finished_at_utc"] for item in summary["history"])
    test = summary["test"]
    assert test["images"] == 81 and test["normal"] == 60 and test["defective"] == 21
    assert test["confusion"]["tp"] + test["confusion"]["fn"] == 21
    assert test["confusion"]["tn"] + test["confusion"]["fp"] == 60
    saved = torch.load(run / "flower_autovi_cnn.pt", map_location="cpu", weights_only=False)
    assert saved["scope"] == "research only"
    model = torch.jit.load(str(run / "flower_autovi_cnn_scripted.pt"), map_location="cpu").eval()
    inputs, labels = [], []
    for row in rows:
        if row["split"] != "test":
            continue
        image = cv2.imread(row["source_path"], cv2.IMREAD_GRAYSCALE)
        inputs.append(cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA))
        labels.append(row["label"] == "defective")
    with torch.no_grad():
        batch = torch.from_numpy(np.stack(inputs)[:, None].copy()).float() / 255
        prediction = torch.sigmoid(model(batch)).numpy().reshape(-1) >= test["threshold"]
    truth = np.array(labels)
    assert int((prediction & truth).sum()) == test["confusion"]["tp"]
    assert int((prediction & ~truth).sum()) == test["confusion"]["fp"]
    return {"verified_unique_images": len(rows), "excluded_duplicates": 13,
            "client_counts": dict(client_counts), "rounds": len(summary["history"]),
            "held_out": test["confusion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
