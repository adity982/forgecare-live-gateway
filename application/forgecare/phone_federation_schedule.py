"""Six-hour local FedAvg attempt using only reviewed, physically distinct phone samples."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .config import DATA
from .db import connect, rows
from .federation import VISUAL_FEATURE_COUNT, aggregate, sigmoid, train_client, visual_schema_hash
from .vision import grouped_splits, image_features


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_rows(site_id: str, asset_id: str) -> tuple[list[dict], dict]:
    with connect() as db:
        records = rows(db, """SELECT c.id,c.image_path,c.sample_id,l.binary_label
            FROM captures c JOIN labels l ON l.capture_id=c.id
            WHERE c.site_id=? AND c.asset_id=? AND c.quality='ok' AND l.binary_label IS NOT NULL
            ORDER BY c.captured_at""", (site_id, asset_id))
    by_sample = defaultdict(set)
    for item in records:
        by_sample[item["sample_id"]].add(int(item["binary_label"]))
    conflicting = {name for name, labels in by_sample.items() if len(labels) != 1}
    eligible = [item for item in records if item["sample_id"] not in conflicting and Path(item["image_path"]).is_file()]
    eligible_samples = {item["sample_id"]: int(item["binary_label"]) for item in eligible}
    group_counts = Counter(eligible_samples.values())
    return eligible, {"reviewed_photos": len(records), "usable_labeled_photos": len(eligible),
                      "distinct_acceptable_phones": group_counts[0], "distinct_damaged_phones": group_counts[1],
                      "conflicting_phone_ids": len(conflicting)}


def two_clients(y: np.ndarray, groups: np.ndarray, train: np.ndarray) -> list[np.ndarray]:
    partitions = [[], []]
    for label in (0, 1):
        sample_ids = sorted(set(groups[train][y[train] == label]))
        for number, sample_id in enumerate(sample_ids):
            client = number % 2
            partitions[client].extend(int(index) for index in train if groups[index] == sample_id)
    if any(not part or set(y[part]) != {0, 1} for part in partitions):
        raise ValueError("Train split cannot give both clients distinct phones in both conditions")
    if set(groups[partitions[0]]) & set(groups[partitions[1]]):
        raise AssertionError("A physical phone was assigned to both clients")
    return [np.asarray(part, dtype=int) for part in partitions]


def confusion(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    predicted = sigmoid(x @ weights[:-1] + weights[-1]) >= .5
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tp = int(((y == 1) & predicted).sum())
    return {"photos": int(len(y)), "damaged_flagged": tp, "damaged_missed": fn,
            "acceptable_flagged": fp, "acceptable_passed": tn,
            "agreement_with_reviewed_labels": (tp + tn) / len(y),
            "damaged_recall": tp / (tp + fn), "acceptable_false_alert_rate": fp / (fp + tn)}


def train_asset(site_id: str, asset: dict, out: Path) -> dict:
    eligible, counts = select_rows(site_id, asset["id"])
    base = {"asset_id": asset["id"], "model": asset["model"], **counts}
    if len(eligible) < 24 or min(counts["distinct_acceptable_phones"], counts["distinct_damaged_phones"]) < 6:
        return {**base, "status": "skipped_insufficient_distinct_reviewed_phones",
                "needed": "At least 24 usable reviewed photos and six distinct physical phones in each condition for this model"}
    x = np.stack([image_features(item["image_path"]) for item in eligible]).astype(np.float64)
    y = np.asarray([int(item["binary_label"]) for item in eligible], dtype=int)
    groups = np.asarray([item["sample_id"] for item in eligible])
    train, validation, test = grouped_splits(y, groups)
    clients = two_clients(y, groups, train)
    current = np.zeros(VISUAL_FEATURE_COUNT + 1, dtype=np.float64)
    schema = visual_schema_hash()
    round_id = str(uuid.uuid4())
    timeline = []
    for number in range(1, 4):
        started = timestamp()
        updates = []
        for index, samples in enumerate(clients, 1):
            local = train_client(current, x[samples], y[samples], epochs=100, rate=.15)
            updates.append({"round_id": round_id, "schema_hash": schema, "client_id": f"local-client-{index}",
                            "count": len(samples), "weights": local})
        current = aggregate(updates, round_id=round_id, schema_hash=schema, start=current)
        timeline.append({"round": number, "started_at_utc": started, "finished_at_utc": timestamp(),
                         "client_train_photos": [len(part) for part in clients]})
    validation_result = confusion(current, x[validation], y[validation])
    test_result = confusion(current, x[test], y[test])
    safe_name = re.sub(r"[^a-z0-9]+", "-", asset["id"].lower()).strip("-")
    model_path = out / f"{safe_name}-{round_id}.joblib"
    source_hashes = [hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest() for item in eligible]
    joblib.dump({"weights": current, "threshold": .5, "feature_version": "gray16-hist16-v1",
                 "source": "reviewed_local_camera_captures", "site_id": site_id, "asset_id": asset["id"],
                 "round_id": round_id, "trained_at_utc": timestamp()}, model_path)
    return {**base, "status": "completed_local_research_model", "rounds": timeline,
            "train_photos": int(len(train)), "validation_photos": int(len(validation)),
            "test_photos": int(len(test)), "validation": validation_result, "held_out": test_result,
            "model_path": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "input_hashes_sha256": hashlib.sha256("\n".join(sorted(source_hashes)).encode()).hexdigest(),
            "limitation": "Two logical clients on one laptop. Review labels are the reference, not verified defect ground truth. No cross-site privacy claim."}


def run_tick(site_id: str | None = None, out: Path | None = None) -> Path:
    target = out or DATA / "scheduled_federation"
    target.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    with connect() as db:
        assets = rows(db, "SELECT id,site_id,model FROM assets WHERE model IN ('iPhone 14 Plus','realme narzo 60 5G')")
    if site_id:
        assets = [item for item in assets if item["site_id"] == site_id]
    results = []
    for asset in assets:
        try:
            results.append(train_asset(asset["site_id"], asset, target))
        except Exception as exc:
            results.append({"asset_id": asset["id"], "model": asset["model"], "status": "failed", "error": str(exc)})
    report = {"started_at_utc": started, "finished_at_utc": timestamp(), "cadence": "6 hours when Windows task runs",
              "destination": str(target), "assets": results}
    path = target / f"attempt-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--site")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(run_tick(args.site, args.out))
