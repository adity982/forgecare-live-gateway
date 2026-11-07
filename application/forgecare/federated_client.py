"""Site-local training/validation CLI. Raw images never enter API payloads."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from .config import OPS_URL
from .federation import VISUAL_FEATURE_COUNT, sigmoid, train_client
from .vision import grouped_splits, image_features


def local_dataset(manifest: Path):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    required = {"image_path", "binary_label", "sample_id"}
    if len(records) < 24 or any(not required.issubset(row) for row in records):
        raise ValueError("Manifest needs at least 24 reviewed rows with image_path,binary_label,sample_id")
    if any(row["binary_label"] not in {"0", "1"} or not row["sample_id"] for row in records):
        raise ValueError("Every row needs a 0/1 reviewed label and physical sample ID")
    paths = [Path(row["image_path"]) for row in records]
    paths = [path if path.is_absolute() else manifest.parent / path for path in paths]
    X = np.stack([image_features(path) for path in paths]).astype(np.float64)
    if X.shape[1] != VISUAL_FEATURE_COUNT:
        raise ValueError("Feature schema mismatch")
    y = np.array([int(row["binary_label"]) for row in records])
    groups = np.array([row["sample_id"] for row in records])
    train, validation, test = grouped_splits(y, groups)
    return X, y, groups, train, validation, test


def metrics(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    start = time.perf_counter()
    scores = sigmoid(X @ weights[:-1] + weights[-1])
    elapsed = (time.perf_counter() - start) * 1000 / max(len(X), 1)
    predicted = scores >= .5
    tp = int(((predicted == 1) & (y == 1)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    return {"recall": tp / (tp + fn) if tp + fn else 0,
            "false_alert_rate": fp / (fp + tn) if fp + tn else 0,
            "latency_ms": elapsed, "sample_count": len(y), "positive_count": int(y.sum())}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "validate"])
    parser.add_argument("--round", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    password = os.environ.get("FORGECARE_PASSWORD")
    if not password:
        raise SystemExit("Set FORGECARE_PASSWORD in this shell; do not put it on the command line")
    session = requests.Session()
    login = session.post(f"{OPS_URL}/auth/login", json={"username": args.username, "password": password}, timeout=15)
    login.raise_for_status()
    session.headers.update({"Authorization": f"Bearer {login.json()['token']}"})
    info_response = session.get(f"{OPS_URL}/federation/rounds/{args.round}", timeout=15)
    info_response.raise_for_status()
    round_data = info_response.json()
    if round_data["site_id"] != args.site:
        raise SystemExit("Round belongs to another site")
    X, y, groups, train, validation, test = local_dataset(args.manifest)
    info = json.loads(round_data["results_json"])
    start = np.asarray(info["initial_weights"], dtype=np.float64)
    local = train_client(start, X[train], y[train], epochs=100, rate=.15)
    if args.action == "train":
        if round_data["status"] != "collecting":
            raise SystemExit("Round is not collecting updates")
        payload = {"site_id": args.site, "client_id": args.client,
                   "schema_hash": round_data["schema_hash"], "sample_count": int(len(train)),
                   "weights": local.tolist()}
        response = session.post(f"{OPS_URL}/federation/rounds/{args.round}/updates", json=payload, timeout=30)
        response.raise_for_status()
        print(json.dumps({"status": "submitted", "client_id": args.client, "train_count": len(train),
                          "held_out_count": len(test), "physical_samples": len(np.unique(groups))}))
    else:
        if round_data["status"] != "awaiting_validation":
            raise SystemExit("Wait for server aggregation before validation")
        fed = np.asarray(info["aggregated_weights"], dtype=np.float64)
        local_results = metrics(local, X[test], y[test])
        fed_results = metrics(fed, X[test], y[test])
        payload = {"site_id": args.site, "client_id": args.client, **fed_results,
                   "local_recall": local_results["recall"],
                   "local_false_alert_rate": local_results["false_alert_rate"]}
        response = session.post(f"{OPS_URL}/federation/rounds/{args.round}/validations", json=payload, timeout=30)
        response.raise_for_status()
        print(json.dumps({"server": response.json(), "local": local_results, "federated": fed_results}))


if __name__ == "__main__":
    run()
