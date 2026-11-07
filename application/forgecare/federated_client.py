"""Site-local training CLI altered for automated 10-round loop execution."""
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
    # ... (Dataset loading logic remains exactly the same) ...
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    required = {"image_path", "binary_label", "sample_id"}
    if len(records) < 24 or any(not required.issubset(row) for row in records):
        raise ValueError("Manifest needs at least 24 reviewed rows with image_path,binary_label,sample_id")
    paths = [Path(row["image_path"]) for row in records]
    paths = [path if path.is_absolute() else manifest.parent / path for path in paths]
    X = np.stack([image_features(path) for path in paths]).astype(np.float64)
    y = np.array([int(row["binary_label"]) for row in records])
    groups = np.array([row["sample_id"] for row in records])
    train, validation, test = grouped_splits(y, groups)
    return X, y, groups, train, validation, test

def metrics(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    # ... (Metrics remain exactly the same) ...
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
            "latency_ms": elapsed, "sample_count": len(y)}

def run():
    parser = argparse.ArgumentParser()
    # Removed specific 'action' and 'round' arguments, now automates 10 rounds
    parser.add_argument("--simulation_id", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    
    password = os.environ.get("FORGECARE_PASSWORD")
    if not password:
        raise SystemExit("Set FORGECARE_PASSWORD in this shell")
        
    session = requests.Session()
    login = session.post(f"{OPS_URL}/auth/login", json={"username": args.username, "password": password}, timeout=15)
    login.raise_for_status()
    session.headers.update({"Authorization": f"Bearer {login.json()['token']}"})
    
    X, y, groups, train, validation, test = local_dataset(args.manifest)

    # LOOP 10 ROUNDS AUTOMATICALLY
    for round_num in range(1, 11):
        round_id = f"{args.simulation_id}-round-{round_num}"
        print(f"--- Starting Round {round_num}/10 ---")
        
        # 1. Fetch round parameters from server
        info_response = session.get(f"{OPS_URL}/federation/rounds/{round_id}", timeout=15)
        info_response.raise_for_status()
        round_data = info_response.json()
        
        # Wait until server is ready (simplified polling loop)
        while round_data["status"] != "collecting":
            print("Waiting for server to open round collection...")
            time.sleep(5)
            round_data = session.get(f"{OPS_URL}/federation/rounds/{round_id}").json()
            
        info = json.loads(round_data["results_json"])
        
        # In Round 1, this will be zeroes. In Round N, this is Round N-1's aggregated weights.
        start_weights = np.asarray(info["initial_weights"], dtype=np.float64)
        
        # 2. Train locally
        local_weights = train_client(start_weights, X[train], y[train], epochs=100, rate=.15)
        
        # 3. Submit Update
        payload = {
            "site_id": args.site, "client_id": args.client,
            "schema_hash": round_data["schema_hash"], "sample_count": int(len(train)),
            "weights": local_weights.tolist()
        }
        response = session.post(f"{OPS_URL}/federation/rounds/{round_id}/updates", json=payload, timeout=30)
        response.raise_for_status()
        print(f"Round {round_num} weights submitted.")

if __name__ == "__main__":
    run()
