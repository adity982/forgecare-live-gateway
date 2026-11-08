"""Synthetic FedAvg teaching simulation and federation helpers."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

VISUAL_FEATURE_COUNT = 275  # gray16 (256), histogram16, edge/mean/std (3)
SYNTHETIC_ROUNDS = 5


def visual_schema_hash() -> str:
    return hashlib.sha256(f"logistic:{VISUAL_FEATURE_COUNT}:gray16-hist16-v1".encode()).hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def train_client(start: np.ndarray, X: np.ndarray, y: np.ndarray, *, epochs: int = 20, rate: float = .05) -> np.ndarray:
    weights = start.copy()
    for _ in range(epochs):
        errors = sigmoid(X @ weights[:-1] + weights[-1]) - y
        gradient = np.concatenate([X.T @ errors / len(y), np.array([errors.mean()])])
        weights -= rate * gradient
    return weights


def aggregate(updates: list[dict], *, round_id: str, schema_hash: str, start: np.ndarray) -> np.ndarray:
    if not updates:
        raise ValueError("No client updates")
    total = 0
    weighted = np.zeros_like(start, dtype=np.float64)
    clients = set()
    for update in updates:
        if update["round_id"] != round_id or update["schema_hash"] != schema_hash:
            raise ValueError("Stale or mismatched round/schema")
        if update["client_id"] in clients:
            raise ValueError("Duplicate client update")
        clients.add(update["client_id"])
        weight = np.asarray(update["weights"], dtype=np.float64)
        count = update["count"]
        if weight.shape != start.shape or not np.all(np.isfinite(weight)) or not isinstance(count, int) or count <= 0:
            raise ValueError("Invalid weight shape, finite values, or sample count")
        weighted += count * weight
        total += count
    return weighted / total


def _synthetic_data(seed: int, positive_fraction: float, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=sample_count,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[1 - positive_fraction, positive_fraction],
        class_sep=1.2,
        random_state=seed,
    )
    return X.astype(np.float64) / 4, y.astype(np.int64)


def _show_progress(label: str, *, animate: bool, pause_seconds: float) -> None:
    if not animate or pause_seconds == 0:
        print(f"  {label} done")
        return
    frames = ("[=     ]", "[===   ]", "[======]")
    if sys.stdout.isatty():
        for frame in frames:
            print(f"  {label} {frame}", end="\r", flush=True)
            time.sleep(pause_seconds / len(frames))
        print(f"  {label} done{' ' * 12}")
    else:
        for frame in frames:
            print(f"  {label} {frame}")
            time.sleep(pause_seconds / len(frames))


def simulate_synthetic_federation(*, animate: bool = False, pause_seconds: float = .3) -> dict:
    """Train two in-memory synthetic clients for five rounds of sample-weighted FedAvg."""
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be non-negative")

    clients = {}
    for name, seed, positive_fraction, sample_count in (
        ("Client A", 17, .25, 200),
        ("Client B", 29, .75, 320),
    ):
        X, y = _synthetic_data(seed, positive_fraction, sample_count)
        train, test = train_test_split(
            np.arange(len(y)), test_size=.25, stratify=y, random_state=seed
        )
        clients[name] = {"X": X, "y": y, "train": train, "test": test}

    test_X, test_y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[.5, .5],
        class_sep=1.2,
        random_state=43,
    )
    test_X = test_X.astype(np.float64) / 4
    test_y = test_y.astype(np.int64)
    current = np.zeros(test_X.shape[1] + 1, dtype=np.float64)
    round_id = str(uuid.uuid4())
    schema_hash = hashlib.sha256(b"logistic:8:synthetic-demo-v1").hexdigest()
    timeline = []
    last_local = {}

    if animate:
        print("FEDERATED LEARNING: a five-round, synthetic-data-only demonstration")
        print("Making fictional examples in memory; no photos, manifests, or existing records are read.")
        print()
        print("Client A [private examples] -- updated weights only -->")
        print("                                                        [SERVER]")
        print("Client A [private examples] <-- new shared model ----- [SERVER]")
        print("Client B [private examples] -- updated weights only -->")
        print("Client B [private examples] <-- new shared model ----- [SERVER]")
        print("Example rows stay with each client; the shared model is sent back after averaging.")
        print()

    for number in range(1, SYNTHETIC_ROUNDS + 1):
        if animate:
            print(f"ROUND {number} OF {SYNTHETIC_ROUNDS}")
            print("  1. Server sends its current shared model to both clients.")
        updates = []
        round_clients = {}
        for name, client in clients.items():
            local = train_client(current, client["X"][client["train"]], client["y"][client["train"]],
                                 epochs=20, rate=.1)
            if animate:
                _show_progress(f"{name} learns from its own {len(client['train'])} private examples", animate=True,
                               pause_seconds=pause_seconds)
            local_pred = sigmoid(client["X"][client["test"]] @ local[:-1] + local[-1]) >= .5
            local_recall = float(recall_score(client["y"][client["test"]], local_pred, zero_division=0))
            updates.append({"round_id": round_id, "schema_hash": schema_hash, "client_id": name,
                            "count": int(len(client["train"])), "weights": local})
            round_clients[name] = {
                "training_examples": int(len(client["train"])),
                "local_accuracy": float(accuracy_score(client["y"][client["test"]], local_pred)),
                "local_recall": local_recall,
            }
            last_local[name] = local

        if animate:
            print("  2. Clients send model updates (not their example rows) to the server.")
            print("     The server gives more influence to the update trained on more examples.")
        current = aggregate(updates, round_id=round_id, schema_hash=schema_hash, start=current)
        global_pred = sigmoid(test_X @ current[:-1] + current[-1]) >= .5
        global_accuracy = float(accuracy_score(test_y, global_pred))
        for name, client in clients.items():
            client_pred = sigmoid(client["X"][client["test"]] @ current[:-1] + current[-1]) >= .5
            round_clients[name]["federated_accuracy"] = float(
                accuracy_score(client["y"][client["test"]], client_pred)
            )
            round_clients[name]["federated_recall"] = float(
                recall_score(client["y"][client["test"]], client_pred, zero_division=0)
            )
        timeline.append({
            "round": number,
            "client_updates": round_clients,
            "global_model_accuracy": global_accuracy,
            "global_model_test_examples": int(len(test_y)),
        })
        if animate:
            _show_progress("Server combines updates into one shared model", animate=True, pause_seconds=pause_seconds)
            print(f"  3. Shared model is ready for round {number + 1 if number < SYNTHETIC_ROUNDS else 'complete'}.")
            print(f"     Score on separate synthetic examples: {global_accuracy:.1%}")
            print()

    final_clients = {}
    for name, client in clients.items():
        local_pred = sigmoid(client["X"][client["test"]] @ last_local[name][:-1] + last_local[name][-1]) >= .5
        fed_pred = sigmoid(client["X"][client["test"]] @ current[:-1] + current[-1]) >= .5
        final_clients[name] = {
            "train_count": int(len(client["train"])),
            "test_count": int(len(client["test"])),
            "local_accuracy": float(accuracy_score(client["y"][client["test"]], local_pred)),
            "local_recall": float(recall_score(client["y"][client["test"]], local_pred, zero_division=0)),
            "federated_accuracy": float(accuracy_score(client["y"][client["test"]], fed_pred)),
            "federated_recall": float(recall_score(client["y"][client["test"]], fed_pred, zero_division=0)),
        }

    final_prediction = sigmoid(test_X @ current[:-1] + current[-1]) >= .5
    summary = {
        "dataset": "newly_generated_synthetic_examples_in_memory",
        "mechanism": "sample_weighted_FedAvg_logistic",
        "claim": "Teaching simulation only; fictional data and two logical clients on one computer.",
        "raw_data_shared": False,
        "round_count": SYNTHETIC_ROUNDS,
        "global_model_accuracy": float(accuracy_score(test_y, final_prediction)),
        "global_model_test_examples": int(len(test_y)),
        "clients": final_clients,
        "rounds": timeline,
        "weights": current.round(6).tolist(),
    }
    if animate:
        print("DONE: the server now has one model updated by both fictional clients.")
        print("In real federated learning, clients would be separate organizations/devices.")
        print("This exercise demonstrates the mechanics; it proves no privacy or real-world accuracy.")
    return summary


def run_federation(site_id: str, round_id: str) -> None:
    from .db import audit, connect

    try:
        summary = simulate_synthetic_federation()
        with connect() as db:
            db.execute("UPDATE federation_rounds SET status='complete',schema_hash=?,participating_json=?,results_json=? WHERE id=?",
                       (hashlib.sha256(b"logistic:8:synthetic-demo-v1").hexdigest(),
                        json.dumps(list(summary["clients"])), json.dumps(summary), round_id))
            audit(db, None, site_id, "fedavg_complete", "federation_round", round_id)
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE federation_rounds SET status='failed',results_json=? WHERE id=?",
                       (json.dumps({"error": str(exc)}), round_id))
            audit(db, None, site_id, "fedavg_failed", "federation_round", round_id, json.dumps({"error": str(exc)}))
