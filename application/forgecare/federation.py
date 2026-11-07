"""Two logical local clients; raw rows stay in client arrays during FedAvg."""
from __future__ import annotations

import hashlib
import json
import uuid

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from .db import audit, connect, now

VISUAL_FEATURE_COUNT = 275  # gray16 (256), histogram16, edge/mean/std (3)


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


def run_federation(site_id: str, round_id: str) -> None:
    try:
        # Explicitly synthetic curriculum fixture until independent site labels exist.
        X, y = make_classification(n_samples=700, n_features=12, n_informative=8, weights=[.65, .35], random_state=42)
        X = X.astype(np.float64) / 5
        client_a = np.arange(0, 350)
        client_b = np.arange(350, 700)
        clients = {}
        for name, indices in (("logical-camera-A", client_a), ("logical-camera-B", client_b)):
            train, test = train_test_split(indices, test_size=.25, stratify=y[indices], random_state=42)
            clients[name] = (train, test)
        start = np.zeros(X.shape[1] + 1, dtype=np.float64)
        schema_hash = hashlib.sha256(f"logistic:{X.shape[1]}:v1".encode()).hexdigest()
        updates = []
        local_results = {}
        for name, (train, test) in clients.items():
            weights = train_client(start, X[train], y[train])
            updates.append({"round_id": round_id, "schema_hash": schema_hash, "client_id": name,
                            "count": int(len(train)), "weights": weights})
            pred = sigmoid(X[test] @ weights[:-1] + weights[-1]) >= .5
            local_results[name] = {"train_count": len(train), "test_count": len(test),
                                   "local_accuracy": float(accuracy_score(y[test], pred)),
                                   "local_recall": float(recall_score(y[test], pred, zero_division=0))}
        fed = aggregate(updates, round_id=round_id, schema_hash=schema_hash, start=start)
        for name, (_, test) in clients.items():
            pred = sigmoid(X[test] @ fed[:-1] + fed[-1]) >= .5
            local_results[name]["federated_accuracy"] = float(accuracy_score(y[test], pred))
            local_results[name]["federated_recall"] = float(recall_score(y[test], pred, zero_division=0))
        summary = {"dataset": "synthetic_learning_fixture", "mechanism": "sample_weighted_FedAvg_logistic",
                   "claim": "mechanics demonstration only; no independent-site evidence", "clients": local_results,
                   "worst_client_federated_recall": min(x["federated_recall"] for x in local_results.values()),
                   "weights": fed.round(6).tolist()}
        with connect() as db:
            db.execute("UPDATE federation_rounds SET status='complete',schema_hash=?,participating_json=?,results_json=? WHERE id=?",
                       (schema_hash, json.dumps(list(clients)), json.dumps(summary), round_id))
            audit(db, None, site_id, "fedavg_complete", "federation_round", round_id)
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE federation_rounds SET status='failed',results_json=? WHERE id=?", (json.dumps({"error": str(exc)}), round_id))
            audit(db, None, site_id, "fedavg_failed", "federation_round", round_id, json.dumps({"error": str(exc)}))
