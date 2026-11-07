"""Simulates 10 rounds of Federated Learning for web visualization."""
from __future__ import annotations

import hashlib
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from .db import audit, connect

VISUAL_FEATURE_COUNT = 275

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
        weighted += count * weight
        total += count
    return weighted / total

# This function can now be connected to a Web API route (e.g., FastAPI or Flask)
# triggered by the website's button click.
def run_10_round_simulation(site_id: str, simulation_id: str) -> dict:
    try:
        X, y = make_classification(n_samples=700, n_features=12, n_informative=8, weights=[.65, .35], random_state=42)
        X = X.astype(np.float64) / 5
        client_a = np.arange(0, 350)
        client_b = np.arange(350, 700)
        clients = {}
        for name, indices in (("logical-camera-A", client_a), ("logical-camera-B", client_b)):
            train, test = train_test_split(indices, test_size=.25, stratify=y[indices], random_state=42)
            clients[name] = (train, test)
            
        start_weights = np.zeros(X.shape[1] + 1, dtype=np.float64)
        schema_hash = hashlib.sha256(f"logistic:{X.shape[1]}:v1".encode()).hexdigest()
        
        simulation_history = []

        # LOOP: Run 10 consecutive rounds
        for round_num in range(1, 11):
            round_id = f"{simulation_id}-round-{round_num}"
            updates = []
            local_results = {}
            
            # 1. Local Training
            for name, (train, test) in clients.items():
                weights = train_client(start_weights, X[train], y[train])
                updates.append({
                    "round_id": round_id, "schema_hash": schema_hash, "client_id": name,
                    "count": int(len(train)), "weights": weights
                })
                pred = sigmoid(X[test] @ weights[:-1] + weights[-1]) >= .5
                local_results[name] = {
                    "train_count": len(train), "test_count": len(test),
                    "local_accuracy": float(accuracy_score(y[test], pred)),
                    "local_recall": float(recall_score(y[test], pred, zero_division=0))
                }
                
            # 2. Server Aggregation
            fed = aggregate(updates, round_id=round_id, schema_hash=schema_hash, start=start_weights)
            
            # 3. Federated Evaluation
            for name, (_, test) in clients.items():
                pred = sigmoid(X[test] @ fed[:-1] + fed[-1]) >= .5
                local_results[name]["federated_accuracy"] = float(accuracy_score(y[test], pred))
                local_results[name]["federated_recall"] = float(recall_score(y[test], pred, zero_division=0))
            
            # 4. Save round history and UPDATE global weights for the next round
            simulation_history.append({
                "round": round_num, 
                "clients": local_results,
                "global_weights": fed.round(6).tolist()
            })
            start_weights = fed  # CRITICAL: Next round starts where this round left off

        summary = {
            "dataset": "synthetic_learning_fixture",
            "mechanism": "sample_weighted_FedAvg_logistic",
            "history": simulation_history,
            "final_federated_recall": min(x["federated_recall"] for x in local_results.values())
        }
        
        with connect() as db:
            db.execute("INSERT INTO federation_simulations (id, status, schema_hash, results_json) VALUES (?, 'complete', ?, ?)",
                       (simulation_id, schema_hash, json.dumps(summary)))
            audit(db, None, site_id, "fedavg_simulation_complete", "federation_simulation", simulation_id)
            
        return {"status": "success", "data": summary}

    except Exception as exc:
        with connect() as db:
            audit(db, None, site_id, "fedavg_simulation_failed", "federation_simulation", simulation_id, json.dumps({"error": str(exc)}))
        return {"status": "error", "message": str(exc)}
