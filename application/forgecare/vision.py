"""Leakage-controlled visual condition comparison and guarded model promotion."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import DATA
from .db import audit, connect, now, rows


def image_features(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image {path}")
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    histogram = cv2.calcHist([image], [0], None, [16], [0, 256]).ravel()
    histogram = histogram / max(histogram.sum(), 1)
    edges = cv2.Canny(image, 80, 180)
    return np.concatenate([small.ravel(), histogram, np.array([edges.mean()/255, image.mean()/255, image.std()/255], dtype=np.float32)])


def grouped_splits(y: np.ndarray, groups: np.ndarray, seed: int = 42):
    if min(np.unique(groups[y == 0]).size, np.unique(groups[y == 1]).size) < 3:
        raise ValueError("Need at least three distinct physical samples in each class; five are recommended")
    indices = np.arange(len(y))
    for offset in range(50):
        outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + offset)
        development, test = next(outer.split(indices, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + offset + 101)
        train_relative, validation_relative = next(inner.split(development, y[development], groups[development]))
        train, validation = development[train_relative], development[validation_relative]
        if all(len(np.unique(y[part])) == 2 for part in (train, validation, test)):
            return train, validation, test
    raise ValueError("Could not form train/validation/test sets containing both classes without reusing a physical sample")


def candidates():
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced")),
        "knn": make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "decision_tree": make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        "svm": make_pipeline(SimpleImputer(), StandardScaler(), SVC(class_weight="balanced", random_state=42)),
        "gaussian_nb": make_pipeline(SimpleImputer(), StandardScaler(), GaussianNB()),
        "random_forest": make_pipeline(SimpleImputer(), RandomForestClassifier(n_estimators=80, max_depth=8, class_weight="balanced", random_state=42, n_jobs=2)),
        "adaboost": make_pipeline(SimpleImputer(), AdaBoostClassifier(n_estimators=60, random_state=42)),
        "perceptron": make_pipeline(SimpleImputer(), StandardScaler(), Perceptron(class_weight="balanced", random_state=42)),
    }


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1 / (1 + np.exp(-np.clip(raw, -30, 30)))
    return model.predict(X).astype(float)


def find_threshold(y, probabilities):
    # Validation-only threshold; no test-set tuning.
    thresholds = np.linspace(.1, .9, 17)
    results = []
    for threshold in thresholds:
        pred = probabilities >= threshold
        fp = ((pred == 1) & (y == 0)).sum()
        negatives = (y == 0).sum()
        fpr = fp / max(negatives, 1)
        results.append((recall_score(y, pred, zero_division=0), -fpr, float(threshold)))
    allowed = [x for x in results if -x[1] <= .20]
    return max(allowed or results)[2]


def report(y, probability, threshold):
    pred = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return {"n": len(y), "positives": int(y.sum()), "threshold": round(float(threshold), 3),
            "average_precision": round(float(average_precision_score(y, probability)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "false_alert_rate": round(fp / max(fp + tn, 1), 4),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def train_visual(site_id: str, model_id: str, family: str, equipment_model: str) -> None:
    try:
        with connect() as db:
            records = rows(db, """SELECT c.image_path,c.sample_id,c.session_id,c.id AS capture_id,l.binary_label
              FROM captures c JOIN labels l ON c.id=l.capture_id JOIN assets a ON a.id=c.asset_id
              WHERE c.site_id=? AND a.family=? AND a.model=? AND c.quality='ok' AND l.binary_label IS NOT NULL ORDER BY c.captured_at""",
                           (site_id, family, equipment_model))
        if len(records) < 24:
            raise ValueError("Need at least 24 quality-accepted, reviewed captures")
        X = np.stack([image_features(r["image_path"]) for r in records])
        y = np.array([r["binary_label"] for r in records], dtype=int)
        groups = np.array([r["sample_id"] for r in records])
        train, val, test = grouped_splits(y, groups)
        dataset_hash = hashlib.sha256("|".join(r["capture_id"] for r in records).encode()).hexdigest()
        results = {}
        trained = {}
        for name, model in candidates().items():
            try:
                model.fit(X[train], y[train])
                prob = scores(model, X[val])
                threshold = find_threshold(y[val], prob)
                results[name] = {"validation": report(y[val], prob, threshold), "threshold": threshold}
                trained[name] = model
            except Exception as exc:
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        viable = [(name, item["validation"]["average_precision"]) for name, item in results.items() if "validation" in item and name != "dummy"]
        if not viable:
            raise ValueError("All non-baseline models failed")
        best_name = max(viable, key=lambda x: x[1])[0]
        best_model = trained[best_name]
        threshold = results[best_name]["threshold"]
        test_report = report(y[test], scores(best_model, X[test]), threshold)
        baseline = results["dummy"]["validation"] if "validation" in results["dummy"] else {}
        # A deliberately conservative demo gate. Independent factory validation is separate.
        passed = test_report["recall"] >= .7 and test_report["false_alert_rate"] <= .2 and test_report["positives"] >= 2
        artifact = DATA / "models" / f"{model_id}.joblib"
        joblib.dump({"model": best_model, "threshold": threshold, "feature_version": "gray16-hist16-v1"}, artifact)
        summary = {"dataset": "reviewed_live_captures", "family": family, "equipment_model": equipment_model,
                   "split": "physical_sample_grouped_60_20_20", "train": len(train), "validation": len(val), "test": len(test),
                   "groups": {"train": int(np.unique(groups[train]).size), "validation": int(np.unique(groups[val]).size), "test": int(np.unique(groups[test]).size)},
                   "models": results, "selected": best_name, "baseline_validation": baseline, "selected_test": test_report,
                   "promotion_gate_passed": passed, "gate": {"recall_min": .7, "false_alert_rate_max": .2, "test_positives_min": 2}}
        with connect() as db:
            db.execute("UPDATE models SET status=?,artifact_path=?,dataset_hash=?,metrics_json=? WHERE id=?",
                       ("candidate" if passed else "research", str(artifact), dataset_hash, json.dumps(summary), model_id))
            audit(db, None, site_id, "train_complete", "model", model_id, json.dumps({"gate_passed": passed}))
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE models SET status='insufficient_data',metrics_json=? WHERE id=?", (json.dumps({"error": f"{type(exc).__name__}: {exc}"}), model_id))
            audit(db, None, site_id, "train_failed", "model", model_id, json.dumps({"error": str(exc)}))


def predict(path: str | Path, artifact_path: str | Path) -> tuple[str, float]:
    bundle = joblib.load(artifact_path)
    if "weights" in bundle:
        from .federation import sigmoid
        features = image_features(path)
        weights = np.asarray(bundle["weights"], dtype=np.float64)
        if len(weights) != len(features) + 1:
            raise ValueError("Federated model feature schema mismatch")
        probability = float(sigmoid(features @ weights[:-1] + weights[-1]))
    else:
        probability = float(scores(bundle["model"], image_features(path)[None, :])[0])
    return ("review_needed" if probability >= bundle["threshold"] else "normal"), probability
