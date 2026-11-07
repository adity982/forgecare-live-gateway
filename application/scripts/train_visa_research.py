"""Reproducible VisA research: patch anomaly baseline and Flower FedAvg CNN.

Runs on a real public industrial dataset. It never registers these models for phone inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import flwr as fl
import joblib
import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
SIDE = 64


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)


def image(path: Path) -> np.ndarray:
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"Unreadable real image: {path}")
    return cv2.cvtColor(cv2.resize(pixels, (SIDE, SIDE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def list_images(root: Path):
    normal = sorted((root / "Data" / "Images" / "Normal").glob("*.JPG"))
    anomaly = sorted((root / "Data" / "Images" / "Anomaly").glob("*.JPG"))
    if len(normal) < 50 or len(anomaly) < 20:
        raise ValueError("Expected a complete VisA category with both real image classes")
    paths = np.array(normal + anomaly, dtype=object)
    labels = np.array([0] * len(normal) + [1] * len(anomaly), dtype=np.int64)
    train, remainder = train_test_split(np.arange(len(paths)), test_size=.4, stratify=labels, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=.5, stratify=labels[remainder], random_state=SEED)
    return paths, labels, {"train": train, "validation": validation, "test": test}


def patch_features(pixels: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    tiny = cv2.resize(hsv, (32, 32), interpolation=cv2.INTER_AREA)
    return tiny.reshape(8, 4, 8, 4, 3).transpose(0, 2, 1, 3, 4).reshape(64, 48)


def anomaly_run(images: np.ndarray, labels: np.ndarray, splits: dict, out: Path) -> dict:
    normal_train = [i for i in splits["train"] if labels[i] == 0]
    patches = np.concatenate([patch_features(images[i]) for i in normal_train])
    rng = np.random.default_rng(SEED)
    subset = patches[rng.choice(len(patches), min(len(patches), 12000), replace=False)]
    model = MiniBatchKMeans(n_clusters=96, batch_size=1024, n_init=3, random_state=SEED)
    model.fit(subset)

    def maps(indices):
        features = np.stack([patch_features(images[i]) for i in indices])
        squared = ((features[:, :, None, :] - model.cluster_centers_[None, None, :, :]) ** 2).sum(axis=-1)
        return np.sqrt(squared.min(axis=-1)).reshape(-1, 8, 8)

    val_maps = maps(splits["validation"])
    val_normal = val_maps[labels[splits["validation"]] == 0].max(axis=(1, 2))
    threshold = float(np.quantile(val_normal, .95))
    test_maps = maps(splits["test"])
    score = test_maps.max(axis=(1, 2))
    metrics = classification_metrics(labels[splits["test"]], score, threshold)
    joblib.dump({"model": model, "threshold": threshold, "side": SIDE,
                 "category": "capsules", "scope": "VisA research only"}, out / "patch_baseline.joblib")
    for index, category in enumerate((0, 1)):
        rows = np.where(labels[splits["test"]] == category)[0]
        sample = int(rows[np.argsort(score[rows])[len(rows) // 2]])
        heatmap = cv2.resize(test_maps[sample], (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlay = cv2.addWeighted(images[splits["test"]][sample], .62,
                                  cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB), .38, 0)
        cv2.imwrite(str(out / f"patch_example_{'normal' if category == 0 else 'anomaly'}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {"method": "normal-only patch k-means baseline (Anomalib-inspired, not Anomalib)",
            "train_normal_images": len(normal_train), "patches_fit": len(subset),
            "patch_size": 4, "prototypes": 96, "threshold_from": "95th percentile of validation normal scores",
            **metrics}


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).flatten()


def parameters(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def load_parameters(model: nn.Module, values: list[np.ndarray]) -> None:
    expected = model.state_dict()
    if len(values) != len(expected):
        raise ValueError("Wrong model parameter count")
    model.load_state_dict({name: torch.from_numpy(value).to(dtype=tensor.dtype)
                           for (name, tensor), value in zip(expected.items(), values)})


class ImageClient(fl.client.NumPyClient):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32) / 255)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.model = SmallCNN()

    def get_parameters(self, config):
        return parameters(self.model)

    def fit(self, values, config):
        load_parameters(self.model, values)
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.002)
        self.model.train()
        losses = []
        for _ in range(2):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return parameters(self.model), len(self.x), {"loss": float(np.mean(losses))}


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"test_images": int(len(y)), "test_normal": int((y == 0).sum()),
            "test_anomaly": int((y == 1).sum()), "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y, score)), "accuracy": float(accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def predict(model: nn.Module, pixels: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255)
        return torch.sigmoid(model(x)).numpy()


def client_partitions(labels: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(train_indices.copy(), test_size=.5,
                            stratify=labels[train_indices], random_state=SEED)


def federated_run(images: np.ndarray, labels: np.ndarray, splits: dict,
                  partitions: tuple[np.ndarray, np.ndarray], out: Path) -> dict:
    client_a, client_b = partitions
    clients = [ImageClient(images[part], labels[part]) for part in (client_a, client_b)]
    model = SmallCNN()
    global_values = parameters(model)
    strategy = FedAvg(inplace=False, accept_failures=False)
    history = []
    for server_round in range(1, 4):
        results = []
        for client in clients:
            values, count, metrics = client.fit(global_values, {})
            results.append((None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(values), count, metrics)))
        combined, _ = strategy.aggregate_fit(server_round, results, [])
        if combined is None:
            raise RuntimeError("Flower refused the client updates")
        global_values = parameters_to_ndarrays(combined)
        load_parameters(model, global_values)
        validation_score = predict(model, images[splits["validation"]])
        history.append({"round": server_round, "client_train_counts": [len(client_a), len(client_b)],
                        "client_losses": [float(result.metrics["loss"]) for _, result in results],
                        "validation_auc": float(roc_auc_score(labels[splits["validation"]], validation_score))})
    validation_score = predict(model, images[splits["validation"]])
    thresholds = np.linspace(.1, .9, 33)
    threshold = float(max(thresholds, key=lambda value: f1_score(labels[splits["validation"]], validation_score >= value)))
    test_score = predict(model, images[splits["test"]])
    torch.save({"state_dict": model.state_dict(), "threshold": threshold, "category": "capsules",
                "scope": "VisA research only"}, out / "flower_fedavg_cnn.pt")
    torch.jit.trace(model.eval(), torch.zeros(1, 3, SIDE, SIDE)).save(str(out / "flower_fedavg_cnn_scripted.pt"))
    client_metrics = []
    for part in (client_a, client_b):
        client_metrics.append({"train_count": int(len(part)), "normal": int((labels[part] == 0).sum()),
                               "anomaly": int((labels[part] == 1).sum())})
    return {"method": "Flower FedAvg with two disjoint local real-image clients", "rounds": 3,
            "local_epochs_per_round": 2, "client_partitions": client_metrics, "history": history,
            "topology": "single laptop, two in-process clients; no independent factories or privacy guarantee",
            **classification_metrics(labels[splits["test"]], test_score, threshold)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Extracted VisA category directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seed_all()
    args.out.mkdir(parents=True, exist_ok=True)
    paths, labels, splits = list_images(args.data)
    partitions = client_partitions(labels, splits["train"])
    images = np.stack([image(path) for path in paths])
    with (args.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_path", "sha256", "split", "label", "client_id", "origin"])
        assignment = {int(index): name for name, indices in splits.items() for index in indices}
        client = {int(index): f"client-{number}" for number, part in enumerate(partitions, start=1)
                  for index in part}
        for index, path in enumerate(paths):
            writer.writerow([str(path), hashlib.sha256(path.read_bytes()).hexdigest(), assignment[index],
                             "anomaly" if labels[index] else "normal", client.get(index, ""),
                             "Amazon VisA 20220922 capsules"])
    patch = anomaly_run(images, labels, splits, args.out)
    print("Patch baseline complete", flush=True)
    flower = federated_run(images, labels, splits, partitions, args.out)
    result = {"dataset": "Amazon VisA 20220922, capsules", "source": "https://github.com/amazon-science/spot-diff",
              "license": "CC BY 4.0", "seed": SEED, "image_size": SIDE,
              "counts": {name: {"normal": int((labels[indices] == 0).sum()),
                                 "anomaly": int((labels[indices] == 1).sum())} for name, indices in splits.items()},
              "patch_baseline": patch, "flower_fedavg": flower,
              "limits": "Research benchmark only. No phone or automotive validation; clients share one laptop."}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
