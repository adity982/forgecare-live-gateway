"""Train a genuine two-client Flower model on unaugmented cast impeller photos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import flwr as fl
import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
SIDE = 96
EPOCHS_PER_ROUND = 3
ROUNDS = 3
SOURCE = "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CastingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 12 * 12, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).flatten()


def load_data(root: Path):
    folder = root / "casting_512x512" / "casting_512x512"
    normal = sorted((folder / "ok_front").glob("*.jpeg"))
    defects = sorted((folder / "def_front").glob("*.jpeg"))
    if len(normal) != 519 or len(defects) != 781:
        raise ValueError(f"Expected unaugmented 519/781 originals, got {len(normal)}/{len(defects)}")
    paths = np.array(normal + defects, dtype=object)
    labels = np.array([0] * len(normal) + [1] * len(defects), dtype=np.int64)
    images = []
    hashes = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        images.append(cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if len(set(hashes)) != len(paths):
        raise ValueError("Duplicate source bytes need grouping before splitting")
    return paths, labels, np.stack(images)[:, None].astype(np.float32) / 255, hashes


def split_data(labels):
    train, remainder = train_test_split(np.arange(len(labels)), test_size=.3,
                                        stratify=labels, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=.5,
                                        stratify=labels[remainder], random_state=SEED)
    first, second = train_test_split(train, test_size=.5, stratify=labels[train], random_state=SEED)
    return {"train": train, "validation": validation, "test": test}, (first, second)


def arrays(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_arrays(model, values):
    current = model.state_dict()
    model.load_state_dict({key: torch.from_numpy(value).to(dtype=original.dtype)
                           for (key, original), value in zip(current.items(), values)})


class CastingClient(fl.client.NumPyClient):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.model = CastingCNN()

    def get_parameters(self, config):
        return arrays(self.model)

    def fit(self, values, config):
        set_arrays(self.model, values)
        loader = DataLoader(TensorDataset(self.images, self.labels), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        positive_weight = torch.tensor([(len(self.labels) - self.labels.sum()) / self.labels.sum()])
        losses = []
        self.model.train()
        for _ in range(EPOCHS_PER_ROUND):
            for inputs, targets in loader:
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self.model(inputs), targets,
                                                                       pos_weight=positive_weight)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return arrays(self.model), len(self.images), {"loss": float(np.mean(losses))}


def scores(model, images):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            result.extend(torch.sigmoid(model(torch.from_numpy(images[start:start + 64]))).numpy().tolist())
    return np.array(result)


def metrics(labels, score, threshold):
    predicted = score >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"images": int(len(labels)), "normal": int((labels == 0).sum()),
            "defective": int((labels == 1).sum()), "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(labels, score)),
            "accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    started_at = utc_now()
    paths, labels, images, hashes = load_data(args.data)
    splits, partitions = split_data(labels)
    assignment = {int(index): name for name, indices in splits.items() for index in indices}
    clients_by_index = {int(index): f"client-{number}" for number, part in enumerate(partitions, 1)
                        for index in part}
    with (args.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["source_path", "sha256", "split", "label", "client_id"])
        for index, path in enumerate(paths):
            writer.writerow([str(path), hashes[index], assignment[index],
                             "defective" if labels[index] else "acceptable", clients_by_index.get(index, "")])
    clients = [CastingClient(images[part], labels[part]) for part in partitions]
    global_model = CastingCNN()
    global_values = arrays(global_model)
    strategy = FedAvg(inplace=False, accept_failures=False)
    history = []
    for number in range(1, ROUNDS + 1):
        round_started = utc_now()
        updates = []
        for client in clients:
            values, count, result = client.fit(global_values, {})
            updates.append((None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(values), count, result)))
        aggregated, _ = strategy.aggregate_fit(number, updates, [])
        if aggregated is None:
            raise RuntimeError("Flower rejected the client updates")
        global_values = parameters_to_ndarrays(aggregated)
        set_arrays(global_model, global_values)
        validation_score = scores(global_model, images[splits["validation"]])
        record = {"round": number, "started_at_utc": round_started, "finished_at_utc": utc_now(),
                  "client_examples": [len(part) for part in partitions],
                  "client_losses": [float(item.metrics["loss"]) for _, item in updates],
                  "validation_auc": float(roc_auc_score(labels[splits["validation"]], validation_score))}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation_score = scores(global_model, images[splits["validation"]])
    threshold = float(max(np.linspace(.1, .9, 81),
                          key=lambda value: f1_score(labels[splits["validation"]], validation_score >= value)))
    test = splits["test"]
    test_score = scores(global_model, images[test])
    torch.save({"state_dict": global_model.state_dict(), "threshold": threshold,
                "dataset": "Kaggle unaugmented cast pump impellers", "scope": "research only"},
               args.out / "flower_casting_cnn.pt")
    torch.jit.trace(global_model.eval(), torch.zeros(1, 1, SIDE, SIDE)).save(str(args.out / "flower_casting_cnn_scripted.pt"))
    test_a, test_b = train_test_split(test, test_size=.5, stratify=labels[test], random_state=SEED)
    result = {"dataset": "Cast pump impeller surface inspection", "source": SOURCE,
              "license": "CC BY-NC-ND 4.0; local educational research only",
              "original_images": len(paths), "image_size": SIDE, "seed": SEED,
              "started_at_utc": started_at, "finished_at_utc": utc_now(),
              "topology": "two in-process Flower clients on one laptop; no site isolation or privacy guarantee",
              "rounds": ROUNDS, "epochs_per_round": EPOCHS_PER_ROUND, "batch_size": 32,
              "optimizer": "Adam 0.001", "threshold_selection": "maximum validation F1",
              "split_counts": {name: {"acceptable": int((labels[part] == 0).sum()),
                                      "defective": int((labels[part] == 1).sum())}
                               for name, part in splits.items()},
              "client_counts": [{"id": f"client-{n}", "images": len(part),
                                 "acceptable": int((labels[part] == 0).sum()),
                                 "defective": int((labels[part] == 1).sum())}
                                for n, part in enumerate(partitions, 1)],
              "history": history, "test": metrics(labels[test], test_score, threshold),
              "client_holdouts": [{"id": f"client-{n}", **metrics(labels[part], scores(global_model, images[part]), threshold)}
                                  for n, part in enumerate((test_a, test_b), 1)],
              "limitations": "Research only. Pump impeller photos are not car- or phone-specific; no independent production line validation. Source says the alternate 300px set is pre-augmented, so this run uses the 512px originals."}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"finished_at_utc": result["finished_at_utc"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
