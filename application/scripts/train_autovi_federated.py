"""Train two real Flower clients on an adapted, grouped AutoVI pipe-clip split."""
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.train_casting_research import CastingCNN, CastingClient, arrays, metrics, scores, set_arrays


SEED = 42
SIDE = 96


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(root: Path):
    category = root / "pipe_clip"
    locations = [(category / "train" / "good", 0), (category / "test" / "good", 0),
                 (category / "test" / "operator", 1), (category / "test" / "unclipped", 1)]
    paths, labels = [], []
    for folder, label in locations:
        found = sorted(folder.glob("*.png"))
        paths.extend(found)
        labels.extend([label] * len(found))
    if len(paths) != 532:
        raise ValueError(f"Expected 532 AutoVI pipe clip photos, got {len(paths)}")
    original_count = len(paths)
    seen = set()
    unique = []
    for path, label in zip(paths, labels):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append((path, label, digest))
    paths, labels, hashes = map(list, zip(*unique))
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        images.append(cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA))
    return paths, np.array(labels), np.stack(images)[:, None].astype(np.float32) / 255, hashes, original_count - len(paths)


def grouped_split(paths, labels):
    # Blocks of ten neighbouring file numbers remain together across all splits.
    groups = [f"{path.parent.relative_to(path.parents[3])}:{int(path.stem) // 10}"
              for path in paths]
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    keys = sorted(members)
    class_by_group = [int(labels[members[key][0]]) for key in keys]
    training, remainder = train_test_split(np.arange(len(keys)), test_size=.3,
                                           stratify=class_by_group, random_state=SEED)
    validation, testing = train_test_split(remainder, test_size=.5,
                                           stratify=np.array(class_by_group)[remainder], random_state=SEED)
    split = {name: np.array([index for key_id in key_ids for index in members[keys[key_id]]])
             for name, key_ids in (("train", training), ("validation", validation), ("test", testing))}
    first, second = train_test_split(split["train"], test_size=.5,
                                     stratify=labels[split["train"]], random_state=SEED)
    return split, (first, second), groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    started = utc_now()
    paths, labels, images, hashes, excluded_duplicates = load(args.data)
    split, clients, groups = grouped_split(paths, labels)
    assignment = {int(index): name for name, ids in split.items() for index in ids}
    client_by_index = {int(index): f"client-{number}" for number, ids in enumerate(clients, 1) for index in ids}
    with (args.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["source_path", "sha256", "split", "label", "group", "client_id"])
        writer.writerows([str(path), hashes[index], assignment[index],
                          "defective" if labels[index] else "acceptable", groups[index],
                          client_by_index.get(index, "")] for index, path in enumerate(paths))
    flower_clients = [CastingClient(images[ids], labels[ids]) for ids in clients]
    model = CastingCNN()
    values = arrays(model)
    strategy = FedAvg(inplace=False, accept_failures=False)
    history = []
    for number in range(1, 4):
        round_started = utc_now()
        updates = []
        for client in flower_clients:
            trained, count, info = client.fit(values, {})
            updates.append((None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(trained), count, info)))
        aggregate, _ = strategy.aggregate_fit(number, updates, [])
        if aggregate is None:
            raise RuntimeError("Flower rejected the client updates")
        values = parameters_to_ndarrays(aggregate)
        set_arrays(model, values)
        validation_score = scores(model, images[split["validation"]])
        record = {"round": number, "started_at_utc": round_started, "finished_at_utc": utc_now(),
                  "client_examples": [len(ids) for ids in clients],
                  "client_losses": [float(item.metrics["loss"]) for _, item in updates],
                  "validation_auc": float(roc_auc_score(labels[split["validation"]], validation_score))}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation_score = scores(model, images[split["validation"]])
    threshold = float(max(np.linspace(.1, .9, 81),
                          key=lambda value: f1_score(labels[split["validation"]], validation_score >= value)))
    test = split["test"]
    test_score = scores(model, images[test])
    torch.save({"state_dict": model.state_dict(), "threshold": threshold,
                "dataset": "Renault AutoVI pipe clip adapted supervised split", "scope": "research only"},
               args.out / "flower_autovi_cnn.pt")
    torch.jit.trace(model.eval(), torch.zeros(1, 1, SIDE, SIDE)).save(str(args.out / "flower_autovi_cnn_scripted.pt"))
    result = {"dataset": "Renault AutoVI pipe clip · adapted supervised split",
              "source": "https://zenodo.org/records/10459003", "license": "CC BY-NC-SA 4.0; local educational research only",
              "source_images": len(paths), "excluded_exact_duplicate_images": excluded_duplicates,
              "seed": SEED, "image_size": SIDE,
              "started_at_utc": started, "finished_at_utc": utc_now(),
              "topology": "two in-process Flower clients on one laptop; no site isolation",
              "method": "96px grayscale CNN, Adam 0.001, three FedAvg rounds, three local epochs per round",
              "split_policy": "Adapted supervised split of original AutoVI photos after excluding exact duplicate bytes; groups of ten adjacent file numbers kept together. Not comparable to original normal-only test protocol.",
              "split_counts": {name: {"acceptable": int((labels[ids] == 0).sum()),
                                      "defective": int((labels[ids] == 1).sum())} for name, ids in split.items()},
              "client_counts": [{"id": f"client-{number}", "images": len(ids)} for number, ids in enumerate(clients, 1)],
              "history": history, "test": metrics(labels[test], test_score, threshold),
              "limitations": "Research only. Grouped adjacent frames reduce but cannot eliminate related-component leakage. No independent factory-site evaluation. Raw images reside together on one laptop."}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"test": result["test"], "split_counts": result["split_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
