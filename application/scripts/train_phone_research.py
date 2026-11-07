"""Fit a crack-screen research classifier on the public phone sample images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


SEED = 42
SOURCE = "https://www.kaggle.com/datasets/axondata/cracked-and-intact-smartphone-images-dataset"


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return ((torch.from_numpy(array).permute(2, 0, 1) -
             torch.tensor([.485, .456, .406])[:, None, None]) /
            torch.tensor([.229, .224, .225])[:, None, None])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra-cracked", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    intact = sorted((args.data / "Image_phones").glob("*.png"))
    cracked = sorted((args.data / "Image_brokenphones").glob("*.PNG"))
    external = sorted((args.extra_cracked / "selected_zip").glob("*.jpg")) if args.extra_cracked else []
    readable_external = []
    unreadable_external = []
    for path in external:
        try:
            with Image.open(path) as source:
                source.load()
        except OSError:
            unreadable_external.append(str(path))
        else:
            readable_external.append(path)
    paths = intact + cracked + readable_external
    labels = np.array([0] * len(intact) + [1] * (len(cracked) + len(readable_external)), dtype=int)
    origins = ["axon_intact"] * len(intact) + ["axon_cracked"] * len(cracked) + ["datacluster_cracked"] * len(readable_external)
    if len(paths) < 50 or len(paths) != len(labels):
        raise ValueError("Expected a usable sample of both intact and cracked phones")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if len(set(hashes)) != len(paths):
        raise ValueError("Duplicate image bytes must be grouped before splitting")
    if readable_external:
        training, validating, testing = [], [], []
        for origin in sorted(set(origins)):
            group = np.array([i for i, item in enumerate(origins) if item == origin])
            group_train, group_rest = train_test_split(group, test_size=.4, random_state=SEED)
            group_val, group_test = train_test_split(group_rest, test_size=.5, random_state=SEED)
            training.extend(group_train)
            validating.extend(group_val)
            testing.extend(group_test)
        train, val, test = np.array(training), np.array(validating), np.array(testing)
    else:
        train, rest = train_test_split(np.arange(len(paths)), test_size=.4, stratify=labels, random_state=SEED)
        val, test = train_test_split(rest, test_size=.5, stratify=labels[rest], random_state=SEED)
    split_by_index = {int(i): name for name, indices in (("train", train), ("validation", val), ("test", test)) for i in indices}
    with (args.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_path", "sha256", "split", "label", "dataset"])
        writer.writerows([str(path), hashes[i], split_by_index[i], "cracked" if labels[i] else "intact", origins[i]]
                         for i, path in enumerate(paths))
    started = datetime.now(timezone.utc).isoformat()
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), 8):
            batch = torch.stack([image_tensor(path) for path in paths[start:start + 8]])
            features.extend(backbone(batch).numpy())
    features = np.asarray(features)
    options = []
    for c in (.01, .1, 1., 10.):
        classifier = LogisticRegression(C=c, max_iter=2000, random_state=SEED,
                                        class_weight="balanced" if readable_external else None)
        classifier.fit(features[train], labels[train])
        probability = classifier.predict_proba(features[val])[:, 1]
        for threshold in (.3, .4, .5, .6, .7):
            options.append((f1_score(labels[val], probability >= threshold), c, threshold, classifier))
    _, selected_c, threshold, classifier = max(options, key=lambda item: (item[0], -abs(item[2] - .5), -item[1]))
    test_probability = classifier.predict_proba(features[test])[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels[test], test_probability >= threshold, labels=[0, 1]).ravel()
    head = nn.Linear(512, 1)
    head.weight.data = torch.tensor(classifier.coef_.astype(np.float32))
    head.bias.data = torch.tensor(classifier.intercept_.astype(np.float32))
    model = nn.Sequential(backbone, head).eval()
    torch.jit.trace(model, torch.zeros(1, 3, 224, 224)).save(str(args.out / "phone_crack_screen_scripted.pt"))
    torch.save({"state_dict": model.state_dict(), "threshold": threshold,
                "scope": "research only; screen crack versus intact; no dent or scratch counts"},
               args.out / "phone_crack_screen.pt")
    result = {
        "dataset": "Public cracked/intact smartphone sample plus cracked-screen photo sample" if readable_external else "Public cracked/intact smartphone sample",
        "source": [SOURCE, "https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset"] if readable_external else [SOURCE],
        "license": "Axon: CC BY-NC 4.0; DataCluster page: CC0 metadata with ownership/licensing caveat in description; local educational research only",
        "unreadable_external_images": unreadable_external,
        "images": len(paths), "split_counts": {name: {"intact": int((labels[idx] == 0).sum()),
                                                       "cracked": int((labels[idx] == 1).sum())}
                                           for name, idx in (("train", train), ("validation", val), ("test", test))},
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "ImageNet ResNet18 frozen features plus fitted balanced logistic regression" if readable_external else "ImageNet ResNet18 frozen features plus fitted logistic regression",
        "seed": SEED, "selected_c": selected_c, "threshold": threshold,
        "test": {"images": len(test), "intact": int((labels[test] == 0).sum()),
                 "cracked": int((labels[test] == 1).sum()),
                 "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                 "roc_auc": float(roc_auc_score(labels[test], test_probability))},
        "test_by_dataset": {origin: {"photos": int(sum(origins[i] == origin for i in test)),
                                     "flagged": int(sum(origins[i] == origin and test_probability[j] >= threshold for j, i in enumerate(test)))}
                            for origin in sorted(set(origins))},
        "limitations": "Public samples lack physical-phone IDs or near-duplicate groups; backgrounds and source differences may drive predictions. The external cracked-only set has no intact controls. No dent or scratch annotations. No iPhone 14 Plus or realme validation."
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
