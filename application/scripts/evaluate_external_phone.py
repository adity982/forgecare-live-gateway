"""Check the saved phone crack classifier on a separate positive-only dataset."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from forgecare.phone_demo import inspect, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.data / "selected_zip").glob("*.jpg"))
    if len(paths) != 300:
        raise ValueError(f"Expected 300 external cracked-screen photos, got {len(paths)}")
    result = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    model = load_model(args.run)
    rows = []
    unreadable = []
    for index, path in enumerate(paths, 1):
        try:
            score, flagged = inspect(path, model, result["threshold"])
        except OSError:
            unreadable.append(str(path))
            continue
        rows.append({"source_path": str(path), "score": round(score, 6), "flagged": int(flagged)})
        if index % 50 == 0:
            print(f"checked {index}/{len(paths)}", flush=True)
    with (args.run / "external_cracked_predictions.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["source_path", "score", "flagged"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset": "DataCluster cracked mobile screens public sample",
               "source": "https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset",
               "images": len(rows), "unreadable_images": unreadable,
               "cracked_flagged": sum(item["flagged"] for item in rows),
               "cracked_missed": sum(not item["flagged"] for item in rows),
               "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "external positive-only check; no specificity or false-alert estimate"}
    (args.run / "external_cracked_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
