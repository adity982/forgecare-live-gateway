"""Check AutoVI source hashes and saved research outputs without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != metrics["source_png_files"]:
        raise ValueError("Source manifest count differs from saved metrics")
    for item in manifest:
        if hashlib.sha256(Path(item["source_path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source image changed: {item['source_path']}")
    actual = metrics["test_confusion"]
    if sum(actual.values()) != 169 or actual["tp"] + actual["fn"] != 71 or actual["fp"] + actual["tn"] != 98:
        raise ValueError("Saved evaluation counts are inconsistent")
    outputs = root / "results" / "Padim" / "AutoVI" / "pipe_clip" / "v0"
    for path in (outputs / "weights" / "lightning" / "model.ckpt",
                 outputs / "images" / "good" / "0001.png",
                 outputs / "images" / "unclipped" / "0000.png"):
        if not path.is_file():
            raise ValueError(f"Missing saved model output: {path}")
    print(json.dumps({"verified_png_files": len(manifest), "test_images": sum(actual.values()),
                      "defects_found": actual["tp"], "defects_missed": actual["fn"],
                      "good_parts_flagged": actual["fp"], "checkpoint": str(outputs / "weights" / "lightning" / "model.ckpt")}, indent=2))


if __name__ == "__main__":
    main()
