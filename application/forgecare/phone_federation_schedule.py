"""Scheduled, synthetic-only five-round FedAvg teaching simulation."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .federation import simulate_synthetic_federation


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_tick(out: Path | None = None, *, animate: bool = False, pause_seconds: float = .3) -> Path:
    target = out or Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / \
        "ForgeCareAI" / "synthetic-federation-demo"
    target.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    simulation = simulate_synthetic_federation(animate=animate, pause_seconds=pause_seconds)
    report = {
        "started_at_utc": started,
        "finished_at_utc": timestamp(),
        "data_source": "newly generated synthetic examples in memory; no existing data read",
        "simulation": simulation,
    }
    path = target / f"synthetic-fedavg-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an educational five-round synthetic FedAvg simulation.")
    parser.add_argument("--out", type=Path, help="Directory for a JSON report (default: user data folder)")
    parser.add_argument("--animate", action="store_true", help="Show the step-by-step teaching animation.")
    parser.add_argument("--pause-seconds", type=float, default=.3, help="Delay between animation frames.")
    args = parser.parse_args()
    report_path = run_tick(args.out, animate=args.animate, pause_seconds=args.pause_seconds)
    print(f"Synthetic simulation report saved to {report_path}")


if __name__ == "__main__":
    main()
