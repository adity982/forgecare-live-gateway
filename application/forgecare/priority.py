from __future__ import annotations

from datetime import datetime, timezone


def priority(criticality: int, reviewed_repeats: int, inspection_interval_days: int,
             last_inspection_at: str | None, weights: tuple[float, float, float],
             at: datetime | None = None) -> tuple[float, dict[str, float]]:
    """A queue urgency score, never a failure probability."""
    at = at or datetime.now(timezone.utc)
    c = min(1.0, max(0.0, criticality / 5))
    r = min(1.0, max(0.0, reviewed_repeats / 3))
    if last_inspection_at:
        try:
            last = datetime.fromisoformat(last_inspection_at)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            days = max(0.0, (at - last).total_seconds() / 86400)
        except ValueError:
            days = inspection_interval_days
    else:
        days = inspection_interval_days
    overdue = min(1.0, max(0.0, days / max(inspection_interval_days, 1)))
    pieces = {"criticality": round(c, 4), "reviewed_repeats": round(r, 4), "inspection_overdue": round(overdue, 4)}
    score = 100 * sum(w * x for w, x in zip(weights, pieces.values()))
    return round(score, 2), pieces


def band(score: float) -> str:
    return "high" if score >= 70 else "medium" if score >= 40 else "low"


def fit_capacity(cases: list[dict], minutes: int, default_minutes: int = 15) -> list[dict]:
    """Greedy, stable priority order; already-open cases remain one task each."""
    remaining = max(0, minutes)
    chosen = []
    for item in sorted(cases, key=lambda c: (-c["priority"], c["created_at"], c["id"])):
        duration = int(item.get("estimated_minutes") or default_minutes)
        if duration > 0 and duration <= remaining:
            chosen.append(item)
            remaining -= duration
    return chosen
