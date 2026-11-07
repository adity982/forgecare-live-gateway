"""Illustrative queue rows for trying capacity planning without creating cases."""

from forgecare.priority import fit_capacity


SAMPLE_CASES = [
    {"id": "SAMPLE-101", "asset_id": "iPhone 14 Plus", "priority": 88,
     "priority_band": "high", "status": "sample only", "estimated_minutes": 25,
     "created_at": "2026-11-01T09:00:00+00:00",
     "scenario": "Reviewed repeat: visible rear-glass damage"},
    {"id": "SAMPLE-102", "asset_id": "realme narzo 60 5G", "priority": 72,
     "priority_band": "high", "status": "sample only", "estimated_minutes": 20,
     "created_at": "2026-11-01T10:00:00+00:00",
     "scenario": "Reviewed repeat: visible charging-port wear"},
    {"id": "SAMPLE-103", "asset_id": "iPhone 14 Plus", "priority": 55,
     "priority_band": "medium", "status": "sample only", "estimated_minutes": 30,
     "created_at": "2026-11-02T09:00:00+00:00",
     "scenario": "Exterior condition review due"},
    {"id": "SAMPLE-104", "asset_id": "realme narzo 60 5G", "priority": 41,
     "priority_band": "medium", "status": "sample only", "estimated_minutes": 15,
     "created_at": "2026-11-02T10:00:00+00:00",
     "scenario": "Routine visible-condition follow-up"},
]


def sample_capacity_plan(minutes: int) -> tuple[list[dict], int]:
    selected = fit_capacity(SAMPLE_CASES, minutes)
    used = sum(int(case["estimated_minutes"]) for case in selected)
    return selected, used
