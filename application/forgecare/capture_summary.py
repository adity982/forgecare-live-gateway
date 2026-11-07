"""Build transparent, editable review suggestions from photo-level model outputs."""
from __future__ import annotations


VIEW_NAMES = {"unassigned": "Unlabeled view", "front": "Front", "back": "Back",
              "left": "Left edge", "right": "Right edge", "top": "Top edge",
              "bottom": "Bottom edge", "detail": "Close-up detail"}


def summarize(photo_results: dict, views: dict) -> dict:
    if not photo_results:
        raise ValueError("At least one analyzed photo is required")
    rows = []
    for photo_id, result in photo_results.items():
        counts = result["counts"]
        rows.append({"photo_id": photo_id, "view": VIEW_NAMES.get(views.get(photo_id), "Unlabeled view"),
                     "broken": counts["broken area"],
                     "scratches": counts["major scratch"] + counts["small scratch"],
                     "major_scratches": counts["major scratch"],
                     "small_scratches": counts["small scratch"],
                     "runtime_ms": result.get("runtime_ms"),
                     "confidence_range": (
                         f"{min(item['confidence'] for item in result['findings']):.2f}–"
                         f"{max(item['confidence'] for item in result['findings']):.2f}"
                         if result["findings"] else "No boxes")})
    scratch_hint = max(row["scratches"] for row in rows)
    broken_hint = max(row["broken"] for row in rows)
    lines = [f"Model-assisted draft from {len(rows)} selected view(s). Review each photo and edit before saving."]
    for row in rows:
        lines.append(f"{row['view']}: {row['scratches']} candidate scratch region(s) and "
                     f"{row['broken']} candidate broken area(s); model box confidence {row['confidence_range']}.")
    lines.append(f"Across views, the largest single-view suggestion is {scratch_hint} scratch region(s) "
                 f"and {broken_hint} broken area(s). Repeated views may show the same mark, so this is not "
                 "a count of unique marks on the phone.")
    lines.append("Dent count could not be established by the local detector. Check the phone edges and back directly.")
    lines.append("Next step: compare the highlighted regions with the physical phone and record only verified marks.")
    return {"rows": rows, "scratch_hint": scratch_hint, "dent_hint": None,
            "broken_hint": broken_hint, "draft_text": "\n".join(lines)}
