"""Business-readable projections of immutable audit records for the dashboard."""
from __future__ import annotations

import json


ACTIVITY = {
    ("capture", "capture"): "Photo captured",
    ("draft", "capture"): "Photo review prepared",
    ("vision_draft", "capture"): "AI photo summary prepared",
    ("label", "capture"): "Technician review saved",
    ("create", "case"): "Work order opened",
    ("auto_create", "case"): "Work order opened by model",
    ("duplicate_link", "case"): "Existing work order reused",
    ("assign", "case"): "Work order assigned",
    ("approve", "case"): "Inspection approved",
    ("close", "case"): "Work order completed",
    ("start", "camera"): "Camera worker started",
    ("stop", "camera"): "Camera worker stopped",
    ("configure", "camera"): "Camera mapping changed",
    ("sample_change", "camera"): "Physical sample changed",
    ("capture_rejected", "camera_event"): "Camera submission rejected",
    ("inference_failed", "model"): "Visual model could not run",
    ("train_start", "model"): "Model training started",
    ("approve", "model"): "Model approved",
    ("rollback", "model"): "Model approval reversed",
    ("create", "rule"): "Priority rules updated",
    ("create", "asset"): "Equipment registered",
    ("create", "camera"): "Camera registered",
    ("create", "site"): "Site created",
    ("create", "user"): "User created",
    ("bootstrap", "site"): "Site initialized",
    ("password_change", "user"): "Password changed",
    ("index", "document"): "Knowledge source indexed",
    ("deactivate", "document"): "Knowledge source deactivated",
    ("ask", "asset"): "Knowledge assistant consulted",
    ("agent_prepare", "asset"): "Inspection guidance prepared",
    ("experiment_start", "suite"): "Research experiment started",
    ("fedavg_start", "federation_round"): "Federated training started",
    ("fed_round_create", "federation_round"): "Federated round opened",
    ("fed_update", "federation_round"): "Training contribution received",
    ("fed_aggregate", "federation_round"): "Client model weights combined",
    ("fed_validate", "federation_round"): "Combined model checked",
    ("fed_gate", "federation_round"): "Model release decision recorded",
}


def _details(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _summary(action: str, kind: str, values: dict, users: dict[int, str]) -> str:
    if action == "capture":
        asset = values.get("asset_id")
        sample = values.get("sample_id")
        subject = f" Equipment: {asset}; physical sample: {sample}." if asset and sample else ""
        return f"Image quality: {values.get('quality', 'not recorded')}.{subject}"
    if action == "draft":
        photos = values.get("source_capture_ids") or []
        return f"{len(photos)} selected photo{'s' if len(photos) != 1 else ''} prepared for review."
    if action == "vision_draft":
        photos = values.get("image_ids") or values.get("source_capture_ids") or []
        counts = f" Suggested scratches: {values.get('proposed_scratch_count') if values.get('proposed_scratch_count') is not None else 'not established'}; dents: {values.get('proposed_dent_count') if values.get('proposed_dent_count') is not None else 'not established'}."
        return (f"AI summary requested for {len(photos)} photo{'s' if len(photos) != 1 else ''}." if photos else "AI summary prepared.") + counts + " Technician confirmation still required."
    if action == "label":
        grade = {"none": "No visible issue", "minor": "Minor visible issue", "significant": "Significant visible issue", "unclear": "Image unclear"}.get(values.get("grade"), "Review saved")
        scratches = values.get("scratch_count")
        dents = values.get("dent_count")
        note = values.get("review_note")
        return f"{grade}. Scratches: {scratches if scratches is not None else 'not assessed'}; dents: {dents if dents is not None else 'not assessed'}." + (f" Review: {note}" if note else "")
    if action == "create" and kind == "case":
        photo = values.get("capture_id")
        detail = f" Linked photo {photo[:8]};" if photo else ""
        attention = values.get("attention")
        return f"Work order opened for a saved technician review.{detail} Attention: {attention or 'see work order'}. Waiting for assignment or approval."
    if action == "auto_create" and kind == "case":
        return "An approved visual model opened this work order for human review."
    if action == "duplicate_link":
        return "This photo was already linked to the referenced work order."
    if action == "assign":
        assigned = users.get(values.get("user_id"), "a site worker")
        return f"Assigned to {assigned}; supervisor approval is the next step."
    if action == "approve" and kind == "case":
        instructions = values.get("decision_note")
        return "Inspection instructions approved for the worker." + (f" Instructions: {instructions}" if instructions else "")
    if action == "close":
        performed = values.get("inspection_performed")
        minutes = values.get("actual_minutes", 0)
        finding = values.get("actual_finding")
        return f"{'Physical inspection completed' if performed else 'Photo-only review completed'}; {minutes} minutes recorded." + (f" Finding: {finding}" if finding else "")
    if action == "start":
        return "Camera worker started; awaiting a fresh frame."
    if action == "stop":
        return "Camera worker stopped; saved photos remain available."
    if action == "configure":
        return f"Inspection subject: {values.get('asset_id', 'not recorded')}; physical sample: {values.get('sample_id', 'not recorded')}."
    if action == "sample_change":
        return f"Physical sample changed to {values.get('sample_id', 'not recorded')}."
    if action == "capture_rejected":
        return f"Camera submission was not accepted (HTTP {values.get('status_code', 'unknown')})."
    if action == "inference_failed":
        return "Model could not analyze the photo; review the camera image manually."
    if action == "fed_update":
        return f"Training contribution received from {values.get('client_id', 'a client')}."
    if action == "fed_gate":
        decision = values.get("passed")
        return "Combined model passed the recorded release checks." if decision is True else "Combined model did not pass the recorded release checks." if decision is False else "Model validation decision recorded; see Federation for the outcome."
    if action == "ask":
        return "Question asked about this equipment; see Knowledge assistant for sources."
    return "Recorded in the inspection workflow."


def business_events(events: list[dict], users: dict[int, str]) -> list[dict[str, str]]:
    result = []
    for event in events:
        action = event.get("action", "")
        kind = event.get("object_type", "")
        values = _details(event.get("details_json"))
        moment = (event.get("at") or "")[:19].replace("T", " ")
        result.append({
            "When (UTC)": moment,
            "Activity": ACTIVITY.get((action, kind), action.replace("_", " ").capitalize()),
            "Record type": kind.replace("_", " ").capitalize(),
            "Reference": event.get("object_id", ""),
            "Who": users.get(event.get("actor_id"), "System" if event.get("actor_id") is None else "Former user"),
            "Business detail": _summary(action, kind, values, users),
        })
    return result
