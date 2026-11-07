"""Optional Groq text/vision client. Keys come only from process environment."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image


ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
TEXT_MODEL = "openai/gpt-oss-20b"
TEXT_FALLBACK_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"


class GroqUnavailable(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def complete(messages: list[dict], *, model: str, max_tokens: int = 350,
             response_format: dict | None = None) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise GroqUnavailable("Set GROQ_API_KEY and restart ForgeCare to enable Groq generation")
    try:
        payload = {"model": model, "messages": messages, "temperature": 0.1,
                   "max_completion_tokens": max_tokens}
        if response_format:
            payload["response_format"] = response_format
        if model.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"
            payload["include_reasoning"] = False
        response = requests.post(ENDPOINT,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload, timeout=55)
        if response.status_code != 200:
            raise GroqUnavailable(f"Groq returned HTTP {response.status_code}; check model access or quota")
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise GroqUnavailable("Groq returned an empty answer")
        return content.strip()
    except requests.RequestException as exc:
        raise GroqUnavailable(f"Groq connection failed: {type(exc).__name__}") from exc
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise GroqUnavailable("Groq response could not be read") from exc


def _photo_data_uri(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1024, 1024))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def describe_selected_photos(photos: list[tuple[Path, str]], subject: str) -> dict:
    if not 1 <= len(photos) <= 3:
        raise ValueError("Groq vision accepts one to three selected photos")
    content = [{"type": "text", "text": (
        f"Inspection subject: {subject}. The following photos are technician-selected views of one physical item. "
        "Return one JSON object with keys summary (string), observations (array of strings), "
        "scratch_count (integer or null), dent_count (integer or null), count_basis (string), "
        "next_view (string), and uncertainty (string). Describe only visible exterior evidence. "
        "Give a count only if separate marks can be seen clearly, and avoid counting the same mark twice "
        "across views; otherwise use null, not zero. Never infer a phone model, hidden damage, "
        "safety clearance, or repair instruction from appearance. State which view supports each count. "
        "This is an editable AI draft, not a technician's finding.") }]
    for number, (path, view) in enumerate(photos, 1):
        content.append({"type": "text", "text": f"Photo {number}, technician view label: {view}."})
        content.append({"type": "image_url", "image_url": {"url": _photo_data_uri(path)}})
    raw = complete([{"role": "user", "content": content}],
                   model=os.environ.get("FORGECARE_GROQ_VISION_MODEL", VISION_MODEL),
                   max_tokens=700, response_format={"type": "json_object"})
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GroqUnavailable("Vision service returned an unreadable draft") from exc
    if not isinstance(result, dict) or not isinstance(result.get("summary"), str):
        raise GroqUnavailable("Vision service returned an incomplete draft")
    for name in ("scratch_count", "dent_count"):
        value = result.get(name)
        result[name] = value if type(value) is int and 0 < value <= 100 else None
    result["observations"] = [str(item)[:400] for item in result.get("observations", [])[:6]] if isinstance(result.get("observations"), list) else []
    for name in ("count_basis", "next_view", "uncertainty"):
        result[name] = str(result.get(name) or "")[:500]
    result["summary"] = result["summary"][:1000]
    return result
