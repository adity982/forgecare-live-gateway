from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "ForgeCareAI" / "data"
DATA = Path(os.environ.get("FORGECARE_DATA", DEFAULT_DATA)).resolve()
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "images").mkdir(exist_ok=True)
(DATA / "models").mkdir(exist_ok=True)
(DATA / "documents").mkdir(exist_ok=True)
DB_PATH = DATA / "forgecare.sqlite3"
KNOWLEDGE_DB = DATA / "knowledge.sqlite3"
OPS_URL = os.environ.get("FORGECARE_OPS_URL", "http://127.0.0.1:8000")
KNOW_URL = os.environ.get("FORGECARE_KNOW_URL", "http://127.0.0.1:8001")


def service_key() -> str:
    """Keep the inter-process key local and stable across restarts."""
    supplied = os.environ.get("FORGECARE_SERVICE_KEY")
    if supplied:
        return supplied
    p = DATA / ".service-key"
    if not p.exists():
        p.write_text(secrets.token_urlsafe(40), encoding="utf-8")
    return p.read_text(encoding="utf-8").strip()


SERVICE_KEY = service_key()
