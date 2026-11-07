"""Streamlit Community Cloud entrypoint for the ForgeCare demonstration.

Cloud storage is ephemeral. Keep real captures and scheduled federation on the
local installation; this entrypoint starts the same UI and APIs for a demo.
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FORGECARE_DATA", str(Path(tempfile.gettempdir()) / "forgecare-cloud-demo"))
OPS_PORT = int(os.environ.get("FORGECARE_CLOUD_OPS_PORT", "8000"))
KNOW_PORT = int(os.environ.get("FORGECARE_CLOUD_KNOW_PORT", "8001"))
os.environ.setdefault("FORGECARE_OPS_URL", f"http://127.0.0.1:{OPS_PORT}")
os.environ.setdefault("FORGECARE_KNOW_URL", f"http://127.0.0.1:{KNOW_PORT}")
try:
    cloud_groq_key = st.secrets.get("GROQ_API_KEY", "")
    admin_password = st.secrets.get("CLOUD_ADMIN_PASSWORD", "")
except FileNotFoundError:
    cloud_groq_key = ""
    admin_password = ""
if cloud_groq_key:
    os.environ["GROQ_API_KEY"] = cloud_groq_key
admin_password = admin_password or os.environ.get("FORGECARE_CLOUD_ADMIN_PASSWORD", "")


def _wait_for(url: str) -> bool:
    for _ in range(40):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(.5)
    return False


def _index_sources() -> None:
    from forgecare.knowledge_api import DocIn, add_document, open_db

    sources = (
        ("iphone-general", "*", "https://support.apple.com/en-us/101575"),
        ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100322"),
        ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100346"),
        ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100328"),
        ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100329"),
        ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100299"),
        ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100298"),
        ("manufacturing-general", "*", "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm"),
        ("automotive-general", "*", "https://www.aiag.org/expertise-areas/quality/quality-core-tools"),
    )
    for family, model, url in sources:
        try:
            with open_db() as db:
                exists = db.execute("SELECT 1 FROM documents WHERE site_id='demo-site' AND url=? AND status='active'", (url,)).fetchone()
            if not exists:
                add_document(DocIn(site_id="demo-site", family=family, model=model,
                                   url=url, revision="official-web-2026-11"))
        except Exception as exc:
            print(f"Source indexing skipped for {url}: {type(exc).__name__}", flush=True)


@st.cache_resource(show_spinner="Starting ForgeCare services…")
def _start_services(admin_password: str):
    processes = []
    for module, port in (("forgecare.ops_api:app", OPS_PORT), ("forgecare.knowledge_api:app", KNOW_PORT)):
        health = f"http://127.0.0.1:{port}/health"
        try:
            already_running = requests.get(health, timeout=1).status_code == 200
        except requests.RequestException:
            already_running = False
        if not already_running:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", module,
                                        "--host", "127.0.0.1", "--port", str(port)],
                                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            processes.append(process)
        if not _wait_for(health):
            raise RuntimeError(f"ForgeCare service on port {port} could not start")
    status = requests.get(f"http://127.0.0.1:{OPS_PORT}/setup/status", timeout=5).json()
    if status.get("needed"):
        response = requests.post(f"http://127.0.0.1:{OPS_PORT}/setup/bootstrap", timeout=10,
                                 json={"site_id": "demo-site", "site_name": "ForgeCare Cloud Demo",
                                       "username": "admin", "password": admin_password})
        response.raise_for_status()
    threading.Thread(target=_index_sources, daemon=True, name="forgecare-source-index").start()
    return processes


if not admin_password or len(admin_password) < 10:
    st.error("Cloud setup is incomplete. Add CLOUD_ADMIN_PASSWORD (10+ characters) in this app's Streamlit Cloud secrets.")
    st.stop()
_start_services(admin_password)
runpy.run_path(str(ROOT / "forgecare" / "ui.py"), run_name="__main__")
