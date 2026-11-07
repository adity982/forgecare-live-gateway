from __future__ import annotations

import json
import importlib.util
import time
import uuid
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from forgecare.config import DATA, OPS_URL
from forgecare.sample_queue import SAMPLE_CASES, sample_capacity_plan
from forgecare.sample_review import SAMPLE_REVIEWS
from forgecare.audit_view import business_events
from forgecare.camera_setup import ip_webcam_direct_source, is_local_view
from forgecare.browser_live_camera import browser_live_camera

RESEARCH_RUN = Path(__file__).resolve().parents[1] / "research" / "visa_capsules_20261105"
CASTING_RUN = Path(__file__).resolve().parents[1] / "research" / "casting_impeller_20261105"
PHONE_RUN = Path(__file__).resolve().parents[1] / "research" / "phone_crack_combined_20261105"
PHONE_DAMAGE_RUN = Path(__file__).resolve().parents[1] / "research" / "phone_damage_20261105"
AUTOVI_RUN = Path(__file__).resolve().parents[1] / "research" / "autovi_pipe_clip_20261105"
AUTOVI_FED_RUN = Path(__file__).resolve().parents[1] / "research" / "autovi_federated_20261105"


def research_metrics():
    path = RESEARCH_RUN / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def casting_metrics():
    path = CASTING_RUN / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def phone_metrics():
    path = PHONE_RUN / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def phone_damage_metrics():
    path = PHONE_DAMAGE_RUN / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def phone_damage_runtime_ready():
    return (PHONE_DAMAGE_RUN / "phone_damage_detector.pt").exists() and importlib.util.find_spec("ultralytics") is not None

def autovi_metrics():
    path = AUTOVI_RUN / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def autovi_fed_metrics():
    path = AUTOVI_FED_RUN / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

@st.cache_resource(show_spinner=False)
def research_models(run_path: str):
    from forgecare.research_demo import load_models
    return load_models(Path(run_path))

@st.cache_resource(show_spinner=False)
def casting_model(run_path: str):
    from forgecare.casting_demo import load_model
    return load_model(Path(run_path))

@st.cache_resource(show_spinner=False)
def autovi_fed_model(run_path: str):
    import torch
    return torch.jit.load(str(Path(run_path) / "flower_autovi_cnn_scripted.pt"), map_location="cpu").eval()

@st.cache_resource(show_spinner=False)
def phone_model(run_path: str):
    from forgecare.phone_demo import load_model
    return load_model(Path(run_path))


@st.cache_resource(show_spinner=False)
def phone_damage_model(weights_path: str):
    from forgecare.phone_damage_demo import load_model

    return load_model(weights_path)

st.set_page_config(page_title="ForgeCare AI", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
  :root {color-scheme:light}
  [data-testid="stAppViewContainer"] {background:#fffefa}
  .block-container {max-width:1240px;padding-top:3.5rem;padding-bottom:5rem}
  [data-testid="stSidebar"] {background:#f3f1ec;border-right:1px solid #e4e1da}
  [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top:2.25rem}
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] label {color:#47443e}
  [data-testid="stSidebar"] [role="radiogroup"] label {border-radius:10px;padding:.34rem .55rem}
  [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#e9e6df}
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#e3dfd6}
  h1,h2,h3 {color:#24231f;letter-spacing:-.025em}
  h2 {margin-top:1.7rem!important;font-size:1.55rem!important}
  [data-testid="stMetric"] {background:#f7f5f0;border:1px solid #ebe8e0;border-radius:14px;padding:17px 20px}
  [data-testid="stMetricLabel"] {color:#67635b}
  [data-testid="stMetricValue"] {color:#24231f;font-size:1.45rem!important;line-height:1.25!important;white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
  [data-testid="stMetricValue"] * {font-size:inherit!important;line-height:inherit!important;white-space:normal!important;overflow-wrap:anywhere!important;text-overflow:clip!important}
  [data-testid="stMetric"] {min-width:0}
  div.stButton button, div.stFormSubmitButton button {border-radius:10px;min-height:2.65rem;font-weight:600}
  div.stButton button[kind="primary"], div.stFormSubmitButton button[kind="primary"] {background:#292723;color:#fff;border-color:#292723}
  div.stButton button[kind="primary"]:hover, div.stFormSubmitButton button[kind="primary"]:hover {background:#454139;color:#fff;border-color:#454139}
  input, textarea, [data-baseweb="select"] {border-radius:9px!important}
  [data-testid="stDataFrame"] {border:1px solid #e7e3da;border-radius:12px;overflow:hidden}
  .brand {font-size:1.2rem;font-weight:750;letter-spacing:-.045em;color:#24231f;margin-bottom:.12rem}
  .brand-mark {display:inline-grid;place-items:center;width:1.65rem;height:1.65rem;margin-right:.45rem;border-radius:7px;background:#292723;color:#fff;font-size:.95rem;vertical-align:middle}
  .brand-note {color:#777168;font-size:.77rem;letter-spacing:.01em;margin:.2rem 0 2rem}
  .sidebar-label {color:#888279;font-size:.7rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin:.5rem 0 .35rem}
  .eyebrow {color:#8a7e70;font-size:.73rem;font-weight:750;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.75rem}
  .hero {border-bottom:1px solid #e8e4dc;padding:0 0 1.6rem;margin:0 0 2rem}
  .hero h1 {font-family:Georgia,'Times New Roman',serif;font-size:clamp(1.9rem,2.5vw,2.5rem);font-weight:400;line-height:1.13;letter-spacing:-.045em;color:#24231f;margin:0}
  .hero p {color:#666159;max-width:56rem;font-size:1.02rem;line-height:1.6;margin:.85rem 0 0}
  .empty-state {border:1px dashed #d9d4ca;background:#faf8f4;border-radius:14px;padding:1.55rem 1.7rem;color:#514c44}
  .empty-state strong {display:block;color:#292723;font-size:1rem;margin-bottom:.25rem}
  .empty-state span {font-size:.93rem;line-height:1.5}
  .sample-note {background:#f4f0e7;border-left:3px solid #ac936d;border-radius:0 10px 10px 0;padding:1rem 1.2rem;color:#4e463a;margin:.7rem 0 1.2rem;line-height:1.5}
  @media (max-width:700px) {.block-container {padding-top:2rem}.hero h1 {font-size:2rem}}
</style>""", unsafe_allow_html=True)


def request(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        res = requests.request(method, f"{OPS_URL}{path}", headers=headers, timeout=120, **kwargs)
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise RuntimeError(str(detail))
        return res.json() if "application/json" in res.headers.get("content-type", "") else res.content
    except requests.ConnectionError:
        raise RuntimeError("Operations service is offline. Start run.ps1.")


def action(method, path, payload, message="Saved"):
    try:
        result = request(method, path, json=payload)
        st.success(message)
        return result
    except Exception as exc:
        st.error(str(exc))


def auth_screen():
    st.markdown('<div class="hero"><div class="eyebrow">Inspection workspace</div><h1>ForgeCare AI</h1><p>Visible condition, inspection priority, and cited guidance in one review record.</p></div>', unsafe_allow_html=True)
    try:
        needed = request("GET", "/setup/status")["needed"]
    except Exception as exc:
        st.error(str(exc)); st.stop()
    if needed:
        st.info("Create the local administrator account to initialize this installation.")
        with st.form("setup"):
            a, b = st.columns(2)
            site_id = a.text_input("Site ID", "demo-site")
            site_name = b.text_input("Site name", "ILP Demonstration")
            username = a.text_input("Admin username", "admin")
            password = b.text_input("Password (10+ characters)", type="password")
            if st.form_submit_button("Initialize ForgeCare", type="primary"):
                result = action("POST", "/setup/bootstrap", {"site_id": site_id, "site_name": site_name, "username": username, "password": password}, "Installation ready")
                if result:
                    st.session_state.token = result["token"]
                    st.rerun()
    else:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                result = action("POST", "/auth/login", {"username": username, "password": password}, "Signed in")
                if result:
                    st.session_state.token = result["token"]
                    st.rerun()
    st.stop()


if not st.session_state.get("token"):
    auth_screen()
try:
    me = request("GET", "/auth/me")
except Exception:
    st.session_state.token = None
    auth_screen()
sites = request("GET", "/sites")
local_camera_view = is_local_view(st.context.url)
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✳</span>ForgeCare</div><div class="brand-note">INSPECTION WORKSPACE</div>', unsafe_allow_html=True)
    choices = {x["name"]: x["id"] for x in sites}
    selected_name = st.selectbox("Site", list(choices))
    site = choices[selected_name]
    st.divider()
    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "Live cameras", "Capture review", "Inspection queue", "Assets & setup", "Knowledge assistant", "Models & experiments", "Federation", "Reports", "Administration", "Audit"], label_visibility="collapsed")
    st.divider()
    st.caption(f"{me['username']} · {me['role']}")
    if st.button("Sign out"):
        st.session_state.token = None
        st.rerun()
    st.caption("Local ILP prototype · Human approval required")


def heading(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">ForgeCare / {escape(page)}</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def table(items, columns=None, empty_title="Nothing here yet", empty_message="Records will appear here when the workflow has data."):
    if not items:
        st.markdown(f'<div class="empty-state"><strong>{escape(empty_title)}</strong><span>{escape(empty_message)}</span></div>', unsafe_allow_html=True)
    else:
        data = pd.DataFrame(items)
        if columns:
            data = data[[x for x in columns if x in data.columns]]
        st.dataframe(data, hide_index=True, use_container_width=True)


def detail_rows(value, prefix=""):
    """Flatten model and monitoring results into labeled values, never raw JSON."""
    rows_out = []
    if isinstance(value, dict):
        for key, item in value.items():
            label = (prefix + " · " if prefix else "") + key.replace("_", " ").capitalize()
            rows_out.extend(detail_rows(item, label))
    elif isinstance(value, list):
        for index, item in enumerate(value, 1):
            rows_out.extend(detail_rows(item, f"{prefix} · Item {index}"))
    else:
        shown = "Not measured" if value is None else "Yes" if value is True else "No" if value is False else str(value)
        rows_out.append({"Measure": prefix, "Recorded value": shown})
    return rows_out


def capture_origin(capture: dict) -> str:
    session_id = capture.get("session_id") or ""
    if session_id.startswith("browser-"):
        return "Viewing device browser"
    if session_id.startswith("relay-"):
        return "Same-network IP Webcam relay"
    return "Earlier server-connected camera"


@st.fragment(run_every="5s")
def live_camera_view(site_id: str):
    cameras = request("GET", f"/cameras?site_id={site_id}")
    captures = (request("GET", f"/captures?site_id={site_id}&limit=100")
                if any(camera["status"] == "live" for camera in cameras) else [])
    st.button("Refresh now", key="refresh-live-cameras")
    st.caption("Only active feeds show an image. Saved photos remain in Capture review. This page refreshes every five seconds.")
    for camera in sorted(cameras, key=lambda item: (item["status"] != "live", item["name"])):
        with st.container(border=True):
            details, preview = st.columns([1, 2])
            details.subheader(camera["name"])
            camera_label = {"live": "Live", "browser_ready": "Ready for photos from this device",
                            "relay_waiting": "Not connected · enter current phone address",
                            "offline": "Offline · camera disconnected or worker unavailable",
                            "no_signal": "No live signal · no new frames", "stopped": "Stopped by operator"}
            quality_label = {"ok": "Clear enough to review", "blurred": "Blurry — retake with a steady camera",
                             "dark": "Too dark — add light", "bright": "Too bright — reduce glare"}
            details.write(f"**Status:** {camera_label.get(camera['status'], camera['status'])}" +
                          (f" · **Image quality:** {quality_label.get(camera['last_quality'], camera['last_quality'])}"
                           if camera["status"] == "live" and camera["last_quality"] else ""))
            details.caption(f"Last frame received: {camera['last_frame_at'] or 'none'} (saved history)")
            if camera["last_error"] and camera["status"] not in ("stopped", "browser_ready"):
                recorded = camera.get("last_error_at")
                when = recorded[:19].replace("T", " ") + " UTC" if recorded else "time unavailable (recorded before this update)"
                details.warning(f"Last recorded camera problem ({when}): {camera['last_error']}")
                if camera["status"] == "offline":
                    details.caption("This is a saved event, not a new analysis of a switched-off phone. Restart the worker after the camera server is available to recheck the feed.")
            matching = next((item for item in captures if camera["status"] == "live"
                             and item["camera_id"] == camera["id"]
                             and (not camera["source"].startswith("relay:")
                                  or item["session_id"].startswith("relay-"))), None)
            recent = [item for item in captures if matching and item["camera_id"] == camera["id"] and item["session_id"] == matching["session_id"]]
            if camera["status"] == "live":
                details.metric("Recent photos received", len(recent))
                details.caption(f"{sum(item['quality'] == 'ok' for item in recent)} passed image quality checks")
                analyzed = [item for item in recent if item.get("model_id") and item.get("predicted_class")]
                details.write(f"**General model alerts:** {sum(item['predicted_class'] == 'review_needed' for item in analyzed)} of {len(analyzed)}" if analyzed else "**General model alerts:** No approved model")
                details.caption("Confirmed scratch and dent counts require technician review. The local research detector can suggest broken and scratch regions on the latest clear frame. Dent detection failed its held-out check.")
            if camera["status"] == "live" and matching:
                try:
                    frame = request("GET", f"/captures/{matching['id']}/image")
                    clarity = {"ok": "Clear enough to review", "blurred": "Blurry — hold the camera steady or improve focus",
                               "dark": "Too dark — add light", "bright": "Too bright — reduce glare"}
                    preview.image(frame, caption=f"Live photo {matching['captured_at']} · {clarity.get(matching['quality'], matching['quality'])}", use_container_width=True)
                    if matching["quality"] == "ok":
                        preview.success("Image check: clear enough for review. Confirm the phone surface is actually in focus before using this photo.")
                    else:
                        preview.warning(f"Image check: {clarity.get(matching['quality'], matching['quality'])}. Retake before reviewing damage.")
                    details.caption(f"Focus measure: {matching['blur']:.0f} · brightness: {matching['brightness']:.0f} / 255" if matching.get("blur") is not None and matching.get("brightness") is not None else "Focus and brightness measurements unavailable")
                    damage_study = phone_damage_metrics()
                    if matching["quality"] == "ok" and damage_study and phone_damage_runtime_ready():
                        result_key = f"live-damage-{camera['id']}"
                        cached = st.session_state.get(result_key)
                        if not cached or cached["capture_id"] != matching["id"]:
                            from forgecare.phone_damage_demo import inspect as inspect_damage
                            detected = inspect_damage(frame,
                                phone_damage_model(str(PHONE_DAMAGE_RUN / "phone_damage_detector.pt")),
                                damage_study["validation"]["threshold"])
                            cached = {"capture_id": matching["id"], "result": detected,
                                      "checked_at": datetime.now().isoformat(timespec="seconds")}
                            st.session_state[result_key] = cached
                        counts = cached["result"]["counts"]
                        details.write(f"**Latest clear frame · model suggestions:** {counts['major scratch'] + counts['small scratch']} scratch regions · {counts['broken area']} broken areas · dents require manual check")
                        details.caption(f"Model checked at {cached['checked_at']}. Counts apply to this photo only; they are unconfirmed.")
                        with preview.expander("See highlighted candidate marks on this frame"):
                            st.image(cached["result"]["overlay_rgb"], use_container_width=True)
                except Exception as exc:
                    preview.warning(str(exc))
            elif camera["status"] == "live":
                preview.info("Waiting for the first capture from this live camera.")
            elif camera["status"] == "browser_ready":
                preview.info("Camera is off. Enable it above to take a new photo. Saved photos are in Capture review.")
            elif camera["status"] == "relay_waiting":
                preview.info("No current Samsung stream. Enter the phone's current IP Webcam address and start it above. Older photos are in Capture review.")
            elif camera["status"] == "stopped":
                preview.info("Camera worker stopped. No new photos are being processed. Saved photos remain in Capture review.")
            elif camera["status"] == "offline":
                preview.warning("No live feed is connected. Switch on the Samsung IP Webcam server, check Wi-Fi, then restart this camera worker in Assets & setup.")
            else:
                preview.warning("No new frames have arrived for more than ten seconds. The stream may be frozen or the phone may be off. Check the phone and restart the worker if needed.")


if page == "Overview":
    heading("Operations overview", "The state of live evidence, open inspections, and reviewed outcomes.")
    stats = request("GET", f"/dashboard?site_id={site}")
    cols = st.columns(6)
    for col, (key, label) in zip(cols, [("assets", "Assets"), ("cameras", "Cameras"), ("captures", "Captures"), ("reviewed_labels", "Labels"), ("open_cases", "Open cases"), ("approved_cases", "Approved tasks")]):
        col.metric(label, stats[key])
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Camera health")
        camera_rows = request("GET", f"/cameras?site_id={site}")
        table([{"Camera": camera["name"], "Current state": camera["status"].replace("_", " ").title(),
                "Last photo received": camera["last_frame_at"] or "None",
                "Last recorded problem": camera["last_error"] if camera["last_error"] not in (None, "Worker stopped") else "None"}
               for camera in camera_rows],
              ["Camera", "Current state", "Last photo received", "Last recorded problem"])
    with right:
        st.subheader("Next inspections")
        table(request("GET", f"/cases/selected?site_id={site}&capacity_minutes=60"), ["id", "asset_id", "priority", "priority_band", "status"])
    st.subheader("Monitoring")
    monitoring = request("GET", f"/monitoring?site_id={site}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Recent captures", monitoring["recent_captures"])
    m2.metric("Recent quality OK", "Insufficient data" if monitoring["quality_ok_newer"] is None else f"{monitoring['quality_ok_newer']:.0%}")
    m3.metric("Quality shift review", "Check inputs" if monitoring["quality_shift_review_signal"] else "No signal")
    if monitoring["reviewed_visual_performance"]:
        performance = monitoring["reviewed_visual_performance"]
        st.subheader("Reviewed model checks")
        table([{"Measure": "Reviewed photos compared", "Recorded value": performance["reviewed_count"]},
               {"Measure": "Damaged photos flagged", "Recorded value": "Not measured" if performance["recall"] is None else f"{performance['recall']:.0%}"},
               {"Measure": "Acceptable photos falsely flagged", "Recorded value": "Not measured" if performance["false_alert_rate"] is None else f"{performance['false_alert_rate']:.0%}"}])
        st.caption("These are agreements with saved technician labels for one approved model, not verified physical defect accuracy.")
    st.caption(monitoring["note"])
    st.caption("A priority score expresses review urgency. It is not a probability of equipment failure.")

elif page == "Live cameras":
    heading("Cameras and photo capture", "Connect a camera, check image clarity, and collect new views of the phone.")
    current_cameras = request("GET", f"/cameras?site_id={site}")
    if local_camera_view:
        st.subheader("Connect a camera on this laptop")
        st.caption("Lenovo uses the built-in webcam on the computer running ForgeCare. Samsung uses IP Webcam directly over the same Wi-Fi. Starting a camera saves a fresh photo about every five seconds; clear and blurry frames are labeled below.")
        local_options = [item for item in current_cameras if item["id"] in ("camera-laptop", "camera-samsung")]
        if local_options:
            selected = st.selectbox("Camera to start", local_options,
                format_func=lambda item: "Lenovo built-in webcam" if item["id"] == "camera-laptop" else "Samsung IP Webcam",
                key="local-live-camera")
            if selected["id"] == "camera-samsung":
                saved_address = selected["source"].removeprefix("relay:").removesuffix("/video")
                if not saved_address.startswith("http://"):
                    saved_address = ""
                phone_address = st.text_input("Samsung IP Webcam address shown on the phone",
                    value=saved_address, placeholder="http://192.168.1.23:8080", key="local-samsung-address")
                st.caption("Start IP Webcam on the Samsung first. Use its current HTTP IPv4 address; both devices must be on the same Wi-Fi. No bridge or pairing code is needed locally.")
                if phone_address:
                    try:
                        st.link_button("Check the moving video on this laptop", ip_webcam_direct_source(phone_address))
                    except ValueError:
                        pass
            else:
                st.caption("The Lenovo webcam opens on this laptop when you press Start. No browser camera permission or separate photo widget is needed.")
            a, b = st.columns(2)
            if a.button("Start selected camera", type="primary", key="start-local-camera"):
                try:
                    wanted_source = ("0" if selected["id"] == "camera-laptop" else
                                     ip_webcam_direct_source(phone_address))
                    if selected["id"] == "camera-samsung":
                        with requests.get(wanted_source, stream=True, timeout=5) as probe:
                            probe.raise_for_status()
                except (ValueError, requests.RequestException) as exc:
                    st.error(str(exc))
                else:
                    ready = True
                    if selected["source"] != wanted_source:
                        ready = bool(action("POST", f"/cameras/{selected['id']}/stop", {}, "Previous camera connection stopped"))
                        if ready:
                            ready = bool(action("PUT", f"/cameras/{selected['id']}",
                                {"asset_id": selected["asset_id"], "source": wanted_source,
                                 "sample_id": selected["sample_id"], "roi": json.loads(selected["roi_json"])},
                                "Camera source saved"))
                    if ready and action("POST", f"/cameras/{selected['id']}/start", {}, "Connecting to the live camera"):
                        st.rerun()
            if b.button("Stop selected camera", key="stop-local-camera"):
                if action("POST", f"/cameras/{selected['id']}/stop", {}, "Camera stopped"):
                    st.rerun()
        else:
            st.info("Register the Lenovo or Samsung camera in Assets & setup first.")
    else:
        browser_cameras = [item for item in current_cameras if item["source"] == "browser"]
        if browser_cameras:
            st.subheader("Camera on this computer")
            st.caption("Press Start once and allow browser camera access. The live preview stays on screen and saves a fresh photo about every five seconds. Stop it when finished.")
            browser_camera = st.selectbox("Photo destination", browser_cameras,
                                         format_func=lambda item: f"{item['name']} · {item['asset_id']} · {item['sample_id']}",
                                         key="browser-camera-destination")
            session_key = f"browser-session-{browser_camera['id']}-{browser_camera['asset_id']}-{browser_camera['sample_id']}"
            if session_key not in st.session_state:
                st.session_state[session_key] = "browser-" + uuid.uuid4().hex
            photo_event = browser_live_camera(
                data={"camera_id": browser_camera["id"], "session_id": st.session_state[session_key]},
                key=f"browser-live-{browser_camera['id']}")
            photo = getattr(photo_event, "photo", None)
            if isinstance(photo, dict) and photo.get("id") != st.session_state.get("last-browser-photo-id"):
                st.session_state["last-browser-photo-id"] = photo["id"]
                result = action("POST", f"/cameras/{browser_camera['id']}/browser-captures",
                    {"session_id": photo["session_id"], "jpeg_b64": photo["jpeg_b64"]},
                    "New photo saved for review")
                if result:
                    st.caption(f"Latest saved photo: {result['quality']} · focus {result['blur']:.0f} · brightness {result['brightness']:.0f}/255")
        samsung = next((item for item in current_cameras if item["id"] == "camera-samsung"), None)
        if samsung:
            st.subheader("Samsung IP Webcam")
            st.caption("Direct connection works only when the laptop hosting ForgeCare and the Samsung share a reachable Wi-Fi network. The hosted web page alone cannot reach a private phone address from a different network.")
            saved = samsung["source"].removeprefix("relay:").removesuffix("/video")
            samsung_address = st.text_input("Phone's current HTTP address", saved if saved.startswith("http://") else "",
                placeholder="http://192.168.1.23:8080", key="hosted-samsung-address")
            a, b = st.columns(2)
            if a.button("Start Samsung stream", key="start-hosted-samsung"):
                try:
                    direct = ip_webcam_direct_source(samsung_address)
                    with requests.get(direct, stream=True, timeout=5) as probe:
                        probe.raise_for_status()
                    ready = True
                    if samsung["source"] != direct:
                        ready = bool(action("POST", "/cameras/camera-samsung/stop", {}, "Old camera connection stopped"))
                        if ready:
                            ready = bool(action("PUT", "/cameras/camera-samsung",
                                {"asset_id": samsung["asset_id"], "source": direct,
                                 "sample_id": samsung["sample_id"], "roi": json.loads(samsung["roi_json"])},
                                "Samsung address saved"))
                    if ready and action("POST", "/cameras/camera-samsung/start", {}, "Connecting to Samsung"):
                        st.rerun()
                except (ValueError, requests.RequestException) as exc:
                    st.error(f"ForgeCare's hosting laptop cannot reach that phone stream: {exc}. Run the local project on the laptop sharing the Samsung's Wi-Fi.")
            if b.button("Stop Samsung stream", key="stop-hosted-samsung"):
                if action("POST", "/cameras/camera-samsung/stop", {}, "Samsung stream stopped"):
                    st.rerun()
        else:
            if not browser_cameras:
                st.info("No browser camera is set up yet. In Assets & setup, set a camera source to browser to use this computer's camera.")
    live_camera_view(site)

elif page == "Capture review":
    heading("Review captures", "Label only visible exterior condition. Keep each physical phone's sample ID consistent.")
    with st.expander("Practice the review with fictional examples"):
        st.caption("These examples are synthetic exercises only. They are never saved as camera captures, technician findings, or training labels.")
        example = st.selectbox("Practice example", SAMPLE_REVIEWS, format_func=lambda item: f"{item['id']} · {item['subject']} · {item['image_quality']}",
                               key="sample-review", index=0)
        st.write(f"**Views:** {example['views']} · **Image quality:** {example['image_quality']}")
        st.write(f"**Automatic draft:** {example['draft']}")
        st.write(f"**Example technician correction:** {example['technician_review']}")
        st.write(f"**Visible condition:** {example['grade']} · **Confirmed scratches:** {example['scratch_count'] if example['scratch_count'] is not None else 'Not assessed'} · **dents:** {example['dent_count'] if example['dent_count'] is not None else 'Not assessed'}")
        st.info(example["next_step"])
    selection_mode = st.selectbox("Show captures", ["Usable images", "All images"],
                                  help="Usable images have passed the automatic focus and brightness checks.")
    quality_filter = "&quality=ok" if selection_mode == "Usable images" else ""
    captures = request("GET", f"/captures?site_id={site}&limit=200{quality_filter}")
    table([{**item, "capture_origin": capture_origin(item)} for item in captures],
          ["id", "asset_id", "camera_id", "sample_id", "captured_at", "capture_origin", "quality", "grade"],
          "No usable captures yet" if selection_mode == "Usable images" else "No captures yet",
          "Position and light the phone, then capture a clear image." if selection_mode == "Usable images" else "Start a camera worker to capture an image.")
    if captures:
        by_id = {x["id"]: x for x in captures}
        capture_ids = list(by_id)
        if st.session_state.get("review-capture-id") not in by_id:
            st.session_state["review-capture-id"] = capture_ids[0]
        capture_id = st.selectbox("Capture", capture_ids, key="review-capture-id",
                                  format_func=lambda item: f"{by_id[item]['captured_at']} · {item[:8]} · {by_id[item]['quality']} · {by_id[item]['asset_id']}")
        selected = by_id[capture_id]
        photo_bytes = request("GET", f"/captures/{capture_id}/image")
        photo_column, context_column = st.columns([1.4, 1])
        photo_column.image(photo_bytes, use_container_width=True)
        with context_column:
            st.write("**Selected camera evidence**")
            st.write(f"Captured: {selected['captured_at']}")
            st.write(f"Source: {capture_origin(selected)}")
            st.write(f"Phone asset: {selected['asset_id']} · physical sample: {selected['sample_id']}")
            st.write(f"Image check: {selected['quality']}")
            st.caption("A clear photo supports a visible-condition review; it does not prove internal or functional condition.")
        phone_study = phone_metrics()
        if selected["quality"] == "ok" and phone_study:
            st.caption("Research screen check: trained on public cracked/intact phone photos. Use only when this image clearly shows the front screen. It cannot find or count scratches and dents.")
            if st.button("Run trained crack-screen check", key=f"screen-check-{capture_id}"):
                from forgecare.phone_demo import inspect as inspect_phone
                score, flagged = inspect_phone(photo_bytes, phone_model(str(PHONE_RUN)), phone_study["threshold"])
                st.session_state[f"screen-result-{capture_id}"] = (score, flagged, datetime.now().isoformat(timespec="seconds"))
            if f"screen-result-{capture_id}" in st.session_state:
                score, flagged, checked_at = st.session_state[f"screen-result-{capture_id}"]
                st.subheader("Screen-check result")
                st.caption(f"Analyzed this selected capture at {checked_at}. The camera does not need to remain on to review a saved photo.")
                if flagged:
                    st.warning("Possible crack in the visible screen · technician review needed")
                else:
                    st.info("No crack alert from this photo · inspect the screen yourself before approving")
                st.write("**What the model did:** compared the saved photo with patterns learned from public cracked and intact phone images. "
                         "The result applies to this photo only; a rear or blurry view cannot establish front-screen condition.")
                st.write("**Next check:** look at the front glass under even light from more than one angle, inspect any suspected line closely, "
                         "and use the technician review below to record the final condition. Capture another clear view if uncertain.")
                with st.expander("How this research model was checked", expanded=True):
                    evaluation = phone_study.get("test", {}).get("confusion", {})
                    st.write(f"The model was trained with **{phone_study['images']} public photos**. In its held-out research set, "
                             f"it flagged **{evaluation.get('tp', 0)} of {phone_study['test']['cracked']} cracked photos**, "
                             f"missed **{evaluation.get('fn', 0)}**, and falsely flagged **{evaluation.get('fp', 0)} of "
                             f"{phone_study['test']['intact']} intact photos**.")
                    st.write(f"Decision score for this image: **{score:.2f}**; alert threshold: **{phone_study['threshold']:.2f}**. "
                             "The score is a model signal, not a calibrated probability of damage or a technician-approved finding.")
                    st.write("The research photos did not validate this model for iPhone 14 Plus or realme narzo 60. "
                             "They have no scratch or dent labels, so this screen check cannot count either defect.")
                    st.markdown("Training sources: [cracked and intact smartphones](https://www.kaggle.com/datasets/axondata/cracked-and-intact-smartphone-images-dataset) · "
                                "[cracked screen photos](https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset)")
        if selected["quality"] != "ok":
            st.warning(f"This capture is {selected['quality']} and cannot support a condition finding or inspection case. Select a usable image or recapture.")
        draft = request("GET", f"/captures/{capture_id}/draft")
        view_options = {"unassigned": "View not labeled", "front": "Front", "back": "Back", "left": "Left edge",
                        "right": "Right edge", "top": "Top edge", "bottom": "Bottom edge", "detail": "Close-up detail"}
        if selected["quality"] == "ok" and not draft:
            with st.expander("Choose photos for one review", expanded=True):
                st.caption("Select different views of this physical phone. The gallery uses this camera and sample from the preceding five minutes; similar consecutive frames may show the same angle.")
                candidates = request("GET", f"/captures/{capture_id}/draft-candidates")[:40]
                by_candidate = {item["id"]: item for item in candidates}
                with st.container(border=True):
                    gallery = st.columns(4)
                    for index, item in enumerate(candidates[:12]):
                        with gallery[index % 4]:
                            st.image(request("GET", f"/captures/{item['id']}/image"), use_container_width=True)
                            st.caption(f"{item['captured_at'][11:19]} · {item['id'][:8]}")
                chosen = st.multiselect("Photos to compare (up to 5)", list(by_candidate), default=[capture_id],
                                        format_func=lambda item: f"{by_candidate[item]['captured_at'][11:19]} · {item[:8]}",
                                        key=f"draft-photos-{capture_id}")
                chosen_views = {}
                for item in chosen[:5]:
                    chosen_views[item] = st.selectbox(f"View for photo {item[:8]}", list(view_options),
                        format_func=lambda view: view_options[view], key=f"draft-view-{capture_id}-{item}")
                if len(chosen) > 5:
                    st.warning("Choose no more than five photos.")
                if capture_id not in chosen:
                    st.warning("Keep the selected main photo in the review set.")
                if st.button("Build draft from selected photos", key=f"draft-{capture_id}",
                             disabled=not chosen or len(chosen) > 5 or capture_id not in chosen):
                    photos = [{"capture_id": item, "view": chosen_views[item]} for item in chosen]
                    if action("POST", f"/captures/{capture_id}/draft", {"photos": photos}, "Multi-photo draft ready for technician review"):
                        st.rerun()
        if draft:
            source_ids = json.loads(draft["source_ids_json"])
            views = json.loads(draft.get("views_json") or "{}")
            st.info(f"**Automated {'multi-photo' if len(source_ids) > 1 else 'single-photo'} draft · "
                    f"{len(source_ids)} photo{'s' if len(source_ids) != 1 else ''}**\n\n{draft['draft_text']}")
            st.caption("This draft has not been approved as a finding. Only the selected photo receives a label; the other photos are supporting views. View names are technician supplied, not image recognition.")
            damage_study = phone_damage_metrics()
            if damage_study and phone_damage_runtime_ready():
                st.caption("Local research detector: highlights possible broken areas and scratches in each selected photo. Dent detection failed the held-out check and is disabled. Repeated views can show the same mark; these are not confirmed totals for the phone.")
                if st.button("Analyze selected photos for visible damage", key=f"damage-review-{capture_id}"):
                    from forgecare.phone_damage_demo import inspect as inspect_damage
                    from forgecare.capture_summary import summarize as summarize_damage
                    photo_results = {}
                    started = time.perf_counter()
                    with st.status("Analyzing selected camera photos", expanded=True) as progress:
                        progress.write("Step 1 of 3 · Loading the saved local model and selected views")
                        time.sleep(.6)  # Keep the real stage visible long enough to read during a demonstration.
                        model = phone_damage_model(str(PHONE_DAMAGE_RUN / "phone_damage_detector.pt"))
                        threshold = damage_study["validation"]["threshold"]
                        for number, source_id in enumerate(source_ids, 1):
                            progress.write(f"Step 2 of 3 · Checking photo {number} of {len(source_ids)} ({view_options.get(views.get(source_id), 'Unlabeled view')})")
                            pixels = request("GET", f"/captures/{source_id}/image")
                            photo_results[source_id] = inspect_damage(pixels, model, threshold)
                            progress.write(f"Photo {number} model inference finished in {photo_results[source_id]['runtime_ms']:.0f} ms")
                        progress.write("Step 3 of 3 · Assembling the editable per-view summary")
                        summary = summarize_damage(photo_results, views)
                        time.sleep(.6)
                        progress.update(label="Analysis finished; technician review is ready", state="complete")
                    checked_at = datetime.now().isoformat(timespec="seconds")
                    st.session_state[f"damage-result-{capture_id}"] = {"at": checked_at, "photos": photo_results,
                        "summary": summary, "elapsed_ms": round((time.perf_counter() - started) * 1000)}
                    note_key = f"note-{capture_id}"
                    if not selected.get("reviewer_note") and not st.session_state.get(note_key):
                        st.session_state[note_key] = summary["draft_text"]
                    scratch_key = f"scratches-{capture_id}"
                    if selected.get("scratch_count") is None and st.session_state.get(scratch_key) is None:
                        st.session_state[scratch_key] = summary["scratch_hint"]
                damage_result = st.session_state.get(f"damage-result-{capture_id}")
                if damage_result:
                    summary = damage_result["summary"]
                    st.info(f"Local detector ran at {damage_result['at']} on {len(damage_result['photos'])} photo(s). Total analysis flow: {damage_result['elapsed_ms'] / 1000:.1f} s; per-photo inference times are measured below. Its draft is editable below.")
                    a, b, c = st.columns(3)
                    a.metric("Selected views analyzed", len(summary["rows"]))
                    b.metric("Scratch regions, largest view", summary["scratch_hint"])
                    c.metric("Dent count", "Needs technician check")
                    st.dataframe(pd.DataFrame([{"View": row["view"], "Scratch regions": row["scratches"],
                        "Broken areas": row["broken"], "Box confidence": row["confidence_range"],
                        "Model time (ms)": row["runtime_ms"]} for row in summary["rows"]]),
                        hide_index=True, use_container_width=True)
                    st.caption("The scratch number uses the largest single-view suggestion; it is not an automatic total for the physical phone. A zero prediction does not prove a mark is absent.")
                    for source_id, result in damage_result["photos"].items():
                        with st.expander(f"Model marks · {view_options.get(views.get(source_id), 'Unlabeled view')} · {source_id[:8]}", expanded=source_id == capture_id):
                            st.image(result["overlay_rgb"], use_container_width=True)
                            counts = result["counts"]
                            st.write(f"Possible broken areas: {counts['broken area']} · major scratches: {counts['major scratch']} · small scratches: {counts['small scratch']} · dents: manual assessment required")
                            st.caption(f"Local inference: {result['runtime_ms']:.0f} ms. Boxes are candidate regions in this photo. Inspect the physical phone before entering confirmed counts.")
            if st.button("Generate AI review from selected photos", key=f"vision-review-{capture_id}"):
                result = action("POST", f"/captures/{capture_id}/vision-review", {}, "AI visual draft generated")
                if result:
                    local_result = st.session_state.get(f"damage-result-{capture_id}")
                    local_note = local_result["summary"]["draft_text"] if local_result else None
                    note_key = f"note-{capture_id}"
                    if not selected.get("reviewer_note") and st.session_state.get(note_key) in (None, "", local_note, draft.get("ai_text")):
                        st.session_state[note_key] = result["ai_text"]
                    scratch_key = f"scratches-{capture_id}"
                    local_scratch = local_result["summary"]["scratch_hint"] if local_result else None
                    if (selected.get("scratch_count") is None and result.get("ai_scratch_count") is not None
                            and st.session_state.get(scratch_key) in (None, local_scratch)):
                        st.session_state[scratch_key] = result["ai_scratch_count"]
                    dent_key = f"dents-{capture_id}"
                    if selected.get("dent_count") is None and result.get("ai_dent_count") is not None and st.session_state.get(dent_key) is None:
                        st.session_state[dent_key] = result["ai_dent_count"]
                    st.rerun()
            st.caption("This optional action sends up to three selected photos to Groq for a tentative visual description. Check the images yourself before saving any finding.")
            if draft.get("ai_text"):
                st.success(draft["ai_text"])
                sent = json.loads(draft.get("ai_image_ids_json") or "[]")
                st.caption(f"AI examined {len(sent)} of {len(source_ids)} selected photos. The model has not been validated for phone damage or defect counts.")
                a, b = st.columns(2)
                a.metric("AI suggested scratches", draft.get("ai_scratch_count") if draft.get("ai_scratch_count") is not None else "Not established")
                b.metric("AI suggested dents", draft.get("ai_dent_count") if draft.get("ai_dent_count") is not None else "Not established")
                if draft.get("ai_count_basis"):
                    st.caption(f"Count basis: {draft['ai_count_basis']}")
            for source_id in source_ids:
                if source_id != capture_id:
                    with st.expander(f"Supporting photo {source_id[:8]} · {view_options.get(views.get(source_id), 'View not labeled')}"):
                        st.image(request("GET", f"/captures/{source_id}/image"), width=520)
        if selected.get("grade"):
            st.caption(f"Saved review: {selected['grade']} · {selected.get('reviewer_note') or 'No note'}")
        with st.form("label"):
            grades = ["none", "minor", "significant", "unclear"] if selected["quality"] == "ok" else ["unclear"]
            grade = st.selectbox("Visible condition", grades, index=grades.index(selected["grade"]) if selected.get("grade") in grades else 0,
                                 key=f"grade-{capture_id}")
            note = st.text_area("Technician's corrected review", value=selected.get("reviewer_note") or (draft.get("ai_text") if draft else "") or "", key=f"note-{capture_id}",
                                help="Describe what you personally observed. The automatic draft is supporting context, not a saved finding.")
            auto_suggestion = bool((draft and draft.get("ai_text")) or st.session_state.get(f"damage-result-{capture_id}"))
            checked_ai = st.checkbox("I checked every selected photo and corrected the suggested review and counts",
                                      value=False, key=f"checked-ai-{capture_id}") if auto_suggestion else True
            st.caption("These editable values may be AI suggestions. They become confirmed findings only after you check the photos and save. Clear an uncertain count rather than treating zero as proof of no damage.")
            a, b = st.columns(2)
            scratch_count = a.number_input("Confirmed visible scratches", min_value=0, max_value=100,
                                           value=selected.get("scratch_count"), key=f"scratches-{capture_id}")
            dent_count = b.number_input("Confirmed visible dents", min_value=0, max_value=100,
                                        value=selected.get("dent_count"), key=f"dents-{capture_id}")
            if st.form_submit_button("Save reviewed label"):
                if not checked_ai:
                    st.error("Check the photos and confirm your correction before saving an AI-assisted review.")
                elif action("POST", f"/captures/{capture_id}/label", {"grade": grade, "note": note,
                    "scratch_count": None if grade == "unclear" else scratch_count,
                    "dent_count": None if grade == "unclear" else dent_count}, "Technician review saved"):
                    if grade != "unclear":
                        linked = action("POST", "/cases", {"capture_id": capture_id}, "Review added to the inspection queue")
                        if linked:
                            st.rerun()
                    else:
                        st.rerun()
        linked_case = next((item for item in request("GET", f"/cases?site_id={site}")
                            if item["observation_id"] == selected["observation_id"]), None)
        if linked_case:
            st.caption(f"Already linked to case {linked_case['id']} ({linked_case['status']}).")
        elif selected["quality"] == "ok" and selected.get("grade") in ("none", "minor", "significant") and st.button("Open inspection case for this capture"):
            action("POST", "/cases", {"capture_id": capture_id}, "Case created or linked to an existing open case")
        elif selected["quality"] == "ok":
            st.info("Save a clear technician review before opening an inspection case.")

elif page == "Inspection queue":
    heading("Inspection queue", "See what needs checking, review the photo and notes, then record what you actually found.")
    st.caption("The attention level helps order work. It does not diagnose a defect or say the phone is unsafe.")
    capacity = st.number_input("Time available for new inspections (minutes)", 0, 1440, 60, 15,
                               help="Used only to suggest which new cases fit your available time. It does not record time worked.")
    cases = request("GET", f"/cases?site_id={site}")
    ready_reviews = request("GET", f"/cases/review-ready?site_id={site}")
    assets_by_id = {item["id"]: item for item in request("GET", f"/assets?site_id={site}")}
    status_labels = {"open": "Needs assignment", "assigned": "Waiting for approval",
                     "approved": "Ready to inspect", "closed": "Completed"}
    attention_labels = {"high": "High attention", "medium": "Medium attention", "low": "Routine attention"}
    def equipment_name(asset_id):
        asset = assets_by_id.get(asset_id, {})
        return asset.get("model") or asset.get("name") or asset_id

    if ready_reviews:
        st.subheader("Reviewed photos awaiting a work order")
        st.caption("These saved technician reviews predate automatic queue linking or had an earlier order for the same equipment. Each photo can now get its own work order.")
        table([{"Phone": equipment_name(item["asset_id"]), "Reviewed": item["reviewed_at"][:19].replace("T", " ") + " UTC",
                "Visible condition": item["grade"].capitalize(), "Scratches": item["scratch_count"] if item["scratch_count"] is not None else "Not assessed",
                "Dents": item["dent_count"] if item["dent_count"] is not None else "Not assessed", "Photo": item["capture_id"][:8]}
               for item in ready_reviews])
        pending = st.selectbox("Review to add", ready_reviews,
            format_func=lambda item: f"{equipment_name(item['asset_id'])} · {item['reviewed_at'][:19].replace('T', ' ')} UTC · {item['capture_id'][:8]}",
            key="review-ready-capture")
        if st.button("Add this reviewed photo to the inspection queue"):
            if action("POST", "/cases", {"capture_id": pending["capture_id"]}, "Work order created from saved review"):
                st.rerun()

    use_samples = st.toggle("Explore with sample cases", value=not bool(cases or ready_reviews),
                            help="Try the capacity planner with illustrative rows. This creates no inspection records.")
    if use_samples:
        chosen, used = sample_capacity_plan(capacity)
        st.markdown('<div class="sample-note"><strong>Practice planning only</strong><br>These examples are not live photos, findings, or work orders.</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Sample cases", len(SAMPLE_CASES))
        b.metric("Minutes scheduled", used)
        c.metric("Minutes remaining", capacity - used)
    else:
        chosen = request("GET", f"/cases/selected?site_id={site}&capacity_minutes={capacity}")
    st.subheader("Suggested for your available time")
    if use_samples:
        table([{"Example": x["id"], "Equipment": x["asset_id"], "Reason to check": x["scenario"],
                "Attention": attention_labels.get(x["priority_band"], x["priority_band"]),
                "Expected minutes": x["estimated_minutes"]} for x in chosen],
              empty_title="No practice cases fit", empty_message="Increase the available minutes to try the planner.")
    else:
        table([{"Equipment": equipment_name(x["asset_id"]), "Attention": attention_labels.get(x["priority_band"], x["priority_band"]),
                "Expected minutes": x["estimated_minutes"], "Case": x["id"][:8]} for x in chosen],
              empty_title="No new cases fit", empty_message="New cases appear here when they fit the available minutes. Assigned and completed work remains below.")
    st.caption("This suggestion uses estimated time and attention order. The supervisor still decides what work to approve.")
    if use_samples:
        with st.expander("All practice examples"):
            table(SAMPLE_CASES, ["id", "asset_id", "scenario", "estimated_minutes"])
    st.subheader("Work orders")
    visible_cases = sorted(cases, key=lambda x: (x["status"] == "closed", -datetime.fromisoformat(x["created_at"]).timestamp()))
    table([{"Equipment": equipment_name(x["asset_id"]), "Attention": attention_labels.get(x["priority_band"], x["priority_band"]),
            "Stage": status_labels.get(x["status"], x["status"]),
            "Opened (UTC)": x["created_at"][:19].replace("T", " "), "Case": x["id"][:8]} for x in visible_cases],
          empty_title="No real work orders yet", empty_message="Review a usable camera photo and open a case to begin.")
    if cases:
        case_options = visible_cases
        case_lookup = {x["id"]: x for x in case_options}
        case_id = st.selectbox("Choose a work order", list(case_lookup),
                               format_func=lambda case_id: f"{equipment_name(case_lookup[case_id]['asset_id'])} · {status_labels.get(case_lookup[case_id]['status'], case_lookup[case_id]['status'])} · {case_id[:8]}")
        case = request("GET", f"/cases/{case_id}")
        st.subheader(f"Work order · {equipment_name(case['asset_id'])}")
        a, b, c = st.columns(3)
        a.metric("Current stage", status_labels.get(case["status"], case["status"]))
        b.metric("Attention", attention_labels.get(case["priority_band"], case["priority_band"]))
        c.metric("Assigned to", case["assignee_name"]["username"] if case.get("assignee_name") else "Not assigned")
        st.caption(f"Case {case['id']} · Opened {case['created_at'][:19].replace('T', ' ')} UTC")
        st.caption("Work order stages: Needs assignment → Waiting for approval → Ready to inspect → Completed. A supervisor may approve an unassigned order directly.")
        if case["status"] == "approved":
            st.info("Approval authorizes the inspection; it does not mark the work complete. Use **Finish this inspection** below to record the actual finding and time, then save. The stage will become **Completed**.")
        elif case["status"] == "assigned":
            st.info("This order is assigned and waiting for a supervisor to approve inspection instructions.")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**What the camera captured**")
            capture = case.get("capture")
            if capture:
                st.write(f"Photo taken {capture['captured_at'][:19].replace('T', ' ')} UTC · Sample {capture['sample_id']} · Image quality {capture['quality']}")
                condition = {"none": "No visible issue noted", "minor": "Minor visible issue",
                             "significant": "Significant visible issue", "unclear": "Could not judge from photo"}
                if capture.get("reviewed_grade"):
                    st.write(f"**Photo review:** {condition.get(capture['reviewed_grade'], capture['reviewed_grade'])}")
                    st.write(f"**Reviewer's note:** {capture.get('reviewer_note') or 'No note provided.'}")
                    st.write(f"**Technician-confirmed scratches:** {capture['scratch_count'] if capture.get('scratch_count') is not None else 'Not assessed'} · **dents:** {capture['dent_count'] if capture.get('dent_count') is not None else 'Not assessed'}")
                    if case.get("review_draft"):
                        st.caption(f"Review used {len(json.loads(case['review_draft']['source_ids_json']))} photos from one camera session. Counts above are technician-entered, not automated detections.")
                else:
                    st.warning("This photo has no saved reviewer label. Check the image yourself before drawing a conclusion.")
            else:
                st.warning("No source photo is linked to this case.")
            st.markdown("**What to do next**")
            if case["status"] in ("open", "assigned"):
                st.write("A supervisor should assign this work and approve clear inspection instructions.")
            elif case["status"] == "approved":
                st.write("Inspect the physical phone, record only what you actually checked, and enter the time spent.")
            else:
                st.write("This work order is complete. The recorded outcome is shown below.")
            if case.get("decision_note"):
                st.write("**Approved task instructions:**")
                st.write(case["decision_note"])
            elif case["status"] in ("approved", "closed"):
                st.warning("No specific task instructions were recorded for this case.")
        with right:
            if case.get("capture"):
                st.image(request("GET", f"/captures/{capture['id']}/image"), caption="Saved camera photo; verify the phone and visible area before inspecting.", use_container_width=True)
        with st.expander("Why this work order was prioritized"):
            st.write("The scheduling order combines equipment importance, reviewed photos with visible issues, and inspection timing. Multiple photos may show the same phone. This is not a diagnosis or chance of failure.")
            asset = case.get("asset") or {}
            reasons = json.loads(case["reason_json"])
            st.write(f"Equipment importance: {asset.get('criticality', 'unknown')} out of 5.")
            st.write("Previously reviewed photos with visible issues affected the order." if reasons.get("reviewed_repeats", 0) else "No previously reviewed visible issues affected the order.")
            st.write(f"Regular inspection interval: {asset.get('inspection_interval_days', 'unknown')} days. "
                     + ("Inspection timing increased the attention level." if reasons.get("inspection_overdue", 0) else "Inspection timing did not increase the attention level."))
            st.caption(f"Scheduling score when opened: {case['priority']:.0f} out of 100. No earlier inspection date is treated as due by this rule.")
        if case["status"] == "open" and me["role"] in ("admin", "supervisor"):
            users = request("GET", f"/users?site_id={site}")
            u = {x["username"]: x["id"] for x in users}
            assignee = st.selectbox("Worker responsible for inspection", list(u))
            if st.button("Assign work order"):
                if action("POST", f"/cases/{case_id}/assign", {"user_id": u[assignee]}, "Work order assigned"):
                    st.rerun()
        if case["status"] in ("open", "assigned") and me["role"] in ("admin", "supervisor"):
            with st.form(f"approve-{case_id}"):
                decision = st.text_area("Inspection instructions for the worker",
                                        value=case.get("decision_note") or "",
                                        placeholder="Example: Check the front screen and edges for visible damage; record what you can and cannot confirm.")
                override = st.text_input("Reason for changing the usual order (optional)")
                if st.form_submit_button("Approve inspection"):
                    if not decision.strip():
                        st.error("Write clear instructions before approving this inspection.")
                    elif action("POST", f"/cases/{case_id}/approve", {"decision_note": decision.strip(), "override_reason": override or None}, "Inspection approved"):
                        st.rerun()
        if case["status"] == "approved":
            st.subheader("Finish this inspection")
            with st.form(f"close-{case_id}"):
                finding = st.text_area("What did you actually find?",
                                       placeholder="State which parts you checked, what you observed, and anything you could not verify.")
                inspection_performed = st.checkbox("I physically inspected this phone", value=False,
                    help="Select only if you handled and inspected the real phone. A photo review alone does not count.")
                minutes = st.number_input("Minutes actually spent inspecting", 0, 1440, 0)
                note = st.text_input("Additional closing note (optional)")
                st.caption("For a photo-only review, leave the checkbox clear, enter 0 minutes, and describe the image limits.")
                if st.form_submit_button("Save finding and complete work order"):
                    if not finding.strip():
                        st.error("Describe what you found before completing the work order.")
                    elif inspection_performed and minutes == 0:
                        st.error("Enter the time actually spent on the physical inspection.")
                    elif not inspection_performed and minutes > 0:
                        st.error("For a photo-only review, enter 0 inspection minutes.")
                    elif action("POST", f"/cases/{case_id}/close", {"actual_finding": finding.strip(), "actual_minutes": minutes,
                        "note": note, "inspection_performed": inspection_performed}, "Finding saved; work order complete"):
                        st.rerun()
        if case["status"] == "closed":
            st.markdown("**Recorded outcome**")
            st.write(case.get("actual_finding") or "No finding recorded.")
            st.caption(f"Physical inspection: {'Yes' if case.get('inspection_performed') else 'No'} · Actual inspection minutes: {case.get('actual_minutes') or 0}")
            if case.get("closure_note"):
                st.write(f"**Closing note:** {case['closure_note']}")

elif page == "Assets & setup":
    heading("Assets and setup", "Register inspection subjects and map each live camera to the phone currently in view.")
    if local_camera_view:
        st.info("Local laptop: Lenovo source 0 opens this computer's webcam. Samsung uses its current HTTP IP Webcam address directly when both devices share Wi-Fi. Start either one from Live cameras.")
    else:
        st.info("For the hosted link, choose source browser. It uses the webcam on the laptop or phone viewing the page. Source 0 and private IP Webcam addresses are opened by the home server, so they cannot access a remote laptop camera or a Samsung on another Wi-Fi network.")
    if me["role"] in ("admin", "supervisor"):
        with st.expander("Add a phone model or equipment type", expanded=False):
            preset = st.selectbox("Inspection subject", ["iPhone 14 Plus", "realme narzo 60 5G", "Custom"])
            defaults = {"iPhone 14 Plus": ("iphone-14-plus", "iPhone 14 Plus exterior", "iphone-exterior", "iPhone 14 Plus"),
                        "realme narzo 60 5G": ("realme-narzo-60", "realme narzo 60 exterior", "realme-exterior", "realme narzo 60 5G"),
                        "Custom": ("asset-01", "Visible exterior sample", "custom-exterior", "")}[preset]
            with st.form("asset"):
                a, b = st.columns(2)
                asset_id = a.text_input("Asset ID", defaults[0], key=f"id-{preset}")
                name = b.text_input("Name", defaults[1], key=f"name-{preset}")
                family = a.text_input("Family", defaults[2], key=f"family-{preset}")
                model = b.text_input("Exact model (when known)", defaults[3], key=f"model-{preset}")
                criticality = a.slider("Criticality", 1, 5, 3)
                interval = b.number_input("Inspection interval (days)", 1, 365, 30)
                if st.form_submit_button("Register asset"):
                    action("POST", "/assets", {"id": asset_id, "site_id": site, "name": name, "family": family, "model": model or None,
                                                  "criticality": criticality, "inspection_interval_days": interval})
    assets = request("GET", f"/assets?site_id={site}")
    table(assets, ["id", "name", "family", "model", "criticality", "last_inspection_at"])
    physical_phones = request("GET", f"/samples?site_id={site}")
    st.subheader("Individual physical phones")
    st.caption("Register each actual device once. Keep its ID when taking more photos from different angles. A second photo of the same device is still one phone for model training.")
    asset_names = {item["id"]: item.get("model") or item["name"] for item in assets}
    table([{"Phone ID": item["id"], "Name": item["display_name"],
            "Model": asset_names.get(item["asset_id"], item["asset_id"]),
            "Registered (UTC)": item["created_at"][:19].replace("T", " ")}
           for item in physical_phones], empty_title="No physical phones registered",
          empty_message="Add a phone model above, then register each separate physical device here.")
    if assets and me["role"] in ("admin", "supervisor"):
        with st.expander("Register another physical phone", expanded=False):
            model_asset = st.selectbox("Phone model", [item["id"] for item in assets],
                format_func=lambda item: asset_names[item], key="new-phone-model")
            existing_ids = {item["id"] for item in physical_phones}
            number = 1
            while f"{model_asset}-unit-{number:02d}" in existing_ids:
                number += 1
            with st.form(f"physical-phone-{model_asset}"):
                phone_id = st.text_input("Unique physical phone ID", f"{model_asset}-unit-{number:02d}")
                display_name = st.text_input("Display name", f"{asset_names[model_asset]} · device {number}")
                st.caption("Use a different ID only for a genuinely different phone. Model training groups all photos by this ID.")
                if st.form_submit_button("Register physical phone"):
                    if action("POST", "/samples", {"id": phone_id.strip(), "site_id": site,
                            "asset_id": model_asset, "display_name": display_name.strip()},
                            "Physical phone registered"):
                        st.rerun()
    if me["role"] == "admin" and assets:
        with st.expander("Add a live camera", expanded=False):
            preset = st.selectbox("Camera device", ["Samsung Galaxy S24 FE stream", "Lenovo webcam", "Custom"])
            defaults = {"Samsung Galaxy S24 FE stream": ("camera-samsung", "Samsung Galaxy S24 FE camera", "browser", "iphone-14-plus", "iphone-physical-01"),
                        "Lenovo webcam": ("camera-laptop", "Lenovo built-in webcam", "0" if local_camera_view else "browser", "realme-narzo-60", "realme-physical-01"),
                        "Custom": ("camera-other", "Other live camera", "", assets[0]["id"], "phone-physical-01")}[preset]
            with st.form("camera"):
                camera_id = st.text_input("Camera ID", defaults[0], key=f"new-camera-id-{preset}")
                camera_name = st.text_input("Name", defaults[1], key=f"new-camera-name-{preset}")
                asset_ids = [x["id"] for x in assets]
                default_asset = defaults[3] if defaults[3] in asset_ids else asset_ids[0]
                asset_id = st.selectbox("Inspected asset", asset_ids, index=asset_ids.index(default_asset), key=f"new-camera-asset-{preset}")
                source = st.text_input("Camera source: browser, local device index, or network URL", defaults[2],
                        placeholder="http://PHONE_IP:8080/video", key=f"new-camera-source-{preset}")
                sample_id = st.text_input("Physical sample ID", defaults[4], key=f"new-camera-sample-{preset}")
                st.caption("On localhost, use 0 for the built-in webcam and the Samsung's http://IP:PORT/video for its stream. On the hosted page, browser or the paired relay is needed. The Samsung is the camera, not an inspected asset.")
                st.caption("Give each physical phone a distinct sample ID. Two simultaneous views need two separate camera sources.")
                if st.form_submit_button("Register camera"):
                    if not source.strip():
                        st.error("Enter browser or a reachable live camera source.")
                    else:
                        action("POST", "/cameras", {"id": camera_id, "site_id": site, "asset_id": asset_id, "name": camera_name, "source": source.strip(), "sample_id": sample_id, "roi": [0, 0, 1, 1]})
    cameras = request("GET", f"/cameras?site_id={site}")
    table(cameras, ["id", "name", "source", "asset_id", "sample_id", "status"])
    if cameras and me["role"] == "admin":
        camera_id = st.selectbox("Worker control", [x["id"] for x in cameras])
        selected_camera = next(x for x in cameras if x["id"] == camera_id)
        if camera_id == "camera-samsung":
            st.subheader("Samsung IP Webcam on the same Wi-Fi as this laptop")
            st.caption("The Samsung and the laptop hosting ForgeCare must share a Wi-Fi network that permits device-to-device traffic. If you open the hosted link from another network, use a local ForgeCare installation on that network for direct capture.")
            saved_ip = selected_camera["source"].removeprefix("relay:").removesuffix("/video")
            if not saved_ip.startswith("http://"):
                saved_ip = ""
            phone_address = st.text_input("Samsung IP Webcam IPv4 address and port", value=saved_ip,
                placeholder="http://192.168.1.23:8080", key="samsung-ip-webcam-address")
            st.caption("Paste the current HTTP address displayed by IP Webcam, such as http://10.73.3.83:8080. Do not paste its HTTPS address. No bridge is needed when the hosting laptop can reach this phone.")
            if st.button("Save Samsung IP Webcam address"):
                try:
                    source = ip_webcam_direct_source(phone_address)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if action("PUT", f"/cameras/{camera_id}", {"asset_id": selected_camera["asset_id"],
                            "source": source, "sample_id": selected_camera["sample_id"],
                            "roi": json.loads(selected_camera["roi_json"])}, "Samsung address saved"):
                        st.rerun()
            if selected_camera["source"].startswith(("relay:http://", "http://")):
                stream_url = selected_camera["source"].removeprefix("relay:")
                st.link_button("Test moving phone video on this laptop", stream_url)
                st.caption("Open Live cameras and press Start. The hosting laptop reads this address directly and automatically saves new frames.")
        st.caption("Stop a running worker before changing its inspection subject or physical sample ID.")
        with st.form(f"camera-config-{camera_id}"):
            ids = [x["id"] for x in assets]
            mapped_asset = st.selectbox("Phone being inspected", ids, index=ids.index(selected_camera["asset_id"]), key=f"camera-asset-{camera_id}")
            source = st.text_input("Camera source (browser, device index, or MJPEG URL)", selected_camera["source"], key=f"camera-source-{camera_id}")
            sample_id = st.text_input("Physical sample ID", selected_camera["sample_id"] or "", key=f"camera-sample-{camera_id}")
            matching_phones = [item for item in physical_phones if item["asset_id"] == mapped_asset]
            if matching_phones:
                st.caption("Registered IDs for this model: " + ", ".join(item["id"] for item in matching_phones[:15]))
            if st.form_submit_button("Save camera mapping"):
                result = action("PUT", f"/cameras/{camera_id}", {"asset_id": mapped_asset, "source": source,
                        "sample_id": sample_id, "roi": json.loads(selected_camera["roi_json"])}, "Camera mapping saved")
                if result:
                    st.rerun()
        if selected_camera["source"] == "browser":
            st.info("Open Live cameras and press Start selected camera to switch this camera to the local source." if local_camera_view else
                    "Browser photo capture is ready. Open Live cameras on the device you want to use; no server worker is needed.")
        elif selected_camera["source"].startswith("relay:"):
            st.info("This camera still has its old relay configuration. Save the current phone HTTP address above to switch to direct capture.")
        else:
            a, b = st.columns(2)
            if a.button("Start live camera worker"):
                action("POST", f"/cameras/{camera_id}/start", {}, "Worker started")
            if b.button("Stop live camera worker"):
                action("POST", f"/cameras/{camera_id}/stop", {}, "Worker stopped")
    if me["role"] in ("admin", "supervisor"):
        st.subheader("Priority rules")
        rules = request("GET", f"/rules?site_id={site}")
        st.caption(f"Current rule version {rules['id']}")
        with st.form("rules"):
            a, b, c = st.columns(3)
            cw = a.number_input("Criticality weight", 0.0, 1.0, float(rules["criticality_weight"]), .05)
            rw = b.number_input("Repeated reviewed condition", 0.0, 1.0, float(rules["repeat_weight"]), .05)
            ow = c.number_input("Inspection overdue", 0.0, 1.0, float(rules["overdue_weight"]), .05)
            duration = st.number_input("Default inspection minutes", 1, 480, int(rules["default_minutes"]))
            if st.form_submit_button("Save new rule version"):
                action("POST", "/rules", {"site_id": site, "criticality_weight": cw, "repeat_weight": rw, "overdue_weight": ow, "default_minutes": duration})

elif page == "Knowledge assistant":
    heading("Cited maintenance assistance", "Procedural guidance comes from applicable approved sources and requires engineer review.")
    assets = request("GET", f"/assets?site_id={site}")
    try:
        docs = request("GET", f"/documents?site_id={site}")
    except Exception as exc:
        docs = []
        st.warning(f"Knowledge service unavailable: {exc}")
    table(docs, ["title", "family", "model", "revision", "status", "url"])
    if docs and me["role"] in ("admin", "reviewer"):
        active_docs = [x for x in docs if x["status"] == "active"]
        if active_docs:
            chosen_document = st.selectbox("Active document to retire", [x["id"] for x in active_docs])
            if st.button("Deactivate selected revision"):
                action("POST", f"/documents/{chosen_document}/deactivate?site_id={site}", {}, "Revision deactivated")
    if me["role"] in ("admin", "reviewer"):
        with st.expander("Index an official phone or manufacturing page"):
            source = st.selectbox("Source preset", ["iPhone 14 Plus battery procedure", "iPhone 14 Plus back glass procedure", "iPhone 14 Plus broken glass safety", "iPhone 14 Plus repair manual", "realme narzo 60 5G specifications", "realme support FAQ", "NIST process control charts", "AIAG automotive quality tools", "Custom"])
            source_defaults = {"iPhone 14 Plus repair manual": ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100322"),
                               "iPhone 14 Plus battery procedure": ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100346"),
                               "iPhone 14 Plus back glass procedure": ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100328"),
                               "iPhone 14 Plus broken glass safety": ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100298"),
                               "realme narzo 60 5G specifications": ("realme-exterior", "realme narzo 60 5G", "https://www.realme.com/in/realme-narzo-60/specs"),
                               "realme support FAQ": ("realme-exterior", "*", "https://www.realme.com/in/support/faq"),
                               "NIST process control charts": ("manufacturing-general", "*", "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm"),
                               "AIAG automotive quality tools": ("automotive-general", "*", "https://www.aiag.org/expertise-areas/quality/quality-core-tools"),
                               "Custom": ("custom-exterior", "*", "")}[source]
            with st.form("doc"):
                family = st.text_input("Equipment family", source_defaults[0], key=f"doc-family-{source}")
                model = st.text_input("Exact model or * for general guidance", source_defaults[1], key=f"doc-model-{source}")
                url = st.text_input("Official support HTTPS URL", source_defaults[2], key=f"doc-url-{source}")
                revision = st.text_input("Revision label", "2026-11")
                st.caption("The realme specifications page describes product facts; it does not provide a repair procedure. Unsupported repair questions must abstain.")
                if st.form_submit_button("Review and index source"):
                    action("POST", "/documents", {"site_id": site, "family": family, "model": model, "url": url, "revision": revision}, "Document indexed")
    if assets:
        asset_names = {x["id"]: x.get("model") or x["name"] for x in assets}
        asset = st.selectbox("Ask about asset", [x["id"] for x in assets], format_func=lambda value: asset_names[value])
        st.subheader("Ask the knowledge assistant")
        st.caption("Ask about the selected phone, manufacturing quality, or automotive production. The assistant cites approved sources and marks gaps when those sources do not answer.")
        area = st.selectbox("Question area", ["Automatic", "This phone", "Manufacturing", "Automotive quality"])
        scope = {"Automatic": "auto", "This phone": "phone", "Manufacturing": "manufacturing", "Automotive quality": "automotive"}[area]
        chat_key = f"knowledge-chat-{site}-{asset}-{scope}"
        messages = st.session_state.setdefault(chat_key, [])
        if st.button("Clear this conversation"):
            messages.clear()
            st.rerun()
        question = st.chat_input("Ask about this equipment or its approved guidance")
        if question:
            history = [{"role": item["role"], "content": item["content"]} for item in messages[-8:]]
            messages.append({"role": "user", "content": question})
            result = action("POST", "/assistant/ask", {"site_id": site, "asset_id": asset,
                               "question": question, "history": history, "scope": scope}, "Guidance checked")
            if result:
                messages.append({"role": "assistant", "content": result.get("answer") or "",
                                 "evidence": result.get("evidence", []), "status": result["status"],
                                 "reason": result.get("reason"), "provider": result.get("provider"),
                                 "scope": result.get("scope"), "asset_model_used": result.get("asset_model_used"),
                                 "searched_sources": result.get("searched_sources", [])})
        if not messages:
            st.info("Try: What should I check for cracked glass? What is a control chart? What does FMEA mean in automotive quality?")
        for item in messages:
            with st.chat_message(item["role"]):
                if item["role"] == "user":
                    st.write(item["content"])
                    continue
                evidence = item.get("evidence", [])
                if item.get("scope") == "manufacturing":
                    st.caption("Manufacturing quality guidance")
                elif item.get("scope") == "automotive":
                    st.caption("Automotive quality guidance")
                elif item.get("asset_model_used"):
                    st.caption(f"Guidance for {item['asset_model_used']}")
                if item.get("content"):
                    linked = item["content"]
                    for index, source in enumerate(evidence, 1):
                        linked = linked.replace(f"[{source['id']}]", f"[Source {index}]({source['url']})")
                    st.markdown(linked)
                    if item.get("provider") == "approved sources":
                        st.caption("Summary of the linked manufacturer procedure · Read the complete guide before acting.")
                    else:
                        st.caption(f"Generated with {item.get('provider') or 'the configured model'} · Check the linked passages before acting.")
                elif item.get("status") == "retrieval_only":
                    st.warning("I found related references, but could not verify a supported answer from them. Check the sources below or try again later.")
                    for source in evidence[:2]:
                        excerpt = " ".join(source["text"].split()[:22])
                        st.markdown(f"> {excerpt}…  [Read the official source]({source['url']})")
                    st.caption(item.get("reason") or "Generation unavailable")
                else:
                    st.warning("I could not find a supported answer in the references searched for this question.")
                    if item.get("reason") and "exact component" in item["reason"].lower():
                        st.caption("An exact component specification needs its approved manual.")
                    searched = item.get("searched_sources", [])
                    if searched:
                        with st.expander(f"References searched · {len(searched)} documents"):
                            for source in searched:
                                st.markdown(f"[{source['title']}]({source['url']})")
                if evidence:
                    with st.expander(f"Sources used or retrieved · {len(evidence)} passages"):
                        for index, source in enumerate(evidence, 1):
                            st.markdown(f"**Source {index}: [{source['title']}]({source['url']})** · revision {source['revision']}")
                            st.write(source["text"])
        st.subheader("Prepare an inspection task")
        st.caption("The local agent may draft text. A supervisor must approve any case task separately.")
        agent_request = st.text_input("Task request", "Prepare a cited visual inspection task for this asset")
        candidate_cases = request("GET", f"/cases?site_id={site}")
        matching_cases = [x for x in candidate_cases if x["asset_id"] == asset and x["status"] in ("open", "assigned")]
        case_labels = {case["id"]: f"Work order {case['id'][:8]} · {case['status'].replace('_', ' ').title()}"
                       for case in matching_cases}
        selected_case = st.selectbox("Related case", ["No case"] + [x["id"] for x in matching_cases],
                                     format_func=lambda value: case_labels.get(value, value))
        if st.button("Run bounded agent"):
            try:
                result = request("POST", "/assistant/prepare", json={"site_id": site, "asset_id": asset,
                                 "case_id": None if selected_case == "No case" else selected_case,
                                 "request": agent_request})
            except Exception as exc:
                st.error(f"Could not prepare guidance: {exc}")
                result = None
            if result:
                if result["status"] == "pending_approval":
                    st.success("A source-cited draft is ready for supervisor review. It has not been approved.")
                elif result["status"] == "pending_approval_extractive":
                    st.info("Approved source passages were found. Review this source packet and write or approve the task yourself.")
                elif result["status"] == "abstained_no_applicable_document":
                    st.warning("The answer to the question was not found based on the references searched. Please find the given resources.")
                    searched = result.get("searched_sources", [])
                    if searched:
                        with st.expander(f"References searched · {len(searched)} documents", expanded=True):
                            for source in searched:
                                st.markdown(f"[{source['title']}]({source['url']})")
                    else:
                        st.caption("No approved reference documents are indexed for this phone yet.")
                else:
                    st.warning("No task was prepared. Check the approved sources and try again.")
                if result.get("draft"):
                    st.text_area("Draft for supervisor review", result["draft"], height=220)
                if result.get("evidence"):
                    with st.expander(f"Approved passages checked · {len(result['evidence'])}"):
                        for source in result["evidence"]:
                            st.markdown(f"**[{source['title']}]({source['url']})** · revision {source['revision']}")
                            st.write(source["text"])
                trace = result.get("trace", [])
                if trace:
                    with st.expander("Steps recorded for this guidance draft"):
                        table([{"Step": item.get("tool", "Unknown step").replace("_", " ").capitalize(),
                                "When (UTC)": (item.get("at") or "")[:19].replace("T", " "),
                                "Evidence found": item.get("result_count", "—")}
                               for item in trace])

elif page == "Models & experiments":
    heading("Model workbench", "Compare curriculum models on reproducible splits; approve only evaluated candidates.")
    automotive = autovi_metrics()
    if automotive:
        st.subheader("Automotive production line · pipe clip inspection")
        st.caption("Renault Group AutoVI · actual automotive assembly-line camera images · Anomalib PaDiM learned from acceptable examples only. Defects in this research dataset were deliberately staged and then corrected by Renault experts.")
        ac = automotive["test_confusion"]
        a, b, c, d = st.columns(4)
        a.metric("Defective clips caught", ac["tp"])
        b.metric("Defective clips missed", ac["fn"])
        c.metric("Good clips flagged", ac["fp"])
        d.metric("Good clips passed", ac["tn"])
        st.warning("Business reading: on 169 held-out automotive images, the model caught 67 of 71 defective clips but would send 67 of 98 acceptable clips to unnecessary review. It is not ready to control a production line.")
        with st.expander("See real automotive photo and model overlay", expanded=True):
            folder = AUTOVI_RUN / "results" / "Padim" / "AutoVI" / "pipe_clip" / "v0" / "images"
            examples = {"Acceptable example": folder / "good" / "0001.png",
                        "Unclipped example": folder / "unclipped" / "0000.png"}
            choice = st.selectbox("Example from the held-out check", list(examples))
            if examples[choice].exists():
                st.image(str(examples[choice]), caption="Left to right: original image, ground truth mask, model anomaly overlay, model mask. These are saved outputs from the actual fitted model.", use_container_width=True)
        with st.expander("Technical evaluation details"):
            measured = automotive["anomalib_metrics"][0]
            st.write(f"Image ROC AUC {measured['image_AUROC']:.3f}; pixel ROC AUC {measured['pixel_AUROC']:.3f}. Training and evaluation timestamps, source hashes, and the saved checkpoint are in the research folder.")
    automotive_fed = autovi_fed_metrics()
    if automotive_fed:
        st.subheader("Automotive pipe clip · two-client Flower study")
        st.caption("A second model trained on genuine AutoVI assembly photos using two in-process Flower clients. This study uses an adapted supervised split that includes staged defect labels; its counts cannot be compared directly with the normal-only PaDiM study above.")
        check = automotive_fed["test"]
        confusion = check["confusion"]
        a, b, c, d = st.columns(4)
        a.metric("Defective clips caught", f"{confusion['tp']} of {check['defective']}")
        b.metric("Defective clips missed", confusion["fn"])
        c.metric("Good clips flagged", f"{confusion['fp']} of {check['normal']}")
        d.metric("Good clips passed", confusion["tn"])
        st.warning("The model sent 19 of 21 defective clips for review, but also flagged 30 of 60 acceptable clips. This is a research demonstration, not a line-release decision.")
        with st.expander("Test the saved Flower model on a held-out automotive photo", expanded=True):
            from forgecare.casting_demo import inspect as inspect_autovi, test_photos as autovi_test_photos
            examples = autovi_test_photos(AUTOVI_FED_RUN)
            if examples:
                choices = {f"{row['label']} · {Path(row['source_path']).parent.name}/{Path(row['source_path']).name}": row for row in examples}
                choice = st.selectbox("Automotive held-out photo", list(choices), key="autovi-fed-photo")
                pixels, score, flagged = inspect_autovi(Path(choices[choice]["source_path"]),
                                                         autovi_fed_model(str(AUTOVI_FED_RUN)), check["threshold"])
                left, right = st.columns([1, 1])
                left.image(pixels, caption=f"Original AutoVI photo · source label: {choices[choice]['label']}", width=350)
                right.write("**Model output:** Send for worker review" if flagged else "**Model output:** No alert")
                right.caption(f"Research score {score:.2f}; validation-selected alert threshold {check['threshold']:.2f}. The worker must inspect the actual part.")
            else:
                st.info("The noncommercial automotive source photos are not bundled with this portable copy. The trained-run measurements remain visible above; download the dataset separately for this preview.")
        with st.expander("Technical training details"):
            st.write(f"Three actual Flower FedAvg rounds; 179 different train photos per in-process client; 13 exact duplicate source photos excluded before splitting. Held-out ROC AUC {check['roc_auc']:.3f}. No separate factory-site training or privacy claim.")
    phone_study = phone_metrics()
    if phone_study:
        st.subheader("Phone screen crack research · separate side study")
        st.caption(f"Public Kaggle cracked/intact phone photos · {phone_study['images']} readable images from two sources · frozen ResNet18 features with a trained classifier. No iPhone 14 Plus or realme test subjects were in this evaluation.")
        check = phone_study["test"]
        a, b, c = st.columns(3)
        a.metric("Cracked screens flagged", f"{check['confusion']['tp']} of {check['cracked']}")
        b.metric("Cracked screens missed", check["confusion"]["fn"])
        c.metric("Intact screens flagged", check["confusion"]["fp"])
        st.warning(f"{check['images']} photos were held out, but only {check['intact']} were intact. The sources lack physical-phone IDs and scratch/dent labels, so these numbers do not establish performance on the demonstration phones or support damage counts.")
        from forgecare.phone_demo import test_photos as phone_test_photos, inspect as inspect_phone
        with st.expander("Try the saved model on a held-out phone photo"):
            rows = phone_test_photos(PHONE_RUN)
            if rows:
                options = {f"{row['label']} · {Path(row['source_path']).name}": row for row in rows}
                chosen = st.selectbox("Held-out phone photo", list(options))
                selected_photo = Path(options[chosen]["source_path"])
                score, flagged = inspect_phone(selected_photo, phone_model(str(PHONE_RUN)), phone_study["threshold"])
                left, right = st.columns([1, 1])
                left.image(str(selected_photo), caption=f"Source label: {options[chosen]['label']}", width=350)
                right.write("**Model output:** Possible crack; review the screen" if flagged else "**Model output:** No crack alert")
                right.caption(f"Research score {score:.2f}. No scratch or dent count is available from these labels.")
            else:
                st.info("Source test photos are not bundled with this portable copy. You can still run the model on a new saved camera photo in Capture review.")
    phone_damage = phone_damage_metrics()
    if phone_damage:
        st.subheader("Phone exterior damage · box detection study")
        st.caption("Public Roboflow phone photos · a locally fine-tuned four-class detector · related frames and duplicate images kept out of the held-out comparison. These are candidate marks per photo, not a diagnosis or a count of unique marks on a physical phone.")
        damage_test = phone_damage["test"]
        damage_classes = damage_test["classes"]
        st.write(f"**Held-out check:** {damage_test['photos']} photos from source groups kept separate from training. A box counts as correctly found only if it matches a labeled region of the same type and overlaps it by at least half.")
        st.caption("The public held-out photos are almost all damaged. This study does not measure how often a clean phone would receive a false alert, and it does not validate a total number of distinct marks across camera views.")
        rows = []
        for label, counts in damage_classes.items():
            rows.append({"Damage type": label, "Labeled marks": counts["annotated_regions"],
                         "Found in right place": counts["localized_correctly"],
                         "Missed": counts["missed_regions"], "Extra marks suggested": counts["false_regions"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        phone_report = PHONE_DAMAGE_RUN / "BUSINESS_SUMMARY.md"
        if phone_report.exists():
            st.download_button("Download plain-language phone result summary", phone_report.read_text(encoding="utf-8"),
                               file_name="phone_damage_business_summary.md", mime="text/markdown")
        dent = damage_classes["Dent"]
        st.warning(f"Dent detection is disabled: of {dent['annotated_regions']} labeled dent regions in the held-out photos, the detector found {dent['localized_correctly']} and missed {dent['missed_regions']}. A worker must assess dents manually and can change the written review and confirmed count.")
        with st.expander("Try the saved detector on a held-out phone photo"):
            prediction_file = PHONE_DAMAGE_RUN / "held_out_predictions.jsonl"
            if prediction_file.exists():
                examples = [json.loads(line) for line in prediction_file.read_text(encoding="utf-8").splitlines() if line]
                names = [Path(row["image_path"]).name for row in examples]
                choice = st.selectbox("Held-out damage photo", range(len(examples)), format_func=lambda index: names[index], key="phone-damage-example")
                example_path = Path(examples[choice]["image_path"])
                if not example_path.is_file():
                    st.info("The local research images are not present in this copy. Use the prepared laptop dataset to run this example.")
                if st.button("Run detector on this photo", key="phone-damage-example-run", disabled=not example_path.is_file() or not phone_damage_runtime_ready()):
                    from forgecare.phone_damage_demo import inspect as inspect_damage
                    result = inspect_damage(example_path.read_bytes(),
                        phone_damage_model(str(PHONE_DAMAGE_RUN / "phone_damage_detector.pt")),
                        phone_damage["validation"]["threshold"])
                    st.session_state["phone-damage-example-result"] = (choice, result)
                selected_result = st.session_state.get("phone-damage-example-result")
                if selected_result and selected_result[0] == choice:
                    st.image(selected_result[1]["overlay_rgb"], caption="Saved local model suggestions on a held-out public photo", width=500)
                    st.write(selected_result[1]["counts"])
        with st.expander("Technical detector details"):
            st.write(f"Confidence selected on validation photos: {phone_damage['validation']['threshold']:.2f}. Box matching uses class and intersection-over-union of at least 0.5. Public source labels contain inconsistencies; those image hashes were excluded before fitting. No ground truth exists yet for this user's iPhone 14 Plus or realme narzo 60.")
    casting = casting_metrics()
    if casting:
        st.subheader("Cast pump impeller inspection · real manufactured parts")
        st.caption("Kaggle casting product dataset · 1,300 unaugmented grayscale originals · two-client Flower research model. These are pump parts, not vehicle-specific or phone images.")
        result = casting["test"]
        a, b, c, d = st.columns(4)
        a.metric("Defective parts caught", f"{result['confusion']['tp']} of {result['defective']}")
        b.metric("Defective parts missed", result["confusion"]["fn"])
        c.metric("Good parts flagged", result["confusion"]["fp"])
        d.metric("Good parts passed", result["confusion"]["tn"])
        caught = result["confusion"]["tp"]
        missed = result["confusion"]["fn"]
        false_alerts = result["confusion"]["fp"]
        alerts = caught + false_alerts
        st.warning(f"**Pilot decision: keep a worker in control.** On {result['images']} held-out photos, {alerts} parts would enter the review queue. Of those, {false_alerts} were acceptable. The model gave no alert on {missed} defective parts, so it must not clear parts automatically.")
        st.write(f"**What the plant team can plan for:** {caught} defective parts reached inspection; {missed} defects were missed; {false_alerts} acceptable parts would consume recheck time. These are counts from this test set, not a forecast for a factory shift.")
        business_report = CASTING_RUN / "BUSINESS_SUMMARY.md"
        if business_report.exists():
            st.download_button("Download plain-language result summary", business_report.read_text(encoding="utf-8"),
                               file_name="cast_impeller_business_summary.md", mime="text/markdown")
        with st.expander("Estimate checking time and possible cost", expanded=True):
            st.caption("Enter your plant's assumptions. The calculation applies only to these 195 held-out photos; no actual factory time, scrap cost, or savings were measured.")
            review_minutes = st.number_input("Minutes to check one flagged part", min_value=0.0, max_value=240.0, value=5.0, step=1.0, key="casting-review-minutes")
            a, b = st.columns(2)
            a.metric("Time to check all alerts", f"{alerts * review_minutes / 60:.1f} hours")
            b.metric("Of that, time on good parts", f"{false_alerts * review_minutes / 60:.1f} hours")
            recheck_cost = st.number_input("Cost to recheck one acceptable part (₹)", min_value=0.0, value=0.0, step=100.0, key="casting-recheck-cost")
            missed_cost = st.number_input("Expected downstream cost per missed alert (₹)", min_value=0.0, value=0.0, step=100.0, key="casting-missed-cost")
            if recheck_cost or missed_cost:
                st.write(f"**Illustrative exposure for this test set:** ₹{false_alerts * recheck_cost + missed * missed_cost:,.0f} = {false_alerts} unnecessary rechecks × ₹{recheck_cost:,.0f} + {missed} missed defects × ₹{missed_cost:,.0f}.")
                st.caption("A real cost case needs plant-specific defect prevalence, labour time, downstream loss, and a comparison with the current inspection process.")
        with st.expander("Technical evaluation details"):
            st.write(f"ROC AUC {result['roc_auc']:.3f} · F1 {result['f1']:.3f} · validation-selected alert threshold {result['threshold']:.2f} · {casting['rounds']} Flower rounds.")
        from forgecare.casting_demo import inspect as inspect_casting, test_photos as casting_test_photos
        test_rows = casting_test_photos(CASTING_RUN)
        if test_rows:
            with st.expander("Test the saved model on a real held-out part image", expanded=True):
                options = {f"{item['label']} · {Path(item['source_path']).name}": item for item in test_rows}
                chosen = st.selectbox("Cast impeller test image", list(options))
                pixels, score, flagged = inspect_casting(Path(options[chosen]["source_path"]),
                                                          casting_model(str(CASTING_RUN)), result["threshold"])
                left, right = st.columns([1, 1])
                left.image(pixels, caption=f"Actual held-out pump impeller · label: {options[chosen]['label']}", width=350)
                right.write("**Model recommendation:** Check this part" if flagged else "**Model recommendation:** No alert")
                right.write(f"**Known source label:** {options[chosen]['label']}")
                right.caption("Compare the recommendation with the known label. A worker must inspect the actual part; the model does not identify root cause or make an acceptance decision.")
                with right.expander("Technical model score"):
                    st.write(f"Review score {score:.3f}; alert threshold {result['threshold']:.2f}.")
    research = research_metrics()
    if research:
        st.subheader("Earlier VisA capsule experiment")
        st.caption("Amazon VisA capsules · CC BY 4.0 · 702 real images. These research models are not approved for live phone or automotive inspection.")
        studies = [("Patch anomaly baseline", research["patch_baseline"]),
                   ("Flower FedAvg CNN", research["flower_fedavg"])]
        anomalib_file = RESEARCH_RUN / "anomalib_padim" / "anomalib_metrics.json"
        if anomalib_file.exists():
            anomalib = json.loads(anomalib_file.read_text(encoding="utf-8"))
            tested = anomalib["test_result"][0]
            st.write(f"**Anomalib PaDiM (separate subset):** image ROC AUC {tested['image_AUROC']:.3f}; "
                     f"image F1 {tested['image_F1Score']:.3f}. Trained on real VisA capsules images; "
                     "its holdout selection differs from the two models below, so the scores are not a direct comparison.")
        table([{"Model": title, "Held-out images": result["test_images"],
                "Anomalies found": f"{result['confusion']['tp']}/{result['test_anomaly']}",
                "False alerts": result["confusion"]["fp"], "ROC AUC": round(result["roc_auc"], 3)}
               for title, result in studies])
        st.warning("The patch and Flower models miss defects or raise too many false alerts. PaDiM also needs a proper target-site evaluation. These weights are research artifacts and do not run on the live cameras.")
        for name in ("patch_example_normal.png", "patch_example_anomaly.png"):
            path = RESEARCH_RUN / name
            if path.exists():
                st.image(str(path), caption=f"Research localization overlay · {name}", width=360)
        padim_example = (RESEARCH_RUN / "anomalib_padim" / "results" / "Padim" /
                         "visa_capsules" / "v0" / "images" / "abnormal_test" / "test-003.JPG")
        if padim_example.exists():
            st.image(str(padim_example), caption="PaDiM example: source photo, anomaly map, predicted mask. Research output on a VisA capsule image.", width=600)
        from forgecare.research_demo import inspect, test_photos
        held_out = test_photos(RESEARCH_RUN)
        if held_out and (RESEARCH_RUN / "flower_fedavg_cnn_scripted.pt").exists():
            with st.expander("Inspect a held-out manufacturing photo", expanded=True):
                st.caption("Choose a real VisA test photo. This read-only research preview never creates a phone finding or inspection case.")
                choices = {f"{row['label']} · {Path(row['source_path']).name}": row for row in held_out}
                selected_name = st.selectbox("Held-out VisA photo", list(choices))
                patch_model, flower_model, study_metrics = research_models(str(RESEARCH_RUN))
                output = inspect(Path(choices[selected_name]["source_path"]), patch_model, flower_model, study_metrics)
                left, right = st.columns(2)
                left.image(output["original"], caption=f"Actual source photo · label: {choices[selected_name]['label']}", use_container_width=True)
                right.image(output["patch_overlay"], caption="Patch-distance heatmap · color scaled within this photo", use_container_width=True)
                a, b = st.columns(2)
                a.metric("Patch anomaly score", f"{output['patch_score']:.3f}", "Review alert" if output["patch_alert"] else "No alert")
                b.metric("Flower CNN score", f"{output['flower_score']:.3f}", "Review alert" if output["flower_alert"] else "No alert")
                st.caption("Scores are model outputs, not defect counts. Both research models had weak held-out performance and must not be used to clear a part or diagnose a phone.")
    st.info("The visual trainer requires reviewed captures from several distinct physical samples in both classes. Same-phone frames remain in one split.")
    model_assets = [x for x in request("GET", f"/assets?site_id={site}") if x.get("model")]
    if me["role"] in ("reviewer", "admin") and model_assets:
        target = st.selectbox("Visual model target", [x["id"] for x in model_assets])
        target_asset = next(x for x in model_assets if x["id"] == target)
        if st.button("Start visual model comparison"):
            action("POST", "/models/train", {"site_id": site, "family": target_asset["family"], "model": target_asset["model"]}, "Training job started")
    models = request("GET", f"/models?site_id={site}")
    for model in models:
        with st.expander(f"{model['id']} · {model['status']} · {model['task']}"):
            st.caption("Research measurements for this saved model. A candidate requires approval before use.")
            table(detail_rows(json.loads(model["metrics_json"])))
            if model["status"] == "candidate" and me["role"] in ("reviewer", "admin"):
                if st.button("Approve model", key=f"approve-{model['id']}"):
                    action("POST", f"/models/{model['id']}/approve", {}, "Model approved")
    st.subheader("Curriculum experiments")
    if me["role"] in ("reviewer", "admin"):
        if st.button("Run reproducible comparison suite"):
            action("POST", "/experiments/run", {"site_id": site}, "Experiment job started")
    experiments = request("GET", f"/experiments?site_id={site}")
    table(experiments, ["id", "task", "status", "dataset", "started_at", "ended_at"])
    for experiment in experiments:
        with st.expander(f"{experiment['task']} · {experiment['status']} · {experiment['id'][:8]}"):
            st.caption("Curriculum experiment results; the dataset and task are recorded above.")
            table(detail_rows(json.loads(experiment["results_json"])))

elif page == "Federation":
    heading("Federated training", "See when each client trained, when weights were combined, and how the final model performed.")
    st.info("Federated learning combines trained model updates. The completed automotive research run used two clients on this laptop; it did not train across separate factories. The iPhone demonstration needs more distinct, reviewed phones before its own federation round.")
    st.subheader("Scheduled phone model check · every six hours")
    attempts = sorted((DATA / "scheduled_federation").glob("attempt-*.json"), reverse=True)
    if attempts:
        latest = json.loads(attempts[0].read_text(encoding="utf-8"))
        st.caption(f"Last attempt: {latest['started_at_utc'][:19].replace('T', ' ')} UTC · combined model files stay on this laptop. Windows runs the task when the account is signed in and the laptop is available.")
        for subject in latest["assets"]:
            with st.container(border=True):
                st.write(f"**{subject['model']} · {'Finished' if subject['status'] == 'completed_local_research_model' else 'Waiting for reviewed phones' if subject['status'].startswith('skipped') else 'Check failed'}**")
                a, b, c = st.columns(3)
                a.metric("Distinct acceptable phones", subject.get("distinct_acceptable_phones", 0))
                b.metric("Distinct damaged phones", subject.get("distinct_damaged_phones", 0))
                c.metric("Reviewed usable photos", subject.get("usable_labeled_photos", 0))
                if subject["status"] == "completed_local_research_model":
                    for round_step in subject["rounds"]:
                        st.write(f"Round {round_step['round']}: started {round_step['started_at_utc'][11:19]} UTC → finished {round_step['finished_at_utc'][11:19]} UTC; both local clients sent model weights.")
                    outcome = subject["held_out"]
                    st.write(f"**Held-out reviewer agreement:** {outcome['agreement_with_reviewed_labels']:.0%} across {outcome['photos']} photos · damaged photos flagged {outcome['damaged_flagged']} · damaged photos missed {outcome['damaged_missed']} · acceptable photos flagged {outcome['acceptable_flagged']}.")
                    st.caption(f"Combined full model saved at {subject['model_path']}. This is agreement with technician labels, not proven diagnostic accuracy.")
                elif subject["status"].startswith("skipped"):
                    st.caption(f"At this attempt: {subject.get('reviewed_photos', 0)} usable photos had saved labels; {subject.get('usable_labeled_photos', 0)} were eligible after sample checks; {subject.get('conflicting_phone_ids', 0)} physical phone ID(s) had opposing labels. Training needs at least 24 eligible photos and six different phones in each condition. Current counts may have changed since this attempt.")
                else:
                    st.error(subject.get("error", "Scheduled check failed"))
    else:
        st.caption("No scheduled phone-model attempt has run yet.")
    automotive_fed = autovi_fed_metrics()
    if automotive_fed:
        st.subheader("Completed run · automotive pipe clips")
        st.caption(f"Renault AutoVI · {automotive_fed['source_images']} unique source photos · {automotive_fed['excluded_exact_duplicate_images']} exact duplicates excluded · three Flower FedAvg rounds")
        started = automotive_fed["started_at_utc"][:19].replace("T", " ")
        finished = automotive_fed["finished_at_utc"][:19].replace("T", " ")
        st.success(f"Started {started} UTC → finished {finished} UTC. Both clients submitted trained weights in all three rounds; the combined model was saved.")
        a, b, c, d = st.columns(4)
        a.metric("Client 1 train photos", automotive_fed["client_counts"][0]["images"])
        b.metric("Client 2 train photos", automotive_fed["client_counts"][1]["images"])
        check = automotive_fed["test"]
        c.metric("Defective clips caught", f"{check['confusion']['tp']} / {check['defective']}")
        d.metric("Good clips flagged", f"{check['confusion']['fp']} / {check['normal']}")
        st.write("**Training timeline**")
        table([{"Round": item["round"], "Started (UTC)": item["started_at_utc"][11:19],
                "Finished (UTC)": item["finished_at_utc"][11:19],
                "Client 1": f"Finished {item['client_examples'][0]} photos",
                "Client 2": f"Finished {item['client_examples'][1]} photos",
                "Combined weights": "Saved"} for item in automotive_fed["history"]])
        st.warning("The model missed 2 of 21 staged defects and flagged 30 of 60 acceptable clips. Its supervised split differs from the normal-only automotive benchmark. Both clients shared this laptop; no cross-factory privacy or production readiness is established.")
        with st.expander("Technical training numbers"):
            table([{"Round": item["round"], "Client 1 loss": round(item["client_losses"][0], 3),
                    "Client 2 loss": round(item["client_losses"][1], 3),
                    "Validation AUC": round(item["validation_auc"], 3)} for item in automotive_fed["history"]])
    casting = casting_metrics()
    if casting:
        st.subheader("Completed run · cast pump impellers")
        st.caption(f"Real unaugmented part images · {casting['original_images']} total · {casting['rounds']} Flower FedAvg rounds · research only")
        started = casting["started_at_utc"][:19].replace("T", " ")
        finished = casting["finished_at_utc"][:19].replace("T", " ")
        st.success(f"Started {started} UTC → finished {finished} UTC. Both clients returned trained weights in every round.")
        a, b, c, d = st.columns(4)
        a.metric("Client 1 train photos", casting["client_counts"][0]["images"])
        b.metric("Client 2 train photos", casting["client_counts"][1]["images"])
        dtest = casting["test"]
        c.metric("Bad parts caught", f"{dtest['confusion']['tp']} / {dtest['defective']}")
        d.metric("Good parts flagged", f"{dtest['confusion']['fp']} / {dtest['normal']}")
        st.write("**Training timeline**")
        table([{"Round": item["round"], "Started (UTC)": item["started_at_utc"][11:19],
                "Finished (UTC)": item["finished_at_utc"][11:19],
                "Client 1": "Finished 455 photos", "Client 2": "Finished 455 photos",
                "Combined weights": "Saved"} for item in casting["history"]])
        with st.expander("Technical training numbers"):
            table([{"Round": item["round"], "Client 1 loss": round(item["client_losses"][0], 3),
                    "Client 2 loss": round(item["client_losses"][1], 3),
                    "Validation AUC": round(item["validation_auc"], 3)} for item in casting["history"]])
        st.write("**Held-out check by client partition**")
        table([{"Client": item["id"], "Test photos": item["images"],
                "Defects found": f"{item['confusion']['tp']}/{item['defective']}",
                "False alerts": f"{item['confusion']['fp']}/{item['normal']}"}
               for item in casting["client_holdouts"]])
        st.warning("The model still missed 21 of 117 defective parts and falsely flagged 23 of 78 acceptable parts. These weights are not approved for a production line or either phone.")
    st.subheader("iPhone 14 Plus live demonstration")
    st.write("**Collect photos → review each physical phone → train locally at two sites → combine weights → test on held-out phones → approve only if it passes.**")
    iphone_assets = [item for item in request("GET", f"/assets?site_id={site}")
                     if item.get("model") == "iPhone 14 Plus"]
    if iphone_assets:
        readiness = request("GET", f"/federation/readiness?site_id={site}&asset_id={iphone_assets[0]['id']}")
        st.write("**Current iPhone data on this laptop**")
        a, b, c = st.columns(3)
        a.metric("Photos captured", readiness["captures"])
        b.metric("Passed image-quality check", readiness["usable_captures"])
        c.metric("Clear photos with saved technician labels", readiness["reviewed_usable_photos"])
        d, e, f = st.columns(3)
        d.metric("Eligible training photos", f"{readiness['eligible_training_photos']} / {readiness['minimum_reviewed_photos']}")
        e.metric("Different acceptable phones", f"{readiness['reviewed_acceptable_samples']} / {readiness['minimum_distinct_phones_per_class']}")
        f.metric("Different damaged phones", f"{readiness['reviewed_defective_samples']} / {readiness['minimum_distinct_phones_per_class']}")
        if not readiness["ready_for_two_clients"]:
            st.info("The six-hour phone training task will keep recording a skipped attempt until all three targets above are met. Photos of the same iPhone add views, but they do not add independent phones for training or held-out testing.")
        if readiness["conflicting_samples"]:
            st.warning(f"{readiness['conflicting_samples']} physical phone ID has both 'no visible issue' and 'visible issue' photo labels. The training selector excludes all photos under that ID because it cannot assign the whole phone one condition. This can happen when different views show different areas. Keep truthful reviews; collect separately identified phones with clear overall condition labels for a phone-specific model.")
    st.subheader("Create an independent-client round")
    st.caption("When two site computers have enough reviewed images of the exact same model, each trains locally and submits weights. Raw images stay at each site. The current laptop demonstration does not claim that cross-site step has happened.")
    fed_assets = [x for x in request("GET", f"/assets?site_id={site}") if x.get("model")]
    if fed_assets and me["role"] in ("reviewer", "admin"):
        selected_asset = st.selectbox("Client round model target", [x["id"] for x in fed_assets])
        selected = next(x for x in fed_assets if x["id"] == selected_asset)
        selected_readiness = request("GET", f"/federation/readiness?site_id={site}&asset_id={selected_asset}")
        if st.button("Create real client round", disabled=not selected_readiness["ready_for_two_clients"]):
            created = action("POST", "/federation/rounds", {"site_id": site, "family": selected["family"], "model": selected["model"]}, "Round ready for client updates")
            if created:
                st.code(created["id"])
        if not selected_readiness["ready_for_two_clients"]:
            st.caption("Round creation is paused until this asset has enough distinct, reviewed samples in both classes.")
    with st.expander("Curriculum-only synthetic exercise"):
        st.caption("This exercises aggregation code using fictional feature rows. It is separate from the completed cast-part training above.")
        if me["role"] in ("reviewer", "admin") and st.button("Run synthetic FedAvg mechanics exercise"):
            action("POST", "/federation/run", {"site_id": site}, "Synthetic exercise started")
    rounds = request("GET", f"/federation?site_id={site}")
    st.subheader("Application round records")
    for item in rounds:
        display_status = {"collecting": "Waiting for client updates", "awaiting_validation": "Checking combined model",
                          "candidate_review": "Awaiting reviewer decision", "complete_rejected": "Did not pass evaluation",
                          "complete": "Finished", "running": "Running"}.get(item["status"], item["status"])
        target = item.get("equipment_model") or "Synthetic curriculum exercise"
        with st.expander(f"{target} · {display_status} · {item['created_at'][:16].replace('T', ' ')} UTC"):
            st.write(f"**Started:** {item['created_at'][:19].replace('T', ' ')} UTC")
            st.write(f"**Target:** {item.get('family') or 'Fictional feature rows'} / {target}")
            report = json.loads(item["results_json"])
            st.write(f"**Data:** {report.get('dataset', 'Waiting for training details')}")
            if report.get("clients"):
                table([{"Client": name, "Train photos/rows": result.get("train_count"),
                        "Held-out rows": result.get("test_count"),
                        "Local recall": round(result.get("local_recall", 0), 3),
                        "Combined recall": round(result.get("federated_recall", 0), 3)}
                       for name, result in report["clients"].items()])
            if report.get("gate"):
                gate = report["gate"]
                st.caption(f"Approval gate: recall at least {gate['recall_min']:.0%}; false alerts at most {gate['false_alert_rate_max']:.0%}; latency at most {gate['latency_ms_max']} ms.")
            if item["status"] in ("collecting", "awaiting_validation", "candidate_review", "complete_rejected"):
                detail = request("GET", f"/federation/rounds/{item['id']}")
                st.write(f"**Client updates received:** {len(detail['updates'])} of 2")
                table([{"Client": update["client_id"], "Training examples": update["sample_count"],
                        "Submitted (UTC)": update["submitted_at"][:19].replace("T", " ")}
                       for update in detail["updates"]], empty_title="No client updates yet")
                if detail["validations"]:
                    st.write("**Held-out checks**")
                    table([{"Client": check["client_id"], "Recall": f"{check['recall']:.0%}",
                            "False alert rate": f"{check['false_alert_rate']:.0%}",
                            "Test examples": check["sample_count"]} for check in detail["validations"]])
                if item["status"] == "collecting" and len(detail["updates"]) >= 2 and me["role"] in ("reviewer", "admin"):
                    if st.button("Aggregate submitted weights", key=f"aggregate-{item['id']}"):
                        action("POST", f"/federation/rounds/{item['id']}/aggregate", {}, "Candidate weights ready for client validation")

elif page == "Reports":
    heading("Reports and measured history", "Exports use recorded site data. Forecasts need regular measured history and an evaluation gate.")
    cases = request("GET", f"/cases?site_id={site}")
    captures = request("GET", f"/captures?site_id={site}&limit=200")
    experiments = request("GET", f"/experiments?site_id={site}")
    rounds = request("GET", f"/federation?site_id={site}")
    closed = [x for x in cases if x["status"] == "closed"]
    physical = [x for x in closed if x.get("inspection_performed")]
    a, b, c, d = st.columns(4)
    a.metric("Closed review cases", len(closed))
    b.metric("Hands-on inspections", len(physical))
    c.metric("Actual inspection minutes", sum(x.get("actual_minutes") or 0 for x in physical))
    d.metric("Reviewed captures", len([x for x in captures if x.get("grade")]))
    if closed:
        st.subheader("Recorded outcomes")
        table(closed, ["id", "asset_id", "inspection_performed", "actual_finding", "actual_minutes", "closure_note"])
    if physical:
        duration = pd.DataFrame(physical)[["asset_id", "actual_minutes", "updated_at", "actual_finding"]]
        st.subheader("Actual inspection duration")
        st.bar_chart(duration.set_index("updated_at")["actual_minutes"])
    else:
        st.info("No measured inspection outcomes yet. Duration regression is research-only.")
    if captures:
        daily = pd.DataFrame(captures)
        daily["day"] = pd.to_datetime(daily["captured_at"]).dt.date
        st.subheader("Captured evidence by day and quality")
        st.bar_chart(daily.groupby(["day", "quality"]).size().unstack(fill_value=0))
    st.info("Operational sensor forecasts are unavailable until regular measured condition history is collected and a walk-forward baseline gate passes.")
    for name, data in (("cases", cases), ("captures", captures), ("experiments", experiments), ("federation", rounds)):
        if data:
            st.download_button(f"Export {name} CSV", pd.DataFrame(data).to_csv(index=False), f"forgecare-{site}-{name}.csv", "text/csv")

elif page == "Administration":
    heading("Site administration", "Manage local roles and site access. Each non-admin user sees only their assigned site.")
    if me["role"] != "admin":
        st.error("Administrator access required.")
    else:
        st.subheader("Users at this site")
        table(request("GET", f"/users?site_id={site}"), ["id", "username", "role", "site_id", "active"])
        with st.form("new-user"):
            username = st.text_input("New username")
            password = st.text_input("Temporary password (10+ characters)", type="password")
            role = st.selectbox("Role", ["technician", "supervisor", "reviewer", "admin"])
            if st.form_submit_button("Create user"):
                action("POST", "/users", {"username": username, "password": password, "role": role, "site_id": site}, "User created")
        with st.form("new-site"):
            site_id = st.text_input("New site ID")
            name = st.text_input("New site name")
            if st.form_submit_button("Create site"):
                action("POST", "/sites", {"id": site_id, "name": name}, "Site created")
    st.subheader("Change my password")
    with st.form("change-password"):
        current = st.text_input("Current password", type="password")
        replacement = st.text_input("New password (10+ characters)", type="password")
        if st.form_submit_button("Change password"):
            result = action("POST", "/auth/change-password", {"current_password": current,
                    "new_password": replacement}, "Password changed. Sign in again.")
            if result:
                st.session_state.token = None
                st.rerun()

elif page == "Audit":
    heading("Business activity and traceability", "A readable record of camera evidence, technician reviews, work orders, and approvals.")
    events = request("GET", f"/audit?site_id={site}")
    users = {item["id"]: item["username"] for item in request("GET", f"/users?site_id={site}")}
    readable = business_events(events, users)
    a, b, c = st.columns(3)
    a.metric("Recorded activities", len(readable))
    b.metric("Photo reviews", sum(item["Activity"] == "Technician review saved" for item in readable))
    c.metric("Work orders opened", sum(item["Activity"] in ("Work order opened", "Work order opened by model") for item in readable))
    table(readable, empty_title="No activity recorded", empty_message="Camera captures and review decisions will appear here with time and person responsible.")
    if readable:
        st.download_button("Export business activity CSV", pd.DataFrame(readable).to_csv(index=False), "forgecare-business-activity.csv", "text/csv")
    if me["role"] == "admin":
        with st.expander("Camera submissions rejected before intake"):
            rejected = business_events(request("GET", "/audit/global"), users)
            table(rejected, empty_title="No rejected camera submissions")
