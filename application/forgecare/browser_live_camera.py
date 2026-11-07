"""Browser webcam component for continuous capture on a hosted page."""
from __future__ import annotations

import streamlit as st


HTML = """
<div class="camera-shell">
  <video autoplay playsinline muted></video>
  <div class="camera-actions">
    <button class="start" type="button">Start this computer's camera</button>
    <button class="stop" type="button">Stop camera</button>
    <span class="message">Camera is off.</span>
  </div>
</div>
"""

CSS = """
.camera-shell { max-width: 760px; font: 15px sans-serif; color: #292824; }
video { display: block; width: 100%; max-height: 450px; background: #171717;
        border-radius: 12px; object-fit: contain; }
.camera-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
button { padding: 9px 14px; border: 1px solid #999; border-radius: 8px;
         background: #f6f4ef; color: #292824; cursor: pointer; }
button.start { background: #252b25; color: white; }
.message { color: #5c5953; }
"""

JS = """
export default function(component) {
  const { parentElement, data, setTriggerValue } = component;
  const video = parentElement.querySelector('video');
  const start = parentElement.querySelector('.start');
  const stop = parentElement.querySelector('.stop');
  const message = parentElement.querySelector('.message');
  const key = 'forgecare-camera-' + data.camera_id;
  window.__forgecareCameras ||= {};
  let state = window.__forgecareCameras[key];
  if (!state) {
    state = { stream: null, timer: null, owner: null, cleanup: null };
    window.__forgecareCameras[key] = state;
  }
  if (state.cleanup) clearTimeout(state.cleanup);
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  state.owner = parentElement;
  const show = (text) => { message.textContent = text; };
  const schedule = () => {
    const canvas = document.createElement('canvas');
    state.timer = setInterval(() => {
      if (!state.stream || video.readyState < 2 || video.videoWidth < 2) return;
      canvas.width = Math.min(video.videoWidth, 960);
      canvas.height = Math.round(canvas.width * video.videoHeight / video.videoWidth);
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const jpeg = canvas.toDataURL('image/jpeg', 0.82).split(',')[1];
      setTriggerValue('photo', {
        id: crypto.randomUUID(), session_id: data.session_id, jpeg_b64: jpeg
      });
    }, 5000);
  };
  if (state.stream && state.stream.getTracks().some((track) => track.readyState === 'live')) {
    video.srcObject = state.stream;
    show('Live — a new photo is saved about every five seconds.');
    schedule();
  }
  const end = () => {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    video.srcObject = null;
    show('Camera is off. Saved photos remain in Capture review.');
  };
  start.onclick = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      show('Camera access requires HTTPS or localhost and a browser that supports camera access.');
      return;
    }
    try {
      end();
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
      });
      video.srcObject = state.stream;
      await video.play();
      show('Live — a new photo is saved about every five seconds.');
      schedule();
    } catch (error) {
      end();
      show('Camera unavailable: ' + (error?.message || 'check browser permission'));
    }
  };
  stop.onclick = end;
  return () => {
    // Streamlit reruns after every frame. Keep the stream through that short remount,
    // then release the device when the component actually leaves the page.
    if (state.owner === parentElement) {
      state.cleanup = setTimeout(() => {
        if (state.owner === parentElement) end();
      }, 1500);
    }
  };
}
"""


browser_live_camera = st.components.v2.component(
    "forgecare_browser_live_camera", html=HTML, css=CSS, js=JS
)
