# ForgeCare fixed public bridge

The presentation address is **https://forgecare-ai-live.onrender.com**. The Render gateway receives one outbound, authenticated WebSocket connection from the prepared laptop. Browser HTTP and Streamlit WebSocket traffic go through that connection to the laptop's loopback dashboard; `/camera-relay/*` reaches the laptop's camera relay. There is no rotating `trycloudflare.com` hostname.

The gateway verifies an Ed25519 signature before accepting the laptop connection. `bridge-public.pem` is public and deployed with this repository; the private signing key stays outside the repository in the prepared laptop's ForgeCare data folder. The gateway stores no inspection records, photos, passwords, or API keys. All four ForgeCare services and their data remain on the laptop.

On the prepared laptop, `ensure-bridge.ps1` checks the local services and starts `bridge_agent.py` if it is not running. `install-gateway-task.ps1` runs that check every minute. The agent reconnects automatically after a transient network or Render restart and sends a heartbeat every 20 seconds. `/gateway-health` reports whether the authenticated laptop bridge is connected. When it is not, the fixed address returns a clear 503 rather than forwarding to an expired origin.

The laptop must remain powered, awake, and online. Render's free service may restart and cannot guarantee uninterrupted availability; the bridge reconnects without changing the URL. For a presentation, keep the laptop plugged in and retain the local `http://127.0.0.1:8501` as a backup. This bridge does not transfer the database or models to Render.
