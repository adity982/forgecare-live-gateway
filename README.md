# ForgeCare live gateway

The stable address is **https://forgecare-ai-live.onrender.com**. This small gateway forwards dashboard traffic to the current Cloudflare quick tunnel and `/camera-relay/*` traffic to the paired camera relay tunnel. It stores no inspection records, photos, API keys, or passwords. The ForgeCare services and their data remain on the host laptop.

The app is available only while that laptop and its tunnels are running. Render's free service may take about a minute to wake after inactivity. `targets.json` is refreshed from this GitHub repository about every 20 seconds. On the prepared laptop, `sync-gateway.ps1` checks both tunnels and publishes updated addresses if either changes. `install-gateway-task.ps1` schedules that check every five minutes while Windows runs the task. The scheduled task and the six-hour phone-training task are separate.

For a new laptop, this gateway does not transfer the ForgeCare database or models. Install and run ForgeCare there first, then change the tunnel source and synchronize the target addresses.
