# ForgeCare live gateway

This small gateway gives the laptop-hosted ForgeCare demonstration a stable public address. It forwards dashboard traffic to the current Cloudflare quick tunnel and `/camera-relay/*` traffic to the paired camera relay tunnel. It stores no inspection records, photos, API keys, or passwords. The ForgeCare services and their data remain on the host laptop.

The app is available only while that laptop and its tunnels are running. Render's free service may take about a minute to wake after inactivity. `targets.json` is refreshed from this GitHub repository about every 20 seconds; update it whenever the quick tunnels are recreated.
