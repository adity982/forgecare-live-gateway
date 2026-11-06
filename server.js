"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const httpProxy = require("http-proxy");

const PORT = Number(process.env.PORT || 10000);
const CONFIG_URL = "https://raw.githubusercontent.com/adity982/forgecare-live-gateway/main/targets.json";
const REFRESH_MS = 20_000;
const initial = JSON.parse(fs.readFileSync(path.join(__dirname, "targets.json"), "utf8"));
let targets = validateTargets(initial);
let refreshedAt = 0;
let refreshing = null;

function validateTargets(value) {
  const out = {};
  for (const name of ["dashboard", "relay"]) {
    const parsed = new URL(value[name]);
    if (parsed.protocol !== "https:" || !parsed.hostname.endsWith(".trycloudflare.com")) {
      throw new Error(`Invalid ${name} tunnel target`);
    }
    out[name] = parsed.origin;
  }
  return out;
}

async function refreshTargets() {
  if (Date.now() - refreshedAt < REFRESH_MS) return;
  if (refreshing) return refreshing;
  refreshing = (async () => {
    try {
      const response = await fetch(`${CONFIG_URL}?t=${Date.now()}`, {
        headers: { "Cache-Control": "no-cache" },
        signal: AbortSignal.timeout(5000)
      });
      if (!response.ok) throw new Error(`Config HTTP ${response.status}`);
      targets = validateTargets(await response.json());
      refreshedAt = Date.now();
    } catch (error) {
      console.error("Could not refresh tunnel addresses:", error.message);
      refreshedAt = Date.now();
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

function route(req) {
  if (req.url === "/camera-relay") req.url = "/";
  else if (req.url.startsWith("/camera-relay/")) req.url = req.url.slice("/camera-relay".length);
  else return targets.dashboard;
  return targets.relay;
}

const proxy = httpProxy.createProxyServer({ ws: true, changeOrigin: true, secure: true,
  timeout: 30_000, proxyTimeout: 30_000 });

proxy.on("proxyReq", (proxyReq, req) => {
  if (req.headers.origin) proxyReq.setHeader("Origin", req.gatewayTarget);
});
proxy.on("proxyReqWs", (proxyReq, req) => {
  if (req.headers.origin) proxyReq.setHeader("Origin", req.gatewayTarget);
});
proxy.on("error", (error, req, res) => {
  console.error("Tunnel target error:", error.message);
  if (res && typeof res.writeHead === "function" && !res.headersSent) {
    res.writeHead(503, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" });
    res.end("ForgeCare is temporarily unavailable. Keep the host laptop and its public tunnel running, then retry.");
  } else if (res && typeof res.destroy === "function") {
    res.destroy();
  }
});

const server = http.createServer(async (req, res) => {
  if (req.url === "/gateway-health") {
    res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    res.end(JSON.stringify({ gateway: "ok", configured: Boolean(targets.dashboard), lastRefresh: refreshedAt }));
    return;
  }
  await refreshTargets();
  req.gatewayTarget = route(req);
  proxy.web(req, res, { target: req.gatewayTarget });
});

server.on("upgrade", async (req, socket, head) => {
  try {
    await refreshTargets();
    req.gatewayTarget = route(req);
    proxy.ws(req, socket, head, { target: req.gatewayTarget });
  } catch (error) {
    console.error("WebSocket upgrade failed:", error.message);
    socket.destroy();
  }
});

server.listen(PORT, "0.0.0.0", () => console.log(`ForgeCare gateway listening on ${PORT}`));
