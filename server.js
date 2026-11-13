"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const crypto = require("node:crypto");
const { WebSocket, WebSocketServer } = require("ws");

const PORT = Number(process.env.PORT || 10000);
const SOURCE_URL = "https://github.com/adity982/forgecare-live-gateway";
const ABOUT_PATH = "/about";
const BRIDGE_PUBLIC_KEY = crypto.createPublicKey(fs.readFileSync(path.join(__dirname, "bridge-public.pem")));
const MAX_BRIDGE_MESSAGE = 32 * 1024 * 1024;
let bridge = null;
let lastBridgeHeartbeat = 0;
const pendingHttp = new Map();
const browserSockets = new Map();
const bridgeWss = new WebSocketServer({ noServer: true, maxPayload: MAX_BRIDGE_MESSAGE });
const browserWss = new WebSocketServer({ noServer: true, maxPayload: MAX_BRIDGE_MESSAGE });

function bridgeReady() {
  return bridge && bridge.readyState === WebSocket.OPEN;
}

function sendBridge(packet) {
  if (!bridgeReady()) return false;
  bridge.send(JSON.stringify(packet));
  return true;
}

function secureHtml(res, status, body, extraHeaders = {}) {
  if (res.headersSent) return res.destroy();
  res.writeHead(status, {
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    ...extraHeaders,
  });
  res.end(body);
}

function unavailable(res) {
  secureHtml(res, 503, `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForgeCare is reconnecting</title>
<style>
  :root { color-scheme: light dark; font: 17px/1.55 system-ui, sans-serif; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #101513; color: #eef7f1; }
  main { width: min(42rem, calc(100% - 3rem)); }
  p { color: #b9c8bf; }
  a { color: #8ee6b2; }
  code { color: #d9ffe7; }
</style>
<main>
  <p><strong>Gateway online · laptop bridge reconnecting</strong></p>
  <h1>ForgeCare is temporarily offline</h1>
  <p>The public gateway is healthy, but the authenticated laptop that runs the private demo data and models is not connected. Retry in about 30 seconds.</p>
  <p>No inspection records, photos, model weights, passwords, or API keys are stored on this gateway.</p>
  <p><a href="${ABOUT_PATH}">Take the always-available project tour</a>, <a href="${SOURCE_URL}">explore the public source</a>, or check <a href="/gateway-health"><code>/gateway-health</code></a>.</p>
</main>`, { "Retry-After": "30" });
}

function about(res) {
  const connected = Boolean(bridgeReady());
  const statusLabel = connected ? "Laptop demo connected" : "Laptop demo reconnecting";
  const statusClass = connected ? "online" : "offline";
  secureHtml(res, 200, `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A source-linked tour of ForgeCare's authenticated, fail-closed AI inspection demo gateway.">
<title>ForgeCare project tour</title>
<style>
  :root { color-scheme: light dark; font: 17px/1.55 system-ui, sans-serif; }
  body { margin: 0; background: #101513; color: #eef7f1; }
  main { width: min(54rem, calc(100% - 3rem)); margin: 0 auto; padding: 4rem 0; }
  p, li { color: #b9c8bf; }
  a { color: #8ee6b2; }
  code { color: #d9ffe7; }
  .eyebrow { color: #8ee6b2; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
  .status { display: inline-block; padding: .35rem .7rem; border: 1px solid #52665a; border-radius: 999px; }
  .online { color: #8ee6b2; }
  .offline { color: #ffd38e; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; margin: 2rem 0; }
  section { border: 1px solid #2d3b33; padding: 1rem 1.2rem; background: #151c18; }
  h1 { font-size: clamp(2.2rem, 6vw, 4.5rem); line-height: 1; max-width: 12ch; }
  h2 { font-size: 1.15rem; }
</style>
<main>
  <p class="eyebrow">ForgeCare AI · public gateway</p>
  <h1>Inspect the architecture before the demo.</h1>
  <p class="status ${statusClass}">${statusLabel}</p>
  <p>This gateway gives a laptop-hosted inspection prototype one stable public address. The operational database, photos, credentials, and model artifacts remain on the laptop.</p>
  <p><a href="/">Open the live Streamlit demo</a> · <a href="/gateway-health">Read gateway status</a> · <a href="${SOURCE_URL}">Browse the source</a></p>
  <div class="grid">
    <section><h2>1. Browser</h2><p>HTTP and Streamlit WebSocket traffic arrive over the public Render URL.</p></section>
    <section><h2>2. Gateway</h2><p>A dependency-light Node service forwards requests only while an Ed25519-authenticated bridge is connected.</p></section>
    <section><h2>3. Laptop</h2><p>The outbound bridge reaches loopback services. No inbound port or rotating tunnel URL is required.</p></section>
  </div>
  <h2>What to verify</h2>
  <ul>
    <li><code>/gateway-health</code> reports gateway and bridge state without host paths or credentials.</li>
    <li>Ordinary requests fail closed with HTTP 503 while the authenticated bridge is offline.</li>
    <li>The public repository includes the gateway tests and a source snapshot of the Python workflow.</li>
  </ul>
  <h2>Evidence boundary</h2>
  <p>The public Node CI validates gateway behavior on Node 20 and 22. It does not validate the Python vision models or establish production inspection accuracy. Research metrics and dataset restrictions are documented in the repository.</p>
</main>`);
}

function closeBridgeClients() {
  for (const [id, item] of pendingHttp) {
    clearTimeout(item.timer);
    unavailable(item.res);
    pendingHttp.delete(id);
  }
  for (const [id, item] of browserSockets) {
    item.ws.close(1012, "Laptop bridge reconnecting");
    browserSockets.delete(id);
  }
}

function handleBridgePacket(raw) {
  let packet;
  try { packet = JSON.parse(raw.toString()); } catch { return; }
  lastBridgeHeartbeat = Date.now();
  if (packet.type === "heartbeat") return;
  if (packet.type === "http_response") {
    const item = pendingHttp.get(packet.id);
    if (!item) return;
    pendingHttp.delete(packet.id);
    clearTimeout(item.timer);
    try {
      const headers = {};
      for (const [key, value] of packet.headers || []) {
        const lower = key.toLowerCase();
        if (["content-length", "content-encoding", "connection", "transfer-encoding", "keep-alive"].includes(lower)) continue;
        if (lower === "set-cookie") {
          if (!headers[key]) headers[key] = [];
          headers[key].push(value);
        } else headers[key] = value;
      }
      const body = Buffer.from(packet.body || "", "base64");
      item.res.writeHead(Number(packet.status) || 502, headers);
      item.res.end(body);
    } catch { unavailable(item.res); }
    return;
  }
  const item = browserSockets.get(packet.id);
  if (!item) return;
  if (packet.type === "ws_ready") {
    item.ready = true;
    for (const message of item.buffer) sendBridge(message);
    item.buffer.length = 0;
  } else if (packet.type === "ws_data" && item.ws.readyState === WebSocket.OPEN) {
    item.ws.send(packet.binary ? Buffer.from(packet.data || "", "base64") : packet.data || "",
                 { binary: Boolean(packet.binary) });
  } else if (packet.type === "ws_close" || packet.type === "ws_error") {
    item.ws.close(packet.type === "ws_error" ? 1011 : 1000);
    browserSockets.delete(packet.id);
  }
}

function acceptBridge(ws) {
  const challenge = crypto.randomBytes(32);
  ws.send(JSON.stringify({ type: "challenge", data: challenge.toString("base64") }));
  const deadline = setTimeout(() => ws.close(4001, "Authentication timeout"), 10000);
  ws.once("message", (raw) => {
    let packet;
    try { packet = JSON.parse(raw.toString()); } catch { ws.close(4003); return; }
    const signature = Buffer.from(packet.signature || "", "base64");
    const signed = Buffer.concat([Buffer.from("forgecare-bridge-v1:"), challenge]);
    if (packet.type !== "auth" || !crypto.verify(null, signed, BRIDGE_PUBLIC_KEY, signature)) {
      ws.close(4003, "Invalid bridge signature");
      return;
    }
    clearTimeout(deadline);
    if (bridgeReady()) bridge.close(4001, "Replaced by new laptop connection");
    bridge = ws;
    lastBridgeHeartbeat = Date.now();
    ws.send(JSON.stringify({ type: "auth_ok" }));
    ws.on("message", handleBridgePacket);
    ws.on("close", () => { if (bridge === ws) { bridge = null; closeBridgeClients(); } });
    ws.on("error", (error) => console.error("Laptop bridge error:", error.message));
    console.log("Authenticated laptop bridge connected");
  });
}

function bridgeHttp(req, res) {
  if (pendingHttp.size >= 64) return unavailable(res);
  const chunks = [];
  let size = 0;
  req.on("data", (chunk) => {
    size += chunk.length;
    if (size > 16 * 1024 * 1024) req.destroy();
    else chunks.push(chunk);
  });
  req.on("end", () => {
    const id = crypto.randomUUID();
    const timer = setTimeout(() => {
      pendingHttp.delete(id);
      unavailable(res);
    }, 90000);
    pendingHttp.set(id, { res, timer });
    const sent = sendBridge({ type: "http_request", id, method: req.method, url: req.url,
                              headers: req.headers, body: Buffer.concat(chunks).toString("base64") });
    if (!sent) { clearTimeout(timer); pendingHttp.delete(id); unavailable(res); }
  });
}

function bridgeBrowser(ws, req) {
  const id = crypto.randomUUID();
  const item = { ws, ready: false, buffer: [] };
  browserSockets.set(id, item);
  if (!sendBridge({ type: "ws_open", id, url: req.url, headers: req.headers })) {
    ws.close(1012, "Laptop bridge unavailable");
    browserSockets.delete(id);
    return;
  }
  ws.on("message", (data, binary) => {
    const packet = { type: "ws_data", id, binary, data: binary ? data.toString("base64") : data.toString() };
    if (item.ready) sendBridge(packet);
    else if (item.buffer.length < 100) item.buffer.push(packet);
    else ws.close(1009, "Bridge buffer full");
  });
  ws.on("close", () => { browserSockets.delete(id); sendBridge({ type: "ws_close", id }); });
  ws.on("error", () => { browserSockets.delete(id); sendBridge({ type: "ws_close", id }); });
}

setInterval(() => {
  if (bridgeReady() && Date.now() - lastBridgeHeartbeat > 65000) {
    console.error("Laptop bridge heartbeat expired; forcing reconnect");
    bridge.terminate();
  }
}, 15000).unref();

const server = http.createServer((req, res) => {
  if (req.url === ABOUT_PATH) {
    about(res);
    return;
  }
  if (req.url === "/gateway-health") {
    res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    res.end(JSON.stringify({ gateway: "ok", laptopBridge: Boolean(bridgeReady()),
                             activeRequests: pendingHttp.size, browserSockets: browserSockets.size,
                             bridgeLastSeen: lastBridgeHeartbeat || null }));
    return;
  }
  if (!bridgeReady()) return unavailable(res);
  bridgeHttp(req, res);
});

server.on("upgrade", (req, socket, head) => {
  try {
    if (req.url === "/bridge") {
      bridgeWss.handleUpgrade(req, socket, head, (ws) => acceptBridge(ws));
      return;
    }
    if (!bridgeReady()) return socket.destroy();
    browserWss.handleUpgrade(req, socket, head, (ws) => bridgeBrowser(ws, req));
  } catch (error) {
    console.error("WebSocket upgrade failed:", error.message);
    socket.destroy();
  }
});

server.listen(PORT, "0.0.0.0", () => console.log(`ForgeCare gateway listening on ${PORT}`));
