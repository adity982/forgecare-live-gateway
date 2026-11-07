"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const crypto = require("node:crypto");
const { WebSocket, WebSocketServer } = require("ws");

const PORT = Number(process.env.PORT || 10000);
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

function unavailable(res) {
  if (res.headersSent) return res.destroy();
  res.writeHead(503, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" });
  res.end("ForgeCare's host laptop is disconnected. The fixed public address is working; the laptop bridge is reconnecting.");
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
