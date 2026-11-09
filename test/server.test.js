"use strict";

const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const net = require("node:net");
const path = require("node:path");
const test = require("node:test");

const repositoryRoot = path.resolve(__dirname, "..");

async function unusedPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close((error) => error ? reject(error) : resolve(address.port));
    });
  });
}

async function startGateway(t) {
  const port = await unusedPort();
  const child = spawn(process.execPath, ["server.js"], {
    cwd: repositoryRoot,
    env: { ...process.env, PORT: String(port) },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const output = [];
  child.stdout.on("data", (chunk) => output.push(chunk.toString()));
  child.stderr.on("data", (chunk) => output.push(chunk.toString()));
  t.after(() => child.kill());

  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Gateway did not start:\n${output.join("")}`)), 5000);
    child.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`Gateway exited with code ${code}:\n${output.join("")}`));
    });
    child.stdout.on("data", (chunk) => {
      if (chunk.toString().includes("ForgeCare gateway listening")) {
        clearTimeout(timer);
        resolve();
      }
    });
  });

  return `http://127.0.0.1:${port}`;
}

test("health endpoint reports a disconnected bridge without leaking host data", async (t) => {
  const baseUrl = await startGateway(t);
  const response = await fetch(`${baseUrl}/gateway-health`);

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.deepEqual(await response.json(), {
    gateway: "ok",
    laptopBridge: false,
    activeRequests: 0,
    browserSockets: 0,
    bridgeLastSeen: null,
  });
});

test("ordinary requests fail closed while the authenticated bridge is offline", async (t) => {
  const baseUrl = await startGateway(t);
  const response = await fetch(`${baseUrl}/inspection-queue`);

  assert.equal(response.status, 503);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.match(await response.text(), /host laptop is disconnected/i);
});
