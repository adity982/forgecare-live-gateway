"""Keep the fixed Render gateway connected to this laptop without changing DNS.

The private signing key stays in the laptop data directory. The public verifier
is deployed with the gateway. The bridge forwards only to ForgeCare's loopback
dashboard and camera relay, and reconnects after transient network failures.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import serialization
from websockets.asyncio.client import connect


LOG = logging.getLogger("forgecare.bridge")
MAX_MESSAGE = 32 * 1024 * 1024
GATEWAY = "wss://forgecare-ai-live.onrender.com/bridge"
HOP_HEADERS = {"host", "connection", "content-length", "transfer-encoding", "accept-encoding",
               "upgrade", "keep-alive", "proxy-connection"}


def destination(url: str) -> tuple[str, str]:
    if not url.startswith("/") or url.startswith("//") or "\x00" in url:
        raise ValueError("Only relative ForgeCare paths are accepted")
    if url == "/camera-relay":
        return "http://127.0.0.1:8502", "/"
    if url.startswith("/camera-relay/"):
        return "http://127.0.0.1:8502", url[len("/camera-relay"):]
    return "http://127.0.0.1:8501", url


async def connected(gateway: str, private_key: Path) -> None:
    key = serialization.load_pem_private_key(private_key.read_bytes(), password=None)
    async with connect(gateway, max_size=MAX_MESSAGE, ping_interval=20, ping_timeout=20) as control:
        challenge = json.loads(await asyncio.wait_for(control.recv(), 12))
        if challenge.get("type") != "challenge":
            raise RuntimeError("Gateway did not provide an authentication challenge")
        signed = b"forgecare-bridge-v1:" + base64.b64decode(challenge["data"], validate=True)
        signature = base64.b64encode(key.sign(signed)).decode("ascii")
        await control.send(json.dumps({"type": "auth", "signature": signature}))
        result = json.loads(await asyncio.wait_for(control.recv(), 12))
        if result.get("type") != "auth_ok":
            raise RuntimeError("Gateway rejected the laptop signature")
        LOG.info("Authenticated fixed gateway connection established")

        send_lock = asyncio.Lock()
        upstream_sockets: dict[str, object] = {}
        tasks: set[asyncio.Task] = set()

        async def send(packet: dict) -> None:
            async with send_lock:
                await control.send(json.dumps(packet, separators=(",", ":")))

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(20)
                await send({"type": "heartbeat"})

        async def serve_http(packet: dict, client: httpx.AsyncClient) -> None:
            request_id = packet.get("id")
            try:
                base, path = destination(str(packet["url"]))
                headers = {name: value for name, value in packet.get("headers", {}).items()
                           if name.lower() not in HOP_HEADERS and isinstance(value, str)}
                headers["accept-encoding"] = "identity"
                body = base64.b64decode(packet.get("body", ""), validate=True)
                response = await client.request(packet["method"], base + path, headers=headers,
                                                content=body, follow_redirects=False)
                await send({"type": "http_response", "id": request_id, "status": response.status_code,
                            "headers": list(response.headers.multi_items()),
                            "body": base64.b64encode(response.content).decode("ascii")})
            except Exception as exc:
                LOG.warning("Local HTTP request failed: %s", type(exc).__name__)
                await send({"type": "http_response", "id": request_id, "status": 502,
                            "headers": [["content-type", "text/plain; charset=utf-8"]],
                            "body": base64.b64encode(b"ForgeCare laptop service temporarily unavailable").decode("ascii")})

        async def serve_socket(packet: dict) -> None:
            socket_id = packet.get("id")
            try:
                base, path = destination(str(packet["url"]))
                endpoint = base.replace("http://", "ws://", 1) + path
                incoming = packet.get("headers", {})
                headers = {name: incoming[name] for name in ("cookie", "authorization", "x-xsrftoken")
                           if isinstance(incoming.get(name), str)}
                protocols = [part.strip() for part in incoming.get("sec-websocket-protocol", "").split(",")
                             if part.strip()]
                async with connect(endpoint, origin=base, additional_headers=headers,
                                   subprotocols=protocols or None, proxy=None,
                                   max_size=MAX_MESSAGE, ping_interval=20, ping_timeout=20) as upstream:
                    upstream_sockets[socket_id] = upstream
                    await send({"type": "ws_ready", "id": socket_id})
                    async for message in upstream:
                        binary = isinstance(message, bytes)
                        await send({"type": "ws_data", "id": socket_id, "binary": binary,
                                    "data": base64.b64encode(message).decode("ascii") if binary else message})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("Local WebSocket failed: %s", type(exc).__name__)
                await send({"type": "ws_error", "id": socket_id})
            finally:
                upstream_sockets.pop(socket_id, None)
                await send({"type": "ws_close", "id": socket_id})

        def track(task: asyncio.Task) -> None:
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            track(asyncio.create_task(heartbeat()))
            try:
                async for raw in control:
                    packet = json.loads(raw)
                    kind = packet.get("type")
                    if kind == "http_request":
                        track(asyncio.create_task(serve_http(packet, client)))
                    elif kind == "ws_open":
                        track(asyncio.create_task(serve_socket(packet)))
                    elif kind == "ws_data":
                        upstream = upstream_sockets.get(packet.get("id"))
                        if upstream:
                            data = (base64.b64decode(packet.get("data", ""), validate=True)
                                    if packet.get("binary") else packet.get("data", ""))
                            await upstream.send(data)
                    elif kind == "ws_close":
                        upstream = upstream_sockets.get(packet.get("id"))
                        if upstream:
                            await upstream.close()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


async def main(gateway: str, private_key: Path) -> None:
    pause = 1
    while True:
        try:
            await connected(gateway, private_key)
            pause = 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Gateway disconnected: %s: %s; retrying in %s seconds",
                        type(exc).__name__, str(exc)[:160], pause)
        await asyncio.sleep(pause)
        pause = min(pause * 2, 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=GATEWAY)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(args.gateway, args.key))
