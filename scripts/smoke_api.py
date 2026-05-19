#!/usr/bin/env python3
"""
Dependency-free smoke tests for the backend and mobile-facing endpoints.

Run on the VM after the backend is already running:

    python scripts/smoke_api.py --base-url http://127.0.0.1:8000

For a deployed HTTPS backend:

    python scripts/smoke_api.py --base-url https://YOUR_BACKEND_URL
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_MESSAGE = "I need a plumber in Karachi Saddar today"


class SmokeFailure(RuntimeError):
    pass


def _json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout: int = 90,
) -> tuple[int, dict[str, Any]]:
    url = base_url.rstrip("/") + path
    if query:
        clean_query = {key: value for key, value in query.items() if value is not None}
        url += "?" + urllib.parse.urlencode(clean_query)

    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            return response.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body


def _assert_status(name: str, status: int, expected: set[int], body: dict[str, Any]) -> None:
    if status not in expected:
        raise SmokeFailure(f"{name} returned HTTP {status}: {json.dumps(body)[:500]}")


def _assert_keys(name: str, body: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in body]
    if missing:
        raise SmokeFailure(f"{name} missing keys {missing}: {json.dumps(body)[:500]}")


def _print_ok(name: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"[OK] {name}{suffix}")


def smoke_health(base_url: str) -> None:
    status, body = _json_request(base_url, "GET", "/api/v1/health", timeout=20)
    _assert_status("health", status, {200}, body)
    _assert_keys("health", body, ["status", "version", "services"])
    _print_ok("health", f"status={body.get('status')} version={body.get('version')}")


def smoke_discovery(base_url: str) -> tuple[str | None, str | None]:
    status, categories = _json_request(base_url, "GET", "/api/v1/discovery/categories", timeout=30)
    _assert_status("discovery categories", status, {200}, categories)
    _assert_keys("discovery categories", categories, ["categories", "total"])

    category_name = None
    if categories.get("categories"):
        category_name = categories["categories"][0].get("name")
    _print_ok("discovery categories", f"total={categories.get('total')} first={category_name}")

    status, cities = _json_request(base_url, "GET", "/api/v1/discovery/cities", timeout=30)
    _assert_status("discovery cities", status, {200}, cities)
    _assert_keys("discovery cities", cities, ["cities", "total"])

    city_name = None
    if cities.get("cities"):
        city_name = cities["cities"][0].get("name")
    _print_ok("discovery cities", f"total={cities.get('total')} first={city_name}")
    return category_name, city_name


def smoke_session(base_url: str) -> str:
    status, body = _json_request(base_url, "POST", "/api/v1/sessions", timeout=20)
    _assert_status("create session", status, {200}, body)
    _assert_keys("create session", body, ["session_id", "created"])
    session_id = str(body["session_id"])
    _print_ok("create session", f"session_id={session_id}")

    status, session = _json_request(base_url, "GET", f"/api/v1/sessions/{session_id}", timeout=20)
    _assert_status("get session", status, {200}, session)
    _assert_keys("get session", session, ["session_id", "turn_count"])
    _print_ok("get session", f"turn_count={session.get('turn_count')}")
    return session_id


def smoke_chat_rest(base_url: str, session_id: str, message: str) -> dict[str, Any]:
    status, body = _json_request(
        base_url,
        "POST",
        "/api/v1/chat",
        {"session_id": session_id, "message": message},
        timeout=180,
    )
    _assert_status("chat REST", status, {200}, body)
    _assert_keys("chat REST", body, ["response", "session_id", "status"])
    _print_ok("chat REST", f"status={body.get('status')} chars={len(body.get('response', ''))}")
    return body


def smoke_providers(base_url: str, category: str | None, city: str | None) -> int | None:
    status, body = _json_request(
        base_url,
        "GET",
        "/api/v1/providers",
        query={"query": category or "plumber", "city": city, "limit": 3},
        timeout=120,
    )
    _assert_status("providers list", status, {200}, body)
    _assert_keys("providers list", body, ["providers", "total", "page", "per_page", "has_next"])
    providers = body.get("providers") or []
    _print_ok("providers list", f"returned={len(providers)} total={body.get('total')}")

    provider_id = None
    if providers:
        provider_id = providers[0].get("provider_id")
        required = ["provider_name", "category", "city", "area", "phone_number", "response_time_minutes"]
        missing = [key for key in required if key not in providers[0]]
        if missing:
            raise SmokeFailure(f"provider payload missing mobile fields {missing}: {providers[0]}")
        _print_ok("provider mobile fields", f"provider_id={provider_id}")

    if provider_id is not None:
        status, detail = _json_request(base_url, "GET", f"/api/v1/providers/{provider_id}", timeout=120)
        _assert_status("provider detail", status, {200}, detail)
        _assert_keys("provider detail", detail, ["provider_id", "provider_name", "phone_number", "email"])
        _print_ok("provider detail", f"provider_id={detail.get('provider_id')}")

    return int(provider_id) if provider_id is not None else None


def smoke_bookings(base_url: str, session_id: str, provider_id: int | None, category: str | None, city: str | None) -> None:
    if provider_id is None:
        print("[SKIP] booking create | no provider_id returned by provider search")
        return

    status, booking = _json_request(
        base_url,
        "POST",
        "/api/v1/bookings",
        {
            "session_id": session_id,
            "provider_id": provider_id,
            "service_type": category or "Service",
            "location": city or "Pakistan",
            "scheduled_time": "tomorrow morning",
            "user_notes": "Smoke test booking from VM",
        },
        timeout=120,
    )
    _assert_status("booking create", status, {200}, booking)
    _assert_keys("booking create", booking, ["booking_id", "status"])
    booking_id = booking["booking_id"]
    _print_ok("booking create", f"booking_id={booking_id} status={booking.get('status')}")

    status, fetched = _json_request(base_url, "GET", f"/api/v1/bookings/{booking_id}", timeout=30)
    _assert_status("booking get", status, {200}, fetched)
    _assert_keys("booking get", fetched, ["booking_id", "status"])
    _print_ok("booking get", f"status={fetched.get('status')}")

    status, history = _json_request(base_url, "GET", f"/api/v1/bookings/session/{session_id}", timeout=30)
    _assert_status("booking session list", status, {200}, history)
    _assert_keys("booking session list", history, ["bookings", "total"])
    _print_ok("booking session list", f"total={history.get('total')}")


def _ws_url_from_base(base_url: str, path: str, session_id: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc
    base_path = parsed.path.rstrip("/")
    query = urllib.parse.urlencode({"session_id": session_id})
    return urllib.parse.urlunparse((scheme, netloc, base_path + path, "", query, ""))


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise SmokeFailure("WebSocket closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _ws_send_text(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    header = bytearray([0x81])
    length = len(data)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
    sock.sendall(bytes(header) + mask + masked)


def _ws_recv_text(sock: socket.socket) -> dict[str, Any]:
    first_two = _recv_exact(sock, 2)
    opcode = first_two[0] & 0x0F
    masked = bool(first_two[1] & 0x80)
    length = first_two[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]

    mask = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

    if opcode == 0x8:
        raise SmokeFailure("WebSocket close frame received before chat response")
    if opcode not in (0x1, 0x2):
        return {}
    return json.loads(payload.decode("utf-8"))


def smoke_chat_websocket(base_url: str, session_id: str, message: str) -> None:
    ws_url = _ws_url_from_base(base_url, "/api/v1/ws/chat", session_id)
    parsed = urllib.parse.urlparse(ws_url)
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    host = parsed.hostname
    if not host:
        raise SmokeFailure(f"Invalid WebSocket URL: {ws_url}")

    raw_sock = socket.create_connection((host, port), timeout=20)
    sock: socket.socket
    if parsed.scheme == "wss":
        sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=host)
    else:
        sock = raw_sock
    sock.settimeout(180)

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    handshake = (
        f"GET {target} HTTP/1.1\r\n"
        f"Host: {parsed.netloc}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(handshake.encode("ascii"))
    response = sock.recv(4096).decode("iso-8859-1", errors="replace")
    if " 101 " not in response.split("\r\n", 1)[0]:
        raise SmokeFailure(f"WebSocket handshake failed: {response.splitlines()[0] if response else 'no response'}")

    accept_expected = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")
    if accept_expected not in response:
        raise SmokeFailure("WebSocket handshake returned an invalid Sec-WebSocket-Accept")

    session_seen = False
    _ws_send_text(sock, {"session_id": session_id, "message": message})
    deadline = time.time() + 180
    while time.time() < deadline:
        payload = _ws_recv_text(sock)
        payload_type = payload.get("type")
        if payload_type == "session":
            session_seen = True
            continue
        if payload_type == "chat_response":
            _assert_keys("chat WebSocket", payload, ["response", "session_id", "status"])
            _print_ok(
                "chat WebSocket",
                f"session_seen={session_seen} status={payload.get('status')} chars={len(payload.get('response', ''))}",
            )
            sock.close()
            return
        if payload_type == "error":
            raise SmokeFailure(f"WebSocket error: {payload}")
    raise SmokeFailure("Timed out waiting for WebSocket chat_response")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test backend and mobile-facing API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Chat smoke message")
    parser.add_argument("--skip-websocket", action="store_true", help="Skip /api/v1/ws/chat")
    parser.add_argument("--skip-booking", action="store_true", help="Skip direct booking create/get smoke")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    try:
        smoke_health(base_url)
        category, city = smoke_discovery(base_url)
        session_id = smoke_session(base_url)
        smoke_chat_rest(base_url, session_id, args.message)
        if not args.skip_websocket:
            smoke_chat_websocket(base_url, session_id, args.message)
        provider_id = smoke_providers(base_url, category, city)
        if not args.skip_booking:
            smoke_bookings(base_url, session_id, provider_id, category, city)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print("[DONE] Smoke tests completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
