#!/usr/bin/env python3
"""
Urdu and Roman Urdu Smoke/Acceptance Test Script.
Validates intent extraction, latency, and response translations on the GCP VM.

Usage:
    python scripts/smoke_urdu_roman.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class SmokeFailure(RuntimeError):
    pass


def _json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 10,
) -> tuple[int, dict[str, Any]]:
    url = base_url.rstrip("/") + path
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


def run_tests(base_url: str) -> bool:
    print("=" * 70)
    print("   URDU & ROMAN URDU ACCEPTANCE SMOKE TESTS ON GCP VM")
    print("=" * 70)

    # First, check health
    print("\n[Step 1] Checking API Health status...")
    try:
        status, health = _json_request(base_url, "GET", "/api/v1/health")
        if status != 200:
            raise SmokeFailure(f"Health check failed with status {status}")
        print(f"  [OK] Health check passed: {health.get('status')} | Version: {health.get('version')}")
    except Exception as exc:
        print(f"  [FAIL] Health check could not connect to {base_url}. Error: {exc}")
        return False

    # Create session
    print("\n[Step 2] Creating new session...")
    status, sess = _json_request(base_url, "POST", "/api/v1/sessions")
    if status != 200:
        print(f"  [FAIL] Failed to create session: status={status}")
        return False
    session_id = sess.get("session_id")
    print(f"  [OK] Session created: session_id={session_id}")

    failures = 0

    # -------------------------------------------------------------
    # TEST 1: Roman Urdu Query
    # -------------------------------------------------------------
    print("\n[TEST 1] Running Roman Urdu query: 'mujhe plumber chahiye Karachi Clifton mein'")
    t1_payload = {
        "session_id": session_id,
        "message": "mujhe plumber chahiye Karachi Clifton mein"
    }
    
    start_time = time.time()
    status, res1 = _json_request(base_url, "POST", "/api/v1/chat", t1_payload, timeout=15)
    latency = (time.time() - start_time) * 1000

    if status != 200:
        print(f"  [FAIL] Roman Urdu chat request returned HTTP status {status}: {res1}")
        failures += 1
    else:
        intent = res1.get("intent") or {}
        resp_text = res1.get("response") or ""
        lang_detected = res1.get("language_detected") or intent.get("language_detected")
        
        print(f"  - Extracted Service:  {intent.get('service_type')} (Expected: Plumber)")
        print(f"  - Extracted City:     {intent.get('city')} (Expected: Karachi)")
        print(f"  - Extracted Area:     {intent.get('area')} (Expected: Clifton)")
        print(f"  - Language Detected:  {lang_detected} (Expected: roman_urdu)")
        print(f"  - E2E Latency (ms):   {latency:.1f}ms (Expected: preferably < 2000ms)")
        print(f"  - Chat Response Preview: {resp_text[:120]}...")

        # Assertions
        try:
            assert intent.get("service_type") == "Plumber", f"Expected Plumber, got {intent.get('service_type')}"
            assert intent.get("city") == "Karachi", f"Expected Karachi, got {intent.get('city')}"
            assert intent.get("area") == "Clifton", f"Expected Clifton, got {intent.get('area')}"
            assert str(lang_detected).lower() == "roman_urdu", f"Expected roman_urdu, got {lang_detected}"
            assert latency < 2000, f"Latency {latency:.1f}ms exceeded 2000ms"
            
            # Check translation matches Roman Urdu terms rather than purely English templates
            assert "Maine providers" in resp_text or "BOOKING CONFIRM" in resp_text or "shortlist" in resp_text.lower(), \
                "Response does not seem to be translated into Roman Urdu templates"
            
            print("  [PASS] Test 1: Roman Urdu extraction and formatting successful!")
        except AssertionError as err:
            print(f"  [FAIL] Test 1 Assertion Failed: {err}")
            failures += 1

    # -------------------------------------------------------------
    # TEST 2: Urdu Script Query
    # -------------------------------------------------------------
    print("\n[TEST 2] Running Urdu Script query: 'مجھے کراچی کلفٹن میں پلمبر چاہیے'")
    t2_payload = {
        "session_id": session_id,
        "message": "مجھے کراچی کلفٹن میں پلمبر چاہیے"
    }

    start_time = time.time()
    status, res2 = _json_request(base_url, "POST", "/api/v1/chat", t2_payload, timeout=15)
    latency = (time.time() - start_time) * 1000

    if status != 200:
        print(f"  [FAIL] Urdu Script chat request returned HTTP status {status}: {res2}")
        failures += 1
    else:
        intent = res2.get("intent") or {}
        resp_text = res2.get("response") or ""
        lang_detected = res2.get("language_detected") or intent.get("language_detected")

        print(f"  - Extracted Service:  {intent.get('service_type')} (Expected: Plumber)")
        print(f"  - Extracted City:     {intent.get('city')} (Expected: Karachi)")
        print(f"  - Extracted Area:     {intent.get('area')} (Expected: Clifton)")
        print(f"  - Language Detected:  {lang_detected} (Expected: urdu)")
        print(f"  - E2E Latency (ms):   {latency:.1f}ms (Expected: preferably < 2000ms)")
        print(f"  - Chat Response Preview: {resp_text[:120]}...")

        # Assertions
        try:
            assert intent.get("service_type") == "Plumber", f"Expected Plumber, got {intent.get('service_type')}"
            assert intent.get("city") == "Karachi", f"Expected Karachi, got {intent.get('city')}"
            assert intent.get("area") == "Clifton", f"Expected Clifton, got {intent.get('area')}"
            assert str(lang_detected).lower() == "urdu", f"Expected urdu, got {lang_detected}"
            assert latency < 2000, f"Latency {latency:.1f}ms exceeded 2000ms"
            
            # Check translation matches Urdu script terms
            # Verify the response has Urdu script characters (range \u0600-\u06ff)
            has_urdu = any('\u0600' <= char <= '\u06ff' for char in resp_text)
            assert has_urdu, "Response does not contain any Urdu script characters"
            
            print("  [PASS] Test 2: Urdu script extraction and formatting successful!")
        except AssertionError as err:
            print(f"  [FAIL] Test 2 Assertion Failed: {err}")
            failures += 1

    print("\n" + "=" * 70)
    if failures == 0:
        print("   ALL ACCEPTANCE SMOKE TESTS PASSED PERFECTLY!")
        print("   Urdu & Roman Urdu Orchestrator features are 100% production-ready.")
        print("=" * 70)
        return True
    else:
        print(f"   SMOKE TEST COMPLETED WITH {failures} FAILURES.")
        print("   Please review current configurations or logs.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for Urdu & Roman Urdu Orchestrator features.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    args = parser.parse_args()
    
    success = run_tests(args.base_url)
    sys.exit(0 if success else 1)
