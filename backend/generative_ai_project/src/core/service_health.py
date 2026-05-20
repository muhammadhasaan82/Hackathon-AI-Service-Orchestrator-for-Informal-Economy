"""Lightweight dependency health checks for submission demos."""

from __future__ import annotations

import os
from typing import Optional


def check_postgres() -> dict:
    """Return PostgreSQL status and provider count when available."""
    urls = _postgres_urls()
    last_error: Optional[str] = None
    for url in urls:
        try:
            import psycopg2

            conn = psycopg2.connect(url, connect_timeout=2)
            try:
                provider_count = _try_provider_count(conn)
                payload = {"status": "connected"}
                if provider_count is not None:
                    payload["provider_count"] = provider_count
                return payload
            finally:
                conn.close()
        except Exception as exc:
            last_error = str(exc)
    return {"status": "error", "error": _safe_error(last_error)}


def check_redis() -> dict:
    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        return {"status": "connected"} if client.ping() else {"status": "error"}
    except Exception as exc:
        return {"status": "error", "error": _safe_error(str(exc))}


def check_weaviate() -> dict:
    try:
        import urllib.request

        base_url = os.getenv("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
        request = urllib.request.Request(f"{base_url}/v1/.well-known/ready", method="GET")
        with urllib.request.urlopen(request, timeout=2) as response:
            if 200 <= response.status < 300:
                return {"status": "connected"}
        return {"status": "error"}
    except Exception as exc:
        return {"status": "error", "error": _safe_error(str(exc))}


def health_payload() -> dict:
    postgres = check_postgres()
    redis = check_redis()
    weaviate = check_weaviate()
    payload = {
        "status": "ok",
        "postgres": postgres["status"],
        "redis": redis["status"],
        "weaviate": weaviate["status"],
    }
    if "provider_count" in postgres:
        payload["provider_count"] = postgres["provider_count"]
    return payload


def _postgres_urls() -> list[str]:
    urls = []
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        urls.append(env_url)
    urls.extend([
        "postgresql://postgres:postgres@localhost:5433/service_orchestrator",
        "postgresql://postgres@localhost:5433/service_orchestrator",
        "postgresql://orchestrator:orchestrator_dev@localhost:5433/orchestrator",
    ])
    deduped = []
    for url in urls:
        if url and url not in deduped:
            deduped.append(url)
    return deduped


def _try_provider_count(conn) -> Optional[int]:
    table_names = ("providers", "service_providers")
    for table in table_names:
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
            return int(row[0]) if row else None
        except Exception:
            conn.rollback()
    return None


def _safe_error(message: Optional[str]) -> str:
    text = str(message or "unknown error")
    if len(text) > 160:
        return text[:157] + "..."
    return text

