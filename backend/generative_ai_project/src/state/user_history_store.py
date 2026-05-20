"""PostgreSQL-backed user interaction history for hackathon demo."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("state.user_history")


class UserHistoryStore:
    """Minimal PostgreSQL history store with graceful no-op fallback."""

    def __init__(self, database_url: Optional[str] = None):
        self._pg_conn_str = None
        self._enabled = False
        for url in self._candidate_urls(database_url):
            try:
                import psycopg2

                conn = psycopg2.connect(url, connect_timeout=2)
                self._init_table(conn)
                conn.close()
                self._pg_conn_str = url
                self._enabled = True
                logger.info("UserHistoryStore: Connected to PostgreSQL")
                break
            except Exception as exc:
                logger.warning("UserHistoryStore PostgreSQL unavailable for %s: %s", self._redact(url), exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def create_entry(self, session_id: str, user_message: str, user_id: Optional[str] = None) -> Optional[int]:
        if not self._enabled:
            return None
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO user_history (user_id, session_id, user_message)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (user_id, session_id, user_message),
            )
            history_id = cur.fetchone()[0]
            conn.commit()
            return int(history_id)
        except Exception as exc:
            conn.rollback()
            logger.warning("Failed to create user_history entry: %s", exc)
            return None
        finally:
            conn.close()

    def update_entry(self, history_id: Optional[int], **fields: Any) -> None:
        if not self._enabled or not history_id:
            return
        allowed = {
            "detected_intent",
            "service_type",
            "location",
            "requested_time",
            "selected_provider_id",
            "selected_provider_name",
            "booking_status",
            "reasoning",
            "workflow_trace",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return

        import psycopg2.extras

        assignments = []
        values = []
        for key, value in updates.items():
            assignments.append(f"{key} = %s")
            if key in {"reasoning", "workflow_trace"}:
                values.append(psycopg2.extras.Json(value))
            else:
                values.append(value)
        values.append(history_id)

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE user_history SET {', '.join(assignments)} WHERE id = %s",
                values,
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("Failed to update user_history entry %s: %s", history_id, exc)
        finally:
            conn.close()

    def get_by_session(self, session_id: str) -> list[dict]:
        if not self._enabled:
            return []
        return self._fetch(
            "SELECT * FROM user_history WHERE session_id = %s ORDER BY created_at ASC, id ASC",
            (session_id,),
        )

    def get_recent(self, limit: int = 20) -> list[dict]:
        if not self._enabled:
            return []
        return self._fetch(
            "SELECT * FROM user_history ORDER BY created_at DESC, id DESC LIMIT %s",
            (limit,),
        )[::-1]

    def _fetch(self, query: str, params: tuple) -> list[dict]:
        import psycopg2.extras

        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._pg_conn_str, connect_timeout=2)

    def _init_table(self, conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NULL,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                detected_intent TEXT NULL,
                service_type TEXT NULL,
                location TEXT NULL,
                requested_time TEXT NULL,
                selected_provider_id INTEGER NULL,
                selected_provider_name TEXT NULL,
                booking_status TEXT NULL,
                reasoning JSONB NULL,
                workflow_trace JSONB NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_history_session ON user_history(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_history_created_at ON user_history(created_at)")
        conn.commit()

    def _candidate_urls(self, database_url: Optional[str]) -> list[str]:
        urls = []
        for url in [
            database_url,
            os.getenv("DATABASE_URL"),
            "postgresql://postgres:postgres@localhost:5433/service_orchestrator",
            "postgresql://postgres@localhost:5433/service_orchestrator",
            "postgresql://orchestrator:orchestrator_dev@localhost:5433/orchestrator",
        ]:
            if url and url not in urls:
                urls.append(url)
        return urls

    def _redact(self, url: str) -> str:
        if "@" not in str(url):
            return str(url)
        prefix, suffix = str(url).split("@", 1)
        scheme = prefix.split("://", 1)[0]
        return f"{scheme}://***@{suffix}"

