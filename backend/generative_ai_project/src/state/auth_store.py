"""PostgreSQL-backed user auth for the Flutter client."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("state.auth")


class DuplicateUserError(Exception):
    """Raised when an email already exists in the users table."""


class AuthStore:
    """Minimal PostgreSQL auth store with bcrypt password hashing and JWTs."""

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
                logger.info("AuthStore: Connected to PostgreSQL")
                break
            except Exception as exc:
                logger.warning("AuthStore PostgreSQL unavailable for %s: %s", self._redact(url), exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def hash_password(self, password: str) -> str:
        return self._password_context().hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        return self._password_context().verify(password, password_hash)

    def create_access_token(self, user: dict) -> str:
        import jwt

        expire_days = int(os.getenv("JWT_EXPIRE_DAYS", "7"))
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user["id"]),
            "email": user["email"],
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=expire_days)).timestamp()),
        }
        return jwt.encode(payload, self._jwt_secret(), algorithm="HS256")

    def decode_access_token(self, token: str) -> dict:
        import jwt

        return jwt.decode(token, self._jwt_secret(), algorithms=["HS256"])

    def create_user(self, name: str, email: str, password_hash: str) -> dict:
        if not self._enabled:
            raise RuntimeError("Auth store is not connected")

        import psycopg2
        import psycopg2.extras

        conn = self._connect()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id, name, email, created_at
                """,
                (name, email, password_hash),
            )
            user = dict(cur.fetchone())
            conn.commit()
            return user
        except psycopg2.IntegrityError as exc:
            conn.rollback()
            if "users_email_key" in str(exc) or "duplicate key" in str(exc).lower():
                raise DuplicateUserError from exc
            raise
        finally:
            conn.close()

    def get_user_by_email(self, email: str, include_hash: bool = False) -> Optional[dict]:
        columns = "id, name, email, password_hash, created_at" if include_hash else "id, name, email, created_at"
        rows = self._fetch(f"SELECT {columns} FROM users WHERE email = %s", (email,))
        return rows[0] if rows else None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        rows = self._fetch("SELECT id, name, email, created_at FROM users WHERE id = %s", (user_id,))
        return rows[0] if rows else None

    def _fetch(self, query: str, params: tuple) -> list[dict]:
        if not self._enabled:
            return []

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
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.commit()

    def _password_context(self):
        from passlib.context import CryptContext

        return CryptContext(schemes=["bcrypt"], deprecated="auto")

    def _jwt_secret(self) -> str:
        secret = os.getenv("JWT_SECRET")
        if not secret:
            raise RuntimeError("JWT_SECRET is not configured")
        return secret

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
