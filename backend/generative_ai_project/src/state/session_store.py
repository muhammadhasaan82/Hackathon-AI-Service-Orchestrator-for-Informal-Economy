"""
Session Store — Redis-backed session management.

Falls back to in-memory dict if Redis is unavailable.
Stores conversation history, agent state, and reasoning traces.
"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("state.session")


class SessionStore:
    """Redis-backed session store with in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._redis = None
        self._fallback: dict[str, dict] = {}

        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info(f"SessionStore: Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Using in-memory fallback.")
            self._redis = None

    def _serialize(self, data: dict) -> str:
        return json.dumps(data, default=str)

    def _deserialize(self, data: str) -> dict:
        return json.loads(data)

    def create_session(self, session_id: str) -> dict:
        """Create a new session."""
        session = {
            "session_id": session_id,
            "conversation_history": [],
            "agent_state": {
                "current_agent": None,
                "pipeline_stage": None,
                "intent": None,
                "candidates": [],
                "ranked_providers": [],
                "selected_provider": None,
                "booking": None,
                "followup": None,
                "awaiting_booking_confirmation": False,
                "last_faq_topics": [],
                "routing": {},
            },
            "reasoning_trace": [],
            "metadata": {
                "created_at": time.time(),
                "last_active": time.time(),
                "language": "english",
                "turn_count": 0,
            },
        }

        if self._redis:
            self._redis.setex(
                f"session:{session_id}",
                self.default_ttl,
                self._serialize(session),
            )
        else:
            self._fallback[session_id] = session

        logger.info(f"Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID."""
        if self._redis:
            data = self._redis.get(f"session:{session_id}")
            if data:
                session = self._deserialize(data)
                session["metadata"]["last_active"] = time.time()
                self._redis.expire(f"session:{session_id}", self.default_ttl)
                return session
            return None
        else:
            session = self._fallback.get(session_id)
            if session:
                session["metadata"]["last_active"] = time.time()
            return session

    def get_or_create(self, session_id: str) -> dict:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        return session

    def update_session(self, session_id: str, updates: dict):
        """Merge updates into session."""
        session = self.get_or_create(session_id)
        for key, value in updates.items():
            if key in session and isinstance(session[key], dict) and isinstance(value, dict):
                session[key].update(value)
            else:
                session[key] = value
        self._save(session_id, session)

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn."""
        session = self.get_or_create(session_id)
        session["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        session["metadata"]["turn_count"] += 1
        self._save(session_id, session)

    def add_trace(self, session_id: str, agent: str, action: str, details: Any = None):
        """Add a reasoning trace entry."""
        session = self.get_or_create(session_id)
        session["reasoning_trace"].append({
            "agent": agent,
            "action": action,
            "details": details,
            "timestamp": time.time(),
        })
        self._save(session_id, session)

    def get_history(self, session_id: str, max_turns: int = 10) -> list[dict]:
        """Get recent conversation history."""
        session = self.get_session(session_id)
        if not session:
            return []
        history = session["conversation_history"]
        return history[-max_turns:] if len(history) > max_turns else history

    def delete_session(self, session_id: str):
        """Delete a session."""
        if self._redis:
            self._redis.delete(f"session:{session_id}")
        else:
            self._fallback.pop(session_id, None)

    def save_session(self, session_id: str, session: dict):
        """Persist a fully-mutated session object."""
        self._save(session_id, session)

    def _save(self, session_id: str, session: dict):
        """Persist session to storage."""
        if self._redis:
            self._redis.setex(
                f"session:{session_id}",
                self.default_ttl,
                self._serialize(session),
            )
        else:
            self._fallback[session_id] = session
