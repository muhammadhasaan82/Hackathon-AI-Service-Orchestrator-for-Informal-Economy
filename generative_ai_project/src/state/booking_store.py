"""
Booking Store — PostgreSQL-backed booking persistence.

Falls back to SQLite if PostgreSQL is unavailable.
ACID guarantees for booking state transitions.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("state.booking")

_SQLITE_PATH = Path(__file__).parent.parent.parent / "data" / "cache" / "bookings.db"


class BookingStore:
    """PostgreSQL booking store with SQLite fallback."""

    VALID_STATES = ["PENDING", "CONFIRMED", "IN_PROGRESS", "COMPLETED", "CANCELLED"]

    def __init__(self, database_url: Optional[str] = None):
        self._pg_pool = None
        self._sqlite_path = str(_SQLITE_PATH)

        db_url = database_url or os.getenv("DATABASE_URL", "")

        if db_url and db_url.startswith("postgresql"):
            try:
                import psycopg2
                self._pg_conn_str = db_url
                conn = psycopg2.connect(db_url)
                self._init_postgres(conn)
                conn.close()
                self._use_pg = True
                logger.info(f"BookingStore: Connected to PostgreSQL")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable ({e}). Using SQLite fallback.")
                self._use_pg = False
                self._init_sqlite()
        else:
            self._use_pg = False
            self._init_sqlite()

    def _init_postgres(self, conn):
        """Create PostgreSQL tables."""
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                provider_id INTEGER NOT NULL,
                provider_name TEXT NOT NULL,
                service_type TEXT NOT NULL,
                location TEXT NOT NULL,
                scheduled_time TEXT,
                status TEXT DEFAULT 'PENDING',
                price_range TEXT,
                provider_phone TEXT,
                provider_email TEXT,
                user_notes TEXT,
                metadata JSONB DEFAULT '{}',
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS booking_events (
                event_id SERIAL PRIMARY KEY,
                booking_id TEXT NOT NULL REFERENCES bookings(booking_id),
                event_type TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT,
                details TEXT,
                timestamp DOUBLE PRECISION NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bookings_session ON bookings(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status)")
        conn.commit()

    def _init_sqlite(self):
        """Create SQLite tables (fallback)."""
        Path(self._sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                provider_id INTEGER NOT NULL,
                provider_name TEXT NOT NULL,
                service_type TEXT NOT NULL,
                location TEXT NOT NULL,
                scheduled_time TEXT,
                status TEXT DEFAULT 'PENDING',
                price_range TEXT,
                provider_phone TEXT,
                provider_email TEXT,
                user_notes TEXT,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS booking_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT,
                details TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (booking_id) REFERENCES bookings(booking_id)
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"BookingStore: Using SQLite at {self._sqlite_path}")

    def _get_conn(self):
        """Get a database connection."""
        if self._use_pg:
            import psycopg2
            return psycopg2.connect(self._pg_conn_str)
        else:
            return sqlite3.connect(self._sqlite_path)

    def create_booking(self, session_id: str, provider: dict, service_type: str, location: str, scheduled_time: Optional[str] = None, user_notes: Optional[str] = None) -> dict:
        """Create a new booking."""
        booking_id = f"BK-{uuid.uuid4().hex[:8].upper()}"
        now = time.time()

        booking = {
            "booking_id": booking_id,
            "session_id": session_id,
            "provider_id": provider.get("provider_id", 0),
            "provider_name": provider.get("provider_name", "Unknown"),
            "service_type": service_type,
            "location": location,
            "scheduled_time": scheduled_time or "To be confirmed",
            "status": "CONFIRMED",
            "price_range": provider.get("price_range", "N/A"),
            "provider_phone": provider.get("phone_number", "N/A"),
            "provider_email": provider.get("email", "N/A"),
            "user_notes": user_notes or "",
            "created_at": now,
            "updated_at": now,
        }

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO bookings
                   (booking_id, session_id, provider_id, provider_name, service_type,
                    location, scheduled_time, status, price_range, provider_phone,
                    provider_email, user_notes, created_at, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""" if self._use_pg else
                """INSERT INTO bookings
                   (booking_id, session_id, provider_id, provider_name, service_type,
                    location, scheduled_time, status, price_range, provider_phone,
                    provider_email, user_notes, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (booking["booking_id"], booking["session_id"],
                 booking["provider_id"], booking["provider_name"],
                 booking["service_type"], booking["location"],
                 booking["scheduled_time"], booking["status"],
                 booking["price_range"], booking["provider_phone"],
                 booking["provider_email"], booking["user_notes"],
                 booking["created_at"], booking["updated_at"]),
            )
            ph = "%s" if self._use_pg else "?"
            cur.execute(
                f"""INSERT INTO booking_events
                   (booking_id, event_type, new_status, details, timestamp)
                   VALUES ({ph},{ph},{ph},{ph},{ph})""",
                (booking_id, "CREATED", "CONFIRMED", json.dumps({"provider": booking["provider_name"]}), now),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Created booking: {booking_id} for {service_type}")
        return booking

    def get_booking(self, booking_id: str) -> Optional[dict]:
        """Get a booking by ID."""
        conn = self._get_conn()
        try:
            if self._use_pg:
                import psycopg2.extras
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT * FROM bookings WHERE booking_id = %s", (booking_id,))
                row = cur.fetchone()
                return dict(row) if row else None
            else:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    def update_status(self, booking_id: str, new_status: str, details: str = "") -> bool:
        """Transition booking to a new status."""
        if new_status not in self.VALID_STATES:
            return False

        conn = self._get_conn()
        try:
            ph = "%s" if self._use_pg else "?"
            cur = conn.cursor()
            cur.execute(f"SELECT status FROM bookings WHERE booking_id = {ph}", (booking_id,))
            old = cur.fetchone()
            if not old:
                return False

            now = time.time()
            cur.execute(f"UPDATE bookings SET status = {ph}, updated_at = {ph} WHERE booking_id = {ph}",
                        (new_status, now, booking_id))
            cur.execute(
                f"""INSERT INTO booking_events
                   (booking_id, event_type, old_status, new_status, details, timestamp)
                   VALUES ({ph},{ph},{ph},{ph},{ph},{ph})""",
                (booking_id, "STATUS_CHANGE", old[0], new_status, details, now),
            )
            conn.commit()
            logger.info(f"Booking {booking_id}: {old[0]} → {new_status}")
            return True
        finally:
            conn.close()

    def get_bookings_by_session(self, session_id: str) -> list[dict]:
        """Get all bookings for a session."""
        conn = self._get_conn()
        try:
            if self._use_pg:
                import psycopg2.extras
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT * FROM bookings WHERE session_id = %s ORDER BY created_at DESC", (session_id,))
                return [dict(r) for r in cur.fetchall()]
            else:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM bookings WHERE session_id = ? ORDER BY created_at DESC", (session_id,)).fetchall()
                return [dict(r) for r in rows]
        finally:
            conn.close()
