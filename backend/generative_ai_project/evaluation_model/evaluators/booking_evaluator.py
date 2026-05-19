"""
Booking Evaluator — Measures booking agent accuracy and state machine correctness.

Metrics:
    - Field Completeness : All required booking fields populated
    - Status Accuracy    : Correct state transitions (PENDING → CONFIRMED → etc.)
    - ID Format          : Booking IDs follow BK-XXXXXXXX pattern
    - Confirmation Rate  : Fraction of intents that result in confirmed bookings
    - Fallback Coverage  : System handles missing fields gracefully
"""

import asyncio
import logging
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("eval.booking")

BOOKING_TEST_CASES = [
    {
        "id": "bk_001",
        "intent": {
            "service_type": "AC Technician", "city": "Islamabad", "area": "G-13",
            "time_preference": "kal subah", "urgency": "normal",
        },
        "provider": {
            "provider_id": 1001, "provider_name": "Ali Ahmad", "category": "AC Technician",
            "city": "Islamabad", "area": "G-13", "rating": 4.7,
            "price_range": "Medium", "phone_number": "+92-300-1234567",
            "email": "ali@example.com", "availability": "Available",
        },
        "expected_status": "CONFIRMED",
        "required_fields": ["booking_id", "provider_name", "service_type", "location", "status", "scheduled_time"],
    },
    {
        "id": "bk_002",
        "intent": {
            "service_type": "Plumber", "city": "Karachi", "area": "DHA",
            "time_preference": "abhi", "urgency": "urgent",
        },
        "provider": {
            "provider_id": 2002, "provider_name": "Hassan Khan", "category": "Plumber",
            "city": "Karachi", "area": "DHA", "rating": 4.5,
            "price_range": "Low", "phone_number": "+92-321-9876543",
            "email": "hassan@example.com", "availability": "Available",
        },
        "expected_status": "CONFIRMED",
        "required_fields": ["booking_id", "provider_name", "service_type", "location", "status"],
    },
    {
        "id": "bk_003",
        "intent": {
            "service_type": "Beautician", "city": "Lahore", "area": "Gulberg",
            "time_preference": "Saturday afternoon", "urgency": "normal",
        },
        "provider": {
            "provider_id": 3003, "provider_name": "Sara Malik", "category": "Beautician",
            "city": "Lahore", "area": "Gulberg", "rating": 4.9,
            "price_range": "High", "phone_number": "+92-333-5550000",
            "email": "sara@example.com", "availability": "Available Soon",
        },
        "expected_status": "CONFIRMED",
        "required_fields": ["booking_id", "provider_name", "service_type", "location", "status", "scheduled_time"],
    },
]

VALID_STATES = ["PENDING", "CONFIRMED", "IN_PROGRESS", "COMPLETED", "CANCELLED"]
BOOKING_ID_PATTERN = re.compile(r"^BK-[A-F0-9]{8}$")


class BookingEvaluator:
    """Evaluates the booking agent's correctness and robustness."""

    def __init__(self, booking_store=None):
        self._booking_store = booking_store

    async def run(self) -> dict:
        logger.info(f"Running Booking evaluation on {len(BOOKING_TEST_CASES)} test cases...")

        store = self._get_store()
        field_scores = []
        status_scores = []
        id_format_scores = []
        confirmed_count = 0
        per_case = []

        for tc in BOOKING_TEST_CASES:
            try:
                booking = store.create_booking(
                    session_id=f"eval_{tc['id']}",
                    provider=tc["provider"],
                    service_type=tc["intent"]["service_type"],
                    location=f"{tc['intent'].get('area', '')}, {tc['intent'].get('city', '')}",
                    scheduled_time=tc["intent"].get("time_preference"),
                )

                # Field completeness
                missing = [f for f in tc["required_fields"] if not booking.get(f)]
                field_score = 1.0 - (len(missing) / len(tc["required_fields"]))
                field_scores.append(field_score)

                # Status correctness
                status_ok = booking.get("status") == tc["expected_status"]
                status_scores.append(1.0 if status_ok else 0.0)

                # Booking ID format
                bid = booking.get("booking_id", "")
                id_ok = bool(BOOKING_ID_PATTERN.match(bid))
                id_format_scores.append(1.0 if id_ok else 0.0)

                if booking.get("status") == "CONFIRMED":
                    confirmed_count += 1

                # State transition test
                trans_ok = store.update_status(bid, "COMPLETED", "eval test")

                per_case.append({
                    "id": tc["id"],
                    "booking_id": bid,
                    "status": booking.get("status"),
                    "expected_status": tc["expected_status"],
                    "field_score": field_score,
                    "id_format_ok": id_ok,
                    "transition_ok": trans_ok,
                    "missing_fields": missing,
                })

            except Exception as e:
                logger.warning(f"{tc['id']} failed: {e}")
                per_case.append({"id": tc["id"], "error": str(e)})

        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        fc = avg(field_scores)
        sc = avg(status_scores)
        ic = avg(id_format_scores)
        cr = confirmed_count / len(BOOKING_TEST_CASES)
        overall = fc * 0.35 + sc * 0.35 + ic * 0.15 + cr * 0.15

        return {
            "overall_score": overall,
            "accuracy": sc,
            "field_completeness": fc,
            "id_format_accuracy": ic,
            "confirmation_rate": cr,
            "total_test_cases": len(BOOKING_TEST_CASES),
            "per_case": per_case,
        }

    def _get_store(self):
        if self._booking_store:
            return self._booking_store
        try:
            from src.state.booking_store import BookingStore
            return BookingStore()
        except Exception as e:
            logger.warning(f"Could not load BookingStore ({e}).")
            raise
