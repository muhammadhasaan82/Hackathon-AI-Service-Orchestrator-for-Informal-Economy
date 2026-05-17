"""
Booking Agent — Simulates booking confirmation and scheduling.
"""

import logging
from typing import Optional
from ..core.base_llm import BaseLLM
from ..state.booking_store import BookingStore
from .tools import generate_time_slot

logger = logging.getLogger("agents.booking")


class BookingAgent:
    def __init__(self, llm: BaseLLM, booking_store: BookingStore, prompts_config: dict, agents_config: dict, guardrails=None):
        self.llm = llm
        self.booking_store = booking_store
        self.prompts_config = prompts_config
        self.config = agents_config.get("agents", {}).get("booking", {})
        self.guardrails = guardrails

    async def create_booking(self, session_id: str, provider: dict, intent: dict, status: Optional[str] = None) -> dict:
        meta = provider.get("metadata", provider)
        time_slot = generate_time_slot(intent.get("time_preference"))
        booking = self.booking_store.create_booking(
            session_id=session_id, provider=meta,
            service_type=intent.get("service_type", "Service"),
            location=f"{meta.get('area', '')}, {meta.get('city', '')}",
            scheduled_time=time_slot,
            status=status or self.config.get("default_status", "CONFIRMED"),
        )
        confirmation = await self._generate_confirmation(booking, intent)
        booking["confirmation_message"] = confirmation
        logger.info(f"Booking created: {booking['booking_id']}")
        return booking

    async def _generate_confirmation(self, booking: dict, intent: dict) -> str:
        lang = intent.get("language_detected", "english")
        language = "Roman Urdu" if lang.lower() in ("roman urdu", "urdu") else "English"
        prompt_template = self.prompts_config.get("booking_confirmation", "Confirm booking {booking_id}")
        prompt = prompt_template.format(
            booking_id=booking["booking_id"], service_type=booking["service_type"],
            provider_name=booking["provider_name"], location=booking["location"],
            scheduled_time=booking["scheduled_time"], price_range=booking.get("price_range", "N/A"),
            phone=booking.get("provider_phone", "N/A"), language=language,
            booking_status=booking.get("status", "CONFIRMED"),
        )
        system_instruction = "Generate a clear booking confirmation."
        if self.guardrails:
            system_instruction = self.guardrails.compose_system_instruction("booking", system_instruction)
        response = await self.llm.generate(prompt=prompt, system_instruction=system_instruction)
        return self.guardrails.sanitize_text(response.text) if self.guardrails else response.text

    async def get_status(self, booking_id: str) -> Optional[dict]:
        return self.booking_store.get_booking(booking_id)

    async def cancel_booking(self, booking_id: str) -> bool:
        return self.booking_store.update_status(booking_id, "CANCELLED", "User cancelled")
