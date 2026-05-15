"""
Follow-Up Agent — Post-booking automation.
"""

import logging
from typing import Optional
from ..core.base_llm import BaseLLM
from ..state.booking_store import BookingStore

logger = logging.getLogger("agents.followup")


class FollowUpAgent:
    def __init__(self, llm: BaseLLM, booking_store: BookingStore, prompts_config: dict, agents_config: dict):
        self.llm = llm
        self.booking_store = booking_store
        self.prompts_config = prompts_config
        self.config = agents_config.get("agents", {}).get("followup", {})

    async def generate_reminder(self, booking_id: str, language: str = "English") -> Optional[str]:
        booking = self.booking_store.get_booking(booking_id)
        if not booking:
            return None
        prompt_template = self.prompts_config.get("followup_reminder", "Reminder for {booking_id}")
        prompt = prompt_template.format(
            booking_id=booking["booking_id"], service_type=booking["service_type"],
            provider_name=booking["provider_name"], scheduled_time=booking["scheduled_time"],
            time_until="1 hour", language=language,
        )
        response = await self.llm.generate(prompt=prompt, system_instruction="Generate a friendly reminder.")
        return response.text

    async def check_status(self, booking_id: str) -> Optional[dict]:
        return self.booking_store.get_booking(booking_id)

    async def complete_booking(self, booking_id: str) -> bool:
        return self.booking_store.update_status(booking_id, "COMPLETED", "Service completed")
