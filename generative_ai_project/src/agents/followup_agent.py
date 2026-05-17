"""
Follow-Up Agent — Post-booking lifecycle automation.

ARCHITECTURE:
    This agent has TWO layers that mirror the project's design philosophy:

    1. DETERMINISTIC layer (schedule_followup):
       - Reads reminder tiers, urgency multipliers, and status timeline
         from agents_config.yaml
       - Computes exact trigger timestamps using booking.scheduled_time
       - Builds notification payloads with deterministic fields
         (booking_id, channel, priority, fire_at_utc, action_buttons)
       - This layer is pure config → output, fully testable

    2. PROBABILISTIC layer (LLM notification text):
       - For each scheduled notification, the LLM generates the
         title and body text by REASONING about context signals:
         service category, urgency, detected language, time of day
       - Uses prompt templates from prompts_config.yaml
       - Same booking can yield different phrasings — that's intentional

    The mobile app receives the full notification payload and can:
    - Schedule local push notifications at fire_at_utc
    - Show in-app notification timeline
    - Render action buttons (call provider, reschedule, cancel, rate)
"""

import datetime
import json
import logging
import time
import uuid
from typing import Optional

from ..core.base_llm import BaseLLM
from ..state.booking_store import BookingStore

logger = logging.getLogger("agents.followup")


class FollowUpAgent:
    """
    Generates actionable follow-up payloads for the mobile app.

    All scheduling logic is config-driven (agents_config.yaml).
    All notification text is LLM-generated (prompts_config.yaml).
    """

    def __init__(
        self,
        llm: BaseLLM,
        booking_store: BookingStore,
        prompts_config: dict,
        agents_config: dict,
        guardrails=None,
    ):
        self.llm = llm
        self.booking_store = booking_store
        self.prompts_config = prompts_config
        self.guardrails = guardrails

        # ── Read all behavior from config (soft-coded) ──────────
        self.config = agents_config.get("agents", {}).get("followup", {})
        self.reminder_tiers = self.config.get("reminder_tiers", [])
        self.urgency_multipliers = self.config.get("urgency_multipliers", {})
        self.status_updates = self.config.get("status_updates", [])
        self.post_completion = self.config.get("post_completion", {})
        self.channels_config = self.config.get("notification_channels", {})
        self.reasoning_directives = self.config.get("reasoning_directives", {})

        logger.info(
            f"FollowUpAgent initialized | "
            f"{len(self.reminder_tiers)} reminder tiers | "
            f"{len(self.status_updates)} status events | "
            f"channels: {[k for k, v in self.channels_config.items() if v.get('enabled')]}"
        )

    # ═══════════════════════════════════════════════════════════
    # PUBLIC: schedule_followup — called by Orchestrator after booking
    # ═══════════════════════════════════════════════════════════

    async def schedule_followup(
        self,
        booking: dict,
        intent: dict,
    ) -> dict:
        """
        Generate the complete follow-up plan for a confirmed booking.

        This is the MAIN entry point called by the orchestrator.
        Returns a dict with all scheduled notifications that the
        mobile app can use to fire local push notifications.

        The output is deterministic in structure (same config → same
        schedule), but the notification text is probabilistic (LLM).

        Args:
            booking: The confirmed booking dict from BookingAgent.
            intent: The extracted user intent (for language, urgency, etc.)

        Returns:
            {
                "followup_id": str,
                "booking_id": str,
                "scheduled_reminders": [...],
                "status_timeline": [...],
                "post_completion_actions": [...],
                "immediate_notification": {...},
                "total_scheduled": int,
            }
        """
        followup_id = f"FU-{uuid.uuid4().hex[:8].upper()}"
        booking_id = booking.get("booking_id", "UNKNOWN")
        urgency = intent.get("urgency", "normal")
        language = self._detect_response_language(intent)
        now = time.time()

        logger.info(
            f"Scheduling follow-up: {followup_id} for booking={booking_id} "
            f"urgency={urgency} language={language}"
        )

        # ── Parse the scheduled appointment time ────────────────
        scheduled_time = booking.get("scheduled_time", "")
        appointment_ts = self._parse_scheduled_time(scheduled_time, now)
        minutes_until = max(0, (appointment_ts - now) / 60)

        # ── DETERMINISTIC: Compute reminder schedule ────────────
        reminders = self._compute_reminder_schedule(
            booking=booking,
            intent=intent,
            appointment_ts=appointment_ts,
            urgency=urgency,
            now=now,
        )

        # ── DETERMINISTIC: Compute status update timeline ───────
        timeline = self._compute_status_timeline(
            booking=booking,
            appointment_ts=appointment_ts,
            now=now,
        )

        # ── DETERMINISTIC: Compute post-completion actions ──────
        post_actions = self._compute_post_completion_actions(
            booking=booking,
            appointment_ts=appointment_ts,
        )

        # ── PROBABILISTIC: Generate immediate notification text ─
        immediate = await self._generate_immediate_notification(
            booking=booking,
            intent=intent,
            language=language,
        )

        # ── PROBABILISTIC: Generate text for first reminder ─────
        # (We generate text for the first reminder eagerly; subsequent
        #  ones will be generated on-demand when the mobile app polls)
        if reminders:
            first_reminder_text = await self._generate_reminder_text(
                booking=booking,
                intent=intent,
                tier=reminders[0],
                language=language,
            )
            reminders[0]["notification_text"] = first_reminder_text

        total = len(reminders) + len(timeline) + len(post_actions)

        result = {
            "followup_id": followup_id,
            "booking_id": booking_id,
            "scheduled_reminders": reminders,
            "status_timeline": timeline,
            "post_completion_actions": post_actions,
            "immediate_notification": immediate,
            "total_scheduled": total,
            "metadata": {
                "urgency": urgency,
                "language": language,
                "minutes_until_appointment": round(minutes_until, 1),
                "appointment_timestamp": appointment_ts,
                "created_at": now,
            },
        }

        logger.info(
            f"Follow-up plan created: {followup_id} | "
            f"{len(reminders)} reminders | {len(timeline)} status events | "
            f"{len(post_actions)} post-completion actions"
        )

        return result

    # ═══════════════════════════════════════════════════════════
    # PUBLIC: On-demand notification generation (for API endpoints)
    # ═══════════════════════════════════════════════════════════

    async def generate_reminder(
        self,
        booking_id: str,
        tier_name: str = "final_reminder",
        language: str = "English",
    ) -> Optional[dict]:
        """
        Generate a reminder notification for a specific booking and tier.

        Called by the /followup/reminders/{booking_id} API endpoint.
        The mobile app polls this to get fresh notification text.
        """
        booking = self.booking_store.get_booking(booking_id)
        if not booking:
            logger.warning(f"generate_reminder: booking {booking_id} not found")
            return None

        # Find the matching tier config
        tier_config = None
        for t in self.reminder_tiers:
            if t.get("tier") == tier_name:
                tier_config = t
                break
        if not tier_config:
            tier_config = self.reminder_tiers[-1] if self.reminder_tiers else {
                "tier": tier_name, "priority": "normal", "channel": "push", "icon": "🔔"
            }

        text = await self._generate_reminder_text(
            booking=booking,
            intent={"urgency": "normal", "language_detected": language.lower()},
            tier={"tier_name": tier_name, **tier_config},
            language=language,
        )

        return {
            "booking_id": booking_id,
            "tier": tier_name,
            "notification": text,
            "channel": tier_config.get("channel", "push"),
            "priority": tier_config.get("priority", "normal"),
        }

    async def generate_status_notification(
        self,
        booking_id: str,
        event_type: str,
        language: str = "English",
    ) -> Optional[dict]:
        """
        Generate a status update notification on demand.

        Called by the /followup/status/{booking_id} API endpoint.
        """
        booking = self.booking_store.get_booking(booking_id)
        if not booking:
            return None

        # Find matching status event config
        event_config = None
        for evt in self.status_updates:
            if evt.get("event") == event_type:
                event_config = evt
                break
        if not event_config:
            event_config = {
                "event": event_type, "status": "IN_PROGRESS",
                "icon": "📋", "message_key": event_type,
                "description": f"Status update: {event_type}",
            }

        text = await self._generate_status_text(
            booking=booking, event=event_config, language=language,
        )

        return {
            "booking_id": booking_id,
            "event_type": event_type,
            "new_status": event_config.get("status"),
            "icon": event_config.get("icon"),
            "notification_text": text,
            "action_buttons": self._get_action_buttons_for_status(event_config.get("status", "")),
        }

    async def check_status(self, booking_id: str) -> Optional[dict]:
        """Get booking status with enriched timeline metadata."""
        booking = self.booking_store.get_booking(booking_id)
        if not booking:
            return None
        # Enrich with follow-up metadata
        booking["followup_available"] = True
        booking["available_actions"] = self._get_action_buttons_for_status(
            booking.get("status", "CONFIRMED")
        )
        return booking

    async def complete_booking(self, booking_id: str) -> bool:
        """Mark a booking as completed."""
        return self.booking_store.update_status(
            booking_id, "COMPLETED", "Service completed"
        )

    # ═══════════════════════════════════════════════════════════
    # PRIVATE: Deterministic scheduling logic
    # ═══════════════════════════════════════════════════════════

    def _compute_reminder_schedule(
        self,
        booking: dict,
        intent: dict,
        appointment_ts: float,
        urgency: str,
        now: float,
    ) -> list[dict]:
        """
        Compute which reminder tiers to schedule based on config.

        DETERMINISTIC: Same (config, booking, urgency) → same schedule.
        The urgency multiplier compresses/expands the timeline:
          urgent  → reminders fire 4x sooner (multiplier 0.25)
          normal  → use configured values (multiplier 1.0)
          flexible → space reminders out (multiplier 1.5)
        """
        multiplier = self.urgency_multipliers.get(urgency, 1.0)
        minutes_until = max(0, (appointment_ts - now) / 60)

        reminders = []
        for tier in self.reminder_tiers:
            # Apply urgency multiplier to the configured minutes_before
            adjusted_minutes = tier["minutes_before"] * multiplier

            # Only schedule if the appointment is far enough away
            if minutes_until >= adjusted_minutes:
                fire_at = appointment_ts - (adjusted_minutes * 60)
                channel = tier.get("channel", "push")

                # Check if channel is enabled in config
                channel_cfg = self.channels_config.get(channel, {})
                if not channel_cfg.get("enabled", True):
                    channel = "in_app"  # fallback to in_app

                reminders.append({
                    "reminder_id": f"REM-{uuid.uuid4().hex[:6].upper()}",
                    "tier_name": tier["tier"],
                    "fire_at_utc": fire_at,
                    "fire_at_iso": datetime.datetime.fromtimestamp(
                        fire_at, tz=datetime.timezone.utc
                    ).isoformat(),
                    "minutes_before_appointment": round(adjusted_minutes, 1),
                    "priority": tier.get("priority", "normal"),
                    "channel": channel,
                    "icon": tier.get("icon", "🔔"),
                    "description": tier.get("description", ""),
                    "action_buttons": ["view_booking", "call_provider", "reschedule"],
                    # notification_text will be filled by LLM (probabilistic)
                    "notification_text": None,
                })

        # Sort by fire_at ascending (earliest first)
        reminders.sort(key=lambda r: r["fire_at_utc"])

        logger.info(
            f"Computed {len(reminders)} reminders "
            f"(urgency={urgency}, multiplier={multiplier}, "
            f"minutes_until={minutes_until:.0f})"
        )
        return reminders

    def _compute_status_timeline(
        self,
        booking: dict,
        appointment_ts: float,
        now: float,
    ) -> list[dict]:
        """
        Build the status update timeline from config.

        DETERMINISTIC: Each event maps to a fixed offset from
        booking creation or appointment time.
        """
        booking_created = booking.get("created_at", now)
        timeline = []

        for event in self.status_updates:
            event_name = event["event"]

            # Compute fire time based on which offset key is present
            if "delay_minutes" in event:
                fire_at = booking_created + (event["delay_minutes"] * 60)
            elif "delay_minutes_before_appointment" in event:
                fire_at = appointment_ts - (event["delay_minutes_before_appointment"] * 60)
            elif "delay_minutes_after_appointment" in event:
                fire_at = appointment_ts + (event["delay_minutes_after_appointment"] * 60)
            else:
                fire_at = booking_created + 300  # default 5 min after booking

            timeline.append({
                "event_id": f"EVT-{uuid.uuid4().hex[:6].upper()}",
                "event_type": event_name,
                "fire_at_utc": fire_at,
                "fire_at_iso": datetime.datetime.fromtimestamp(
                    fire_at, tz=datetime.timezone.utc
                ).isoformat(),
                "status_transition": event.get("status", "IN_PROGRESS"),
                "icon": event.get("icon", "📋"),
                "description": event.get("description", event_name),
                "message_key": event.get("message_key", event_name),
                "action_buttons": self._get_action_buttons_for_status(
                    event.get("status", "IN_PROGRESS")
                ),
            })

        # Sort by fire_at ascending
        timeline.sort(key=lambda e: e["fire_at_utc"])
        return timeline

    def _compute_post_completion_actions(
        self,
        booking: dict,
        appointment_ts: float,
    ) -> list[dict]:
        """
        Build post-completion action schedule from config.

        DETERMINISTIC: Rating prompt and re-booking suggestion
        are scheduled at fixed offsets after the appointment.
        """
        # Estimate completion time (appointment + 1 hour for service)
        estimated_completion = appointment_ts + 3600  # 1 hour for service

        actions = []

        # Rating prompt
        rating_delay = self.post_completion.get("rating_prompt_delay_minutes", 5)
        rating_fire = estimated_completion + (rating_delay * 60)
        actions.append({
            "action_id": f"ACT-{uuid.uuid4().hex[:6].upper()}",
            "action_type": "rating_prompt",
            "fire_at_utc": rating_fire,
            "fire_at_iso": datetime.datetime.fromtimestamp(
                rating_fire, tz=datetime.timezone.utc
            ).isoformat(),
            "icon": self.post_completion.get("rating_prompt_icon", "⭐"),
            "channel": "push",
            "priority": "normal",
            "action_buttons": ["rate_service", "view_booking"],
        })

        # Re-booking suggestion
        rebook_delay_hours = self.post_completion.get(
            "rebooking_suggestion_delay_hours", 168
        )
        rebook_fire = estimated_completion + (rebook_delay_hours * 3600)
        actions.append({
            "action_id": f"ACT-{uuid.uuid4().hex[:6].upper()}",
            "action_type": "rebooking_suggestion",
            "fire_at_utc": rebook_fire,
            "fire_at_iso": datetime.datetime.fromtimestamp(
                rebook_fire, tz=datetime.timezone.utc
            ).isoformat(),
            "icon": self.post_completion.get("rebooking_suggestion_icon", "🔁"),
            "channel": "push",
            "priority": "low",
            "action_buttons": ["view_booking"],
        })

        return actions

    # ═══════════════════════════════════════════════════════════
    # PRIVATE: Probabilistic LLM text generation
    # ═══════════════════════════════════════════════════════════

    async def _generate_immediate_notification(
        self,
        booking: dict,
        intent: dict,
        language: str,
    ) -> dict:
        """
        Generate the immediate 'booking confirmed' notification.

        This is shown to the user right away and also stored as
        the first entry in the notification timeline.
        """
        tone = self.reasoning_directives.get("tone", "warm, professional, concise")
        channel_cfg = self.channels_config.get("push", {})

        prompt_template = self.prompts_config.get(
            "followup_notification",
            "Generate a booking confirmation notification for {booking_id}.",
        )

        # Determine time-of-day context for personalization
        now_hour = datetime.datetime.now().hour
        if now_hour < 12:
            time_context = "morning"
        elif now_hour < 17:
            time_context = "afternoon"
        elif now_hour < 21:
            time_context = "evening"
        else:
            time_context = "night"

        try:
            prompt = prompt_template.format(
                booking_id=booking.get("booking_id", ""),
                service_type=booking.get("service_type", ""),
                provider_name=booking.get("provider_name", ""),
                location=booking.get("location", ""),
                scheduled_time=booking.get("scheduled_time", ""),
                event_type="booking_confirmed",
                event_description="Your booking has been confirmed and the provider has been notified",
                language=language,
                urgency=intent.get("urgency", "normal"),
                service_category=booking.get("service_type", ""),
                time_context=time_context,
                tone=tone,
                personalization_signals=", ".join(
                    self.reasoning_directives.get("personalization_signals", [])
                ),
                max_title_length=channel_cfg.get("max_title_length", 50),
                max_body_length=channel_cfg.get("max_body_length", 200),
            )

            system_instruction = (
                "You are a notification text generator for a mobile service booking app. "
                "Generate concise, actionable notification content. Respond in valid JSON only."
            )
            if self.guardrails:
                system_instruction = self.guardrails.compose_system_instruction("followup", system_instruction)

            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=system_instruction,
            )

            # Parse LLM response
            notif = self._parse_notification_json(response.text)

        except Exception as e:
            logger.warning(f"LLM notification generation failed: {e}. Using fallback.")
            notif = self._fallback_notification(booking, "booking_confirmed", language)

        # Build full notification payload
        payload = {
            "notification_id": f"NOTIF-{uuid.uuid4().hex[:6].upper()}",
            "type": "booking_confirmed",
            "channel": "push",
            "priority": "high",
            "icon": "✅",
            "title": notif.get("title", f"Booking Confirmed — {booking.get('service_type', 'Service')}"),
            "body": notif.get("body", f"Your {booking.get('service_type', 'service')} booking with {booking.get('provider_name', 'provider')} is confirmed."),
            "reasoning": notif.get("reasoning", "Standard confirmation notification"),
            "sound": channel_cfg.get("sound", "default"),
            "vibrate": channel_cfg.get("vibrate", True),
            "action_buttons": [
                {"action": "view_booking", "label": "View Booking", "icon": "📋"},
                {"action": "call_provider", "label": "Call Provider", "icon": "📞"},
                {"action": "cancel", "label": "Cancel", "icon": "❌"},
            ],
            "data": {
                "booking_id": booking.get("booking_id"),
                "provider_phone": booking.get("provider_phone"),
                "deep_link": f"/bookings/{booking.get('booking_id')}",
            },
            "created_at": time.time(),
        }
        return self.guardrails.sanitize_payload(payload) if self.guardrails else payload

    async def _generate_reminder_text(
        self,
        booking: dict,
        intent: dict,
        tier: dict,
        language: str,
    ) -> dict:
        """Generate LLM-reasoned notification text for a reminder tier."""
        tone = self.reasoning_directives.get("tone", "warm, professional, concise")
        tier_name = tier.get("tier_name", tier.get("tier", "reminder"))

        prompt_template = self.prompts_config.get(
            "followup_reminder",
            "Reminder for {booking_id}: {service_type} with {provider_name} at {scheduled_time}.",
        )

        try:
            prompt = prompt_template.format(
                booking_id=booking.get("booking_id", ""),
                service_type=booking.get("service_type", ""),
                provider_name=booking.get("provider_name", ""),
                scheduled_time=booking.get("scheduled_time", ""),
                time_until=f"{tier.get('minutes_before_appointment', 60):.0f} minutes",
                reminder_tier=tier_name,
                urgency=intent.get("urgency", "normal"),
                language=language,
                tone=tone,
            )

            system_instruction = "Generate a brief, friendly reminder notification. Keep it under 200 characters."
            if self.guardrails:
                system_instruction = self.guardrails.compose_system_instruction("followup", system_instruction)

            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=system_instruction,
            )

            payload = {
                "title": f"{tier.get('icon', '🔔')} Reminder — {booking.get('service_type', 'Service')}",
                "body": response.text.strip(),
                "tier": tier_name,
            }
            return self.guardrails.sanitize_payload(payload) if self.guardrails else payload
        except Exception as e:
            logger.warning(f"LLM reminder generation failed: {e}. Using fallback.")
            payload = {
                "title": f"{tier.get('icon', '🔔')} Reminder",
                "body": f"Your {booking.get('service_type', 'service')} with {booking.get('provider_name', '')} is in {tier.get('minutes_before_appointment', 60):.0f} minutes.",
                "tier": tier_name,
            }
            return self.guardrails.sanitize_payload(payload) if self.guardrails else payload

    async def _generate_status_text(
        self,
        booking: dict,
        event: dict,
        language: str,
    ) -> str:
        """Generate LLM-reasoned notification text for a status update."""
        tone = self.reasoning_directives.get("tone", "warm, professional, concise")
        prompt_template = self.prompts_config.get(
            "followup_status_update",
            "Status update for {booking_id}: {event_type}.",
        )

        try:
            prompt = prompt_template.format(
                booking_id=booking.get("booking_id", ""),
                service_type=booking.get("service_type", ""),
                provider_name=booking.get("provider_name", ""),
                event_type=event.get("event", "update"),
                event_description=event.get("description", "Status changed"),
                new_status=event.get("status", "IN_PROGRESS"),
                provider_phone=booking.get("provider_phone", "N/A"),
                language=language,
                tone=tone,
            )

            system_instruction = "Generate a concise status update notification. Max 200 characters."
            if self.guardrails:
                system_instruction = self.guardrails.compose_system_instruction("followup", system_instruction)

            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=system_instruction,
            )
            return self.guardrails.sanitize_text(response.text.strip()) if self.guardrails else response.text.strip()
        except Exception as e:
            logger.warning(f"LLM status text failed: {e}. Using fallback.")
            text = f"{event.get('icon', '📋')} {event.get('description', 'Status updated')} — {booking.get('provider_name', 'Provider')}"
            return self.guardrails.sanitize_text(text) if self.guardrails else text

    # ═══════════════════════════════════════════════════════════
    # PRIVATE: Utility methods
    # ═══════════════════════════════════════════════════════════

    def _parse_scheduled_time(self, time_str: str, fallback_now: float) -> float:
        """Parse the scheduled time string into a Unix timestamp."""
        # Try common datetime formats
        formats = [
            "%Y-%m-%d %I:%M %p",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(time_str.strip(), fmt)
                return dt.timestamp()
            except (ValueError, AttributeError):
                continue

        # Fallback: assume 2 hours from now
        logger.warning(f"Could not parse scheduled_time '{time_str}'. Using now + 2h.")
        return fallback_now + 7200

    def _detect_response_language(self, intent: dict) -> str:
        """Map detected language to the response language string."""
        lang = intent.get("language_detected", "english")
        if lang.lower() in ("roman urdu", "urdu"):
            return "Roman Urdu (romanized Urdu in Latin script)"
        return "English"

    def _parse_notification_json(self, text: str) -> dict:
        """Extract JSON from LLM response (handles markdown wrapping)."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return {}

    def _fallback_notification(self, booking: dict, event_type: str, language: str) -> dict:
        """Generate a deterministic fallback notification when LLM fails."""
        return {
            "title": f"Booking Update — {booking.get('service_type', 'Service')}",
            "body": f"Your booking {booking.get('booking_id', '')} with {booking.get('provider_name', 'your provider')} has been updated.",
            "reasoning": "Fallback notification — LLM was unavailable",
        }

    def _get_action_buttons_for_status(self, status: str) -> list[dict]:
        """
        Return context-appropriate action buttons based on booking status.

        The button set changes as the booking progresses through its lifecycle.
        This is deterministic — same status always yields same buttons.
        """
        # Read allowed actions from config, with sensible defaults
        all_actions = self.reasoning_directives.get("button_actions", [
            "view_booking", "call_provider", "reschedule", "cancel", "rate_service",
        ])

        button_defs = {
            "view_booking": {"action": "view_booking", "label": "View Booking", "icon": "📋"},
            "call_provider": {"action": "call_provider", "label": "Call Provider", "icon": "📞"},
            "reschedule": {"action": "reschedule", "label": "Reschedule", "icon": "🔄"},
            "cancel": {"action": "cancel", "label": "Cancel Booking", "icon": "❌"},
            "rate_service": {"action": "rate_service", "label": "Rate Service", "icon": "⭐"},
        }

        # Status → which actions are available (deterministic mapping)
        status_actions = {
            "PENDING": ["view_booking", "call_provider", "cancel"],
            "CONFIRMED": ["view_booking", "call_provider", "reschedule", "cancel"],
            "IN_PROGRESS": ["view_booking", "call_provider"],
            "COMPLETED": ["view_booking", "rate_service"],
            "CANCELLED": ["view_booking"],
        }

        allowed = status_actions.get(status, ["view_booking"])
        return [
            button_defs[a]
            for a in allowed
            if a in button_defs and a in all_actions
        ]
