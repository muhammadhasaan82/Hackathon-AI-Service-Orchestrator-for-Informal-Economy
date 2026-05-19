import copy
import re
from typing import Optional


class GuardrailEngine:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.input_config = self.config.get("input", {})
        self.intent_config = self.config.get("intent", {})
        self.booking_config = self.config.get("booking", {})
        self.output_config = self.config.get("output", {})
        self.llm_instructions = self.config.get("llm_instructions", {})

    def evaluate_input(self, user_message: str) -> Optional[dict]:
        message = (user_message or "").strip()
        min_chars = self.input_config.get("min_message_chars", 1)
        max_chars = self.input_config.get("max_message_chars", 2000)
        messages = self.input_config.get("messages", {})

        if len(message) < min_chars:
            return {
                "status": "needs_clarification",
                "action": "clarify",
                "message": messages.get("too_short", "Please provide more details."),
                "policy": "message_length_min",
            }

        if len(message) > max_chars:
            return {
                "status": "input_rejected",
                "action": "block",
                "message": messages.get("too_long", "Please send a shorter message."),
                "policy": "message_length_max",
            }

        for name, policy in self.input_config.get("policies", {}).items():
            if not policy.get("enabled", False):
                continue
            for pattern in policy.get("patterns", []):
                if re.search(pattern, message):
                    action = policy.get("action", "block")
                    status = {
                        "block": "input_rejected",
                        "redirect": "needs_clarification",
                        "escalate": "human_handoff_recommended",
                    }.get(action, "input_rejected")
                    return {
                        "status": status,
                        "action": action,
                        "message": policy.get("message", messages.get("too_short", "Please provide more details.")),
                        "policy": name,
                    }
        return None

    def compose_system_instruction(self, scope: str, base_instruction: Optional[str] = None) -> Optional[str]:
        lines = []
        if base_instruction:
            lines.append(base_instruction.strip())
        shared = self.llm_instructions.get("shared", [])
        scoped = self.llm_instructions.get(scope, [])
        directives = [line for line in [*shared, *scoped] if line]
        if directives:
            lines.append("Runtime guardrails:\n" + "\n".join(f"- {line}" for line in directives))
        final_text = "\n\n".join(part for part in lines if part)
        return final_text or None

    def evaluate_intent(self, intent: Optional[dict]) -> dict:
        payload = intent or {}
        required_fields = self.intent_config.get("required_fields", [])
        threshold = self.intent_config.get("confidence_threshold", 0.65)
        escalation_threshold = self.intent_config.get("escalation_threshold", 0.35)
        confidence = payload.get("confidence", {}) or {}

        missing_fields = [field for field in required_fields if not payload.get(field)]
        low_confidence_fields = []
        escalation_fields = []

        for field in required_fields:
            score = self._coerce_float(confidence.get(field))
            if score is None:
                continue
            if score < threshold:
                low_confidence_fields.append(field)
            if score < escalation_threshold:
                escalation_fields.append(field)

        needs_clarification = []
        for field in payload.get("needs_clarification", []) or []:
            if field not in needs_clarification:
                needs_clarification.append(field)
        for field in [*missing_fields, *low_confidence_fields]:
            if field not in needs_clarification:
                needs_clarification.append(field)

        return {
            "missing_fields": missing_fields,
            "low_confidence_fields": low_confidence_fields,
            "needs_clarification": needs_clarification,
            "should_escalate": bool(escalation_fields) and len(escalation_fields) == len(required_fields),
        }

    def booking_mode(self) -> str:
        return self.booking_config.get("mode", "auto_book")

    def booking_status(self, key: str, default: str) -> str:
        return self.booking_config.get("statuses", {}).get(key, default)

    def min_top_provider_score(self) -> float:
        return float(self.booking_config.get("min_top_provider_score", 0.0))

    def confirmation_prompt(self, provider_count: int, top_score: Optional[float] = None) -> str:
        messages = self.booking_config.get("messages", {})
        if top_score is not None and top_score < self.min_top_provider_score():
            template = messages.get(
                "low_score_confirmation",
                "I found a few possible matches. Reply with the option number you want to book, or share more details so I can refine the search.",
            )
        else:
            template = messages.get(
                "confirmation_required",
                "Reply with the option number you want to book.",
            )
        return template.format(provider_count=provider_count)

    def rejection_message(self) -> str:
        return self.booking_config.get("messages", {}).get(
            "rejection_acknowledged",
            "No problem. I can refine the search if you want.",
        )

    def missing_selection_message(self) -> str:
        return self.booking_config.get("messages", {}).get(
            "missing_previous_selection",
            "I need to shortlist providers before I can confirm a booking.",
        )

    def resolve_booking_confirmation(self, user_message: str, ranked_providers: Optional[list[dict]]) -> dict:
        text = (user_message or "").strip()
        providers = ranked_providers or []

        if not providers:
            return {"action": "missing_context"}

        if self._matches_any(text, self.booking_config.get("rejection_patterns", [])):
            return {"action": "reject"}

        confirmation_signal = self._matches_any(text, self.booking_config.get("confirmation_patterns", []))

        if text.isdigit():
            index = int(text) - 1
            if 0 <= index < len(providers):
                return {
                    "action": "confirm",
                    "provider": providers[index],
                    "selected_index": index,
                    "selection_type": "simple_numeric",
                }

        numeric_pattern = self.booking_config.get("selection_patterns", {}).get("numeric")
        if numeric_pattern:
            match = re.search(numeric_pattern, text)
            if match:
                index = int(match.group("index")) - 1
                if 0 <= index < len(providers):
                    return {
                        "action": "confirm",
                        "provider": providers[index],
                        "selected_index": index,
                        "selection_type": "numeric",
                    }

        if confirmation_signal:
            simple_number_match = re.search(r"\b([1-5])\b", text)
            if simple_number_match:
                index = int(simple_number_match.group(1)) - 1
                if 0 <= index < len(providers):
                    return {
                        "action": "confirm",
                        "provider": providers[index],
                        "selected_index": index,
                        "selection_type": "confirmation_numeric",
                    }

        ordinals = self.booking_config.get("selection_patterns", {}).get("ordinals", {})
        lowered = text.lower()
        for key, terms in ordinals.items():
            if any(term.lower() in lowered for term in terms):
                index = int(key) - 1
                if 0 <= index < len(providers):
                    return {
                        "action": "confirm",
                        "provider": providers[index],
                        "selected_index": index,
                        "selection_type": "ordinal",
                    }

        for index, provider in enumerate(providers):
            name = provider.get("metadata", provider).get("provider_name", "")
            if name and name.lower() in lowered:
                return {
                    "action": "confirm",
                    "provider": provider,
                    "selected_index": index,
                    "selection_type": "provider_name",
                }

        if confirmation_signal and len(providers) == 1:
            return {
                "action": "confirm",
                "provider": providers[0],
                "selected_index": 0,
                "selection_type": "single_option",
            }

        if confirmation_signal:
            return {"action": "confirm_ambiguous"}

        return {"action": "none"}

    def sanitize_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        sanitized = text
        replacement = self.output_config.get("redact_with", "[redacted]")
        for pattern in self.output_config.get("redact_patterns", []):
            sanitized = re.sub(pattern, replacement, sanitized)
        max_chars = self.output_config.get("max_response_chars")
        suffix = self.output_config.get("truncation_suffix", "...")
        if max_chars and len(sanitized) > max_chars:
            cutoff = max(0, max_chars - len(suffix))
            sanitized = sanitized[:cutoff].rstrip() + suffix
        return sanitized.strip()

    def sanitize_payload(self, payload):
        cloned = copy.deepcopy(payload)
        return self._sanitize_value(cloned)

    def _sanitize_value(self, value):
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize_value(item) for key, item in value.items()}
        return value

    def _matches_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def _coerce_float(self, value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
