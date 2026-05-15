"""
Response Parser — Structured output parsing.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger("inference.parser")


def parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning(f"Failed to parse JSON from: {text[:200]}")
    return None
