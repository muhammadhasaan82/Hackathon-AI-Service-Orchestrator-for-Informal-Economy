"""
Data Preparation — Converts service booking scenarios to SFT training format.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fine_tuning.data_prep")


# Sample training conversations for informal economy domain
TRAINING_EXAMPLES = [
    {
        "instruction": "User says: 'Mujhe kal subah G-13 mein AC technician chahiye'",
        "response": '{"service_type": "AC Technician", "city": "Islamabad", "area": "G-13", "time_preference": "kal subah", "urgency": "normal", "language_detected": "roman_urdu", "confidence": {"service_type": 0.95, "city": 0.90, "area": 0.95}}'
    },
    {
        "instruction": "User says: 'plumber chahiye abhi Gulberg Lahore mein, pipe leak ho raha hai'",
        "response": '{"service_type": "Plumber", "city": "Lahore", "area": "Gulberg", "time_preference": "abhi", "urgency": "urgent", "language_detected": "roman_urdu", "confidence": {"service_type": 0.98, "city": 0.95, "area": 0.95}}'
    },
    {
        "instruction": "User says: 'I need a good electrician in DHA Karachi tomorrow evening'",
        "response": '{"service_type": "Electrician", "city": "Karachi", "area": "DHA", "time_preference": "tomorrow evening", "urgency": "normal", "language_detected": "english", "confidence": {"service_type": 0.95, "city": 0.95, "area": 0.90}}'
    },
    {
        "instruction": "User says: 'sasta wala painter chahiye Faisalabad mein'",
        "response": '{"service_type": "Painter", "city": "Faisalabad", "area": null, "time_preference": null, "urgency": "normal", "language_detected": "roman_urdu", "price_preference": "budget", "confidence": {"service_type": 0.90, "city": 0.95}}'
    },
    {
        "instruction": "User says: 'beautician for bridal makeup in Model Town Lahore next Saturday'",
        "response": '{"service_type": "Beautician", "city": "Lahore", "area": "Model Town", "time_preference": "next Saturday", "urgency": "normal", "language_detected": "english", "confidence": {"service_type": 0.95, "city": 0.95, "area": 0.95}}'
    },
    {
        "instruction": "User says: 'computer repair wala chahiye, laptop ki screen broken hai Rawalpindi mein'",
        "response": '{"service_type": "Computer Technician", "city": "Rawalpindi", "area": null, "time_preference": null, "urgency": "normal", "language_detected": "roman_urdu", "confidence": {"service_type": 0.85, "city": 0.95}}'
    },
    {
        "instruction": "User says: 'mobile repair nazdeek F-10'",
        "response": '{"service_type": "Mobile Repair", "city": "Islamabad", "area": "F-10", "time_preference": null, "urgency": "normal", "language_detected": "roman_urdu", "price_preference": null, "confidence": {"service_type": 0.90, "city": 0.80, "area": 0.90}}'
    },
    {
        "instruction": "User says: 'home tutor for class 10 math Peshawar'",
        "response": '{"service_type": "Home Tutor", "city": "Peshawar", "area": null, "time_preference": null, "urgency": "normal", "language_detected": "english", "confidence": {"service_type": 0.90, "city": 0.95}}'
    },
]


def generate_training_data(
    output_path: Optional[str] = None,
    examples: Optional[list[dict]] = None,
) -> str:
    """
    Generate JSONL training data file.

    Each entry is an instruction-response pair formatted for SFT.
    """
    data = examples or TRAINING_EXAMPLES
    out_path = output_path or str(
        Path(__file__).parent.parent.parent / "data" / "fine_tuned" / "training_data.jsonl"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for example in data:
            entry = {
                "instruction": example["instruction"],
                "input": "",
                "output": example["response"],
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Generated {len(data)} training examples → {out_path}")
    return out_path
