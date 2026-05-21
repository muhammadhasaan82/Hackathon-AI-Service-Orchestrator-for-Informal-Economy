import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_urdu_roman_fixes import (
    test_roman_urdu_response_translation,
    test_urdu_intent_extraction_plumber,
    test_roman_urdu_intent_extraction_plumber
)

def run():
    print("==================================================")
    print("Running Urdu/Roman Urdu Response Translation test...")
    test_roman_urdu_response_translation()
    print("SUCCESS: Response translation test PASSED!")
    print("--------------------------------------------------")

    print("Running Urdu Intent Extraction test...")
    asyncio.run(test_urdu_intent_extraction_plumber())
    print("SUCCESS: Urdu intent extraction test PASSED!")
    print("--------------------------------------------------")

    print("Running Roman Urdu Intent Extraction test...")
    asyncio.run(test_roman_urdu_intent_extraction_plumber())
    print("SUCCESS: Roman Urdu intent extraction test PASSED!")
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run()
