import re
from typing import Dict, Any, Optional

# Supported service categories from the dataset
SUPPORTED_CATEGORIES = [
    "AC Technician", "Appliance Repair", "Beautician", "Carpenter", 
    "Cleaning Service", "Computer Technician", "Electrician", 
    "Home Tutor", "Mechanic", "Mobile Repair", "Painter", 
    "Plumber", "Tutor", "Water Tank Cleaner"
]

# Lowercase supported categories for exact/partial matching
SUPPORTED_CATEGORIES_LOWER = [c.lower() for c in SUPPORTED_CATEGORIES]

# Common unsupported informal services to explicitly capture and reject
UNSUPPORTED_SERVICES = {
    # English
    "photographer", "photography", "photo", "barber", "driver", "driving", "tailor", "stitching", "darzi", 
    "chef", "cook", "cooking", "gardener", "gardening", "guard", "security", "laundry", "dhobi", 
    "massage", "haircut", "makeup artist", "nail", "dentist", "doctor", "lawyer", "advocate", 
    "accountant", "carpenter helper", "delivery", "courier", "cleaning lady", "maid", "nanny", 
    "babysitter", "dog walker", "pet groomer", "mover", "packers", "electrician helper",
    # Roman Urdu
    "bawarchi", "mali", "choukidar", "chowkidar", "maalish", "malish", "hajaam", "hajam", "wakeel",
    # Urdu script
    "فوٹوگرافر", "ڈرائیور", "درزی", "باورچی", "مالی", "چوکیدار", "دھوبی", "حجام", "نائی", "وکیل", "ڈاکٹر"
}

# Unsafe categories and pattern regex lists
UNSAFE_POLICIES = {
    "weapons_guns": [
        r"\bweapons?\b", r"\bguns?\b", r"\bpistols?\b", r"\bbullets?\b", r"\bammunitions?\b", 
        r"\brifles?\b", r"\bbombs?\b", r"\bexplosives?\b", r"\bfirearms?\b", r"ak-47", r"kalashnikov",
        r"\bbandook\b", r"\bbandooq\b", r"\bgoli\b", r"\bgolian\b", r"\baslaha\b", r"\bhatyar\b", r"\bhathyar\b",
        r"بندوق", r"پستول", r"گولی", r"اسلحہ", r"ہتھیار"
    ],
    "drugs_substances": [
        r"\bdrugs?\b", r"\bcocaine\b", r"\bheroin\b", r"\bmarijuana\b", r"\bweed\b", r"\bmeth\b", 
        r"\bnarcotics?\b", r"\becstasy\b", r"\bcannabis\b", r"\billegal substance\b",
        r"\bcharas\b", r"\bafeem\b", r"\bsharaab\b", r"\bsharab\b", r"\bnasha\b", r"\bnasheeli\b",
        r"چرس", r"افیون", r"شراب", r"نشہ"
    ],
    "hacking_cyber": [
        r"\bhack\b", r"\bhacking\b", r"\bcyberattacks?\b", r"cyber\s+attacks?", r"\bddos\b", 
        r"\bphishing\b", r"\bmalware\b", r"\btrojan\b", r"\bspyware\b", r"bypass\s+password", r"crack\s+software",
        r"hack\s+karna", r"password\s+chori", r"cyber\s+hamla",
        r"ہیک"
    ],
    "fraud_scams": [
        r"fake\s+degrees?", r"fake\s+passports?", r"fake\s+documents?", r"fake\s+ids?", r"\bscams?\b", 
        r"\bfraud\b", r"\bcounterfeits?\b", r"\bforgery\b", r"identity\s+theft", r"money\s+laundering",
        r"\bjaali\b", r"\bjali\b", r"\bdhoka\b",
        r"جعلی", r"دھوکہ"
    ],
    "violence_harm": [
        r"\bkill\b", r"\bmurder\b", r"\bsuicide\b", r"\babuses?\b", r"\bassaults?\b", r"\bharm\b", 
        r"beat\s+up", r"\bfights?\b", r"\bthreaten\b", r"\bextortion\b",
        r"\bmaar\b", r"\bqatl\b", r"\bkhoon\b", r"\bmarna\b", r"\bmaarna\b", r"\bkhudkushi\b", r"\bdhamki\b",
        r"قتل", r"خودکشی", r"دھمکی", r"مارنا"
    ],
    "sexual_adult": [
        r"\bsex\b", r"\bporn\b", r"adult\s+services?", r"\bprostitutes?\b", r"\bescorts?\b", 
        r"sensual\s+massage", r"\bstrip\b", r"\bbrothels?\b",
        r"\bjinsi\b", r"\bfaashi\b", r"\bfashi\b",
        r"جنسی", r"فحش"
    ],
    "medical_emergency": [
        r"heart\s+attacks?", r"\bstrokes?\b", r"medical\s+emergenc(y|ies)", r"emergency\s+doctors?", 
        r"\bambulances?\b", r"\bcpr\b", r"bleeding\s+out", r"\bpoisoned\b", r"\bchoking\b", r"chest\s+pain",
        r"dil\s+ka\s+daura", r"dil\s+ka\s+dora",
        r"ایمبولینس", r"دل\s+کا\s+دورہ"
    ]
}

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"reveal\s+system\s+prompt",
    r"show\s+hidden\s+prompt",
    r"bypass\s+polic(y|ies)",
    r"developer\s+messages?",
    r"system\s+messages?",
    r"ignore\s+policies",
    r"reveal\s+instructions",
    r"what\s+is\s+your\s+system\s+prompt",
    r"show\s+system\s+prompt",
    r"reveal\s+prompt",
    r"instruction\s+ignore",
    r"system\s+prompt\s+dikhao",
    r"prompt\s+bypass"
]

# Privacy invasion patterns
PRIVACY_PATTERNS = [
    r"reveal\s+passwords?",
    r"get\s+user\s+history\s+of",
    r"other\s+user'?s?\s+phone",
    r"hack\s+account\s+of",
    r"private\s+details\s+of",
    r"expose\s+user\s+details",
    r"social\s+security\s+number",
    r"\bssn\b",
    r"credit\s+card\s+number",
    r"apna\s+system\s+details",
    r"shinaxti\s+card",
    r"nic\s+number"
]

# Out of scope patterns (general random tasks completely unrelated to booking/platforms)
OUT_OF_SCOPE_PATTERNS = [
    r"write\s+a?\s*python\s+code", r"write\s+a?\s*java\s+code", r"programming", r"weather\s+in",
    r"\bpolitics\b", r"\bcricket\b", r"\bsports\b", r"\bnews\b", r"\bmovie\b", r"\bsong\b",
    r"buy\s+a?\s*cars?", r"math\s+problems?", r"write\s+an?\s*essay", r"\btranslate\b",
    r"who\s+is\s+the\s+prime\s+minister", r"who\s+is\s+the\s+president"
]

# Language-specific refusal messages
REFUSALS = {
    "unsafe_illegal": {
        "english": "I cannot assist with this request as it involves unsafe, illegal, or hazardous activities.",
        "roman_urdu": "Main is request mein madad nahi kar sakta kyunki ye unsafe ya illegal activities ke mutaliq hai.",
        "urdu": "میں اس درخواست میں مدد نہیں کر سکتا کیونکہ یہ غیر محفوظ یا غیر قانونی سرگرمیوں کے متعلق ہے۔"
    },
    "prompt_injection": {
        "english": "I cannot reveal my internal system instructions or bypass security policies. How can I help you with supported home services today?",
        "roman_urdu": "Main apni internal system instructions ya security policies ko bypass nahi kar sakta. Aaj main aap ki supported home services mein kaise madad kar sakta hoon?",
        "urdu": "میں اپنی اندرونی سسٹم کی ہدایات یا سیکیورٹی پالیسیوں کو بائی پاس نہیں کر سکتا۔ آج میں آپ کی سپورٹڈ ہوم سروسز میں کیسے مدد کر سکتا ہوں؟"
    },
    "privacy": {
        "english": "I cannot assist with requests that compromise user privacy or expose personal information.",
        "roman_urdu": "Main aisi requests mein madad nahi kar sakta jo user privacy ko compromise karein ya personal information expose karein.",
        "urdu": "میں ان درخواستوں میں مدد نہیں کر سکتا جو صارف کی رازداری پر سمجھوتہ کرتی ہیں یا ذاتی معلومات کو ظاہر کرتی ہیں۔"
    },
    "unsupported_service": {
        "english": "Sorry, this service is not available yet. Please choose one of the supported services.",
        "roman_urdu": "Maazrat, ye service abhi available nahi hai. Barah-e-karam supported services mein se choose karein.",
        "urdu": "معذرت، یہ service ابھی available نہیں ہے۔ براہ کرم supported services میں سے منتخب کریں۔"
    },
    "out_of_scope": {
        "english": "This request is outside the scope of our home and informal services platform. Please ask about booking a supported service like a Plumber or Electrician.",
        "roman_urdu": "Ye request hamare home aur informal services platform ke scope se bahar hai. Barah-e-karam supported services jaise ke Plumber ya Electrician ke baare mein poochein.",
        "urdu": "یہ درخواست ہمارے ہوم اور غیر رسمی سروسز پلیٹفرم کے دائرہ کار سے باہر ہے۔ براہ کرم پلمبر یا الیکٹریشن جیسی سپورٹڈ سروس بک کرنے کے بارے میں پوچھ کریں۔"
    }
}

def detect_language(text: str) -> str:
    """
    Deterministic language detector.
    Urdu script characters -> "urdu"
    Common Roman Urdu vocabulary -> "roman_urdu"
    Otherwise -> "english"
    """
    # Check Urdu script range
    if any('\u0600' <= char <= '\u06FF' for char in text):
        return "urdu"
    
    # Roman Urdu keywords
    roman_urdu_words = {
        "mujhe", "chahiye", "hai", "nahi", "he", "krna", "karna", "kr", "kar", "karo", "kia", "kiya",
        "kaise", "kese", "kahan", "kab", "kuch", "bhye", "bhai", "baji", "aap", "ap", "tum", "ko",
        "hi", "ho", "gaya", "gaye", "gya", "rha", "raha", "rhi", "rahi", "sath", "saath", "liye", "lye",
        "lekin", "magar", "aur", "ya", "bhi", "he", "hi", "ki", "ke", "ka", "se", "par", "pe", "per",
        "karo", "karein", "karen", "krn", "zaroorat", "zarurat", "chahye", "chahyay", "mil", "sakte", "skti",
        "sakta", "skta", "sktay", "hain", "han", "hun", "hoon", "mein", "shukriya"
    }
    
    words = re.findall(r'[a-zA-Z]+', text.lower())
    roman_count = sum(1 for w in words if w in roman_urdu_words)
    if roman_count >= 1 or (len(words) > 0 and roman_count / len(words) >= 0.2):
        return "roman_urdu"
    return "english"

def evaluate_guardrails(user_message: str, language: str = "") -> dict:
    """
    Evaluate user message against all guardrails:
    1. Unsafe/Illegal
    2. Prompt Injection
    3. Privacy Violations
    4. Unsupported safe services
    5. Out of scope
    
    Returns:
    {
      "allowed": bool,
      "reason": str,
      "category": str,
      "message": str,
      "language_detected": str
    }
    """
    msg = (user_message or "").strip()
    msg_lower = msg.lower()
    
    # 1. Detect language
    lang = detect_language(msg)
    lang_key = lang if lang in ["urdu", "roman_urdu"] else "english"
    
    # 2. Prompt Injection Guardrail
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, msg_lower):
            return {
                "allowed": False,
                "reason": "Prompt injection attempt detected.",
                "category": "prompt_injection",
                "message": REFUSALS["prompt_injection"][lang_key],
                "language_detected": lang
            }
            
    # 3. Privacy Guardrail
    for pattern in PRIVACY_PATTERNS:
        if re.search(pattern, msg_lower):
            return {
                "allowed": False,
                "reason": "Attempt to access user privacy or internal system credentials.",
                "category": "privacy",
                "message": REFUSALS["privacy"][lang_key],
                "language_detected": lang
            }
            
    # 4. Unsafe/Illegal Guardrail
    for policy, patterns in UNSAFE_POLICIES.items():
        for pattern in patterns:
            if re.search(pattern, msg_lower):
                return {
                    "allowed": False,
                    "reason": f"Request involves unsafe/illegal activity ({policy}).",
                    "category": "unsafe_illegal",
                    "message": REFUSALS["unsafe_illegal"][lang_key],
                    "language_detected": lang
                }
                
    # 5. Unsupported Service Guardrail
    # If the user asks for a known unsupported service (e.g. photographer)
    for service in UNSUPPORTED_SERVICES:
        if re.search(r'\b' + re.escape(service) + r'\b', msg_lower) or service in msg_lower:
            # Confirm it's not actually containing a supported category
            has_supported = any(sc in msg_lower for sc in SUPPORTED_CATEGORIES_LOWER)
            if not has_supported:
                return {
                    "allowed": False,
                    "reason": f"Service '{service}' is not supported on this platform.",
                    "category": "unsupported_service",
                    "message": REFUSALS["unsupported_service"][lang_key],
                    "language_detected": lang
                }
                
    # 6. Out of Scope Guardrail
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, msg_lower):
            return {
                "allowed": False,
                "reason": "Request is completely out of scope of home services.",
                "category": "out_of_scope",
                "message": REFUSALS["out_of_scope"][lang_key],
                "language_detected": lang
            }
            
    # Passed all guardrails
    return {
        "allowed": True,
        "reason": "Passed safety and scope checks.",
        "category": "safe",
        "message": "Safe request.",
        "language_detected": lang
    }


# Translation mappings for standard system/orchestrator responses to keep them multilingual.
TRANSLATIONS = {
    "urdu": {
        "Hi! Please tell me what service you need and your city/area.": 
            "ہیلو! براہ کرم مجھے بتائیں کہ آپ کو کس سروس کی ضرورت ہے اور آپ کس شہر/علاقے میں ہیں۔",
        "Wa alaikum salam! Aap ko kis service ki zaroorat hai aur kis city/area mein?":
            "وعلیکم السلام! آپ کو کس سروس کی ضرورت ہے اور کس شہر/علاقے میں؟",
        "Please tell me what service you need and your location.":
            "براہ کرم مجھے بتائیں کہ آپ کو کس سروس کی ضرورت ہے اور آپ کا مقام کیا ہے۔",
        "What kind of service do you need?":
            "آپ کو کس قسم کی سروس کی ضرورت ہے؟",
        "Which city are you in?":
            "آپ کس شہر میں ہیں؟",
        "No problem. I can refine the search if you want.":
            "کوئی بات نہیں۔ اگر آپ چاہیں تو میں تلاش کو مزید بہتر کر سکتا ہوں۔",
        "Your booking is confirmed!":
            "آپ کی بکنگ کی تصدیق ہو گئی ہے!",
        "I need to shortlist providers before I can confirm a booking.":
            "بکنگ کی تصدیق کرنے سے پہلے مجھے فراہم کنندگان کو شارٹ لسٹ کرنے کی ضرورت ہے۔",
        "I found a few possible matches. Reply with the option number you want to book, or share more details so I can refine the search.":
            "مجھے چند ممکنہ میچ ملے ہیں۔ براہ کرم اس آپشن نمبر کے ساتھ جواب دیں جسے آپ بک کرنا چاہتے ہیں، یا مزید تفصیلات شیئر کریں تاکہ میں تلاش کو بہتر کر سکوں۔",
        "Reply with the option number you want to book.":
            "براہ کرم اس آپشن نمبر کے ساتھ جواب دیں جسے آپ بک کرنا چاہتے ہیں۔",
        "I found the following providers for":
            "مجھے درج ذیل فراہم کنندگان ملے ہیں برائے",
        "in":
            "بقام",
        "Option":
            "آپشن",
        "Experience:":
            "تجربہ:",
        "years":
            "سال",
        "Jobs:":
            "کام:",
        "Rating:":
            "ریٹنگ:",
        "Response:":
            "جوابی وقت:",
        "mins":
            "منٹ",
        "Price:":
            "قیمت:",
        "Status:":
            "حیثیت:",
        "Phone:":
            "فون:"
    },
    "roman_urdu": {
        "Hi! Please tell me what service you need and your city/area.": 
            "Hi! Please tell me what service you need and your city/area.",
        "Wa alaikum salam! Aap ko kis service ki zaroorat hai aur kis city/area mein?":
            "Wa alaikum salam! Aap ko kis service ki zaroorat hai aur kis city/area mein?",
        "Please tell me what service you need and your location.":
            "Aap ko kis service ki zaroorat hai aur aap kis city/area mein hain?",
        "What kind of service do you need?":
            "Aap ko kis kisam ki service chahiye?",
        "Which city are you in?":
            "Aap kis city mein hain?",
        "No problem. I can refine the search if you want.":
            "Koi baat nahi. Agar aap chahein to main search ko mazeed behtar kar sakta hoon.",
        "Your booking is confirmed!":
            "Aap ki booking confirm ho gayi hai!",
        "I need to shortlist providers before I can confirm a booking.":
            "Booking confirm karne se pehle mujhe providers ko shortlist karna hoga.",
        "I found a few possible matches. Reply with the option number you want to book, or share more details so I can refine the search.":
            "Mujhe kuch matches mile hain. Aap jis option ko book karna chahte hain uska number reply karein, ya mazeed details share karein taake main search refine kar sakoon.",
        "Reply with the option number you want to book.":
            "Aap jis option ko book karna chahte hain uska number reply karein.",
        "I found the following providers for":
            "Mujhe ye providers mile hain for",
        "in":
            "in",
        "Option":
            "Option",
        "Experience:":
            "Experience:",
        "years":
            "years",
        "Jobs:":
            "Jobs:",
        "Rating:":
            "Rating:",
        "Response:":
            "Response:",
        "mins":
            "mins",
        "Price:":
            "Price:",
        "Status:":
            "Status:",
        "Phone:":
            "Phone:"
    }
}

def translate_response(text: str, language: str) -> str:
    """
    Translate dynamic system messages in response based on user language.
    Does not translate provider names, cities, or categories.
    """
    if not text or language not in TRANSLATIONS:
        return text
    
    translated = text
    lang_map = TRANSLATIONS[language]
    
    # Sort keys by length descending to replace longer phrases first
    sorted_keys = sorted(lang_map.keys(), key=len, reverse=True)
    
    for eng_str in sorted_keys:
        tr_str = lang_map[eng_str]
        # Use word boundaries for short words like "in" to avoid matching inside other words (e.g. "Rating:")
        if eng_str.isalpha() and len(eng_str) <= 3:
            translated = re.sub(r'\b' + re.escape(eng_str) + r'\b', tr_str, translated)
        else:
            translated = translated.replace(eng_str, tr_str)
            
    return translated
