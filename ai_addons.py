# ai_addons.py — FINAL VERSION WITH TRANSLATION CACHE
# WORKS WITH DOUBLE-TRANSLATION IN app.py WITHOUT MODIFYING IT

import requests

try:
    from deep_translator import GoogleTranslator
    DEEP_OK = True
except:
    DEEP_OK = False

try:
    from googletrans import Translator
    GOOGLE_OK = True
except:
    GOOGLE_OK = False


# -------------------------------
# NEW: translation cache
# -------------------------------
# This stores:
#   (original_text + lang) → translated_text
TRANSLATION_CACHE = {}

# remembers current language applied
CURRENT_LANGUAGE = "english"


def _attempt_translate(text, lang):
    """Attempts translation using 3 methods."""
    # 1) deep translator
    if DEEP_OK:
        try:
            return GoogleTranslator(source="auto", target=lang).translate(text)
        except:
            pass

    # 2) googletrans fallback
    if GOOGLE_OK:
        try:
            t = Translator()
            return t.translate(text, dest=lang[:2]).text
        except:
            pass

    # 3) LibreTranslate free API
    try:
        r = requests.post(
            "https://libretranslate.de/translate",
            json={"q": text, "source": "auto", "target": lang[:2]},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()["translatedText"]
    except:
        pass

    return text


def translate_text(text, target_language):
    global CURRENT_LANGUAGE

    if not text:
        return ""

    target_language = target_language.lower()

    # ----------------------------------
    # CASE 1: Already translated earlier
    # ----------------------------------
    key = (text, target_language)
    if key in TRANSLATION_CACHE:
        CURRENT_LANGUAGE = target_language
        return TRANSLATION_CACHE[key]

    # ----------------------------------
    # CASE 2: First time translation
    # ----------------------------------
    translated = _attempt_translate(text, target_language)

    # save in cache
    TRANSLATION_CACHE[key] = translated

    # remember current language
    CURRENT_LANGUAGE = target_language

    return translated
