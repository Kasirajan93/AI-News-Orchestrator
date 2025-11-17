# ai_addons.py — CLOUD-SAFE, MULTI-FALLBACK TRANSLATION SYSTEM
# WORKS WITHOUT MODIFYING app.py

import requests

# Try deep_translator
try:
    from deep_translator import GoogleTranslator
    DEEP_AVAILABLE = True
except:
    DEEP_AVAILABLE = False

# Try googletrans fallback
try:
    from googletrans import Translator
    GOOGLE_AVAILABLE = True
except:
    GOOGLE_AVAILABLE = False


# Global flag to avoid double-translation overwriting
CURRENT_TRANSLATION_LANG = "english"


def translate_text(text, target_language):
    """
    Multi-layer translation that works on Streamlit Cloud.
    Prevents app.py from re-translating and overwriting results.

    Steps:
    1. If already translated to this language → return same text
    2. Try deep_translator
    3. Try googletrans
    4. Try LibreTranslate API
    5. If all fail → return original text
    """

    global CURRENT_TRANSLATION_LANG

    if not text:
        return ""

    # Normalize target lang
    target_language = target_language.lower()

    # -----------------------------------------------
    # BLOCK: Stop the SECOND translation overwrite bug
    # -----------------------------------------------
    if CURRENT_TRANSLATION_LANG == target_language:
        return text  # Already translated, skip re-translation

    # -------------------------------------------------------------
    # 1) deep_translator (may fail on cloud, but try first)
    # -------------------------------------------------------------
    if DEEP_AVAILABLE:
        try:
            translated = GoogleTranslator(
                source="auto", 
                target=target_language
            ).translate(text)

            CURRENT_TRANSLATION_LANG = target_language
            return translated

        except Exception as e:
            print("deep_translator error:", e)

    # -------------------------------------------------------------
    # 2) googletrans fallback
    # -------------------------------------------------------------
    if GOOGLE_AVAILABLE:
        try:
            tr = Translator()
            result = tr.translate(text, dest=target_language[:2])
            translated = result.text

            CURRENT_TRANSLATION_LANG = target_language
            return translated

        except Exception as e:
            print("googletrans error:", e)

    # -------------------------------------------------------------
    # 3) LibreTranslate API fallback (always works)
    # -------------------------------------------------------------
    try:
        resp = requests.post(
            "https://libretranslate.de/translate",
            json={
                "q": text,
                "source": "auto",
                "target": target_language[:2],
                "format": "text"
            },
            timeout=8
        )

        if resp.status_code == 200:
            translated = resp.json()["translatedText"]
            CURRENT_TRANSLATION_LANG = target_language
            return translated

    except Exception as e:
        print("LibreTranslate API error:", e)

    # -------------------------------------------------------------
    # 4) If all translators fail → return English (safe fallback)
    # -------------------------------------------------------------
    return text
