# ai_addons.py — Cloud Safe Translation Layer

# Primary translator
try:
    from deep_translator import GoogleTranslator
    DEEP_AVAILABLE = True
except:
    DEEP_AVAILABLE = False

# Secondary fallback: googletrans
try:
    from googletrans import Translator
    GOOGLE_AVAILABLE = True
except:
    GOOGLE_AVAILABLE = False

# Third fallback: LibreTranslate API
try:
    import requests
    LIBRE_AVAILABLE = True
except:
    LIBRE_AVAILABLE = False


def translate_text(text, target_language):
    """
    CLOUD-SAFE TRANSLATION:
    1. Try deep_translator (may fail on Streamlit Cloud)
    2. Try googletrans (works more reliably)
    3. Try LibreTranslate (public API)
    4. If all fail → return original English text
    """
    if not text:
        return ""

    lang = target_language.lower()

    # ------------------------
    # 1. Try deep_translator
    # ------------------------
    if DEEP_AVAILABLE:
        try:
            return GoogleTranslator(source="auto", target=lang).translate(text)
        except Exception as e:
            print("deep_translator error:", e)

    # ------------------------
    # 2. Try googletrans
    # ------------------------
    if GOOGLE_AVAILABLE:
        try:
            translator = Translator()
            result = translator.translate(text, dest=lang[:2])
            return result.text
        except Exception as e:
            print("googletrans error:", e)

    # ------------------------
    # 3. Try LibreTranslate public API
    # ------------------------
    if LIBRE_AVAILABLE:
        try:
            r = requests.post("https://libretranslate.de/translate", json={
                "q": text,
                "source": "auto",
                "target": lang[:2],
                "format": "text"
            })
            if r.status_code == 200:
                return r.json()["translatedText"]
        except Exception as e:
            print("LibreTranslate error:", e)

    # ------------------------
    # 4. Last fallback
    # ------------------------
    return text

