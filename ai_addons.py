from deep_translator import GoogleTranslator

def translate_text(text, target_language):
    if not text:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception:
        return text  # fail silently (no crash)
