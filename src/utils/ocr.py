import re

class OCR:
    """
    Tries RapidOCR (fast CPU). If not available, returns empty text gracefully.
    You can switch to EasyOCR by setting engine='easyocr' in config.yaml.
    """
    def __init__(self, engine="rapidocr", languages=None, uppercase=True, alnum_only=True, min_text_len=3):
        self.engine = engine
        self.languages = languages or ["en"]
        self.uppercase = uppercase
        self.alnum_only = alnum_only
        self.min_len = int(min_text_len)
        self.reader = None

        if engine == "easyocr":
            try:
                import easyocr
                self.reader = easyocr.Reader(self.languages, gpu=False)
            except Exception as e:
                print("[WARN] EasyOCR not available, falling back to RapidOCR:", e)
                self.engine = "rapidocr"

        if self.engine == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.reader = RapidOCR()
            except Exception as e:
                print("[WARN] RapidOCR not available. OCR will return empty strings:", e)
                self.reader = None

    def _clean(self, text: str):
        if self.uppercase:
            text = text.upper()
        if self.alnum_only:
            text = re.sub(r"[^A-Z0-9]", "", text)
        return text if len(text) >= self.min_len else ""

    def infer(self, image_bgr):
        if self.reader is None:
            return ""
        try:
            if self.engine == "rapidocr":
                res, _ = self.reader(image_bgr)
                cand = [it[1] for it in (res or []) if len(it) >= 2]
                return self._clean(max(cand, key=len) if cand else "")
            else:
                out = self.reader.readtext(image_bgr)
                cand = [t for (_, t, _) in out]
                return self._clean(max(cand, key=len) if cand else "")
        except Exception:
            return ""
