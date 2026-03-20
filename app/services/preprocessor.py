"""
Arabic text preprocessing service.
All operations run in pure Python — no heavy NLP libraries required.
"""

from __future__ import annotations

import re
import unicodedata

# ---- Unicode ranges ----
# Arabic diacritics (tashkeel / harakat)
_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A"   # Arabic supplemental diacritics
    r"\u064B-\u065F"    # Standard harakat (fathah, dammah, kasrah, etc.)
    r"\u0670"           # Superscript alef
    r"\u06D6-\u06DC"    # Quranic annotation signs
    r"\u06DF-\u06E4"
    r"\u06E7\u06E8"
    r"\u06EA-\u06ED]"
)

# Arabic punctuation + common punctuation retained in Arabic text
_PUNCTUATION_RE = re.compile(
    r'[،؛؟!\u0022\u0027\u201c\u201d\u2018\u2019()\[\]{}\-\u2013\u2014_/\\|<>.,;:?!@#$%^&*+=~`]'
)

# Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) and Western numerals
_NUMBERS_RE = re.compile(r"[\u0660-\u0669\d]+")

# Extra whitespace
_WHITESPACE_RE = re.compile(r"\s+")

# Tatweel (kashida) — stretching character used for stylistic elongation
_TATWEEL_RE = re.compile(r"\u0640+")


# ---- Letter normalisation maps ----
# Alef variants -> plain alef (ا)
_ALEF_RE = re.compile(r"[إأآٱ]")
# Teh marbuta -> heh
_TEH_MARBUTA_RE = re.compile(r"ة")
# Yeh variants -> yeh (ي)
_YEH_RE = re.compile(r"[ىئ]")
# Waw with hamza above -> waw
_WAW_HAMZA_RE = re.compile(r"ؤ")
# Alef with hamza below -> alef
_ALEF_HAMZA_BELOW_RE = re.compile(r"إ")


class ArabicPreprocessor:
    """
    Stateless Arabic text preprocessor.
    All methods are pure functions operating on strings.
    """

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Strip tashkeel marks."""
        return _DIACRITICS_RE.sub("", text)

    @staticmethod
    def remove_tatweel(text: str) -> str:
        """Remove kashida (tatweel) elongation characters."""
        return _TATWEEL_RE.sub("", text)

    @staticmethod
    def normalize_letters(text: str) -> str:
        """
        Normalise Arabic letter variants:
        - All alef forms -> ا
        - ة -> ه
        - ى / ئ -> ي
        - ؤ -> و
        """
        text = _ALEF_RE.sub("ا", text)
        text = _TEH_MARBUTA_RE.sub("ه", text)
        text = _YEH_RE.sub("ي", text)
        text = _WAW_HAMZA_RE.sub("و", text)
        return text

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation marks."""
        return _PUNCTUATION_RE.sub(" ", text)

    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove Arabic-Indic and Western digit sequences."""
        return _NUMBERS_RE.sub(" ", text)

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Collapse runs of whitespace to a single space and strip edges."""
        return _WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer — returns non-empty tokens."""
        return [t for t in text.split() if t]

    # ------------------------------------------------------------------
    # Compound pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        text: str,
        *,
        normalize: bool = True,
        remove_diacritics: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        tokenize: bool = True,
    ) -> dict:
        """
        Run the full preprocessing pipeline according to caller flags.
        Returns a dict compatible with PreprocessResponse.
        """
        operations_applied: list[str] = []
        processed = text

        # Always remove tatweel — it never adds information
        processed = self.remove_tatweel(processed)

        if remove_diacritics:
            processed = self.remove_diacritics(processed)
            operations_applied.append("remove_diacritics")

        if normalize:
            processed = self.normalize_letters(processed)
            operations_applied.append("normalize_letters")

        if remove_punctuation:
            processed = self.remove_punctuation(processed)
            operations_applied.append("remove_punctuation")

        if remove_numbers:
            processed = self.remove_numbers(processed)
            operations_applied.append("remove_numbers")

        processed = self.collapse_whitespace(processed)
        operations_applied.append("collapse_whitespace")

        tokens = self.tokenize(processed) if tokenize else []
        if tokenize:
            operations_applied.append("tokenize")

        return {
            "original": text,
            "processed": processed,
            "tokens": tokens,
            "token_count": len(tokens),
            "operations_applied": operations_applied,
        }


# Singleton — instantiate once at import time
preprocessor = ArabicPreprocessor()
