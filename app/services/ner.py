"""
Arabic Named Entity Recognition Service.

Approach: Multi-pass rule-based tagging using gazetteers + regex patterns.

Pass order (later passes do not overwrite earlier matches):
  1. DATE entities  — regex for years, month names, day names
  2. NUMBER entities — Arabic-Indic and Western digit sequences
  3. PERSON entities — title + name patterns, then standalone first names
  4. ORGANIZATION entities — org prefix + proper noun sequence
  5. LOCATION entities — gazetteer lookup

After all passes, overlapping spans are resolved (longest span wins).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from data.ner_gazetteer import (
    ALL_LOCATIONS,
    ALL_MONTHS,
    DAYS_OF_WEEK,
    ORGANIZATIONS,
    ORG_PREFIXES,
    PERSON_NAMES,
    PERSON_TITLES,
)
from app.services.preprocessor import preprocessor


@dataclass
class Span:
    start: int
    end: int
    text: str
    entity_type: str
    confidence: float


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy longest-span-first overlap resolution."""
    spans.sort(key=lambda s: (s.end - s.start), reverse=True)
    accepted: list[Span] = []
    for span in spans:
        if not any(_overlaps(span, kept) for kept in accepted):
            accepted.append(span)
    accepted.sort(key=lambda s: s.start)
    return accepted


class ArabicNER:

    # ---- Compiled patterns ----
    _YEAR_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3}|١[٠-٩]{3}|٢[٠-٩]{3})\b")
    _NUM_RE = re.compile(r"[\u0660-\u0669\d]+(?:[.,،][\u0660-\u0669\d]+)*")
    # A "proper noun token" in Arabic: starts with an Arabic letter, length >= 2
    _AR_WORD_RE = re.compile(r"[\u0600-\u06FF][\u0600-\u06FF\u0640]+")

    def _find_dates(self, text: str, norm: str) -> list[Span]:
        spans: list[Span] = []

        # Year patterns on original text
        for m in self._YEAR_RE.finditer(text):
            spans.append(Span(m.start(), m.end(), m.group(), "DATE", 0.85))

        # Month name matches on normalised text, mapped back to original offsets.
        # Because normalisation may change character counts we search on original.
        for month in ALL_MONTHS:
            for m in re.finditer(re.escape(month), text):
                spans.append(Span(m.start(), m.end(), m.group(), "DATE", 0.90))

        for day in DAYS_OF_WEEK:
            for m in re.finditer(re.escape(day), text):
                spans.append(Span(m.start(), m.end(), m.group(), "DATE", 0.88))

        return spans

    def _find_numbers(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for m in self._NUM_RE.finditer(text):
            # Skip single-digit standalone numbers (too ambiguous)
            if len(m.group()) > 1:
                spans.append(Span(m.start(), m.end(), m.group(), "NUMBER", 0.92))
        return spans

    # Arabic function words / prepositions that terminate a name span
    _NAME_STOPWORDS = frozenset({
        "في", "من", "إلى", "على", "عن", "مع", "بعد", "قبل", "حول",
        "بين", "خلال", "ضد", "عند", "حتى", "منذ", "إن", "أن", "لأن",
        "لكن", "أو", "و", "أم", "عن", "لـ", "بـ", "كـ", "وـ",
        "الذي", "التي", "الذين", "اللاتي", "هذا", "هذه", "ذلك",
        "يوم", "عام", "شهر", "أسبوع",
    })

    def _find_persons(self, text: str) -> list[Span]:
        spans: list[Span] = []

        # Build a stop-word alternation for negative lookahead inside the name
        # Use a regex that grabs ONLY proper-looking name tokens (3+ chars, not stopwords)
        # Pattern: TITLE + 1-3 name words (each word must NOT be a stopword)
        _sw = "|".join(re.escape(w) for w in sorted(self._NAME_STOPWORDS, key=len, reverse=True))
        # An "acceptable" name token: Arabic word that is NOT a stopword
        _name_tok = r"(?!(?:" + _sw + r")(?:\s|$))[\u0600-\u06FF][\u0600-\u06FF\u0640]*"
        _name_seq = r"(" + _name_tok + r"(?:\s+" + _name_tok + r"){0,2})"

        title_pattern = "(" + "|".join(re.escape(t) for t in sorted(PERSON_TITLES, key=len, reverse=True)) + ")"
        title_re = re.compile(title_pattern + r"\s+" + _name_seq)

        for m in title_re.finditer(text):
            spans.append(Span(m.start(), m.end(), m.group(), "PERSON", 0.88))

        # Standalone first name gazetteer
        for name in PERSON_NAMES:
            for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
                spans.append(Span(m.start(), m.end(), m.group(), "PERSON", 0.72))

        return spans

    def _find_organizations(self, text: str) -> list[Span]:
        spans: list[Span] = []

        # Known organisations
        for org in sorted(ORGANIZATIONS, key=len, reverse=True):
            for m in re.finditer(re.escape(org), text):
                spans.append(Span(m.start(), m.end(), m.group(), "ORGANIZATION", 0.85))

        # Org prefix + Arabic proper noun sequence
        prefix_pattern = "(" + "|".join(re.escape(p) for p in sorted(ORG_PREFIXES, key=len, reverse=True)) + ")"
        org_re = re.compile(
            prefix_pattern
            + r"\s+([\u0600-\u06FF][\u0600-\u06FF\u0640]*(?:\s+[\u0600-\u06FF][\u0600-\u06FF\u0640]*){0,3})"
        )
        for m in org_re.finditer(text):
            spans.append(Span(m.start(), m.end(), m.group(), "ORGANIZATION", 0.78))

        return spans

    def _find_locations(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for loc in sorted(ALL_LOCATIONS, key=len, reverse=True):
            for m in re.finditer(r"(?<!\w)" + re.escape(loc) + r"(?!\w)", text):
                spans.append(Span(m.start(), m.end(), m.group(), "LOCATION", 0.88))
        return spans

    def extract(self, text: str) -> dict:
        t0 = time.perf_counter()

        # Normalise for pattern matching (keep original for span offsets)
        norm_prep = preprocessor.process(
            text,
            normalize=True,
            remove_diacritics=True,
            remove_punctuation=False,
            tokenize=False,
        )
        norm_text: str = norm_prep["processed"]

        # Run all passes against original text so character offsets are correct
        all_spans: list[Span] = []
        all_spans.extend(self._find_dates(text, norm_text))
        all_spans.extend(self._find_numbers(text))
        all_spans.extend(self._find_persons(text))
        all_spans.extend(self._find_organizations(text))
        all_spans.extend(self._find_locations(text))

        # Also run person/org/location passes on normalised text for better recall
        all_spans.extend(self._find_persons(norm_text))
        all_spans.extend(self._find_organizations(norm_text))
        all_spans.extend(self._find_locations(norm_text))

        resolved = _resolve_overlaps(all_spans)

        entities = [
            {
                "text": s.text,
                "entity_type": s.entity_type,
                "start": s.start,
                "end": s.end,
                "confidence": round(s.confidence, 4),
            }
            for s in resolved
        ]

        types_found = sorted({e["entity_type"] for e in entities})
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "entities": entities,
            "entity_count": len(entities),
            "entity_types_found": types_found,
            "meta": {
                "char_count": len(text),
                "processing_time_ms": elapsed_ms,
            },
        }


ner_service = ArabicNER()
