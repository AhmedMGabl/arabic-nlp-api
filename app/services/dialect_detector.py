"""
Arabic Dialect Detection Service.

Algorithm:
1. Normalise and tokenise the input.
2. Score each dialect profile by:
   a) Keyword hits (high weight).
   b) Character bigram and trigram overlap (medium weight).
3. Apply Laplace smoothing and dialect-specific priors.
4. Softmax the raw scores into probabilities.

Dialects: MSA, EGY (Egyptian), GULF, LEV (Levantine), MAG (Maghrebi).
"""

from __future__ import annotations

import math
import time

from data.dialect_features import DIALECT_PROFILES, DialectProfile
from app.services.preprocessor import preprocessor


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _extract_char_ngrams(text: str, n: int) -> set[str]:
    """Extract all character n-grams from text (no padding)."""
    return {text[i : i + n] for i in range(len(text) - n + 1)}


class DialectDetector:

    # Scoring weights
    KEYWORD_WEIGHT = 3.0
    BIGRAM_WEIGHT = 1.0
    TRIGRAM_WEIGHT = 1.5
    PRIOR_WEIGHT = 0.5

    def detect(self, text: str) -> dict:
        t0 = time.perf_counter()

        # Preprocess: normalise, remove diacritics, lowercase is not relevant
        # for Arabic (no case), but strip punctuation for keyword matching.
        prep = preprocessor.process(
            text,
            normalize=True,
            remove_diacritics=True,
            remove_punctuation=True,
            tokenize=True,
        )
        tokens: set[str] = set(prep["tokens"])
        processed_text: str = prep["processed"]

        # Character n-gram sets from the cleaned text
        bigrams_in_text = _extract_char_ngrams(processed_text, 2)
        trigrams_in_text = _extract_char_ngrams(processed_text, 3)

        raw_scores: list[float] = []

        for profile in DIALECT_PROFILES:
            score = 0.0

            # --- Keyword hits ---
            keyword_hits = sum(
                1 for kw in profile.keywords if kw in tokens or kw in processed_text
            )
            score += keyword_hits * self.KEYWORD_WEIGHT

            # --- Bigram overlap ---
            bg_hits = sum(1 for bg in profile.bigrams if bg in bigrams_in_text)
            bg_coverage = bg_hits / max(len(profile.bigrams), 1)
            score += bg_coverage * self.BIGRAM_WEIGHT

            # --- Trigram overlap ---
            tg_hits = sum(1 for tg in profile.trigrams if tg in trigrams_in_text)
            tg_coverage = tg_hits / max(len(profile.trigrams), 1)
            score += tg_coverage * self.TRIGRAM_WEIGHT

            # --- Prior (log scale to not dominate) ---
            score += math.log(profile.weight + 1e-9) * self.PRIOR_WEIGHT

            # Laplace smoothing floor
            score = max(score, 0.01)
            raw_scores.append(score)

        probs = _softmax(raw_scores)

        # Sort dialects by probability descending
        ranked = sorted(
            zip(DIALECT_PROFILES, probs),
            key=lambda x: x[1],
            reverse=True,
        )

        top_profile, top_prob = ranked[0]

        # Confidence: difference between top and second probability
        # (reflects how decisive the prediction is)
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = round(min(top_prob + (top_prob - second_prob) * 0.5, 1.0), 4)

        all_scores = [
            {
                "code": p.code,
                "name_en": p.name_en,
                "name_ar": p.name_ar,
                "score": round(prob, 4),
            }
            for p, prob in ranked
        ]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "dialect": top_profile.code,
            "dialect_name_en": top_profile.name_en,
            "dialect_name_ar": top_profile.name_ar,
            "confidence": confidence,
            "all_scores": all_scores,
            "meta": {
                "char_count": len(text),
                "processing_time_ms": elapsed_ms,
            },
        }


dialect_detector = DialectDetector()
