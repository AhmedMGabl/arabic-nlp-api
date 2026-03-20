"""
Arabic Sentiment Analysis Service.

Algorithm:
1. Preprocess text (normalise + remove diacritics).
2. Build a sliding window over tokens; detect negation within a 3-token
   look-back window.
3. Look up each token in the positive/negative lexicons.
4. Apply intensifier multipliers when detected in the token's context.
5. Aggregate into a signed score, then map to label + calibrated scores.

All logic is pure Python — no GPU, no heavy models.
"""

from __future__ import annotations

import math
import re
import time

from data.sentiment_lexicon import (
    INTENSIFIER_WORDS,
    NEGATION_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
)
from app.services.preprocessor import ArabicPreprocessor, preprocessor


# Maximum tokens we consider to limit CPU on very long texts
_MAX_TOKENS = 800

# Negation scope: how many tokens after a negation word does it apply?
_NEGATION_SCOPE = 3

# ---- Normalise all lexicon keys once at import time ----
# This ensures lookups work in the same orthographic space as the preprocessed tokens.
_p = ArabicPreprocessor()


def _norm(s: str) -> str:
    return _p.normalize_letters(_p.remove_diacritics(s))


_POSITIVE_NORM: dict[str, float] = {_norm(k): v for k, v in POSITIVE_WORDS.items()}
_NEGATIVE_NORM: dict[str, float] = {_norm(k): v for k, v in NEGATIVE_WORDS.items()}
_NEGATION_NORM: set[str] = {_norm(w) for w in NEGATION_WORDS}
_INTENSIFIER_NORM: dict[str, float] = {_norm(k): v for k, v in INTENSIFIER_WORDS.items()}


class SentimentAnalyser:
    """Rule-based Arabic sentiment analyser backed by a curated lexicon."""

    def analyse(self, text: str) -> dict:
        t0 = time.perf_counter()

        # Preprocess: normalise + remove diacritics, keep punctuation
        # (we strip punctuation separately after tokenising so we can
        #  detect sentence boundaries for negation reset)
        prep = preprocessor.process(
            text,
            normalize=True,
            remove_diacritics=True,
            remove_punctuation=False,
            remove_numbers=False,
            tokenize=True,
        )
        tokens: list[str] = prep["tokens"][:_MAX_TOKENS]

        # Strip trailing punctuation from each token for lexicon lookup
        clean_tokens = [re.sub(r"[^\w\u0600-\u06FF]", "", t) for t in tokens]

        pos_total: float = 0.0
        neg_total: float = 0.0
        matched_words: list[tuple[str, float]] = []  # (word, score)

        negation_counter = 0  # counts down from _NEGATION_SCOPE
        sentence_end = re.compile(r"[.!?؟،;]")

        for i, token in enumerate(clean_tokens):
            raw_token = tokens[i]

            # Reset negation at sentence boundaries
            if sentence_end.search(raw_token):
                negation_counter = 0

            # Detect negation
            if token in _NEGATION_NORM:
                negation_counter = _NEGATION_SCOPE + 1

            if negation_counter > 0:
                negation_counter -= 1

            is_negated = negation_counter > 0

            # Detect intensifier in look-back window (up to 2 tokens before)
            intensifier_mult = 1.0
            for j in range(max(0, i - 2), i):
                back_tok = clean_tokens[j]
                if back_tok in _INTENSIFIER_NORM:
                    intensifier_mult = max(intensifier_mult, _INTENSIFIER_NORM[back_tok])
            # Also check multi-word intensifiers (2-gram)
            if i >= 1:
                bigram = f"{clean_tokens[i-1]} {token}"
                if bigram in _INTENSIFIER_NORM:
                    intensifier_mult = max(intensifier_mult, _INTENSIFIER_NORM[bigram])

            # Lexicon lookup (normalised keys).
            # Also try stripping a trailing ه (normalised ة) to handle
            # adjective surface forms like سيئه -> سيء, فظيعه -> فظيع.
            score: float | None = None
            candidates = [token]
            if token.endswith("ه") and len(token) > 2:
                candidates.append(token[:-1])   # strip trailing ه (was ة)
            if token.endswith("ي") and len(token) > 2:
                candidates.append(token[:-1])   # strip yeh suffix

            for candidate in candidates:
                if candidate in _POSITIVE_NORM:
                    score = _POSITIVE_NORM[candidate]
                    break
                if candidate in _NEGATIVE_NORM:
                    score = _NEGATIVE_NORM[candidate]
                    break

            if score is not None:
                effective_score = score * intensifier_mult
                if is_negated:
                    effective_score = -effective_score * 0.8  # flip + dampen

                if effective_score > 0:
                    pos_total += effective_score
                else:
                    neg_total += abs(effective_score)

                matched_words.append((token, effective_score))

        # ---- Map raw scores to calibrated probabilities ----
        raw_net = pos_total - neg_total
        total_signal = pos_total + neg_total

        if total_signal == 0:
            # No lexicon hits → neutral with moderate confidence
            pos_prob = 0.05
            neg_prob = 0.05
            neu_prob = 0.90
            confidence = 0.55
            label = "neutral"
        else:
            # Normalise net score to [-1, 1] then apply sigmoid.
            # Use a saturation factor so even a single strong hit yields a
            # decisive label rather than getting averaged away by neutral mass.
            norm_net = raw_net / (total_signal + 1e-9)      # in [-1, 1]
            sigmoid = 1 / (1 + math.exp(-norm_net * 5))     # sharpened sigmoid

            # Coverage: how much of the sentence was covered by the lexicon.
            # A single hit in a 20-word sentence should still classify, but
            # with lower confidence than a densely opinionated sentence.
            coverage = min(1.0, total_signal / 3.0)          # saturates at 3 signal units

            # Distribute probability: polarity splits coverage, neutral gets the rest
            polarity_mass = coverage                          # 0..1
            pos_prob = round(sigmoid * polarity_mass, 4)
            neg_prob = round((1 - sigmoid) * polarity_mass, 4)
            neu_prob = round(max(0.0, 1.0 - pos_prob - neg_prob), 4)

            if pos_prob > neg_prob and pos_prob > neu_prob:
                label = "positive"
                confidence = round(pos_prob, 4)
            elif neg_prob > pos_prob and neg_prob > neu_prob:
                label = "negative"
                confidence = round(neg_prob, 4)
            else:
                label = "neutral"
                confidence = round(neu_prob, 4)

        # Top contributing words (sorted by absolute effective score)
        matched_words.sort(key=lambda x: abs(x[1]), reverse=True)
        key_words = [w for w, _ in matched_words[:5]]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "sentiment": label,
            "confidence": confidence,
            "positive_score": round(pos_prob, 4),
            "negative_score": round(neg_prob, 4),
            "neutral_score": round(neu_prob, 4),
            "key_words": key_words,
            "meta": {
                "char_count": len(text),
                "processing_time_ms": elapsed_ms,
            },
        }


sentiment_analyser = SentimentAnalyser()
