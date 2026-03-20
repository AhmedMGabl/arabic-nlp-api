"""
Arabic NLP API — Python SDK

Install: pip install httpx
Usage:
    from arabic_nlp import ArabicNLP
    nlp = ArabicNLP()  # or ArabicNLP(api_key="your-rapidapi-key")

    result = nlp.sentiment("هذا المنتج ممتاز")
    print(result["sentiment"])  # "positive"

    result = nlp.detect_dialect("ازيك يا باشا عامل ايه")
    print(result["dialect"])  # "EGY"

Author: Ahmed Abogabl (github.com/AhmedMGabl)
"""

import httpx


class ArabicNLP:
    """Python client for the Arabic NLP API."""

    DEFAULT_BASE_URL = "https://thorough-perception-production-8028.up.railway.app"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-RapidAPI-Key"] = api_key

    def _post(self, endpoint: str, text: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/{endpoint}",
                json={"text": text},
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def sentiment(self, text: str) -> dict:
        """Analyze sentiment of Arabic text.

        Returns:
            {"sentiment": "positive"|"negative"|"neutral", "confidence": 0.87, ...}
        """
        return self._post("sentiment", text)

    def detect_dialect(self, text: str) -> dict:
        """Detect Arabic dialect.

        Returns:
            {"dialect": "EGY"|"MSA"|"GULF"|"LEV"|"MAG", "confidence": 0.95, ...}
        """
        return self._post("detect-dialect", text)

    def preprocess(self, text: str) -> dict:
        """Normalize and tokenize Arabic text.

        Returns:
            {"processed": "...", "tokens": [...], "token_count": N, ...}
        """
        return self._post("preprocess", text)

    def entities(self, text: str) -> dict:
        """Extract named entities from Arabic text.

        Returns:
            {"entities": [{"text": "...", "type": "PERSON"|"LOCATION"|...}, ...]}
        """
        return self._post("entities", text)

    def health(self) -> dict:
        """Check API health."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


# Quick usage example
if __name__ == "__main__":
    nlp = ArabicNLP()

    print("=== Health ===")
    print(nlp.health())

    print("\n=== Sentiment ===")
    print(nlp.sentiment("هذا المنتج ممتاز جداً وأنصح الجميع بشرائه"))

    print("\n=== Dialect Detection ===")
    print(nlp.detect_dialect("ازيك يا باشا عامل ايه النهارده"))

    print("\n=== Preprocessing ===")
    print(nlp.preprocess("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"))

    print("\n=== NER ===")
    print(nlp.entities("سافر أحمد من القاهرة إلى دبي يوم الخميس"))
