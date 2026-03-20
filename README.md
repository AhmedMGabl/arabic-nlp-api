# Arabic NLP API

Production-ready Arabic Natural Language Processing API built with FastAPI. Provides sentiment analysis, dialect detection, text preprocessing, and named entity recognition for Arabic text.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/sentiment` | POST | Classify Arabic text as positive, negative, or neutral with confidence scores |
| `/v1/detect-dialect` | POST | Identify Arabic dialect: MSA, Egyptian, Gulf, Levantine, or Maghrebi |
| `/v1/preprocess` | POST | Normalize, remove diacritics, and tokenize Arabic text |
| `/v1/entities` | POST | Extract persons, locations, organizations, dates, and numbers |
| `/v1/health` | GET | Service health check |
| `/docs` | GET | Interactive Swagger UI documentation |

## Quick Start

```bash
git clone https://github.com/AhmedMGabl/arabic-nlp-api.git
cd arabic-nlp-api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

## Example Usage

```python
import httpx

# Sentiment Analysis
response = httpx.post("http://localhost:8000/v1/sentiment", json={
    "text": "هذا المنتج ممتاز جداً وأنصح الجميع بشرائه"
})
# {"sentiment": "positive", "confidence": 0.87, ...}

# Dialect Detection
response = httpx.post("http://localhost:8000/v1/detect-dialect", json={
    "text": "إزيك يا باشا، عامل إيه النهاردة؟"
})
# {"dialect": "egyptian", "confidence": 0.92, ...}
```

## Deploy

**Railway:** Pre-configured with `Procfile` and `railway.json`. Connect repo and deploy.

**RapidAPI:** Deploy anywhere, then list on RapidAPI using `/openapi.json`. Set `RAPIDAPI_PROXY_SECRET` env var for auth.

## Features

- No GPU required — runs on any $5/month server
- Arabic-first — built by a native Arabic speaker
- RapidAPI proxy auth middleware built in
- Rate limiting (slowapi)
- CORS enabled
- Pydantic v2 typed models

## Author

**Ahmed Abogabl** — AI Engineer, Cairo, Egypt
- GitHub: [@AhmedMGabl](https://github.com/AhmedMGabl)
- LinkedIn: [ahmedabogabl](https://linkedin.com/in/ahmedabogabl)

MIT License
