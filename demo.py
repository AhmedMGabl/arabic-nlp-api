"""
Arabic NLP API — Interactive Demo (Streamlit)

Run: pip install streamlit && streamlit run demo.py
Screenshot the results for Fiverr/Upwork portfolio.
"""

import streamlit as st
import httpx
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Arabic NLP API Demo", page_icon="🌍", layout="wide")

st.title("🌍 Arabic NLP API")
st.caption("Sentiment Analysis • Dialect Detection • Text Preprocessing • Named Entity Recognition")

tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Dialect Detection", "Preprocessing", "NER"])

with tab1:
    st.subheader("Arabic Sentiment Analysis")
    text = st.text_area(
        "Enter Arabic text:",
        value="هذا المنتج ممتاز جداً وأنصح الجميع بشرائه",
        key="sentiment",
        height=100,
    )
    if st.button("Analyze Sentiment", key="btn_sentiment"):
        try:
            r = httpx.post(f"{API_URL}/v1/sentiment", json={"text": text}, timeout=10)
            result = r.json()
            col1, col2, col3 = st.columns(3)
            sentiment = result.get("sentiment", "unknown")
            confidence = result.get("confidence", 0)
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "❓")
            col1.metric("Sentiment", f"{emoji} {sentiment.title()}")
            col2.metric("Confidence", f"{confidence:.0%}")
            col3.json(result.get("scores", {}))
        except Exception as e:
            st.error(f"API Error: {e}. Make sure the API is running on {API_URL}")

with tab2:
    st.subheader("Arabic Dialect Detection")
    text2 = st.text_area(
        "Enter Arabic text:",
        value="إزيك يا باشا، عامل إيه النهاردة؟",
        key="dialect",
        height=100,
    )
    if st.button("Detect Dialect", key="btn_dialect"):
        try:
            r = httpx.post(f"{API_URL}/v1/detect-dialect", json={"text": text2}, timeout=10)
            result = r.json()
            dialect = result.get("dialect", "unknown")
            confidence = result.get("confidence", 0)
            flags = {"msa": "📖", "egyptian": "🇪🇬", "gulf": "🇸🇦", "levantine": "🇱🇧", "maghrebi": "🇲🇦"}
            st.metric("Detected Dialect", f"{flags.get(dialect, '🌍')} {dialect.title()}", f"{confidence:.0%} confidence")
            if "scores" in result:
                st.bar_chart(result["scores"])
        except Exception as e:
            st.error(f"API Error: {e}. Make sure the API is running on {API_URL}")

with tab3:
    st.subheader("Arabic Text Preprocessing")
    text3 = st.text_area(
        "Enter Arabic text:",
        value="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ - هذا نصٌّ عربيٌّ للتجربة!",
        key="preprocess",
        height=100,
    )
    if st.button("Preprocess Text", key="btn_preprocess"):
        try:
            r = httpx.post(f"{API_URL}/v1/preprocess", json={"text": text3}, timeout=10)
            result = r.json()
            st.text_area("Normalized:", value=result.get("normalized", ""), height=80)
            st.write("**Tokens:**", result.get("tokens", []))
        except Exception as e:
            st.error(f"API Error: {e}. Make sure the API is running on {API_URL}")

with tab4:
    st.subheader("Named Entity Recognition")
    text4 = st.text_area(
        "Enter Arabic text:",
        value="سافر أحمد من القاهرة إلى دبي يوم الخميس للعمل في شركة جوجل",
        key="ner",
        height=100,
    )
    if st.button("Extract Entities", key="btn_ner"):
        try:
            r = httpx.post(f"{API_URL}/v1/entities", json={"text": text4}, timeout=10)
            result = r.json()
            entities = result.get("entities", [])
            if entities:
                for ent in entities:
                    st.write(f"**{ent.get('text', '')}** → `{ent.get('type', '')}`")
            else:
                st.info("No entities found.")
        except Exception as e:
            st.error(f"API Error: {e}. Make sure the API is running on {API_URL}")

st.divider()
st.caption("Built by Ahmed Abogabl | github.com/AhmedMGabl | Cairo, Egypt")
