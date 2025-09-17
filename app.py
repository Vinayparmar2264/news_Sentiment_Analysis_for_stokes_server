# app.py
import os
import time
import logging
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
from newsapi import NewsApiClient
from transformers import pipeline
import yfinance as yf

# -------------------------
# Basic logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Config / Environment
# -------------------------
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY") or "97ea8e7278d845fa92b3a912717969f8"
MODEL_PREFER = os.environ.get("SENTIMENT_MODEL") or "ProsusAI/finbert"  # prefer finbert
MAX_ARTICLES = 7

# -------------------------
# Flask app init
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# NewsAPI client
# -------------------------
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# -------------------------
# Sentiment pipeline load with fallback
# -------------------------
sentiment_pipeline = None
loaded_model_name = None

def load_sentiment_pipeline():
    global sentiment_pipeline, loaded_model_name
    tried = []
    candidates = [MODEL_PREFER, "distilbert-base-uncased-finetuned-sst-2-english"]  # fallback
    for model_name in candidates:
        try:
            logger.info(f"Loading sentiment model: {model_name} ... (this may take a bit on first run)")
            sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            loaded_model_name = model_name
            logger.info(f"Loaded sentiment model: {model_name}")
            return
        except Exception as exc:
            logger.warning(f"Failed to load model {model_name}: {exc}")
            tried.append((model_name, str(exc)))
    # if here, none loaded
    sentiment_pipeline = None
    loaded_model_name = None
    logger.error("Could not load any sentiment model. Tried: %s", tried)

# Load on startup
load_sentiment_pipeline()

# -------------------------
# Utility: basic text clean / combine
# -------------------------
def combine_text_for_analysis(article):
    # Use title + description + source name for better context (when available)
    parts = []
    title = article.get("title") or ""
    desc = article.get("description") or ""
    src = article.get("source", {}).get("name") or ""
    for p in (title.strip(), desc.strip(), src.strip()):
        if p:
            parts.append(p)
    combined = " . ".join(parts)
    # Ensure not too long for model; truncate if necessary
    if len(combined) > 512:
        combined = combined[:500] + "..."
    return combined or title or desc or "No text available"

# -------------------------
# Utility: get recent price movement via yfinance
# -------------------------
def get_recent_price_change(ticker):
    try:
        # Download last 3 trading days to compute simple pct change
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d")[['Close']]
        if hist.empty or len(hist) < 2:
            return None
        # use most recent two closes
        closes = hist['Close'].dropna()
        if len(closes) < 2:
            return None
        last = closes.iloc[-1]
        prev = closes.iloc[-2]
        pct_change = ((last - prev) / prev) * 100.0
        return {
            "last_close": float(last),
            "prev_close": float(prev),
            "pct_change": float(pct_change)
        }
    except Exception as e:
        logger.warning("yfinance error for %s: %s", ticker, e)
        return None

# -------------------------
# Routes
# -------------------------
@app.route("/general_news", methods=["GET"])
def get_general_news():
    try:
        # Use a stable source (Wall Street Journal) as before
        top_headlines = newsapi.get_top_headlines(sources="the-wall-street-journal")
        articles = []
        for art in top_headlines.get("articles", []):
            # safe-get title/url
            title = art.get("title")
            url = art.get("url")
            if title and url:
                articles.append({"title": title, "url": url})
        return jsonify(articles)
    except Exception as e:
        logger.exception("Error fetching general news")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["GET"])
def analyze_ticker():
    # Early checks
    if not sentiment_pipeline:
        return jsonify({"error": "Sentiment analysis model is not available. Check server logs."}), 503

    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400

    ticker = ticker.strip().upper()
    try:
        # 1) fetch news for the ticker
        query = f"{ticker} stock market"
        logger.info("Searching news for query: %s", query)
        news_result = newsapi.get_everything(q=query, sort_by="relevancy", language="en", page_size=30)
        articles = news_result.get("articles", [])

        if not articles:
            return jsonify({
                "ticker": ticker,
                "overall_sentiment": "Neutral",
                "price": None,
                "articles": [{"title": f"No recent news found for {ticker}.", "url": "#", "source": "System", "sentiment": "neutral"}]
            })

        processed = []
        sentiments = []

        # 2) analyze top MAX_ARTICLES articles but prefer ones with text
        count = 0
        for a in articles:
            if count >= MAX_ARTICLES:
                break
            title = a.get("title") or ""
            if not title:
                # skip articles without title as before
                continue
            text = combine_text_for_analysis(a)
            try:
                res = sentiment_pipeline(text)[0]
                label = res.get("label")
            except Exception as e:
                logger.warning("Sentiment pipeline failed on text: %s | %s", text[:80], e)
                label = "neutral"

            sentiments.append(label)
            processed.append({
                "title": title,
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "sentiment": label
            })
            count += 1

        # 3) compute overall sentiment (most common)
        if sentiments:
            overall = Counter(sentiments).most_common(1)[0][0]
        else:
            overall = "Neutral"

        # 4) fetch recent price info using yfinance (best-effort)
        price_info = get_recent_price_change(ticker)

        response = {
            "ticker": ticker,
            "overall_sentiment": str(overall).capitalize(),
            "model_used": loaded_model_name,
            "price": price_info,
            "articles": processed
        }
        return jsonify(response)

    except Exception as e:
        logger.exception("An error occurred during analysis")
        return jsonify({"error": "An internal error occurred during analysis."}), 500

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # Bind to all interfaces if you want to open from other devices on the LAN; keep 127.0.0.1 for local only
    app.run(debug=True, host="127.0.0.1", port=5000)
