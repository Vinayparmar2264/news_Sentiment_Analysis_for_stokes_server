import os
import csv
import json
import logging
import requests
import difflib
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from flask import Flask, request, jsonify
from flask_cors import CORS
from newsapi import NewsApiClient
from transformers import pipeline
import yfinance as yf
import re

# ---------- logging CONFIGURATION ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Optional libs
try:
    import spacy

    spacy_nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    logger.warning("spaCy not found or model not downloaded. Keyword extraction will be limited.")
    spacy_nlp = None

# ---------- config ----------
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY") or "97ea8e7278d845fa92b3a912717969f8"
MODEL_A = os.environ.get("FIN_MODEL") or "ProsusAI/finbert"
MODEL_B = os.environ.get("GEN_MODEL") or "distilbert-base-uncased-finetuned-sst-2-english"
FETCH_PAGE_SIZE = int(os.environ.get("FETCH_PAGE_SIZE", "100"))
MAX_ARTICLES = int(os.environ.get("MAX_ARTICLES", "7"))
HALF_LIFE_HOURS = float(os.environ.get("HALF_LIFE_HOURS", "72"))
TICKERS_CSV = "tickers.csv"  # Define the path for the CSV file

# ---------- flask + clients ----------
app = Flask(__name__)
CORS(app)
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ---------- model loading ----------
sentiment_a = pipeline("sentiment-analysis", model=MODEL_A)
sentiment_b = pipeline("sentiment-analysis", model=MODEL_B)
loaded_models = [MODEL_A, MODEL_B]

# --- START: CSV FALLBACK LOGIC (re-introduced for accuracy and reliability) ---
company_to_tickers = {}
name_index = []


def load_tickers_csv(path=TICKERS_CSV):
    global company_to_tickers, name_index
    if not os.path.exists(path):
        logger.warning(f"Tickers CSV not found at '{path}'. The CSV fallback search will be unavailable.")
        return
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    ticker, company_name = row[0].strip(), row[1].strip()
                    if company_name and ticker:
                        company_to_tickers[company_name.lower()] = ticker
                        name_index.append(company_name)
        logger.info(f"Loaded {len(company_to_tickers)} companies from '{path}' for fallback search.")
    except Exception as e:
        logger.exception(f"Failed to load tickers CSV: {e}")


load_tickers_csv()


@lru_cache(maxsize=512)
def map_name_to_ticker_from_csv(query_name):
    if not query_name: return None
    query_lower = query_name.lower()

    if query_lower in company_to_tickers:
        return company_to_tickers[query_lower]

    scored_matches = []
    for name in name_index:
        name_lower = name.lower()
        similarity = difflib.SequenceMatcher(None, query_lower, name_lower).ratio()
        score = similarity
        if all(token in name_lower for token in query_lower.split()): score += 0.3
        if name_lower.startswith(query_lower): score += 0.2
        scored_matches.append((name, score))

    if not scored_matches: return None
    scored_matches.sort(key=lambda x: x[1], reverse=True)
    best_match_name, best_score = scored_matches[0]

    if best_score >= 0.7:  # Using a confidence threshold
        logger.info(f"CSV search found '{best_match_name}' for query '{query_name}' with score {best_score:.2f}")
        return company_to_tickers[best_match_name.lower()]
    return None


# --- END: CSV FALLBACK LOGIC ---


# ---------- Ticker Resolution (API-First, CSV-Fallback) ----------
@lru_cache(maxsize=512)
def search_ticker_via_api(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        results = data.get('quotes', [])
        if not results: return None

        best_equity = None
        for item in results:
            if item.get('quoteType') == 'EQUITY' and item.get('symbol'):
                if '.' in item['symbol']: return item['symbol']
                if best_equity is None: best_equity = item
        if best_equity: return best_equity['symbol']
        return None
    except Exception as e:
        logger.error(f"API call to Yahoo Finance search failed: {e}")
        return None


def resolve_input_to_ticker(query):
    logger.info(f"Resolving '{query}' using hybrid model (API first).")

    # 1. Try the live API first for the most current data
    api_result = search_ticker_via_api(query)
    if api_result:
        logger.info(f"Successfully resolved '{query}' to '{api_result}' via API.")
        return api_result

    # 2. If API fails or finds nothing, fall back to the local CSV
    logger.warning(f"API lookup failed for '{query}'. Falling back to local CSV search.")
    csv_result = map_name_to_ticker_from_csv(query)
    if csv_result:
        logger.info(f"Successfully resolved '{query}' to '{csv_result}' via local CSV.")
        return csv_result

    logger.error(f"Failed to resolve '{query}' using both API and local CSV.")
    return None


# ---------- (The rest of the file is the same) ----------
@lru_cache(maxsize=128)
def get_company_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get("shortName") or info.get("longName")
        hist = t.history(period="5d", interval="1d")
        price_info = None
        if not hist.empty:
            closes = hist["Close"].dropna()
            if len(closes) >= 1:
                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
                pct = ((last - prev) / prev) * 100.0 if prev != 0 else 0.0
                price_info = {"last_close": last, "pct_change": pct}
        return name, price_info
    except Exception as e:
        logger.error(f"yfinance failed for ticker '{ticker}': {e}")
        return ticker, None


@lru_cache(maxsize=256)
def fetch_articles(query):
    try:
        res = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=FETCH_PAGE_SIZE)
        return res.get("articles", []) or []
    except Exception as e:
        logger.warning(f"NewsAPI get_everything failed for '{query}': {e}")
        return []


def parse_published_at(ts):
    if not ts: return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


@app.route("/analyze", methods=["GET"])
def analyze_ticker():
    query = request.args.get("ticker", "").strip()
    if not query: return jsonify({"error": "Ticker or company name required"}), 400

    resolved_ticker = resolve_input_to_ticker(query)

    if not resolved_ticker:
        return jsonify(
            {"error": f"Could not find a valid company or ticker for '{query}'. Please try a different name."}), 404

    company_name, price_info = get_company_info(resolved_ticker)
    search_queries = f'"{company_name}" OR "{resolved_ticker}"'
    articles = fetch_articles(search_queries)

    if not articles:
        return jsonify(
            {"query": query, "ticker": resolved_ticker, "company_name": company_name, "overall_sentiment": "Neutral",
             "price": price_info, "articles": []})

    weighted_votes = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    processed_articles = []

    for art in articles[:MAX_ARTICLES]:
        text_to_analyze = f"{(art.get('title') or '')}. {(art.get('description') or '')}"
        sentiment_result = sentiment_a(text_to_analyze)[0]
        sentiment = sentiment_result['label'].lower()
        pub_date = parse_published_at(art.get("publishedAt"))
        age_hours = (datetime.now(timezone.utc) - pub_date).total_seconds() / 3600 if pub_date else 24 * 7
        weight = 0.5 ** (age_hours / HALF_LIFE_HOURS)
        weighted_votes[sentiment] += weight
        processed_articles.append(
            {"title": art.get("title"), "url": art.get("url"), "description": art.get("description"),
             "source": (art.get("source") or {}).get("name"), "sentiment": sentiment,
             "publishedAt": art.get("publishedAt")})

    overall_sentiment = "Neutral"
    if any(weighted_votes.values()):
        overall_sentiment = max(weighted_votes, key=weighted_votes.get).capitalize()

    response = {"query": query, "ticker": resolved_ticker, "company_name": company_name,
                "overall_sentiment": overall_sentiment, "models_used": loaded_models, "price": price_info,
                "articles": processed_articles}
    return jsonify(response)


@app.route("/impact", methods=["POST"])
def analyze_impact_on_company():
    payload = request.get_json()
    ticker = payload.get("ticker")
    if not ticker: return jsonify({"error": "Ticker is required for impact analysis."}), 400

    company_name, price_info = get_company_info(ticker)
    if not company_name: return jsonify({"error": f"Could not resolve company name for ticker {ticker}."}), 404

    title = payload.get("title", "")
    description = payload.get("description", "")
    full_text = f"{title}. {description}"
    sentiment_result = sentiment_a(full_text)[0]
    sentiment = sentiment_result['label'].lower()

    result = {"impact_on": {"name": company_name, "ticker": ticker}, "sentiment": sentiment,
              "evidence": [description or title], "key_topics": [], "price": price_info}
    return jsonify(result)


@app.route("/general_news", methods=["GET"])
def general_news():
    try:
        res = newsapi.get_everything(q="NIFTY 50 OR Sensex", language="en", sort_by="publishedAt")
        return jsonify(res.get("articles", []))
    except Exception as e:
        logger.exception("Error fetching general news")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
