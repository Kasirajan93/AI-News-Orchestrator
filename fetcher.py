# fetcher.py — CLOUD SAFE VERSION (NO extractor, NO newspaper3k)
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
from db import upsert_article, article_exists
from summarizer import summarize_text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# --------------------------
# Helper: extract readable content
# --------------------------
def extract_content(url):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")

        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        clean = "\n".join([p for p in paragraphs if len(p) > 40])

        return clean[:4000] if clean else ""
    except:
        return ""


# --------------------------
# Tag extractor
# --------------------------
def simple_tags(text, top_n=5):
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vec.fit_transform([text])
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        top = terms[scores.argsort()[-top_n:]][::-1]
        return list(top)
    except:
        return []


# --------------------------
# Sentiment, quality, category
# --------------------------
def compute_sentiment(text):
    try:
        s = analyzer.polarity_scores(text)
        return float(s["compound"])
    except:
        return 0.0


def compute_quality(text):
    try:
        return min(1000, len(text.split()))
    except:
        return 0


def basic_category_from_text(text):
    t = text.lower()
    if "election" in t or "minister" in t or "president" in t:
        return "Politics"
    if "tech" in t or "ai" in t:
        return "Technology"
    if "health" in t or "covid" in t:
        return "Health"
    if "movie" in t or "film" in t:
        return "Entertainment"
    return "World"


# --------------------------
# MAIN FEED PROCESSOR
# --------------------------
def process_feed(feed_url, max_items=20):
    print(f"[FEED] Processing: {feed_url}")

    parsed = feedparser.parse(feed_url)
    if not parsed.entries:
        print("[FEED] No entries found.")
        return 0

    saved = 0

    for entry in parsed.entries[:max_items]:
        url = entry.get("link")
        if not url or article_exists(url):
            continue

        title = entry.get("title", "")
        published = entry.get("published", "")
        source = urlparse(url).netloc

        content = extract_content(url)
        if not content:
            continue

        summary = summarize_text(content)
        tags = simple_tags(content)
        sentiment = compute_sentiment(content)
        quality = compute_quality(content)
        category = basic_category_from_text(content)

        article = {
            "url": url,
            "title": title,
            "published": published,
            "source": source,
            "content": content,
            "summary": summary,
            "tags": tags,
            "quality": quality,
            "sentiment": sentiment,
            "category": category,
            "fetched_at": datetime.utcnow().isoformat()
        }

        upsert_article(article)
        saved += 1
        print(f"[FEED] Saved: {title}")

    print(f"[FEED] Done. Saved = {saved}")
    return saved
