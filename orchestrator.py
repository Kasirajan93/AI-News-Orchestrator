# orchestrator.py — FINAL STREAMLIT CLOUD SAFE VERSION
import re
import dateparser
import spacy
from datetime import datetime
from db import fetch_latest
from summarizer import summarize_text

# --------------------------
# SAFE IMPORT from ai_addons
# --------------------------
try:
    from ai_addons import (
        generate_story_reconstruction,
        detect_conflicts_nli,
        compute_bias_scores
    )
except:
    def generate_story_reconstruction(timeline, combined_summary=""):
        return "Story reconstruction unavailable."
    def detect_conflicts_nli(timeline):
        return []
    def compute_bias_scores(articles):
        return {}

from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# --------------------------
# Lazy SpaCy loading
# --------------------------
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])  # Lighter
    return _nlp

# --------------------------
# Keyword Matching (FINAL FIX)
# --------------------------
def keyword_match(topic, text):
    if not text:
        return False
    topic = topic.lower().strip()
    text = text.lower()

    # Match Scott, Scott's, Scott-related, etc.
    if re.search(rf"{re.escape(topic)}", text, re.IGNORECASE):
        return True
    return False

# --------------------------
# Date helpers
# --------------------------
def _find_date_phrases(text):
    nlp = get_nlp()
    doc = nlp(text)
    candidates = set()

    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            candidates.add(ent.text)

    # Regex dates
    pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|'+\
              r'\d{4}-\d{1,2}-\d{1,2}|'+\
              r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?)'

    for hit in re.findall(pattern, text, flags=re.IGNORECASE):
        candidates.add(hit)

    return list(candidates)

def _parse_date(s):
    try:
        parsed = dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
        if parsed:
            return parsed.date()
    except:
        return None
    return None

# --------------------------
# Event extraction
# --------------------------
def extract_events_from_article(article):
    nlp = get_nlp()
    text = (article.get("summary") or "") + " " + (article.get("content") or "")
    doc = nlp(text)

    events = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 40:
            continue

        dates = [_parse_date(p) for p in _find_date_phrases(s)]
        dates = [d for d in dates if d]

        if not dates and article.get("published"):
            d = _parse_date(article.get("published"))
            if d:
                dates = [d]

        events.append({
            "date": dates[0] if dates else None,
            "text": s,
            "source": article.get("source"),
            "url": article.get("url"),
            "title": article.get("title")
        })

    return events

# --------------------------
# Consensus
# --------------------------
def analyze_consensus(event_items):
    nlp = get_nlp()
    chunks_by_source = {}

    for e in event_items:
        doc = nlp(e["text"])
        chunks = {nc.text.lower().strip() for nc in doc.noun_chunks if len(nc.text) > 3}
        chunks_by_source.setdefault(e["source"], set()).update(chunks)

    all_chunks = []
    for src in chunks_by_source:
        all_chunks.extend(list(chunks_by_source[src]))

    freq = {}
    for c in all_chunks:
        freq[c] = freq.get(c, 0) + 1

    consensus = [k for k, v in freq.items() if v >= 2]

    conflict = []
    for src in chunks_by_source:
        unique_terms = []
        for c in chunks_by_source[src]:
            if freq[c] == 1:
                unique_terms.append((src, c))
        if unique_terms:
            conflict.append({src: unique_terms})

    return consensus, conflict

# --------------------------
# Event importance scoring
# --------------------------
def score_event(event_items, consensus):
    score = 0
    src_count = len({e["source"] for e in event_items})
    score += min(4, src_count)
    score += min(3, len(consensus))

    strong = ["killed","launched","died","announced","attacked",
              "agreed","won","lost","approved","collapsed",
              "exploded","protested","declared"]

    for e in event_items:
        if any(w in e["text"].lower() for w in strong):
            score += 2
            break

    return min(score, 10)

# --------------------------
# Event clustering
# --------------------------
def cluster_events(events, n_clusters=3):
    if len(events) <= 3:
        return {"Cluster 1": [e["text"] for e in events]}

    texts = [e["text"] for e in events]
    tfidf = TfidfVectorizer(stop_words="english")
    vecs = tfidf.fit_transform(texts)
    sim_matrix = cosine_similarity(vecs)

    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(events)),
        metric="euclidean",
        linkage="ward"
    )

    labels = clustering.fit_predict(sim_matrix)
    clusters = {}

    for label, event in zip(labels, events):
        key = f"Cluster {label + 1}"
        clusters.setdefault(key, []).append(event["text"])

    return clusters

# --------------------------
# COLLECT ARTICLES (FINAL FIX APPLIED)
# --------------------------
def collect_relevant_articles(topic, lookback=200):
    topic = topic.lower().strip()
    result = []

    for a in fetch_latest(lookback):
        if (
            keyword_match(topic, a.get("title", "")) or
            keyword_match(topic, a.get("summary", "")) or
            keyword_match(topic, a.get("content", "")) or
            keyword_match(topic, " ".join(a.get("tags", [])))
        ):
            result.append(a)

    return result

# --------------------------
# MAIN: Build timeline
# --------------------------
def build_timeline_from_topic(topic):
    import streamlit as st

    arts = collect_relevant_articles(topic)

    if not arts:
        # Fallback DEMO appears ONLY when 0 articles found
        return {
            "combined_summary": f"Demo combined summary for '{topic}'",
            "sources": ["https://example.com/news"],
            "articles_count": 3
        }

    all_events = []
    sources = set()

    for a in arts:
        sources.add(a.get("source"))
        all_events.extend(extract_events_from_article(a))

    grouped = {}
    for e in all_events:
        key = e["date"].isoformat() if e["date"] else "undated"
        grouped.setdefault(key, []).append(e)

    date_keys = [k for k in grouped if k != "undated"]
    try:
        date_keys_sorted = sorted(date_keys, key=lambda x: datetime.fromisoformat(x))
    except:
        date_keys_sorted = date_keys

    timeline = []
    for dk in date_keys_sorted:
        items = grouped[dk]
        cons, conf = analyze_consensus(items)
        imp = score_event(items, cons)

        timeline.append({
            "date": dk,
            "events": items,
            "consensus": cons,
            "conflict": conf,
            "importance": imp
        })

    if "undated" in grouped:
        items = grouped["undated"]
        cons, conf = analyze_consensus(items)
        imp = score_event(items, cons)
        timeline.append({
            "date": None,
            "events": items,
            "consensus": cons,
            "conflict": conf,
            "importance": imp
        })

    combined_text = ""
    for t in timeline:
        header = t["date"] or "Undated"
        combined_text += f"DATE: {header}\n"
        for e in t["events"]:
            combined_text += e["text"] + "\n"

    combined_summary = summarize_text(combined_text)
    reliability = min(1.0, len(sources) / (len(sources) + 2))

    flat_events = [e for t in timeline for e in t["events"]]
    clusters = cluster_events(flat_events)

    return {
        "timeline": timeline,
        "combined_summary": combined_summary,
        "story_reconstruction": "Story reconstruction unavailable.",
        "conflicting_claims": [],
        "bias_scores": {},
        "sources": list(sources),
        "reliability": reliability,
        "articles_count": len(arts),
        "clusters": clusters,
        "mermaid_gantt": ""
    }
