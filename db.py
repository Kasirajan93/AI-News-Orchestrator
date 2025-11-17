# db.py — Final working version
import sqlite3
from datetime import datetime
import os

DB_PATH = "ainews.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            content TEXT,
            source TEXT,
            published TEXT,
            tags TEXT,
            quality INTEGER,
            sentiment REAL,
            category TEXT,
            fetched_at TEXT
        )
    """)

    conn.commit()
    conn.close()

# call on import
init_db()

def upsert_article(article):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO articles
        (url, title, summary, content, source, published, tags,
         quality, sentiment, category, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        article.get("url"),
        article.get("title"),
        article.get("summary"),
        article.get("content"),
        article.get("source"),
        article.get("published"),
        ",".join(article.get("tags", [])),
        article.get("quality"),
        article.get("sentiment"),
        article.get("category"),
        article.get("fetched_at")
    ))

    conn.commit()
    conn.close()

def fetch_latest(n=200):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT * FROM articles
        ORDER BY fetched_at DESC
        LIMIT ?
    """, (n,))

    rows = c.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def article_exists(url):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT 1 FROM articles WHERE url=?", (url,))
    exists = c.fetchone() is not None
    conn.close()
    return exists
