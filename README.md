# 📰 AI News Orchestrator

Event Timeline Generator using Multi-Source News Aggregation, NLP & AI Summarization
-
<div align="center">








</div>

# 🚀 Project Overview

The AI News Orchestrator reconstructs the full story of any event by aggregating, verifying, and summarizing news from multiple sources — producing a single, chronological, credible event timeline.

Users enter a topic (e.g., "Moon Mission", "COP30", "US Elections").
The system:

Collects articles

Extracts key events

Detects consensus vs conflicts

Generates an AI-driven timeline

Reconstructs the story

Visualizes it with a Gantt-style timeline

This project was built for the GUVI x HCL Hackathon – AI News Orchestrator Track.

# ✨ Key Features

# 🔍 1. Multi-source News Aggregation

RSS feeds / external news sources

Stores scraped articles into local DB

# 🧠 2. Event Extraction

SpaCy NLP sentence analysis

Auto date parsing with dateparser

Entity extraction (People, Organizations)

# ⏳ 3. Chronological Timeline Generation

Groups events by date

Calculates event importance score

Detects milestone patterns

# 📊 4. Gantt Visualization (Mermaid.js)

A clean auto-generated timeline chart like:

gantt
dateFormat YYYY-MM-DD
section 2024-02-12
Govt announces relief : 2024-02-12, 1d

# 🧩 5. Semantic Clustering

Groups similar events using:

TF-IDF

Cosine Similarity

Agglomerative Clustering

# 🧠 6. AI Story Reconstruction (LLM based)

A clean narrative that explains:

What happened

Why

How the story evolved

# ⚠ 7. Fact Conflict Detection (NLI-based)

Detects contradictions among articles.

# 🎭 8. Clickbait / Bias Scoring

Identifies:

sensationalism

subjective tone

biased framing

# 🌍 9. Multi-Language Translation

Instant output translation using GoogleTranslator into:

Tamil

Hindi

French

Spanish

Arabic

German

Chinese

# 📈 10. Deluxe Analytics Dashboard

Includes:

Source reliability board

Emotion trend

Event-density chart

Key actors (NER)

Compression score

Cross-source alignment

# 🏗 Architecture Diagram

# Mermaid Diagram (auto-renders in GitHub):
flowchart LR

A[User Input Topic] --> B[Fetch Latest News Articles from DB/Feeds]
B --> C[NLP Event Extraction<br> (SpaCy + DateParser)]
C --> D[Event Grouping<br>by Date]
D --> E[Consensus + Conflict Detection]
E --> F[Event Importance Scoring]
F --> G[AI Combined Summary<br> & Story Reconstruction]
G --> H[Timeline Construction]
H --> I[Mermaid Gantt Visualization]
G --> J[Deluxe Analytics Dashboard]

# 📁 Folder Structure

AI-News-Orchestrator/
│
├── app.py                 # Streamlit UI
├── orchestrator.py        # Timeline & NLP engine
├── ai_addons.py           # Translation / NLI / Bias scoring
├── fetcher.py             # RSS article ingestion
├── db.py                  # Local DB functions
│
├── feeds.txt              # RSS feeds list
├── requirements.txt       
├── README.md              
│
├── stats/                 # Metrics
├── logs/                  # Log files
└── tmp/                   # Cached timelines

# 🛠 Tech Stack

Python 3.10+

Streamlit Cloud (Deployment)

SpaCy (NLP)

dateparser

scikit-learn (Clustering)

GoogleTranslator (deep-translator)

Matplotlib / Pandas

Mermaid.js
