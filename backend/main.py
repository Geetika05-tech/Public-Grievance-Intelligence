"""
Day 9 — Connect Everything

One FastAPI app that ties every stage together:

    Complaint -> Preprocessing -> Classification -> NER -> Urgency/Sentiment
              -> Summary -> Embedding -> Similarity -> Department -> Database

Run with (from the project root):
    uvicorn backend.main:app --reload --port 8000

Then open frontend/index.html in a browser (or serve it with any static
file server) and it will call this API at http://localhost:8000.
"""

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from backend.nlp.preprocessing import clean_text, basic_sanity_check
from backend.nlp.classifier import classify_complaint
from backend.nlp.ner import extract_entities
from backend.nlp.sentiment import score_sentiment, score_urgency
from backend.nlp.summarizer import summarize, extract_key_phrases
from backend.nlp.embeddings import embed_text, embed_batch
from backend.nlp.similarity import top_k_similar
from backend.clustering.clustering import cluster_embeddings, majority_category_labels
from backend.clustering.emerging_issues import weekly_counts_by_cluster, detect_emerging_issues
from backend.database.database import (
    init_db, insert_complaint, get_all_complaints, update_cluster_assignments,
)

app = FastAPI(title="Public Grievance Intelligence API")

# Allow the plain HTML/JS frontend (served from file:// or a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEPARTMENT_MAP = {
    "Water Supply": "Water Department",
    "Roads": "Roads Department",
    "Electricity": "Electrical Department",
    "Sanitation": "Sanitation Department",
}


@app.on_event("startup")
def startup():
    init_db()


class ComplaintIn(BaseModel):
    text: str


@app.post("/api/complaints")
def submit_complaint(payload: ComplaintIn):
    """
    The core vertical-slice endpoint (Day 1 milestone target, fully
    wired by Day 9): takes raw complaint text, runs it through the
    whole pipeline, stores it, and returns the full analysis.
    """
    raw_text = payload.text
    text = clean_text(raw_text)

    if not basic_sanity_check(text):
        return {"error": "Complaint text is too short or empty."}

    classification = classify_complaint(text)
    entities = extract_entities(text)
    sentiment = score_sentiment(text)
    urgency = score_urgency(text)
    summary = summarize(text)
    key_phrases = extract_key_phrases(text)
    embedding = embed_text(text)

    department = DEPARTMENT_MAP.get(classification["category"], "General Department")

    # Similar-complaint search against everything stored so far.
    existing = get_all_complaints()
    existing_texts = [c["text"] for c in existing]
    existing_embeddings = np.array([c["embedding"] for c in existing]) if existing else np.empty((0, len(embedding)))
    similar = top_k_similar(text, existing_texts, existing_embeddings, k=5) if existing else []

    record = {
        "text": text,
        "date": datetime.now(timezone.utc).isoformat(),
        "category": classification["category"],
        "locations": entities["locations"],
        "duration": entities["duration"] or (entities["dates"][0] if entities["dates"] else None),
        "urgency": urgency["urgency"],
        "sentiment": sentiment["label"],
        "summary": summary,
        "key_phrases": key_phrases,
        "embedding": embedding.tolist(),
        "cluster_id": None,   # assigned later by /api/cluster
        "department": department,
    }
    complaint_id = insert_complaint(record)

    return {
        "complaint_id": complaint_id,
        "category": classification["category"],
        "confidence": classification["confidence"],
        "entities": entities,
        "urgency": urgency,
        "sentiment": sentiment,
        "summary": summary,
        "key_phrases": key_phrases,
        "department": department,
        "similar_complaints": similar,
    }


@app.post("/api/cluster")
def run_clustering(n_clusters: int = 4):
    """
    Re-cluster every stored complaint. Call this periodically (e.g. a
    cron job, or a button on the admin dashboard) rather than on every
    single submission — clustering needs a reasonable batch of data.
    """
    complaints = get_all_complaints()
    if len(complaints) < 2:
        return {"error": "Not enough complaints to cluster yet."}

    embeddings = np.array([c["embedding"] for c in complaints])
    labels, _ = cluster_embeddings(embeddings, n_clusters=n_clusters)
    categories = [c["category"] for c in complaints]
    cluster_names = majority_category_labels(categories, labels)

    cluster_map = {c["complaint_id"]: int(l) for c, l in zip(complaints, labels)}
    update_cluster_assignments(cluster_map)

    return {"clusters": cluster_names, "assignments": cluster_map}


@app.get("/api/emerging-issues")
def emerging_issues(min_complaints: int = 10, growth_threshold: float = 0.3):
    complaints = get_all_complaints()
    if not complaints:
        return {"issues": []}

    categories = [c["category"] for c in complaints]
    labels = [c["cluster_id"] for c in complaints]
    cluster_names = {}
    for cat, label in zip(categories, labels):
        if label is not None:
            cluster_names.setdefault(label, cat)

    weekly = weekly_counts_by_cluster(complaints)
    issues = detect_emerging_issues(weekly, cluster_names, min_complaints, growth_threshold)
    return {"issues": issues}


@app.get("/api/dashboard")
def dashboard_summary():
    """Aggregate stats for Screen 3 — the admin dashboard."""
    complaints = get_all_complaints()

    total = len(complaints)
    by_urgency = {"High": 0, "Medium": 0, "Low": 0}
    by_category = {}

    for c in complaints:
        if c["urgency"] in by_urgency:
            by_urgency[c["urgency"]] += 1
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1

    return {
        "total_complaints": total,
        "by_urgency": by_urgency,
        "by_category": by_category,
    }


@app.get("/api/complaints")
def list_complaints():
    return {"complaints": get_all_complaints()}
