"""
STAGE 6 — Similar Complaint Detection
Day 5 of the build plan.

Given a new complaint, find the most similar past complaints by
comparing embeddings with cosine similarity.

Also includes a TF-IDF baseline (`tfidf_top_k`) purely so you can run
the "TF-IDF vs Sentence Transformer" comparison the plan calls out for
Day 5 / Day 13 — this is genuinely good material for a report or
interview because it shows you understand *why* embeddings beat
keyword overlap on paraphrased complaints.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from .embeddings import embed_text, embed_batch


def top_k_similar(new_text: str, corpus_texts: list[str], corpus_embeddings: np.ndarray, k: int = 5):
    """
    Embedding-based similarity search.

    corpus_embeddings must already be normalized (embed_batch does this).
    Returns a list of {"text": ..., "index": ..., "similarity": ...} sorted
    by similarity, highest first.
    """
    if len(corpus_texts) == 0:
        return []

    query_vec = embed_text(new_text)
    sims = corpus_embeddings @ query_vec  # cosine sim, since vectors are normalized

    top_idx = np.argsort(-sims)[:k]
    return [
        {"index": int(i), "text": corpus_texts[i], "similarity": round(float(sims[i]), 3)}
        for i in top_idx
    ]


def tfidf_top_k(new_text: str, corpus_texts: list[str], k: int = 5):
    """
    Baseline similarity using TF-IDF + cosine similarity — no semantic
    understanding, only shared words/n-grams. Use this side-by-side with
    top_k_similar() to demonstrate the difference in your evaluation.
    """
    if len(corpus_texts) == 0:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    query_vec = vectorizer.transform([new_text])

    sims = sk_cosine(query_vec, corpus_matrix)[0]
    top_idx = np.argsort(-sims)[:k]
    return [
        {"index": int(i), "text": corpus_texts[i], "similarity": round(float(sims[i]), 3)}
        for i in top_idx
    ]


if __name__ == "__main__":
    corpus = [
        "There has been no water supply in MVP Colony for four days.",
        "Our street has had zero water pressure for a week now.",
        "Massive pothole near the school gate is a danger to children.",
        "Power has been going out every evening for the past week.",
    ]
    corpus_emb = embed_batch(corpus)
    query = "No drinking water for three days in our area, please help."

    print("=== Sentence Transformer results ===")
    for r in top_k_similar(query, corpus, corpus_emb, k=3):
        print(r)

    print("\n=== TF-IDF baseline results ===")
    for r in tfidf_top_k(query, corpus, k=3):
        print(r)
