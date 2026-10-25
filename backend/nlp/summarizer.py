"""
STAGE 7 — Summarization + Key Phrase Extraction
Day 7 of the build plan.

Rather than pulling in a heavy abstractive model (and the fine-tuning
time that comes with it), this is a self-contained extractive
summarizer: split the complaint into sentences, embed each one, and
rank sentences by how "central" they are to the whole complaint
(a small TextRank variant using cosine similarity instead of raw word
overlap). The 1-2 most central sentences become the summary.

Key phrases use a simple TF-IDF-over-single-document trick: score each
candidate n-gram by its TF-IDF weight against a background corpus.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import embed_batch


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter — good enough for short complaints.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def summarize(text: str, max_sentences: int = 2) -> str:
    """
    Extractive summary: pick the most central sentence(s) by embedding
    similarity to every other sentence (TextRank-style), preserving
    their original order in the output.
    """
    sentences = _split_sentences(text)

    if len(sentences) <= max_sentences:
        return text.strip()

    embeddings = embed_batch(sentences)
    sim_matrix = embeddings @ embeddings.T  # cosine sim (already normalized)
    np.fill_diagonal(sim_matrix, 0)

    scores = sim_matrix.sum(axis=1)  # "centrality" score per sentence
    top_idx = sorted(np.argsort(-scores)[:max_sentences])  # keep original order

    return " ".join(sentences[i] for i in top_idx)


def extract_key_phrases(text: str, background_corpus: list[str] | None = None, top_n: int = 5) -> list[str]:
    """
    Score candidate 1-3 word phrases by TF-IDF weight. If a background
    corpus (e.g. a sample of past complaints) is supplied, it's used to
    compute realistic IDF weights; otherwise falls back to a single-doc
    heuristic.
    """
    corpus = (background_corpus or []) + [text]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=2000)
    matrix = vectorizer.fit_transform(corpus)

    doc_vector = matrix[-1].toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    top_idx = np.argsort(-doc_vector)[:top_n]
    return [feature_names[i] for i in top_idx if doc_vector[i] > 0]


if __name__ == "__main__":
    sample = (
        "There has been no water supply in MVP Colony for four days. "
        "I have complained to the local office twice but nothing has been done. "
        "This is affecting daily life for the entire street."
    )
    print("Summary:", summarize(sample))
    print("Key phrases:", extract_key_phrases(sample))
