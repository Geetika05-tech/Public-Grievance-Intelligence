"""
STAGE 8 — Complaint Clustering
Day 6 of the build plan.

Similarity (Day 5) answers "what's like THIS complaint?".
Clustering answers "what groups of issues exist across ALL complaints?".

Uses K-Means on the sentence embeddings — simplest thing that works
when you have a rough idea of how many issue-groups exist (or can just
try a few values of k). HDBSCAN is noted as an alternative for when you
don't want to fix the number of clusters ahead of time; swap it in here
if you have time.
"""

import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    """
    Run K-Means over complaint embeddings.
    Returns (labels, model). labels[i] is the cluster id of embeddings[i].
    """
    if len(embeddings) < n_clusters:
        # Not enough data yet — put everything in one cluster.
        return np.zeros(len(embeddings), dtype=int), None

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(embeddings)
    return labels, model


def choose_k_by_silhouette(embeddings: np.ndarray, k_range=range(2, 8)):
    """
    Try a range of k values and return the one with the best silhouette
    score — a simple, defensible way to pick "how many issue groups
    exist" instead of guessing. Useful for your Day 13 evaluation.
    """
    best_k, best_score = None, -1
    for k in k_range:
        if k >= len(embeddings):
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def label_clusters(texts: list[str], labels: np.ndarray, key_phrase_fn=None) -> dict[int, str]:
    """
    Give each cluster a human-readable name for the dashboard by looking
    at the most frequent category-ish words across its complaints.
    If you already have a `category` field from the classifier, prefer
    using the majority category per cluster instead of this.
    """
    cluster_names = {}
    for cluster_id in sorted(set(labels)):
        cluster_texts = [t for t, l in zip(texts, labels) if l == cluster_id]
        words = Counter()
        for t in cluster_texts:
            for w in t.lower().split():
                w = w.strip(".,!?")
                if len(w) > 4:
                    words[w] += 1
        top_word = words.most_common(1)[0][0] if words else "misc"
        cluster_names[int(cluster_id)] = top_word.capitalize()
    return cluster_names


def majority_category_labels(categories: list[str], labels: np.ndarray) -> dict[int, str]:
    """
    Preferred labeling approach once you have classifier categories:
    name each cluster after its most common category, e.g. Cluster 0 -> "Water Supply".
    """
    cluster_names = {}
    for cluster_id in sorted(set(labels)):
        cats = [c for c, l in zip(categories, labels) if l == cluster_id]
        cluster_names[int(cluster_id)] = Counter(cats).most_common(1)[0][0] if cats else "Unknown"
    return cluster_names


if __name__ == "__main__":
    from ..nlp.embeddings import embed_batch  # run as: python -m backend.clustering.clustering

    texts = [
        "No water supply in MVP Colony for four days.",
        "Water pressure is zero on our street for a week.",
        "Massive pothole near the school gate.",
        "Road full of cracks after the rains.",
        "Power cuts every evening this week.",
        "Transformer sparking at night, unsafe.",
    ]
    categories = ["Water Supply", "Water Supply", "Roads", "Roads", "Electricity", "Electricity"]

    emb = embed_batch(texts)
    labels, _ = cluster_embeddings(emb, n_clusters=3)
    print("Cluster labels:", labels)
    print("Named by category:", majority_category_labels(categories, labels))
