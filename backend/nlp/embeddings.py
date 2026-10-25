"""
STAGE 5 — Sentence Embeddings
Day 4 of the build plan.

Converts complaint text into a dense vector using a pretrained
Sentence Transformer (no training needed). Similar-meaning complaints
end up close together in this vector space — that's what powers
similarity search (Day 5) and clustering (Day 6).

Model choice: 'all-MiniLM-L6-v2' — small, fast on CPU, good enough
quality for an MVP. Swap to a larger model later if you have GPU time.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single complaint string. Returns a 1D numpy array."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed many complaints at once (much faster than one-by-one)."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)


if __name__ == "__main__":
    a = embed_text("No water supply in MVP Colony for four days.")
    b = embed_text("Water pipeline burst near the main road, water is being wasted.")
    print("Embedding shape:", a.shape)
    cosine_sim = float(np.dot(a, b))  # vectors are already normalized
    print("Cosine similarity between two water complaints:", round(cosine_sim, 3))
