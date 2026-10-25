"""
STAGE 1 — Text Preprocessing
Day 2 of the build plan.

Transformer / sentence-embedding models work best on natural, un-mangled text,
so this module intentionally stays light: whitespace cleanup, basic
normalization, and light punctuation fixes. No stemming, no stopword removal,
no lowercasing-everything — that would hurt the downstream classifier,
NER, and embedding quality.
"""

import re


def clean_text(raw_text: str) -> str:
    """
    Normalize a raw complaint string before it enters the NLP pipeline.

    - Collapses extra whitespace/newlines
    - Strips leading/trailing spaces
    - Fixes repeated punctuation ("!!!" -> "!")
    - Keeps casing and stopwords intact (transformers use them as signal)
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Collapse multiple whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)

    # Collapse repeated punctuation, e.g. "!!!" -> "!", "??" -> "?"
    text = re.sub(r"([!?.])\1{1,}", r"\1", text)

    # Remove control / non-printable characters
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)

    return text.strip()


def basic_sanity_check(text: str) -> bool:
    """
    Very light guard so empty/garbage submissions don't reach the models.
    Returns True if the complaint text is usable.
    """
    if not text:
        return False
    if len(text.strip()) < 5:
        return False
    return True


if __name__ == "__main__":
    sample = "  There has been   no water supply!!! in MVP Colony  for four days.\n\n"
    print(repr(clean_text(sample)))
