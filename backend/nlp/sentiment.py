"""
STAGE 4 — Urgency & Sentiment
Day 3 of the build plan.

The planning doc is explicit: don't over-engineer this for the MVP.
There is no labeled urgency dataset, so instead of training a model on
guesses, this module combines:

  1. A small negative-sentiment lexicon (proxy for "how upset" the
     complaint sounds)
  2. A handful of clearly-documented severity signals (repeat
     complaints, safety/health keywords, long durations, "urgent"
     language)

into a transparent, explainable urgency score. This is easy to defend
in an interview: every signal is named, not hidden inside a black box.
Swap this for a trained classifier later if you get labeled data.
"""

import re

NEGATIVE_WORDS = {
    "no", "not", "never", "broken", "unsafe", "dangerous", "overflowing",
    "smell", "foul", "sparking", "leaking", "flooded", "damaged", "dirty",
    "unhygienic", "muddy", "cracked", "fell", "exposed", "hazard", "urgent",
    "emergency", "worse", "worst", "again", "twice", "repeatedly",
}

SAFETY_HEALTH_WORDS = {
    "sparking", "exposed", "accident", "danger", "dangerous", "unsafe",
    "health", "hazard", "sewage", "disease", "mosquitoes", "fire", "electrocut",
}

REPEAT_COMPLAINT_PATTERN = re.compile(
    r"\b(twice|again|repeatedly|multiple times|several times|many times)\b",
    re.IGNORECASE,
)

LONG_DURATION_PATTERN = re.compile(
    r"\b(\d+|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\b",
    re.IGNORECASE,
)


def score_sentiment(text: str) -> dict:
    """
    Very small lexicon-based negativity score in [0, 1].
    Higher = more negative / distressed tone.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return {"negativity": 0.0, "label": "Neutral"}

    hits = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = min(1.0, hits / max(3, len(words) * 0.15))

    if score >= 0.5:
        label = "Negative"
    elif score >= 0.2:
        label = "Mixed"
    else:
        label = "Neutral"

    return {"negativity": round(score, 2), "label": label}


def score_urgency(text: str) -> dict:
    """
    Combine documented signals into a Low / Medium / High urgency label.

    Signals (each adds points, capped):
      +2  safety/health keyword present
      +1  repeat-complaint language ("twice", "again")
      +1  long duration mentioned (5+ days/weeks/months)
      +1  strongly negative sentiment (>= 0.5)
    """
    lower = text.lower()
    points = 0
    signals = []

    if any(word in lower for word in SAFETY_HEALTH_WORDS):
        points += 2
        signals.append("safety_or_health_keyword")

    if REPEAT_COMPLAINT_PATTERN.search(lower):
        points += 1
        signals.append("repeat_complaint_language")

    if LONG_DURATION_PATTERN.search(lower):
        points += 1
        signals.append("long_duration")

    sentiment = score_sentiment(text)
    if sentiment["negativity"] >= 0.5:
        points += 1
        signals.append("strongly_negative_tone")

    if points >= 3:
        level = "High"
    elif points >= 1:
        level = "Medium"
    else:
        level = "Low"

    return {"urgency": level, "score": points, "signals": signals}


if __name__ == "__main__":
    sample = (
        "There has been no water supply in MVP Colony for four days. "
        "I have complained to the local office twice but nothing has been done."
    )
    print(score_sentiment(sample))
    print(score_urgency(sample))
