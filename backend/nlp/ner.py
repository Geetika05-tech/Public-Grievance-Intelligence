"""
STAGE 3 — Named Entity Recognition / Key Information Extraction
Day 2 of the build plan.

Uses spaCy's pretrained en_core_web_sm model for generic entities
(location, dates/durations) plus a small set of regex rules for
things spaCy doesn't reliably catch out of the box, like
"four days" style durations phrased informally.

Install the model once with:
    python -m spacy download en_core_web_sm
"""

import re
import spacy

_nlp = None

DURATION_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(day|days|week|weeks|month|months|hour|hours)\b",
    re.IGNORECASE,
)


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Run: "
                "python -m spacy download en_core_web_sm"
            ) from e
    return _nlp


def extract_entities(text: str) -> dict:
    """
    Extract structured info from a complaint.

    Returns:
        {
            "locations": [...],
            "duration": "four days" | None,
            "dates": [...],
            "organizations": [...],
        }
    """
    nlp = _get_nlp()
    doc = nlp(text)

    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    duration_match = DURATION_PATTERN.search(text)
    duration = duration_match.group(0) if duration_match else None

    return {
        "locations": list(dict.fromkeys(locations)),   # de-duplicate, keep order
        "duration": duration,
        "dates": list(dict.fromkeys(dates)),
        "organizations": list(dict.fromkeys(organizations)),
    }


if __name__ == "__main__":
    sample = (
        "There has been no water supply in MVP Colony for four days. "
        "I have complained to the local office twice but nothing has been done."
    )
    print(extract_entities(sample))
