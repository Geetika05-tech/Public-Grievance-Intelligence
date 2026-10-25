"""
Database layer — SQLite (zero setup, fine for an MVP).

Stores exactly the record shape from the plan:
complaint_id, text, date, category, location, duration, urgency,
sentiment, summary, key_phrases, embedding, cluster_id, department.

Embeddings are stored as a JSON-encoded list of floats — simplest thing
that works for a few thousand rows. If the dataset grows a lot, swap
this for a vector DB (e.g. FAISS/Chroma) without changing the API layer.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "grievances.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    date TEXT NOT NULL,
    category TEXT,
    locations TEXT,       -- JSON list
    duration TEXT,
    urgency TEXT,
    sentiment TEXT,
    summary TEXT,
    key_phrases TEXT,     -- JSON list
    embedding TEXT,        -- JSON list of floats
    cluster_id INTEGER,
    department TEXT
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def insert_complaint(record: dict) -> int:
    """record should contain the analysis output for one complaint."""
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO complaints
            (text, date, category, locations, duration, urgency, sentiment,
             summary, key_phrases, embedding, cluster_id, department)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["text"],
            record.get("date", datetime.now(timezone.utc).isoformat()),
            record.get("category"),
            json.dumps(record.get("locations", [])),
            record.get("duration"),
            record.get("urgency"),
            record.get("sentiment"),
            record.get("summary"),
            json.dumps(record.get("key_phrases", [])),
            json.dumps(record.get("embedding", [])),
            record.get("cluster_id"),
            record.get("department"),
        ),
    )
    conn.commit()
    complaint_id = cur.lastrowid
    conn.close()
    return complaint_id


def get_all_complaints() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM complaints ORDER BY date ASC").fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["locations"] = json.loads(d["locations"] or "[]")
        d["key_phrases"] = json.loads(d["key_phrases"] or "[]")
        d["embedding"] = json.loads(d["embedding"] or "[]")
        result.append(d)
    return result


def update_cluster_assignments(cluster_map: dict[int, int]):
    """cluster_map: {complaint_id: cluster_id}"""
    conn = get_connection()
    conn.executemany(
        "UPDATE complaints SET cluster_id = ? WHERE complaint_id = ?",
        [(cluster_id, complaint_id) for complaint_id, cluster_id in cluster_map.items()],
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
