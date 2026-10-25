# Public Grievance Intelligence

An NLP pipeline that takes a public complaint, classifies it, extracts key
information, scores urgency/sentiment, summarizes it, finds similar past
complaints, clusters complaints into issue groups, and detects emerging
issues over time — with a 3-screen frontend and an admin dashboard.

This repo is a **working starter build** of the pipeline described in the
project plan. Every stage has a runnable file with a `if __name__ ==
"__main__"` demo so you can test it in isolation before wiring it into the
API.

## 0. One-time setup

```bash
cd Public-Grievance-Intelligence
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

## 1. Test each stage on its own (recommended order)

```bash
python backend/nlp/preprocessing.py
python backend/nlp/classifier.py        # trains + saves the model on first run
python backend/nlp/ner.py
python backend/nlp/sentiment.py
python backend/nlp/embeddings.py        # downloads the sentence-transformer model on first run
python backend/nlp/similarity.py
python backend/nlp/summarizer.py
python -m backend.clustering.clustering
python -m backend.clustering.emerging_issues
python backend/database/database.py     # creates backend/database/grievances.db
```

If any of these fail with a missing-package error, re-check step 0 — the
`requirements.txt` pins the exact versions this code was written against.

## 2. Run the full pipeline via the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (double-click it, or
serve the `frontend/` folder with any static file server). Submit a
complaint on Screen 1, check Screen 2 for similar complaints, and Screen 3
for the dashboard.

Seed a bit of history first so similarity/clustering have something to work
with — submit 8-10 varied complaints (water, roads, electricity, sanitation)
before expecting interesting similarity or cluster results, and hit
"Run Clustering" on the dashboard once you have ~10+ complaints stored.

## 3. Project structure

```
Public-Grievance-Intelligence/
├── backend/
│   ├── main.py                    # FastAPI app — wires every stage together
│   ├── nlp/
│   │   ├── preprocessing.py       # Stage 1 — text cleaning
│   │   ├── classifier.py          # Stage 2 — category classification (TF-IDF + LogReg)
│   │   ├── ner.py                 # Stage 3 — location/duration/entity extraction (spaCy)
│   │   ├── sentiment.py           # Stage 4 — urgency + sentiment scoring
│   │   ├── summarizer.py          # Stage 7 — extractive summary + key phrases
│   │   ├── embeddings.py          # Stage 5 — sentence embeddings
│   │   └── similarity.py          # Stage 6 — similar complaint search (+ TF-IDF baseline)
│   ├── clustering/
│   │   ├── clustering.py          # Stage 8 — K-Means issue clustering
│   │   └── emerging_issues.py     # Stage 9 — weekly counts + growth-based flagging
│   ├── database/
│   │   └── database.py            # SQLite storage layer
│   └── requirements.txt
│
├── frontend/
│   ├── index.html                 # Screen 1 — Submit & Analyze
│   ├── similarity.html            # Screen 2 — Similar Complaints
│   ├── dashboard.html             # Screen 3 — Admin Dashboard
│   ├── css/style.css
│   └── js/
│       ├── app.js
│       ├── similarity.js
│       └── dashboard.js
│
├── data/
│   └── complaints.csv             # sample training/seed data (28 rows, 4 categories)
│
├── notebooks/                     # put your exploration/evaluation notebooks here
├── README.md
└── .gitignore
```

## 4. Your 14-day build checklist

Work through this in order. Each day builds on the previous one — resist
the urge to jump to the frontend early.

### Days 1-3 — Dataset + Core NLP
- [ ] **Day 1** — Set up the environment (done above). Confirm your real
      complaint dataset — replace/extend `data/complaints.csv` with your
      actual data (target 500-5,000 rows if you have them; the sample CSV
      here is just a seed).
- [ ] **Day 2** — Get `preprocessing.py` → `classifier.py` → `ner.py`
      working end-to-end on a few sample complaints from the command line.
- [ ] **Day 3** — Get `sentiment.py` (urgency + sentiment) working. Don't
      chase perfect accuracy — get the whole basic pipeline running first.

### Days 4-6 — Strongest NLP features
- [ ] **Day 4** — `embeddings.py` — confirm two similar complaints score a
      high cosine similarity and two unrelated ones score low.
- [ ] **Day 5** — `similarity.py` — retrieve top-5 similar complaints for a
      new complaint. Also run the TF-IDF baseline (`tfidf_top_k`) side by
      side — this comparison is genuinely good report/interview material.
- [ ] **Day 6** — `clustering.py` — group your dataset into clusters. Use
      `choose_k_by_silhouette()` to pick a sensible number of clusters
      instead of guessing, and inspect a few complaints per cluster to
      sanity-check the grouping.

### Days 7-9 — Summarization + Emerging Issues
- [ ] **Day 7** — `summarizer.py` — check that summaries and key phrases
      look reasonable on a handful of longer complaints.
- [ ] **Day 8** — `emerging_issues.py` — start with the simple rule
      (`min_complaints` + `growth_threshold`), test it against the
      simulated example in the file's `__main__` block, then try it on
      real weekly counts once you have enough data.
- [ ] **Day 9** — Wire everything into `main.py` and confirm one API
      request (`POST /api/complaints`) runs the full pipeline and returns
      a complete result.

### Days 10-12 — Frontend
- [ ] **Day 10** — Get Screen 1 (`index.html` + `app.js`) submitting to the
      API and showing category/urgency/entities/summary.
- [ ] **Day 11** — Get Screen 2 (`similarity.html`) showing similar
      complaints and the issue cluster.
- [ ] **Day 12** — Get Screen 3 (`dashboard.html`) showing totals, the
      category chart, and emerging issues. Hit `POST /api/cluster`
      periodically (there's already a button for it) rather than on every
      submission.

### Days 13-14 — Testing + Resume Quality
- [ ] **Day 13** — Evaluate, don't just eyeball:
  - Classification: accuracy/precision/recall/F1 (already printed by
    `classifier.py` on every training run)
  - Similarity: TF-IDF vs Sentence Transformer, side by side
  - Clustering: silhouette score + manual inspection of cluster coherence
  - Emerging issues: construct a synthetic spike (like the `__main__` demo
    in `emerging_issues.py`) and confirm it gets flagged
- [ ] **Day 14** — Polish: fix loading/error states, empty inputs,
      duplicate-looking results; write up the README/architecture diagram;
      take screenshots; record a 2-3 minute demo.

## 5. Explicitly out of scope for the 2-week MVP

Don't add these — they inflate scope without strengthening the NLP:
login/auth, email/SMS notifications, a mobile app, live government API
integration, complex maps, training a transformer from scratch, or a
microservices architecture. A vector database (FAISS/Chroma) is a
reasonable post-MVP upgrade once SQLite's JSON-blob embeddings feel slow.
