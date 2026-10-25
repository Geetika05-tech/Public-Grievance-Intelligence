# Public Grievance Intelligence

An NLP pipeline that takes a public complaint, classifies it, extracts key
information, scores urgency/sentiment, summarizes it, finds similar past
complaints, clusters complaints into issue groups, and detects emerging
issues over time — with a 3-screen frontend and an admin dashboard.
 
 ## initial setup

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

