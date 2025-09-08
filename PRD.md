# Technical design (easy-to-build option) — Resume Matching Platform

Great — you want a practical, **easy-to-build technical design** for the resume-matching platform (the one-day / MVP-friendly route). Below I give a clear, minimal architecture that you can implement quickly, the rationale for why it’s easiest, concrete component choices, APIs, data flow, security, testing/acceptance criteria, and a compact implementation plan with estimated effort for a small team.

---

# 1 — Recommendation (single-sentence)

Build a **single FastAPI Python service + ChromaDB vector database + Sentence-Transformers embeddings + simple parser** with S3 for storage and PostgreSQL for metadata. This monolithic-but-modular approach is fastest to implement and easiest to demo, yet productionizable later.

Why easiest:

* Everything is Python — one language and environment.
* Uses pre-trained models (no training required).
* ChromaDB provides persistent storage with SQL-like queries and built-in metadata filtering.
* ChromaDB offers better scalability and production-readiness than FAISS for growing datasets.
* Minimal infra: single VM or managed app (GCP App Engine / AWS EC2 / Heroku).
* ChromaDB supports collections, metadata filtering, and easy migration to cloud deployments.

---

# 2 — High-level architecture (MVP)

```
[Upload UI / ATS Connector] --> [API (FastAPI)] --> 
  ├─ Resume Parser (spaCy / PyPDF2 / docx2txt) → raw_text
  ├─ Embedding Service (sentence-transformers) → vector
  ├─ Metadata DB (Postgres) ← store candidate metadata
  └─ Vector DB (ChromaDB) ← store vectors + metadata with collections
API serves: /upload, /match, /explain, /audit
Storage: S3 (resumes, precomputed artifacts)
Monitoring: Prometheus + Grafana (optional)
Logging: ELK or cloud logs
```

---

# 3 — Component choices (MVP, easiest to implement)

* **API layer**: Python + FastAPI (simple async, auto docs via Swagger).
* **Resume parsing**: `PyPDF2` + `python-docx` (or `docx2txt`) + simple spaCy pipeline for entities.
* **Embeddings**: `sentence-transformers` (model: `all-MiniLM-L6-v2` for speed) — no training.
* **Vector store**: ChromaDB with persistent storage and metadata filtering. Supports collections for organized data management.
* **Metadata DB**: PostgreSQL with JSONB columns for resume fields.
* **Storage**: S3-compatible (AWS S3, GCS, or MinIO for local).
* **Explainability**: Rule-based reasons (skill overlap + semantic contribution) + optional SHAP on small interpretable model later.
* **Bias checks**: Rule-based audits and statistical checks (distribution of top-k by simulated/available demographics). Use `fairlearn` later if needed.
* **Authentication**: API keys / OAuth for ATS integration.
* **Hosting (fastest)**: Single cloud VM or PaaS (Heroku / GCP App Engine). Use Docker for portability.

---

# 3.1 — ChromaDB Integration Details

### Why ChromaDB over FAISS:

* **Persistent Storage**: ChromaDB provides built-in persistence, eliminating the need for custom serialization
* **Metadata Filtering**: Native support for filtering by metadata (skills, experience, etc.) during similarity search
* **Collections**: Organized data management with collections for different data types or versions
* **Production Ready**: Built for production with scalability and reliability features
* **SQL-like Queries**: Familiar query interface with WHERE clauses for metadata filtering
* **Multi-modal Support**: Future-ready for text, image, and other embeddings
* **Built-in Analytics**: Query performance metrics and collection statistics

### ChromaDB Setup Configuration:

```python
# chromadb_client.py example configuration
import chromadb
from chromadb.config import Settings

# Local persistent storage
client = chromadb.PersistentClient(
    path="./data/chromadb",
    settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./data/chromadb"
    )
)

# Create collections
resume_collection = client.get_or_create_collection(
    name="resume_vectors",
    metadata={"description": "Resume embeddings with skills and experience metadata"}
)
```

### Metadata Schema for ChromaDB:

```python
# Metadata stored with each vector in ChromaDB
metadata_schema = {
    "candidate_id": "uuid-string",
    "skills": ["Python", "Machine Learning", "AWS"],
    "experience_years": 5,
    "filename": "john_doe_resume.pdf",
    "job_title": "Data Scientist",
    "education_level": "Masters",
    "location": "San Francisco, CA",
    "created_at": "2024-01-15T10:30:00Z"
}
```

### Sample ChromaDB Operations:

```python
# Add resume to ChromaDB
resume_collection.add(
    ids=[candidate_id],
    embeddings=[embedding_vector],
    metadatas=[metadata_dict],
    documents=[resume_text]
)

# Search with metadata filtering
results = resume_collection.query(
    query_embeddings=[job_embedding],
    n_results=10,
    where={"experience_years": {"$gte": 3}},
    where_document={"$contains": "Python"}
)
```

---

# 4 — Data model (simplified)

Postgres table `candidates`:

* `candidate_id` (uuid)
* `name` (nullable)
* `filename`
* `parsed_text` (text)
* `skills` (jsonb list)
* `experience_years` (int/nullable)
* `created_at`
* `s3_path` (resume file)
* `metadata` (jsonb)

ChromaDB collection `resume_vectors`:

* `ids`: candidate_id (string)
* `embeddings`: vector embeddings from sentence-transformers
* `metadatas`: dict with skills, experience_years, filename, etc.
* `documents`: parsed resume text chunks
* Collection supports filtering by metadata fields for efficient querying

---

# 5 — API design (MVP endpoints)

* `POST /upload`

  * body: file or S3 path, optional candidate metadata
  * returns: `candidate_id`, parsing summary, skills extracted

* `POST /match`

  * body: `{ "job_description": "...", "top_k": 10, "filters": {...} }`
  * returns: ranked list: `[ {candidate_id, score, skills, brief_explain} ]`

* `GET /candidate/{id}`

  * returns: parsed resume, metadata, versioned explain logs

* `GET /audit/fairness?job_id=...`

  * returns: basic fairness metrics for top-k (distribution across simulated/provided groups)

* `GET /health` and `GET /metrics` (Prometheus)

---

# 6 — Matching & explainability (simple, deterministic, easy to implement)

Scoring formula (transparent):

1. `semantic_score` = cosine(embedding(job), embedding(candidate)) ∈ \[0,1]
2. `keyword_score` = (#matching\_skill\_tokens) / (#job\_skill\_tokens) ∈ \[0,1]
3. `experience_score` = clamp(years\_experience / target\_years, 0,1)

Final score = `0.65 * semantic_score + 0.30 * keyword_score + 0.05 * experience_score`

Explainability (return with each candidate):

* top 5 overlapping tokens (skills/keywords),
* semantic\_score value,
* keyword\_score value,
* short template text: “Matched on {Python, AWS}. Semantic similarity: 0.83. Keyword overlap: 0.67.”

This rule-based explainability is easy, audit-friendly, and avoids heavy XAI toolchain for MVP.

---

# 7 — Bias & compliance (MVP approach)

* **Remove explicit sensitive fields** during matching (do not use name, photo, address).
* **Mask** any demographic fields in model inputs.
* **Audit**: After any match run, produce top-k distribution reports (counts) by any known demographic fields (if TechCorp supplies them) or by proxies (unreliable). Flag disparity ratios > threshold (e.g., 1.2).
* **Logging**: Store `request_id`, job\_text, top\_k candidates, timestamp, and explainability payload for each match (audit trail).
* **Policy**: Store retention policy and DPA in product docs.

For full legal compliance, propose regular fairness audits and involve legal/HR — but for MVP, implement audits and logging.

---

# 8 — Non-functional requirements (MVP targets)

* Single-node latency: match request < 500ms (with precomputed embeddings and small index)
* Concurrency: support 50 concurrent users for demo (scale later)
* Storage: resilient S3 + daily backups
* Observability: basic logs and Prometheus metrics
* Security: TLS, RBAC, API keys

---

# 9 — Implementation plan (minimal team: 1–2 devs, 1 day sprint split into phases)

Below are **practical steps** and **estimated time** (hours) to produce a demo-ready MVP. Adjust based on team size & skill.

1. **Environment & skeleton (0.5–1h)**

   * Setup repo, venv, Dockerfile, basic FastAPI app + Swagger.

2. **Resume parsing & ingestion (1–2h)**

   * Implement `/upload`, text extraction for txt/pdf/docx, naive skill extractor (list-based matching).
   * Store raw text + metadata in Postgres / local JSON (for tiny demo).

3. **Embedding pipeline (1h)**

   * Integrate sentence-transformers, produce embedding on upload.
   * Initialize ChromaDB client and create collection for resume vectors.

4. **Vector database + search (1h)**

   * Setup ChromaDB collection, add vectors with metadata, implement similarity search wrapper.

5. **Matching endpoint + scoring (1h)**

   * Implement `/match` using ChromaDB similarity search with metadata filtering and scoring formula.

6. **Explainability & audit logging (1h)**

   * Return rule-based explains. Store audit logs in file/DB.

7. **Simple demo script or minimal UI (1h)**

   * `scripts/demo_upload.py` + sample resumes + curl examples. Optionally simple HTML that calls API.

8. **Testing & polish (1h)**

   * Basic unit tests, API smoke tests, run demo, record fallback video/screenshots.

Total MVP time: \~7–9 hours (one focused day for one developer; faster with two).

---

# 10 — Upgrade path (when ready to scale)

If/when you outgrow MVP:

* Scale ChromaDB to distributed mode or migrate to ChromaDB Cloud for enterprise features.
* Move embedding service to an async microservice or serverless with caching.
* Introduce retraining/fine-tuning or use a larger embedding model where accuracy requires it.
* Add SHAP/Model Cards + richer XAI for compliance teams.
* Add RBAC, SSO (SAML/OIDC), and SOC2 controls.
* Implement ChromaDB's advanced features like custom distance functions and batch operations.

---

# 11 — Testing & acceptance criteria (MVP)

Acceptance tests:

* Upload 50 sample resumes to ChromaDB collection, and `/match` returns top-5 within 1 second.
* ChromaDB collection properly stores embeddings with metadata (skills, experience, etc.).
* Metadata filtering works: search for candidates with specific skills or experience levels.
* Explain payload included for each returned candidate with tokens and contribution scores.
* Audit endpoint returns top-k distribution and logs match requests.
* Basic security: all endpoints under TLS and require an API key.
* ChromaDB persistence: data survives application restarts.

---

# 12 — Checklist to start coding (copy-paste)

* [ ] Create repo + venv + `requirements.txt` (fastapi, uvicorn, sentence-transformers, chromadb, psycopg2-binary or sqlite for demo, spacy, PyPDF2, python-docx)
* [ ] Implement parser script + `/upload` endpoint
* [ ] Add `embeddings.py` wrapper using `sentence-transformers`
* [ ] Setup ChromaDB client and create collections + persist configuration
* [ ] Implement `/match` endpoint with scoring + explain function using ChromaDB similarity search
* [ ] Add basic logging + `/audit` endpoint
* [ ] Create `scripts/demo_upload.py` and sample resumes
* [ ] Dockerize and run locally. Record demo.

---

# 13 — Final recommendation (which is easiest)

Use the **Monolith MVP (FastAPI + Sentence-Transformers + ChromaDB + Postgres + S3)**. ChromaDB provides better persistence, metadata filtering, and scalability compared to FAISS while maintaining simplicity. It’s the fastest path from zero → demoable prototype with a clean path to production hardening.

---

# 14 — Project Folder Structure

Here's the recommended folder structure for the AI Resume Matcher project:

```
ai-resume-matcher/
│
├── README.md                           # Project overview and setup instructions
├── PRD.md                             # Product Requirements Document
├── requirements.txt                    # Python dependencies
├── Dockerfile                         # Container configuration
├── docker-compose.yml                 # Multi-service orchestration
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── src/                               # Main application code
│   ├── __init__.py
│   ├── main.py                        # FastAPI application entry point
│   ├── config/                        # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py                # Application settings
│   │   └── database.py                # Database configurations
│   │
│   ├── api/                           # API layer
│   │   ├── __init__.py
│   │   ├── routes/                    # API route handlers
│   │   │   ├── __init__.py
│   │   │   ├── upload.py              # Resume upload endpoints
│   │   │   ├── match.py               # Matching endpoints
│   │   │   ├── candidates.py          # Candidate management
│   │   │   └── audit.py               # Audit and compliance endpoints
│   │   └── middleware/                # Custom middleware
│   │       ├── __init__.py
│   │       ├── auth.py                # Authentication middleware
│   │       └── logging.py             # Request logging
│   │
│   ├── core/                          # Core business logic
│   │   ├── __init__.py
│   │   ├── resume_parser.py           # Resume parsing logic
│   │   ├── embeddings.py              # Embedding generation
│   │   ├── vector_store.py            # ChromaDB operations
│   │   ├── matching.py                # Matching algorithms
│   │   ├── scoring.py                 # Scoring logic
│   │   └── explainability.py         # Explanation generation
│   │
│   ├── models/                        # Data models and schemas
│   │   ├── __init__.py
│   │   ├── candidate.py               # Candidate data models
│   │   ├── job.py                     # Job description models
│   │   ├── match.py                   # Match result models
│   │   └── schemas.py                 # Pydantic schemas for API
│   │
│   ├── services/                      # External service integrations
│   │   ├── __init__.py
│   │   ├── storage.py                 # S3/cloud storage service
│   │   ├── chromadb_client.py         # ChromaDB client wrapper
│   │   ├── postgres_client.py         # PostgreSQL client
│   │   └── ml_models.py               # ML model management
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── file_utils.py              # File handling utilities
│       ├── text_processing.py         # Text processing helpers
│       ├── validation.py              # Data validation utilities
│       └── logger.py                  # Logging configuration
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration
│   ├── unit/                          # Unit tests
│   │   ├── test_resume_parser.py
│   │   ├── test_embeddings.py
│   │   ├── test_matching.py
│   │   └── test_vector_store.py
│   ├── integration/                   # Integration tests
│   │   ├── test_api_endpoints.py
│   │   ├── test_chromadb_integration.py
│   │   └── test_postgres_integration.py
│   └── fixtures/                      # Test data
│       ├── sample_resumes/
│       │   ├── resume1.pdf
│       │   ├── resume2.docx
│       │   └── resume3.txt
│       └── sample_jobs.json
│
├── scripts/                           # Utility scripts
│   ├── setup_environment.py           # Environment setup
│   ├── demo_upload.py                 # Demo data upload
│   ├── migrate_data.py                # Data migration tools
│   ├── performance_test.py            # Performance testing
│   └── backup_chromadb.py             # Database backup utility
│
├── data/                              # Data storage (gitignored)
│   ├── chromadb/                      # ChromaDB persistent storage
│   ├── uploads/                       # Temporary file uploads
│   └── logs/                          # Application logs
│
├── docs/                              # Documentation
│   ├── api_documentation.md           # API documentation
│   ├── deployment_guide.md            # Deployment instructions
│   ├── troubleshooting.md             # Common issues and solutions
│   └── model_cards/                   # ML model documentation
│       └── embedding_model_card.md
│
├── monitoring/                        # Monitoring and observability
│   ├── prometheus.yml                 # Prometheus configuration
│   ├── grafana/                       # Grafana dashboards
│   │   └── dashboards/
│   └── alerts/                        # Alert configurations
│
└── deployment/                        # Deployment configurations
    ├── kubernetes/                    # K8s manifests
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── configmap.yaml
    ├── terraform/                     # Infrastructure as code
    │   ├── main.tf
    │   └── variables.tf
    └── ci-cd/                         # CI/CD pipeline configs
        ├── .github/
        │   └── workflows/
        │       ├── test.yml
        │       └── deploy.yml
        └── jenkinsfile
```

### Key Folder Purposes:

- **src/**: Main application code organized by functional layers
- **src/core/**: Core business logic including ChromaDB integration
- **src/services/**: External service clients and integrations
- **tests/**: Comprehensive test suite with fixtures
- **scripts/**: Utility scripts for setup, demo, and maintenance
- **data/**: Runtime data storage (excluded from git)
- **docs/**: Project documentation and guides
- **monitoring/**: Observability and monitoring configurations
- **deployment/**: Infrastructure and deployment configurations

### ChromaDB Integration Points:

- **src/services/chromadb_client.py**: ChromaDB client wrapper and configuration
- **src/core/vector_store.py**: Vector operations and collection management
- **data/chromadb/**: Persistent storage for ChromaDB data
- **scripts/backup_chromadb.py**: Database backup and recovery utilities

---

If you want, I can now:

* produce a **detailed file-level scaffold** (file names + starter code for each file), or
* generate a **30–60 minute developer runbook** with exact shell commands and copy-paste-ready files to implement the MVP.

Which one should I generate next?
