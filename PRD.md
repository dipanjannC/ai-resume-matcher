# Technical design (easy-to-build option) — AI-Powered Resume Matching Platform

This is a practical, **AI-enhanced technical design** for an intelligent resume-matching platform that leverages **LangChain for structured data extraction** from unstructured CVs. The design uses LLMs with system and role prompts to extract structured information and match candidates effectively.

## Final recommendation

Use the **Monolith MVP (FastAPI + LangChain + Sentence-Transformers + ChromaDB + Postgres + S3)**. It minimizes moving parts, uses a single language, relies on off-the-shelf components, and gives immediate, explainable results. **LangChain** provides intelligent data extraction from unstructured resumes using LLM-powered prompts. ChromaDB provides a persistent vector database with built-in metadata storage, simplifying the architecture. It's the fastest path from zero → demoable prototype with a clean path to production hardening.hnical design (easy-to-build option) — Resume Matching Platform

Great — you want a practical, **easy-to-build technical design** for the resume-matching platform (the one-day / MVP-friendly route). Below I give a # 13 — Final recommendation (which is easiest)

Use the **Monolith MVP (FastAPI + Sentence-Transformers + ChromaDB + Postgres + S3)**. It minimizes moving parts, uses a single language, relies on off-the-shelf components, and gives immediate, explainable results. ChromaDB provides a persistent vector database with built-in metadata storage, simplifying the architecture. It's the fastest path from zero → demoable prototype with a clean path to production hardening.ar, minimal architecture that you can implement quickly, the rationale for why it’s easiest, concrete component choices, APIs, data flow, security, testing/acceptance criteria, and a compact implementation plan with estimated effort for a small team.

---

# 1 — Recommendation (single-sentence)

Build a **single FastAPI Python service + LangChain + ChromaDB vector database + Sentence-Transformers embeddings + intelligent resume parser** with S3 for storage and PostgreSQL for metadata. This monolithic-but-modular approach uses **LangChain system prompts** for structured data extraction from unstructured CVs, making it fastest to implement while providing intelligent resume parsing.

Why easiest:

* Everything is Python — one language and environment.
* **LangChain** provides structured data extraction using LLM prompts (no manual parsing rules).
* Uses pre-trained models (no training required).
* ChromaDB provides a simple, persistent vector database with built-in metadata storage.
* Minimal infra: single VM or managed app (GCP App Engine / AWS EC2 / Heroku).
* Clear upgrade path to managed vector DBs, microservices, or Kubernetes later.

---

# 2 — High-level architecture (MVP with LangChain)

```ascii
[Upload UI / ATS Connector] --> [API (FastAPI)] --> 
  ├─ LangChain Resume Parser (LLM + System Prompts) → structured_data
  ├─ Embedding Service (sentence-transformers) → vector
  ├─ Metadata DB (Postgres) ← store candidate metadata
  └─ Vector DB (ChromaDB) ← store vectors + metadata

LangChain Extraction Pipeline:
Resume Text → [System Prompt + Role Prompt] → LLM → Structured Output:
{
  "profile": { "name": "...", "email": "...", "phone": "..." },
  "experience": { "years": 5, "positions": [...] },
  "skills": { "technical": [...], "soft": [...] },
  "topics": { "domains": [...], "specializations": [...] },
  "tools_libraries": { "languages": [...], "frameworks": [...], "tools": [...] },
  "summary": "Professional summary extracted from CV"
}

API serves: /upload, /match, /explain, /audit
Storage: S3 (resumes, precomputed artifacts)
Monitoring: Prometheus + Grafana (optional)
Logging: ELK or cloud logs
```

---

# 3 — Component choices (MVP, LangChain-enhanced)

* **API layer**: Python + FastAPI (simple async, auto docs via Swagger).
* **AI-Powered Parsing**: **LangChain** with OpenAI/Anthropic/Local LLM for structured extraction using system and role prompts.
* **Resume parsing (fallback)**: `PyPDF2` + `python-docx` + spaCy for text extraction.
* **Embeddings**: `sentence-transformers` (model: `all-MiniLM-L6-v2` for speed) — no training.
* **Vector store**: ChromaDB (persistent, embeddable vector database with metadata storage). Scalable and easy to implement.
* **Metadata DB**: PostgreSQL with JSONB columns for structured resume data and user management.
* **Storage**: S3-compatible (AWS S3, GCS, or MinIO for local).
* **LLM Integration**: LangChain + OpenAI API (or Ollama for local LLMs).
* **Explainability**: Rule-based reasons (skill overlap + semantic contribution) + LangChain reasoning chains.
* **Bias checks**: Rule-based audits and statistical checks (distribution of top-k by simulated/available demographics).
* **Authentication**: API keys / OAuth for ATS integration.
* **Hosting (fastest)**: Single cloud VM or PaaS (Heroku / GCP App Engine). Use Docker for portability.

---

# 4 — Data model (LangChain-enhanced structured data)

Postgres table `candidates`:

* `candidate_id` (uuid)
* `profile` (jsonb) - `{ "name": "...", "email": "...", "phone": "...", "location": "..." }`
* `experience` (jsonb) - `{ "years": 5, "positions": [...], "companies": [...] }`
* `skills` (jsonb) - `{ "technical": [...], "soft": [...], "certifications": [...] }`
* `topics` (jsonb) - `{ "domains": [...], "specializations": [...], "industries": [...] }`
* `tools_libraries` (jsonb) - `{ "languages": [...], "frameworks": [...], "tools": [...], "databases": [...] }`
* `summary` (text) - AI-extracted professional summary
* `filename` (varchar)
* `parsed_text` (text) - raw extracted text
* `created_at` (timestamp)
* `updated_at` (timestamp)
* `s3_path` (varchar) - resume file location
* `extraction_metadata` (jsonb) - LangChain processing metadata

ChromaDB collection:

* `embeddings`: vector embeddings of resume content and summary
* `documents`: processed text combining summary + key skills + experience
* `metadatas`: structured data from LangChain extraction (candidate_id, experience_years, top_skills, etc.)
* `ids`: unique identifiers matching candidate_id in PostgreSQL

Job Requirements table `job_requirements`:

* `job_id` (uuid)
* `title` (varchar)
* `description` (text)
* `required_experience` (jsonb) - `{ "years": 3, "roles": [...] }`
* `required_skills` (jsonb) - `{ "technical": [...], "soft": [...] }`
* `preferred_topics` (jsonb) - `{ "domains": [...], "specializations": [...] }`
* `required_tools` (jsonb) - `{ "languages": [...], "frameworks": [...], "tools": [...] }`
* `created_at` (timestamp)

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

   * Integrate sentence-transformers, produce embedding on upload (or batch precompute).
   * Store vector in memory and persist to FAISS file.

4. **Vector index + search (1h)**

   * Create FAISS index, add vectors, implement search wrapper.

5. **Matching endpoint + scoring (1h)**

   * Implement `/match` using scoring formula and simple filtering.

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

* Scale ChromaDB with distributed deployment or consider migrating to Milvus/Pinecone/Weaviate for larger-scale production needs.
* Move embedding service to an async microservice or serverless with caching.
* Introduce retraining/fine-tuning or use a larger embedding model where accuracy requires it.
* Add SHAP/Model Cards + richer XAI for compliance teams.
* Add RBAC, SSO (SAML/OIDC), and SOC2 controls.

---

# 11 — Testing & acceptance criteria (MVP)

Acceptance tests:

* Upload 50 sample resumes, build index, and `/match` returns top-5 within 1 second.
* Explain payload included for each returned candidate with tokens and contribution scores.
* Audit endpoint returns top-k distribution and logs match requests.
* Basic security: all endpoints under TLS and require an API key.

---

# 12 — Checklist to start coding (copy-paste)

* [ ] Create repo + venv + `requirements.txt` (fastapi, uvicorn, sentence-transformers, chromadb, psycopg2-binary or sqlite for demo, spacy, PyPDF2, python-docx)
* [ ] Implement parser script + `/upload` endpoint
* [ ] Add `embeddings.py` wrapper using `sentence-transformers`
* [ ] Set up ChromaDB collection + persistence
* [ ] Implement `/match` endpoint with scoring + explain function
* [ ] Add basic logging + `/audit` endpoint
* [ ] Create `scripts/demo_upload.py` and sample resumes
* [ ] Dockerize and run locally. Record demo.

---

# 13 — Final recommendation (which is easiest)

Use the **Monolith MVP (FastAPI + Sentence-Transformers + FAISS + Postgres + S3)**. It minimizes moving parts, uses a single language, relies on off-the-shelf components, and gives immediate, explainable results. It’s the fastest path from zero → demoable prototype with a clean path to production hardening.

---

# 14 — Folder structure

```ascii
ai-resume-matcher/
├── .github/
│   └── workflows/              # CI/CD workflows
├── app/
│   ├── api/                    # API routes 
│   │   ├── __init__.py
│   │   ├── candidates.py       # Candidate routes
│   │   ├── jobs.py             # Job routes
│   │   ├── matching.py         # Matching routes
│   │   └── auth.py             # Authentication routes
│   ├── core/                   # Core application code
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration settings
│   │   ├── security.py         # Security utilities
│   │   └── exceptions.py       # Custom exceptions
│   ├── db/                     # Database related code
│   │   ├── __init__.py
│   │   ├── models.py           # SQLAlchemy models
│   │   └── session.py          # Database session management
│   ├── services/               # Business logic services
│   │   ├── __init__.py
│   │   ├── parser.py           # Resume parsing logic
│   │   ├── embeddings.py       # Embedding generation service
│   │   ├── vector_store.py     # ChromaDB integration
│   │   ├── matcher.py          # Matching algorithm
│   │   └── explainer.py        # Explanation generation
│   ├── schemas/                # Pydantic models for validation
│   │   ├── __init__.py
│   │   ├── candidates.py
│   │   ├── jobs.py
│   │   └── matching.py
│   └── main.py                 # FastAPI application entry point
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py             # Test fixtures
│   ├── test_parser.py
│   ├── test_embeddings.py
│   └── test_matcher.py
├── scripts/                    # Utility scripts
│   ├── demo_upload.py          # Demo script for uploading resumes
│   └── generate_sample_data.py # Generate sample data
├── data/                       # Data storage
│   ├── resumes/                # Sample resumes for testing
│   ├── vectordb/               # ChromaDB persistent storage
│   └── logs/                   # Application logs
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   └── setup/                  # Setup instructions
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore file
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker compose configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

This folder structure follows best practices for Python applications and provides a clean separation of concerns. The modular design allows for easy extension and maintenance as the project grows.
