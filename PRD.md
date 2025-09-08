# Technical design (easy-to-build option) — Resume Matching Platform

Great — you want a practical, **easy-to-build technical design** for the resume-matching platform (the one-day / MVP-friendly route). Below I give a clear, minimal architecture that you can implement quickly, the rationale for why it’s easiest, concrete component choices, APIs, data flow, security, testing/acceptance criteria, and a compact implementation plan with estimated effort for a small team.

---

# 1 — Recommendation (single-sentence)

Build a **single FastAPI Python service + FAISS local vector index + Sentence-Transformers embeddings + simple parser** with S3 for storage and PostgreSQL for metadata. This monolithic-but-modular approach is fastest to implement and easiest to demo, yet productionizable later.

Why easiest:

* Everything is Python — one language and environment.
* Uses pre-trained models (no training required).
* FAISS is simple to run locally for MVP (no external DB).
* Minimal infra: single VM or managed app (GCP App Engine / AWS EC2 / Heroku).
* Clear upgrade path to managed vector DBs, microservices, or Kubernetes later.

---

# 2 — High-level architecture (MVP)

```
[Upload UI / ATS Connector] --> [API (FastAPI)] --> 
  ├─ Resume Parser (spaCy / PyPDF2 / docx2txt) → raw_text
  ├─ Embedding Service (sentence-transformers) → vector
  ├─ Metadata DB (Postgres) ← store candidate metadata
  └─ Vector Index (FAISS) ← store vectors + id mapping
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
* **Vector store**: FAISS (IndexFlatIP or IndexFlatL2). For >100k vectors later, upgrade to Milvus/Pinecone/Weaviate.
* **Metadata DB**: PostgreSQL with JSONB columns for resume fields.
* **Storage**: S3-compatible (AWS S3, GCS, or MinIO for local).
* **Explainability**: Rule-based reasons (skill overlap + semantic contribution) + optional SHAP on small interpretable model later.
* **Bias checks**: Rule-based audits and statistical checks (distribution of top-k by simulated/available demographics). Use `fairlearn` later if needed.
* **Authentication**: API keys / OAuth for ATS integration.
* **Hosting (fastest)**: Single cloud VM or PaaS (Heroku / GCP App Engine). Use Docker for portability.

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

FAISS index:

* vector index of embeddings
* mapping: FAISS index id ↔ `candidate_id` (store mapping in Postgres or separate `meta.json`)

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

* Replace FAISS with Milvus/Pinecone/Weaviate for horizontal scale and persistence.
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

* [ ] Create repo + venv + `requirements.txt` (fastapi, uvicorn, sentence-transformers, faiss-cpu, psycopg2-binary or sqlite for demo, spacy, PyPDF2, python-docx)
* [ ] Implement parser script + `/upload` endpoint
* [ ] Add `embeddings.py` wrapper using `sentence-transformers`
* [ ] Build FAISS index + `index.save()` + `index.load()`
* [ ] Implement `/match` endpoint with scoring + explain function
* [ ] Add basic logging + `/audit` endpoint
* [ ] Create `scripts/demo_upload.py` and sample resumes
* [ ] Dockerize and run locally. Record demo.

---

# 13 — Final recommendation (which is easiest)

Use the **Monolith MVP (FastAPI + Sentence-Transformers + FAISS + Postgres + S3)**. It minimizes moving parts, uses a single language, relies on off-the-shelf components, and gives immediate, explainable results. It’s the fastest path from zero → demoable prototype with a clean path to production hardening.

---

If you want, I can now:

* produce a **detailed file-level scaffold** (file names + starter code for each file), or
* generate a **30–60 minute developer runbook** with exact shell commands and copy-paste-ready files to implement the MVP.

Which one should I generate next?
