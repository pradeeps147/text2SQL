# Tenarai logistics intelligence

An end-to-end logistics intelligence workspace: upload multiple inconsistent
legacy shipment CSVs, clean them into a validated DuckDB table, and ask business
questions through a retrieval-grounded, sandboxed Text-to-SQL chat workflow.
The FastAPI backend owns ingestion, querying, natural-language synthesis, and
structured operational logs; Streamlit provides the Tenarai user experience.

## Architecture

```
Streamlit UI ──────────────────────────────────────────────────────┐
                                                                 ▼
POST /api/v1/datasets/upload (one or more CSV files)       FastAPI backend
        │
        ▼
src/pipeline/ingest.py      – in-memory column reconciliation across source systems
        │
        ▼
src/pipeline/cleaning.py    – date parsing, missing-field policy, fuzzy
        │                      location standardization, dedupe
        ▼
src/pipeline/load.py        – validated rows -> DuckDB `shipments` table
        │                      + data_quality_report.json
        ▼
src/nlq/retrieval.py        – retrieve relevant metric/glossary definitions
        │
        ▼
src/nlq/text_to_sql.py      – question + live schema + retrieved context -> Ollama
        │                      -> structured JSON {sql, explanation}
        ▼
src/nlq/guardrails.py       – sqlglot allowlist validation (SELECT-only,
        │                      table/column allowlist, forced LIMIT)
        ▼
src/nlq/executor.py         – read-only execution w/ timeout + self-healing
        │                      retry loop
        ▼
src/nlq/answer_synthesis.py – bounded DuckDB result -> natural-language answer
        │
        ├──► /api/v1/chat response (answer + SQL + rows + retrieval evidence)
        └──► structured rotating logs + /api/v1/logs monitor endpoint
```

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) installed locally.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Install & start Ollama, then pull a model (any tool-capable chat model works):
ollama serve &
ollama pull llama3.1:8b
```

## Run it

```bash
# 1. Generate synthetic messy sample data (3 "legacy system" CSVs)
python scripts/generate_sample_data.py

# 2. Run the ingestion/cleaning/load pipeline (CLI, prints a summary)
python -m src.pipeline.run_pipeline

# 3. Start the backend API
uvicorn src.api.app:app --reload --port 8000

# 4. In another terminal, launch the UI
streamlit run src/app/app.py
```

Open `http://localhost:8501`. Use **Data workspace** to upload one or more CSVs,
then use **Ask Tenarai** for questions such as *"Which route had the highest
delay rate last quarter?"*. Interactive API docs are available at
`http://localhost:8000/docs`.

Set `TENARAI_API_URL` to point the UI at a backend on a different host and set
`LLM_MODEL` / `OLLAMA_HOST` to configure the local LLM service.

## Docker

The same non-root production image runs either the FastAPI backend or the
Streamlit UI. For local development, keep Ollama running on the host and start
both application containers with:

```bash
docker compose up --build
```

Then open:

- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

Compose keeps DuckDB, uploaded files, and backend logs in named volumes. To
build or run just the default backend image:

```bash
docker build -t tenarai .
docker run --rm -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  tenarai
```

The container health check automatically selects `/api/v1/health` for the
backend and `/_stcore/health` for the UI based on `SERVICE_ROLE`.

## GitHub Actions

Two workflows are included:

- `.github/workflows/ci.yml` runs the complete test suite, compiles Python,
  validates `compose.yaml`, and builds the image on pull requests and `main`.
- `.github/workflows/publish-container.yml` publishes multi-architecture images
  with provenance and SBOM attestations to `ghcr.io/<owner>/<repository>` on
  `main`, version tags, or manual dispatch. It uses the repository-scoped
  `GITHUB_TOKEN`; no registry password needs to be committed.

## Public deployment on Render

`render.yaml` defines two isolated public web services:

- `tenarai-api`: FastAPI, hosted LLM access, API health checks, generated admin
  key, and a public-chat rate limit.
- `tenarai-ui`: Streamlit, connected to the API over Render's private network.

Deployment steps:

1. Push this directory to a GitHub repository.
2. In Render, create a **Blueprint** from that repository. Render detects
   `render.yaml`.
3. Enter `OPENAI_API_KEY` when Render prompts for the `sync: false` secret.
4. After deployment, open the `tenarai-ui` `onrender.com` URL. Copy the generated
   `TENARAI_ADMIN_KEY` from the API service only when an administrator needs to
   upload a dataset or inspect logs.

The Blueprint uses free web-service plans so it does not silently create paid
resources. Free service filesystems are ephemeral: public uploads remain active
until the backend restarts, then the bundled demonstration dataset is restored.
For durable production uploads, attach a paid persistent disk at `/app/data` or
move dataset storage to an external object/database service.

For hosted deployments, `LLM_PROVIDER=openai` uses the Chat Completions API with
JSON Schema structured outputs. Local Compose continues to use
`LLM_PROVIDER=ollama`. Operational endpoints (`upload` and `logs`) are protected
when `TENARAI_ADMIN_KEY` is configured, and public chat is limited per client.

## Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Backend and active-dataset readiness |
| `GET` | `/api/v1/dataset` | Row count, sources, coverage, delay rate, quality report |
| `POST` | `/api/v1/datasets/upload` | Multipart upload of up to 20 CSV files (25 MB each) |
| `POST` | `/api/v1/chat` | Retrieval-grounded Text-to-SQL and natural answer |
| `GET` | `/api/v1/rag` | Machine-readable explanation of the RAG pipeline |
| `GET` | `/api/v1/logs` | Recent structured backend events, with level/limit filters |

Uploaded batches are validated and built in a staging DuckDB file. The active
database is replaced atomically only after a batch succeeds. Original uploads
are retained under `data/uploads/<batch-id>/` for local provenance.

## Tests

```bash
python -m pytest -q
```

Covers date-parsing edge cases, location fuzzy-matching, missing/duplicate row
handling, and — most importantly — adversarial SQL guardrail tests (multi-statement
injection, DROP/INSERT/UPDATE/DELETE/PRAGMA/ATTACH, `information_schema` /
`duckdb_tables()` / `read_csv()` filesystem-access attempts, disallowed columns),
plus an end-to-end executor test using a mocked LLM.

## Data cleaning approach

- **Inconsistent date formats**: `cleaning.parse_flexible_date` uses `dateutil`
  to parse formats like `2025-01-05`, `01/05/2025`, `05-01-2025`, `Jan 5, 2025`.
  Garbage tokens (`N/A`, `TBD`, `0000-00-00`, blank) are treated as missing, not
  as errors.
- **Missing fields**: `shipment_id`, `origin`, `destination`, and a parseable
  `ship_date` are required — rows missing any of these are rejected and logged
  with a reason (not silently dropped). Optional fields (`carrier`, `route_id`,
  delivery dates) are flagged in the data-quality report but don't reject the row.
- **Unstandardized locations**: exact alias table lookup first (`NYC` → `New York`),
  then fuzzy matching (RapidFuzz `WRatio`, threshold 78) against a canonical
  location list; anything below the threshold is kept as-is and flagged as
  "unmatched" for manual review.
- **Duplicates**: exact-duplicate rows (simulating re-exported/overlapping
  extracts) are dropped.
- All of the above is summarized in `data/processed/data_quality_report.json`
  after every pipeline run.

## How RAG works in Text-to-SQL

This implementation uses a compact, auditable form of RAG rather than a vector
database:

1. The backend reads the live DuckDB schema and representative rows.
2. `src/nlq/retrieval.py` ranks business-glossary documents using the words and
   phrases in the question. The corpus defines delay rate, route/lane meaning,
   carrier performance, status semantics, relative time, and provenance.
3. The top three documents are injected into the SQL-generation system prompt
   alongside the live schema. This tells the model both *what columns exist* and
   *what the business terms mean*.
4. Generated SQL is parsed and allowlisted before a read-only DuckDB execution.
5. A bounded result set (maximum 100 rows sent to the model) is passed to a
   separate synthesis prompt. That second LLM step returns the natural-language
   business answer shown above the evidence table.

The chat response includes the retrieved document IDs, scores, generated SQL,
and execution attempts. The UI exposes this evidence under each answer and the
**RAG & governance** tab explains the same pipeline at runtime.

## Defensive Text-to-SQL design

The LLM is never trusted to execute anything directly. Every generated query
passes through:

1. **Structured output** — the model is asked for JSON matching a pydantic
   schema (`{sql, explanation}`), with a regex-based fallback parser if a model
   ignores the JSON constraint.
2. **Single-statement check** — rejects `SELECT ...; DROP TABLE ...`-style
   stacked queries.
3. **Statement-type allowlist** — only `SELECT`/`UNION` ASTs are accepted;
   `INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`CREATE`/`ATTACH`/`PRAGMA`/`COPY`
   either fail to parse as a `Select` or are rejected outright.
4. **Table & column allowlists** — blocks reads from `information_schema`,
   `duckdb_tables()`, or table-valued functions like `read_csv('/etc/passwd')`
   that could otherwise read arbitrary files off disk, and blocks references
   to any column not in the known schema.
5. **Forced LIMIT** — a `LIMIT` is added (or capped) on every query to bound
   result size.
6. **Read-only connection + timeout** — the executor opens DuckDB in
   read-only mode and runs each query on a worker thread that's interrupted if
   it exceeds a timeout (defends against accidental runaway queries).
7. **Self-healing retries** — if validation or execution fails, the error is
   fed back to the model (up to 2 retries) to correct itself; if it still can't
   produce a valid answer, the UI surfaces a clear error instead of guessing.
8. **Result synthesis boundary** — only the bounded SQL result, question, and
   executed SQL are sent to the answer-synthesis call. If synthesis fails, the
   successful table is preserved and a deterministic fallback is returned.
9. **Prompt-injection resistance** — the system prompt explicitly instructs the
   model to treat schema, retrieved text, questions, SQL, and cell values as
   untrusted data, not instructions.

## Backend monitoring

The backend records JSON events for HTTP requests, upload batches, retrieved RAG
documents, SQL rejection/execution, and answer synthesis. Logs are written to a
5 MB rotating file at `logs/tenarai-backend.log` (three backups) and to an
in-memory 2,000-event tail. The **Backend monitor** UI reads `/api/v1/logs` and
supports severity filtering.

## Known limitations / not in scope

- Single fact table (`shipments`); no multi-table joins or a full star schema.
- Location canonicalization list is a small hardcoded demo set, not a real
  geo-master-data service.
- No authentication/multi-user authorization; deploy behind an authenticated
  gateway before exposing the API outside a trusted network.
- The RAG corpus is an in-process audited glossary, not an embedding/vector
  index. This is deliberate for the current small domain and should be upgraded
  when the knowledge base becomes large or frequently changing.
- Text-to-Pandas was intentionally out of scope in favor of the more easily
  sandboxed Text-to-SQL path.
