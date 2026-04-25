# AeroSports RAG Pipeline

A two-service Retrieval-Augmented Generation (RAG) system that powers an AI voice agent ("Maya") for **AeroSports Scarborough** — a trampoline / family entertainment park. Customers call a Twilio number, the bot answers questions using the park's knowledge base (Google Sheet), and escalates booking changes to a human via a deterministic capture flow.

---

## 1. High-Level Overview

### What it does
- Accepts inbound phone calls via **Twilio ConversationRelay** (ASR + TTS).
- Streams the caller's transcribed speech into a **local Ollama LLM** to generate a spoken reply.
- Grounds every answer with documents pulled from a **pgvector**-backed RAG service.
- Pulls source-of-truth content from a Google Sheet, parses each worksheet into Markdown, runs an LLM enrichment pass that rewrites tables into voice-friendly prose, then embeds and indexes it.
- Logs every call (transcripts, RAG hits, LLM context, final reply) to per-session log files and a SQLite call log.

### Problems it solves
- **24/7 phone coverage** without dedicated human staff for routine questions (hours, prices, attractions, packages).
- **Grounded answers** — eliminates LLM hallucination by enforcing "facts only from the knowledge base context."
- **Single source of truth** — operations team edits a Google Sheet; the resync endpoint re-pulls, re-enriches, and re-embeds without code changes.
- **Graceful human handoff** — booking changes / complaints are captured (name, phone, details) and flagged for callback.

---

## 2. Project Structure

```
rag_pipeline/
├── chatbot/                       # Twilio voice server (the "front" of the system)
│   ├── server.py                  # FastAPI app: /voice/inbound, /voice/ws, /api/health
│   ├── voice_handler.py           # System prompt, query rewrite, RAG call, TTS chunking, end-of-call classifier
│   ├── conversation.py            # Thread-safe in-memory per-call message store with TTL
│   ├── llm.py                     # Ollama client (sync + async) with retry/backoff
│   └── rag_client.py              # Async HTTP client → core/rag /rag/retrieve
│
├── core/rag/                      # Standalone RAG microservice (the "brain")
│   ├── Dockerfile, docker-compose.yaml   # Containerized backend + pgvector
│   ├── .env, .env.example         # Backend config (PG, embedding model, defaults)
│   ├── src/                       # FastAPI backend
│   │   ├── main.py                # App bootstrap, lifespan, DB wait, auto-ingest, CORS
│   │   ├── config.py              # Pydantic settings (APP_* env vars), DATA_FOLDER paths
│   │   ├── schemas.py             # Pydantic request/response models
│   │   ├── dependencies.py        # FastAPI DI for RAGRepository / RAGService
│   │   ├── rag/                   # Retrieval + ingest + resync
│   │   │   ├── routes.py          # /rag/query, /rag/retrieve, /rag/resync, /rag/health
│   │   │   ├── services.py        # Resync orchestration (wipe → parse → enrich → re-index)
│   │   │   └── repositories.py    # PGVectorStore, MarkdownNodeParser, Ollama embed/LLM
│   │   ├── history/               # Query-history SQLModel + repo + routes (/history/*)
│   │   └── llm_config/            # Runtime-mutable chat-LLM config (/settings/llm)
│   │
│   └── parser/                    # Knowledge-base ingestion pipeline
│       ├── main.py, run_parsers.py     # CLI entry: download → parse → optional enrich
│       ├── merge_output.py        # Utility: concatenate per-sheet markdowns
│       ├── data/                  # Cached Google Sheet (.xlsx) + .sheet_meta.json
│       ├── output/                # Raw parser output (one .md per worksheet)
│       │   ├── enriched/          # LLM-rewritten, voicebot-friendly prose (RAG ingest source)
│       │   └── original/          # Pristine copy of raw outputs (for diffing)
│       └── src/
│           ├── sheet_downloader.py     # Pulls public Google Sheet → .xlsx (hash-cached)
│           ├── enricher.py             # Two-phase LLM enrichment (understanding + rewrite)
│           └── parsers/                # One BaseParser subclass per worksheet
│               ├── base.py             # Abstract pandas → markdown contract
│               ├── attractions.py, faqs.py, jump_prices.py, passes.py, ...
│               └── (14 worksheet-specific parsers)
│
├── src/                           # Shared utilities used by the chatbot side
│   ├── ai/localllm.py             # Async/sync wrapper around OllamaLLM (LangChain)
│   ├── utils/pipeline_logger.py   # Per-call structured pipeline log
│   └── logs/                      # Voice session logs (auto-created per call)
│
├── credentials/google_service_account.json   # GCP key for Sheets (gitignored)
├── db.py                          # SQLite call log (calls + messages tables)
├── calls.db                       # SQLite DB for call/message records
├── pyproject.toml, uv.lock        # uv-managed dependencies (Python 3.13)
├── requirements.txt               # Pip-style dependency list (legacy)
├── run.sh                         # Convenience launcher for chatbot.server
├── logs/                          # Per-call .log files (Twilio CallSid filename)
└── .env                           # Top-level env (Twilio creds, BASE_URL, embedding keys)
```

---

## 3. Execution Flow

### Two services, one pipeline

```
 ┌────────────────┐   HTTPS   ┌─────────────────────┐   HTTP   ┌────────────────────┐
 │ Twilio (PSTN)  │──────────▶│  chatbot/server.py  │─────────▶│  core/rag (FastAPI)│
 │ ConversationRel│ WebSocket │  (port 3232 / 8001) │ /retrieve│  (port 8000)       │
 └────────────────┘           └─────────┬───────────┘          └─────────┬──────────┘
                                        │                                │
                                        │ Ollama OpenAI-compat API       │ Ollama embeddings + LLM
                                        ▼                                ▼
                              ┌────────────────────┐          ┌────────────────────┐
                              │   Ollama server    │          │  Postgres + pgvector│
                              │  (LLM + embed)     │          │  (port 5433)        │
                              └────────────────────┘          └────────────────────┘
```

### Per-call timeline (voice path)

1. **Twilio webhook** → `POST /voice/inbound` returns TwiML that opens a `<ConversationRelay>` WebSocket.
2. **WebSocket setup** ([chatbot/server.py:324](chatbot/server.py#L324)) — Twilio sends `{type: "setup", callSid, from}`.
   - `db.start_call()` writes a row to `calls.db`.
   - A `PipelineLogger` opens `logs/<CallSid>.log`.
   - Session state initialised in `_voice_sessions` dict.
3. **User speaks** → Twilio ASR → sends `{type: "prompt", voicePrompt: "..."}`.
4. **Booking-change capture** ([voice_handler.py:191-230](chatbot/voice_handler.py#L191-L230)) — if utterance matches `_BOOKING_CAPTURE_TRIGGERS`, the LLM is bypassed and a deterministic state machine asks for name → phone → details, then writes a flagged row.
5. **Otherwise → LLM path** in `prepare_voice_stream()` ([voice_handler.py:508](chatbot/voice_handler.py#L508)):
   - **Query rewrite** — turn the follow-up into a self-contained search query (Ollama).
   - **Skip-RAG heuristic** — pure acknowledgments ("yeah", "okay") bypass retrieval.
   - **`POST /rag/retrieve`** to the core/rag service with `top_k=7`.
   - Build a system prompt = `VOICE_SYSTEM_PROMPT` + current Toronto time + dedup'd RAG context + last 20 turns.
6. **Stream tokens** ([voice_handler.py:553](chatbot/voice_handler.py#L553)) — `stream_voice_tokens()` flushes the first chunk on the first comma/period or after ~8 words to minimise time-to-first-audio. Every `{type: "text", token: "...", last: false}` is sent to Twilio for incremental TTS.
7. **End-of-turn** — server sends `{token: "", last: true}`. Cleaned reply is appended to `conversation_store`.
8. **End-of-call detection**:
   - Cheap keyword filter (`check_end_keywords`) → "definite" goodbye words finalize the call without an LLM hop.
   - "Maybe" matches go through `classify_turn_for_end()` (LLM JSON classifier) for nuance.
9. **Interrupt path** — if Twilio sends `{type: "interrupt"}`, the in-flight LLM streaming task is cancelled and `conversation_store.replace_last_assistant()` truncates history to what was actually spoken.
10. **Hangup / disconnect** — `db.end_call()` writes the full transcript JSON, summary, and `needs_human` flag.

### Per-resync timeline (knowledge-base path)

1. Operator (or scheduled job) calls `POST /rag/resync`.
2. **Drop and recreate** the `data_documents` pgvector table.
3. Subprocess `parser/run_parsers.py --enrich --force`:
   - **Download** Google Sheet as `.xlsx` (skip if md5 unchanged) — [sheet_downloader.py](core/rag/parser/src/sheet_downloader.py).
   - **Parse** each worksheet using its dedicated `BaseParser` subclass → `parser/output/<name>.md`.
   - **Enrich** ([enricher.py](core/rag/parser/src/enricher.py)):
     - Step 1: feed every workbook to a single Ollama session and ask for a holistic `data_understanding.md`.
     - Step 2: per workbook, prime a fresh session with that understanding, then rewrite the markdown into voice-friendly prose → `parser/output/enriched/<name>.md`.
4. **Re-index** `parser/output/enriched/*.md` via `MarkdownNodeParser` + `SentenceSplitter` (chunk_size=1000, overlap=200) → embed with `OllamaEmbedding` → write to pgvector.
5. Endpoint returns `{parser_ok, documents_indexed, document_count}`.

---

## 4. Key Components

### Voice service (`chatbot/`)
| File | Purpose |
|------|---------|
| [server.py](chatbot/server.py) | FastAPI app + Twilio webhooks + WebSocket handler. Owns the booking-capture state machine and per-session task lifecycle. |
| [voice_handler.py](chatbot/voice_handler.py) | All per-turn pipeline logic: system prompt, query rewriting, RAG-skip heuristic, end-of-call classifier, sentence-boundary token streaming, TTS cleaning. |
| [conversation.py](chatbot/conversation.py) | `ConversationStore` — thread-safe in-memory per-`session_id` history with 30-min TTL and trim to `MAX_CONVERSATION_TURNS`. |
| [llm.py](chatbot/llm.py) | Ollama client factory (sync + async OpenAI-compatible). 3-attempt retry with exponential backoff on 429/503. |
| [rag_client.py](chatbot/rag_client.py) | Reusable `httpx.AsyncClient` for `POST /rag/retrieve`. |

### RAG backend (`core/rag/src/`)
| File | Purpose |
|------|---------|
| [main.py](core/rag/src/main.py) | App bootstrap: wait-for-DB, ensure pgvector ext, create SQLModel tables, bootstrap auto-ingest, mount routers, CORS, frontend SPA fallthrough. |
| [config.py](core/rag/src/config.py) | `Settings` (Pydantic) — `APP_*` env vars: PG creds, embedding model/dim, default LLM, data folder paths. |
| [rag/repositories.py](core/rag/src/rag/repositories.py) | `RAGRepository` — owns the pgvector store, embedding probe, indexing pipeline, query engine, hot-reload of chat model. |
| [rag/services.py](core/rag/src/rag/services.py) | `RAGService` — wraps repo, runs the resync subprocess + log pump, persists query history. |
| [rag/routes.py](core/rag/src/rag/routes.py) | `/rag/query`, `/rag/retrieve`, `/rag/health`, `/rag/documents/count`, `/rag/resync`, `/rag/resync-raw`. |
| [history/](core/rag/src/history/) | SQLModel-backed query history (`QueryHistory`, `SourceDocumentHistory`) and `/history/*` routes. |
| [llm_config/](core/rag/src/llm_config/) | Singleton `LLMConfig` row + `GET/PUT /settings/llm` for runtime LLM swapping (hot-reloads `RAGRepository`). |

### Parser pipeline (`core/rag/parser/`)
| File | Purpose |
|------|---------|
| [run_parsers.py](core/rag/parser/run_parsers.py) | CLI entry — orchestrates download → 14 parsers → optional enrichment. |
| [src/sheet_downloader.py](core/rag/parser/src/sheet_downloader.py) | Public Sheet → .xlsx with md5-based change detection. |
| [src/enricher.py](core/rag/parser/src/enricher.py) | Two-phase LLM rewrite (Understanding doc → per-workbook voice-friendly prose). |
| [src/parsers/base.py](core/rag/parser/src/parsers/base.py) | `BaseParser` abstract class — pandas read + markdown write contract. |
| [src/parsers/*.py](core/rag/parser/src/parsers/) | One subclass per worksheet (Attractions, Jump Prices, Birthday Parties, Passes, FAQs, Voice Call Scripts, etc.). |

### Cross-cutting utilities (`src/`)
| File | Purpose |
|------|---------|
| [src/ai/localllm.py](src/ai/localllm.py) | LangChain-based `OllamaChat` wrapper with persistent SQL-backed history (used by enricher). |
| [src/utils/pipeline_logger.py](src/utils/pipeline_logger.py) | Per-call structured log: transcript → refined query → RAG hits → LLM context → final reply. |
| [db.py](db.py) | SQLite call log — `calls` and `messages` tables, plus booking-change persistence. |

### External dependencies
- **FastAPI / Uvicorn** — web framework & ASGI server (both services).
- **LlamaIndex** (`llama-index`, `llama-index-vector-stores-postgres`, `llama-index-llms-ollama`, `llama-index-embeddings-ollama`) — retrieval + indexing.
- **SQLModel / SQLAlchemy** — DB ORM for history & LLM config.
- **pgvector / psycopg2** — vector store.
- **Twilio Python SDK** — TwiML + ConversationRelay.
- **OpenAI SDK** — used to talk to Ollama's OpenAI-compatible endpoint.
- **LangChain** (core/community/ollama) — used by `OllamaChat` wrapper for the enricher.
- **pandas / openpyxl / pymupdf** — parsing Excel exports.
- **httpx** — async client for service-to-service calls.

---

## 5. Architecture Summary

### Pattern
**Two-service microservice** with a **layered (Repository → Service → Routes)** architecture inside the RAG backend.

- **`chatbot/`** — stateless edge service speaking Twilio's ConversationRelay protocol. Holds ephemeral per-call state in memory.
- **`core/rag/`** — stateful retrieval service backed by pgvector + Postgres. All knowledge-base mutations and ingest run here.
- **`core/rag/parser/`** — invoked as a subprocess by the resync endpoint; not a long-running service.

### Notable architectural decisions
- **Wipe-first resync** — `RAGService.resync()` drops the vector table *before* running the parser. Trade-off: a parser failure leaves the store empty, but guarantees the next ingest writes into a clean schema (avoids dimension/conflict bugs).
- **Subprocess parser** — the parser package's `src/` shadows the backend's `src/`, so it runs in a separate Python process via `subprocess.Popen` with a live stdout pump.
- **Two outputs from the parser** — raw `parser/output/*.md` (source of truth) and voice-rewritten `parser/output/enriched/*.md` (RAG ingest source). `/rag/resync-raw` exists for fast iteration without LLM enrichment.
- **Static vs. runtime config** — embedding model + DB are static (changing them requires a vector-table rebuild). Chat model is mutable via `PUT /settings/llm` and hot-reloads `Settings.llm` in `RAGRepository`.
- **Two LLM hops per turn** — query rewriter (small/fast model) + main reply (larger model). Set independently via `VOICE_FAST_MODEL` and `VOICE_LLM_MODEL`.
- **First-chunk fast flush** — voice handler flushes the first TTS chunk on the first punctuation/word-count threshold (≈8 words) to minimise silence-to-first-reply latency, then switches to clean sentence boundaries.
- **Deterministic booking capture** — high-stakes flow (booking changes) is *not* trusted to the LLM; matched by keyword and run as a 4-state machine that always saves and flags `needs_human=1`.
- **In-memory conversation store** — explicitly chosen over a DB for per-turn latency. Sessions expire after 30 min and are cleared on disconnect.

---

## 6. Setup & Running Instructions

### Prerequisites
- **Python 3.13** (chatbot side; pyproject pins this) — `core/rag/` Dockerfile uses 3.12.
- **Ollama** running locally or on a reachable host. Pull at minimum:
  ```bash
  ollama pull phi4:latest         # default chat / voice model
  ollama pull embeddinggemma      # default embedding model (768-dim)
  ```
- **Postgres + pgvector** (provided by `core/rag/docker-compose.yaml`).
- **Twilio account** with a phone number and ConversationRelay enabled.
- A public HTTPS tunnel (e.g. **zrok**, ngrok) so Twilio can reach the local server.

### Install dependencies (top-level chatbot)
```bash
# Using uv (preferred)
uv sync

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

`/.env` (chatbot side):
```bash
BASE_URL=https://<your-zrok-or-ngrok-url>      # public URL Twilio dials
RAG_API_URL=http://localhost:8000              # core/rag service
# Chatbot uses Ollama's OpenAI-compatible API — note the trailing /v1.
OLLAMA_BASE_URL=http://192.168.50.150:11434/v1
LLM_MODEL=phi4:latest
VOICE_LLM_MODEL=phi4:latest
VOICE_FAST_MODEL=phi4:latest
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
SESSION_LOG_DIR=logs
```

`core/rag/.env` (RAG backend):
```bash
APP_PG_HOST=localhost
APP_PG_PORT=5433
APP_PG_USER=postgres
APP_PG_PASSWORD=password
APP_PG_DATABASE=ultimate_advisor
APP_VECTOR_TABLE_NAME=documents
APP_EMBEDDING_MODEL=embeddinggemma
APP_EMBED_DIM=768
# RAG side uses LlamaIndex's native Ollama client — NO /v1 suffix.
APP_DEFAULT_OLLAMA_BASE_URL=http://192.168.50.150:11434
APP_DEFAULT_CHAT_MODEL=phi4:latest
APP_AUTO_INGEST=true
```

Place the Google service-account JSON at `credentials/google_service_account.json` (the file is gitignored).

### Start the RAG backend

```bash
cd core/rag
docker compose up -d db        # Postgres + pgvector on :5433
docker compose up -d backend   # FastAPI on :8000

# Or run the backend natively without Docker:
cd core/rag
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Build the knowledge base (first run only — auto-ingest will pick up enriched/ if present)

```bash
cd core/rag/parser
python run_parsers.py --enrich      # download → parse → LLM enrich
# or trigger remotely:
curl -X POST http://localhost:8000/rag/resync -d '{"force_download": true}' -H 'Content-Type: application/json'
```

### Start the voice server

```bash
./run.sh                                                # uses port 3232
# or manually:
uvicorn chatbot.server:app --host 0.0.0.0 --port 8001 --reload
```

### Expose to Twilio

```bash
zrok share public 3232          # or: ngrok http 3232
# Copy the HTTPS URL into BASE_URL in .env, then in Twilio Console:
#   Phone Numbers → your number → "A call comes in" → Webhook
#   POST {BASE_URL}/voice/inbound
```

### Smoke tests

```bash
curl http://localhost:3232/api/health          # {"status":"ok"}
curl http://localhost:8000/health              # {"status":"healthy", ...}
curl http://localhost:8000/rag/documents/count # row count after ingest
curl -X POST http://localhost:8000/rag/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"how much is a birthday party","top_k":5}'
```

---

## 7. Improvements / Issues

### Bugs / typos to fix
- [core/rag/src/rag/repositories.py:54](core/rag/src/rag/repositories.py#L54) — `tempreature = 0.3` is a typo (should be `temperature`) **and** is passed as a kwarg to `Ollama(...)` that the constructor likely ignores → temperature is effectively unset.
- [core/rag/src/rag/repositories.py:263](core/rag/src/rag/repositories.py#L263) — stray `print(optimized_top_k)` left in production code.
- [core/rag/src/rag/repositories.py:300](core/rag/src/rag/repositories.py#L300) — `similarity_cutoff=0.6` is hard-coded; the schemas in [schemas.py](core/rag/src/schemas.py) don't surface it as a request field even though the existing README hints at exposing it.
- [core/rag/parser/run_parsers.py:65](core/rag/parser/run_parsers.py#L65) — defaults are now consistent (LAN IP), but the URL is still hardcoded; should derive from an env var so a different host doesn't require a code edit.

### Security / config concerns
- **Secrets committed to `.env`** (Twilio, HuggingFace, Voyage AI keys) — these should be rotated and pulled from a secret manager. `.gitignore` *does* list `.env`, but the file is currently tracked-and-modified per `git status`. Verify nothing leaks before pushing.
- **`credentials/google_service_account.json`** — gitignored, but verify it's not in any past commit.
- **CORS** in [chatbot/server.py:144](chatbot/server.py#L144) defaults to `*` — fine for dev, but `CHATBOT_CORS_ORIGINS` should be set in prod.
- **No auth on RAG endpoints** — anyone who reaches `:8000` can `/rag/resync` (very expensive) or query history. Add a simple shared-secret header or put it behind a private network.

### Architectural improvements
- **Split or document the two `src/` packages** — top-level `src/` (chatbot utilities) and `core/rag/parser/src/` (parsers) collide on import path; the enricher already works around this with `importlib.util`. A rename (e.g. `chatbot_utils/`, `parser_lib/`) would remove the workaround.
- **Resync is blocking** — the FastAPI worker stalls for the entire enrichment run. Move it behind a background task (`BackgroundTasks` or a job queue) and expose `/rag/resync/status`.
- **Hot-reload is partial** — `PUT /settings/llm` reloads the LLM in `RAGRepository`, but the chatbot's own Ollama clients (`chatbot/llm.py`) are still the singletons created at import time and read env vars only once.
- **No tests** — there's no `tests/` directory or CI. End-of-call classifier, query rewriter, and TTS cleaner are all good candidates for unit tests with recorded fixtures.
- **Two separate LLM clients** — the chatbot uses raw OpenAI SDK against Ollama, while the enricher uses LangChain. Pick one or document the split.
- **`requirements.txt` ≠ `pyproject.toml`** — they have different (overlapping) dependency lists. The `requirements.txt` references `voyageai`, `gspread`, `sentence-transformers` etc. that don't appear in `pyproject.toml`. Decide which is canonical.

### Operational nits
- **Log rotation** — `logs/*.log` and `src/logs/*.log` accumulate per call with no rotation policy.
- **`calls.db` and `enrichment_memory.db`** are SQLite files committed to git (per `git status`); they should be in `.gitignore` alongside `.env`.
- **Hard-coded Spreadsheet ID** in [run_parsers.py:37](core/rag/parser/run_parsers.py#L37) — should be an env var (`GOOGLE_SHEET_ID` already exists in `.env` but isn't used here).
- **Embedding-dim probe** ([repositories.py:67](core/rag/src/rag/repositories.py#L67)) only warns on mismatch but still proceeds with the model dim — good. But the warning never bubbles up to a health endpoint.

### Suggested next steps
1. Fix the typo + `print` in `repositories.py`, drop a regression test that asserts temperature is plumbed through.
2. Move resync to a background task + status endpoint.
3. Add an auth header to `core/rag` (even just `X-Internal-Token`) and require it from `chatbot/rag_client.py`.
4. Consolidate dependencies into `pyproject.toml`, delete `requirements.txt`.
5. Add `calls.db`, `*.xlsx`, `enrichment_memory.db` to `.gitignore` and untrack them.
6. Wire up the `mailer` module hinted at in [server.py:565](chatbot/server.py#L565) so `needs_human=True` actually pages someone.
