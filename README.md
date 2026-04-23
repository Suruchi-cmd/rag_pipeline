# AeroSports Scarborough — AI Receptionist (Voice + Chat)

A FastAPI server that answers customer questions for **AeroSports Scarborough**
(trampoline park, Ontario) both through a web chat widget and through real
phone calls via Twilio. Knowledge retrieval is delegated to an **external
RAG service** (`RAG_API_URL`), and text generation runs on a local
**Ollama** LLM.

---

## 1. High-Level Overview

### What this project does
- Serves a streaming **chat widget** over SSE.
- Accepts **inbound phone calls** through Twilio **ConversationRelay**
  (Deepgram STT + ElevenLabs TTS) and streams LLM replies back over a
  WebSocket in real time, with barge-in / interrupt support.
- For every user turn, it:
  1. Posts the query to an **external RAG API**
     (`POST {RAG_API_URL}/rag/retrieve`) to fetch `source_documents`.
  2. Builds a voice-tuned prompt (persona + voice rules + retrieved context +
     recent conversation history).
  3. Streams tokens from a local **Ollama** model (`phi4:latest` by default).
  4. For voice: cleans tokens for TTS, detects end-of-call cues, and can
     enter a deterministic booking-change capture state machine.
  5. Persists the call, the per-message transcript, and any captured
     booking-change request to **SQLite** (`calls.db`).

### What it does NOT do (important)
- **This repo does not own the vector store or the embeddings.** There are
  several legacy ingestion/embedding/search modules at the root of the
  repo (`ingest.py`, `embedding.py`, `chunk_builder.py`, `search.py`,
  `setup_db.py`, `sync.py`). They are **not part of the live runtime** —
  the live voice pipeline calls the **external** RAG API at
  `RAG_API_URL` (default `https://aeroscbadvisor.share.zrok.io`), which
  hosts the pgvector store, embedding model, and chunking. See §7.

### Problem it solves
- Reduces front-desk phone load by handling repetitive questions
  (hours, pricing, packages, promos) consistently in the park's voice.
- Keeps answers **factually grounded** — the LLM is only allowed to use
  what the external RAG API returns; hallucinations are deflected to a
  callback.
- Captures booking-change requests and logs every call for staff follow-up.

---

## 2. Project Structure

```
rag_pipeline/
├── README.md                  ← this file
├── CLAUDE.md                  Project notes for Claude Code (partly stale)
├── CONTEXT.md, SETUP.md       Long-form context / setup docs
├── requirements.txt
├── run.sh                     Convenience launcher for the server
├── .env                       Secrets & runtime config (NOT commit)
│
├── chatbot/                   ← THE LIVE APP
│   ├── server.py              FastAPI app: /api/chat, SSE, /voice/*, WS /voice/ws
│   ├── voice_handler.py       Twilio voice pipeline — calls external RAG API,
│   │                          streams LLM tokens, barge-in, booking capture
│   ├── chat_handler.py        Web-chat pipeline (also defined here; see §7 note)
│   ├── call_handler.py        Alt. tool-calling voice flow (not wired up)
│   ├── llm.py                 Ollama OpenAI-compatible client (sync + async)
│   ├── prompt_builder.py      Web-chat system prompt + context assembler
│   ├── conversation.py        In-memory per-session history (30-min TTL)
│   ├── fallback.py            Post-response CTA detection (booking / phone / email)
│   └── static/                Embeddable widget (HTML / CSS / JS)
│
├── utils/
│   └── pipeline_logger.py     Per-call structured voice-pipeline log
│
├── db.py                      SQLite call log (calls / messages / booking_change)
├── mailer.py                  Flag-alert stub (writes logs/flag_alerts.log)
├── calls.db                   SQLite data file
├── logs/                      Per-call .log files
│
├── credentials/               Google service-account JSON (legacy, unused in live flow)
├── data/                      Legacy KB JSON (unused in live flow)
│
└── [legacy, NOT part of the live runtime] — see §7
    ├── config.py              DB pool / Sheets client / embedding constants
    ├── models.py              ChunkRecord, SearchResult, ChangeLogEntry
    ├── embedding.py           Voyage / local sentence-transformer embeddings
    ├── setup_db.py            One-shot pgvector schema setup
    ├── ingest.py              Sheets → chunks → embeddings → pgvector
    ├── sync.py                Incremental Change Log → pgvector sync
    ├── chunk_builder.py       Per-sheet chunk builders
    ├── search.py              semantic / hybrid / voice search SQL
    └── test.py
```

### Purpose of each live module

| File | Purpose |
| --- | --- |
| [chatbot/server.py](chatbot/server.py) | FastAPI routes (HTTP, SSE, Twilio webhook, WebSocket), lifespan (DB init, periodic session cleanup). |
| [chatbot/voice_handler.py](chatbot/voice_handler.py) | The voice pipeline. `_query_rag_api` → external RAG; `prepare_voice_stream` / `stream_voice_tokens` run the LLM; end-of-call classifiers and booking-capture helpers. |
| [chatbot/chat_handler.py](chatbot/chat_handler.py) | Web-chat pipeline (still imports local `search.py`; see Issues). |
| [chatbot/llm.py](chatbot/llm.py) | Thin wrapper over Ollama's OpenAI-compatible API; sync + async clients with 3× retry. |
| [chatbot/prompt_builder.py](chatbot/prompt_builder.py) | XML-tagged persona + voice/knowledge rules + RAG context + time awareness. |
| [chatbot/conversation.py](chatbot/conversation.py) | Thread-safe in-memory store, 30-min TTL, 30-turn cap — keyed by session_id (web) or Twilio CallSid (voice). |
| [chatbot/fallback.py](chatbot/fallback.py) | Rule-based CTAs appended after the LLM reply. |
| [db.py](db.py) | SQLite call log: `start_call`, `log_message`, `end_call`, `save_booking_change`. |
| [mailer.py](mailer.py) | Flag-alert stub, writes to `logs/flag_alerts.log`. |
| [utils/pipeline_logger.py](utils/pipeline_logger.py) | Per-call structured trace: transcript → refined query → RAG docs → LLM context → raw response → TTS-clean output. |

---

## 3. Execution Flow

### Voice call (the live path)

```
 Caller ─► Twilio ─POST─► /voice/inbound                       (server.py)
                               │
                               └── TwiML <ConversationRelay wss://…/voice/ws>

 Twilio STT ─WebSocket─► /voice/ws                             (server.py + voice_handler)

 setup     : open session, db.start_call(), open per-call log
 prompt    : user turn arrives as transcribed text
             ├─ booking-capture triggered?  → canned state machine
             │                                (name → phone → details → save)
             │
             └─ else:
                 ├─ prepare_voice_stream(call_sid, user_text)
                 │     ├─ _should_skip_rag?  (acks like "yes/thanks/hi")
                 │     ├─ _rewrite_query()   (resolve follow-ups)
                 │     ├─ _query_rag_api()  ─► POST {RAG_API_URL}/rag/retrieve
                 │     │                       body: {"query":…, "top_k":7}
                 │     │                       resp: {"source_documents":[…]}
                 │     └─ _build_voice_messages()   (system + rag + history + user)
                 │
                 └─ stream_voice_tokens()   (Ollama streaming, TTS-cleaned)
                     └─► ws.send {"type":"text","token":"…","last":false}
                         ws.send {"type":"text","token":"",  "last":true}

 interrupt : cancel the in-flight asyncio.Task; save only what Twilio
             actually spoke aloud ("utteranceUntilInterrupt") to history.

 end-of-turn classifiers decide whether to hang up or let the caller
 speak again; booking changes are persisted via db.save_booking_change().
```

### Web chat (SSE)

```
 Browser (widget.js) ─GET─► /api/chat/stream?message=…          (server.py)
                                   │
                                   ▼
                          chat_handler.handle_message()         (chat_handler.py)
                                   │
                    NOTE: this path currently still imports the
                          LOCAL search.py / embedding.py and expects
                          a local pgvector. See Issues §7.
                                   │
                                   ▼
                          prompt_builder.build_messages()
                                   ▼
                          llm.generate_response()  ← Ollama
                                   ▼
                          SSE events: {token, done, sources}
```

### Entry points
- **Server**: `uvicorn chatbot.server:app --host 0.0.0.0 --port 3232`
  (or `./run.sh`). Exposes:
  - `GET  /`                — test page (static HTML)
  - `GET  /widget`          — iframe-ready widget shell
  - `POST /api/chat`        — non-streaming chat
  - `GET  /api/chat/stream` — SSE streaming chat
  - `POST /api/chat/reset`  — clear a session's history
  - `GET  /api/health`      — DB + HF-token + CRUD check
  - `POST /voice/inbound`   — Twilio webhook (returns TwiML)
  - `POST /voice/action`    — Twilio action webhook (hangup)
  - `WS   /voice/ws`        — Twilio ConversationRelay socket

---

## 4. Key Components

### External RAG API (what's actually doing retrieval)
- **Config**: `RAG_API_URL` env var. Default in code:
  `https://aeroscbadvisor.share.zrok.io`
  ([chatbot/voice_handler.py:114](chatbot/voice_handler.py#L114)).
- **Call site**: `_query_rag_api()` in
  [chatbot/voice_handler.py:148](chatbot/voice_handler.py#L148).
- **HTTP contract**:
  - `POST {RAG_API_URL}/rag/retrieve`
  - Request: `{"query": "<string>", "top_k": <int>}`
  - Response: `{"source_documents": [{"content": "...", "score": 0.87, "metadata": {...}}, ...]}`
- **Reliability**: `httpx.AsyncClient` with a persistent connection (avoids
  TLS re-handshake per call), 15 s timeout, returns an empty list on
  failure so the LLM still has a chance to deflect gracefully.

### Voice pipeline
- **`prepare_voice_stream`** ([voice_handler.py:885](chatbot/voice_handler.py#L885))
  — refines the query, calls the RAG API, builds the LLM messages list,
  records the user turn to `conversation_store`, logs each step via
  `PipelineLogger`.
- **`stream_voice_tokens`** ([voice_handler.py:1061](chatbot/voice_handler.py#L1061))
  — async generator that yields TTS-safe sentences from Ollama; the
  server pushes each one over the Twilio WS.
- **Barge-in**: on a Twilio `interrupt` frame, the server cancels the
  streaming `asyncio.Task`; the `utteranceUntilInterrupt` field is what
  actually gets persisted to history (what was spoken aloud).
- **Booking-change capture**: deterministic 3-step FSM
  (`name → phone → details`) in
  [chatbot/server.py:540](chatbot/server.py#L540) bypassing the LLM
  entirely, then saved via `db.save_booking_change`.

### Web chat pipeline
- **`handle_message`** ([chatbot/chat_handler.py:167](chatbot/chat_handler.py#L167))
  — orchestrates retrieval, prompt building, streaming, CTA append, and
  persistence. Note: it currently imports `search.py` at the repo root
  rather than calling the external RAG API — see §7.

### LLM layer
- **Ollama** exposing the OpenAI-compatible API at
  `http://localhost:11434/v1` ([chatbot/llm.py:31](chatbot/llm.py#L31)).
- Default model `phi4:latest` (`LLM_MODEL` env var).
- Streaming via the sync `OpenAI()` client for web chat (bridged to
  asyncio via a daemon thread + queue) and the native `AsyncOpenAI()`
  client for voice.

### Conversation store
- **`ConversationStore`** ([chatbot/conversation.py:26](chatbot/conversation.py#L26))
  — singleton, thread-safe, 30-minute TTL, 30-turn cap. Keyed by
  `session_id` for web and `CallSid` for voice — both surfaces share the
  same primitive.

### Call log (SQLite)
- [db.py](db.py) tables:
  - `calls(id, phone_number, started_at, ended_at, transcript_json, summary, needs_human, flag_reason, booking_change_json, metadata_json)`
  - `messages(id, call_id, role, content, created_at)`
- `db.end_call` serialises the full transcript into `transcript_json`
  when the session ends.

### Dependencies
| Concern | Library | Service |
| --- | --- | --- |
| Web framework | `fastapi`, `uvicorn`, `sse-starlette` | — |
| HTTP client | `httpx` | External RAG API |
| LLM SDK | `openai` | **Ollama** on `localhost:11434` |
| Telephony | `twilio` | Twilio ConversationRelay (Deepgram + ElevenLabs) |
| Call storage | stdlib `sqlite3` | `calls.db` |

---

## 5. Architecture Summary

### Design pattern — client of an external RAG service

```
┌────────────────────────────────────────────────────────────────┐
│  Transport      widget.js / SSE      •      Twilio WS           │
├────────────────────────────────────────────────────────────────┤
│  Orchestration  chat_handler.py      •      voice_handler.py    │
├────────────────────────────────────────────────────────────────┤
│  Prompt + LLM   prompt_builder.py    •      llm.py (Ollama)     │
├────────────────────────────────────────────────────────────────┤
│  Retrieval                                                      │
│     voice: httpx ─► {RAG_API_URL}/rag/retrieve   ◄── external   │
│     web : (currently local search.py — see Issues)              │
├────────────────────────────────────────────────────────────────┤
│  Local storage      calls.db (SQLite)   •   in-memory sessions  │
└────────────────────────────────────────────────────────────────┘
```

### Notable decisions
- **RAG is a network service, not a library.** The chatbot talks to
  it over HTTP so the vector store, embedding model, and chunking can
  evolve independently (and be hosted elsewhere). That also keeps this
  repo's runtime lean.
- **Persistent `httpx.AsyncClient`** for the RAG call — single TLS handshake,
  re-used across the whole process.
- **Streaming first** — both SSE (chat) and ConversationRelay (voice)
  push tokens as soon as Ollama emits them. Voice additionally buffers
  to sentence boundaries so TTS doesn't stutter.
- **Barge-in via `asyncio.Task.cancel()`** — only what was actually spoken
  (`utteranceUntilInterrupt`) is persisted to history.
- **Deterministic capture FSM** for PII (name / phone / booking details) —
  the LLM is bypassed entirely to eliminate hallucinated confirmations.
- **Shared `conversation_store`** across web and voice — a single
  in-memory primitive keyed by whichever session identifier the surface
  provides.

---

## 6. Setup & Running

### Prerequisites
- Python 3.11+
- A reachable **RAG API** with `POST /rag/retrieve` (see §4 contract).
- [Ollama](https://ollama.com) running locally with the chosen model pulled
  (default `phi4:latest`).
- (Voice only) Twilio account + a number configured for ConversationRelay,
  plus a public HTTPS tunnel (ngrok, zrok, etc.) whose URL goes in `BASE_URL`.

### Install
```bash
git clone <this-repo>
cd rag_pipeline

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull phi4     # or whichever model you set as LLM_MODEL
```

### `.env`
```ini
# RAG (external service)
RAG_API_URL=https://aeroscbadvisor.share.zrok.io

# LLM (local Ollama)
LLM_MODEL=phi4:latest
OLLAMA_BASE_URL=http://localhost:11434/v1
VOICE_FAST_MODEL=phi4:latest       # optional override for voice

# Server
CHATBOT_CORS_ORIGINS=*             # comma-separated origins in prod
MAX_CONVERSATION_TURNS=30
CHUNK_RELEVANCE_THRESHOLD=0.55
SESSION_LOG_DIR=logs

# Twilio (voice only)
BASE_URL=https://your-public-host
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
```

### Run the server
```bash
./run.sh
# or
uvicorn chatbot.server:app --host 0.0.0.0 --port 3232 --reload
```

Then:
- [http://localhost:3232/](http://localhost:3232/) — test page
- [http://localhost:3232/widget](http://localhost:3232/widget) — iframe widget
- `GET http://localhost:3232/api/health` — sanity check

### Embed the widget
```html
<iframe src="https://your-domain/widget"
        width="400" height="600"
        style="border:none;border-radius:16px;"
        title="AeroBot Chat"></iframe>
```

### Wire up Twilio (voice)
1. Expose `:3232` publicly (ngrok/zrok) and set `BASE_URL` to that URL.
2. In the Twilio console, set your number's **"A call comes in"** webhook to
   `POST {BASE_URL}/voice/inbound`.
3. Call the number. `/voice/inbound` returns TwiML that starts
   `ConversationRelay` pointing at `wss://{BASE_URL}/voice/ws`.

---

## 7. Improvements / Issues

### Dead-code / confusing layout
- **Legacy local-RAG files at the repo root are not used by the live
  voice runtime.** [config.py](config.py), [embedding.py](embedding.py),
  [ingest.py](ingest.py), [sync.py](sync.py), [search.py](search.py),
  [setup_db.py](setup_db.py), [chunk_builder.py](chunk_builder.py),
  [models.py](models.py), [data/](data/), [credentials/](credentials/).
  They correspond to the external RAG service's original codebase and
  were left behind when RAG was split out. Recommend either:
  - **Delete them** from this repo entirely and keep the RAG service
    code in its own repo; or
  - **Move them under `legacy/`** with a one-line README explaining
    they're historical.
- **`chatbot/chat_handler.py` still uses the local `search.py`.** The web
  chat path imports and calls `semantic_search` / `hybrid_search` against
  a local pgvector that probably no longer exists on this host. Either:
  - Switch web chat to call the same external `/rag/retrieve` endpoint
    via `httpx` (recommended — unifies both surfaces), or
  - Deprecate `/api/chat*` and point the widget at a gateway that hits
    the RAG API directly.
- **`chatbot/call_handler.py`** is not imported anywhere, references an
  un-imported `db` module (line 76), and duplicates part of
  `voice_handler.py`. Delete it, or document what flow it represents.
- **Persona disagreement**: the system prompt calls the voice persona
  "Dan" but the Twilio welcome greeting (server.py:347) says "this is
  **Rajan**." Pick one and align both.
- **`CLAUDE.md` is stale** — it describes a HuggingFace
  `meta-llama/Llama-3.1-8B-Instruct` integration, but the code uses
  Ollama (`phi4:latest`). Update it.

### Security
- **Secrets are committed in `.env`** (Twilio auth token, HF token, Voyage
  key). Rotate them and confirm `.env` is in `.gitignore`.
- **CORS defaults to `*`** ([server.py:177](chatbot/server.py#L177)). Lock
  `CHATBOT_CORS_ORIGINS` to known origins in production.
- **No auth on any endpoint.** Anyone who can reach `:3232` can chat or
  spoof Twilio webhooks. Add:
  - An API key on `/api/chat*` (or move to cookie-auth).
  - **Twilio signature validation** on `/voice/inbound` and `/voice/ws`
    — right now either can be hit by anyone.
- **SQLite call logger opens a new connection per message**
  ([db.py](db.py)). Fine at low volume, but a single long-lived
  connection (or `aiosqlite`) would be faster and safer under concurrent
  traffic.

### Reliability
- **In-memory `conversation_store`.** A process restart wipes all active
  sessions. For voice that's usually fine; for the web widget it's a UX
  regression. Swap for Redis with the same interface.
- **No retries on the RAG API.** `_query_rag_api` swallows exceptions
  and returns `[]` — which makes the LLM deflect, but masks real
  outages. Log the error class and add a single retry with short
  backoff; surface repeated failures on `/api/health`.
- **`/api/health`** currently returns `degraded` if the local pgvector
  pool can't be reached. Since the live path uses the external RAG, the
  check should instead ping `{RAG_API_URL}/rag/retrieve` (with a trivial
  query) and report on that.
- **Session log files (`logs/`) are unbounded.** Add rotation or a
  cleanup job.

### DX / testing
- No unit tests. At minimum cover `fallback.detect_fallback`,
  `_rewrite_query`, and the booking-capture FSM — all pure or
  near-pure.
- Replace the hand-rolled `sys.path` insert in `chat_handler.py` /
  `voice_handler.py` with a proper package layout (`chatbot/` has
  `__init__.py` already; pull `db.py`, `mailer.py`, `utils/` into it or
  convert to a `src/` layout).
- Consider a lightweight admin view on top of `calls.db` (e.g. a
  `/admin/calls` route behind auth) so staff can triage `needs_human=1`
  calls without SQL.

---

## Questions for you
1. **What's the contract of the external RAG API beyond
   `/rag/retrieve`?** Does it expose other endpoints (metadata,
   health, reindex) we should call from here?
2. **Should the web chat (`/api/chat*`) switch to the same external
   RAG**, or is the local-pgvector path deliberate (e.g. a fallback)?
3. **Is the legacy RAG code at the root safe to delete / move?** And
   where does the live pgvector + ingestion actually live now — in a
   sibling repo?
4. **Persona name** — "Dan" (prompt) or "Rajan" (Twilio greeting)?
5. **Deployment target** — single VM behind a tunnel, or eventually
   behind a load balancer? That drives whether the in-memory session
   store needs to move to Redis.
