# CONTEXT.md — AeroSports Scarborough RAG + Chatbot System

## Overview

This is a **RAG (Retrieval-Augmented Generation) pipeline + customer-facing chatbot** for **AeroSports Scarborough**, a trampoline park in Ontario. It combines a PostgreSQL/pgvector knowledge base (~120+ chunks across 14 sheets) with a streaming Llama 3.1 chatbot served via FastAPI, plus Twilio voice calling integration.

**Architecture: Sheet-first.** The Google Sheet is the single source of truth. `ingest.py` reads all sheets, chunks them using voice-optimized builders, generates a local JSON backup (`data/knowledge_base.json`), embeds, and upserts into pgvector.

---

## Architecture Diagram

```
User (Web Widget / Phone Call)
        │
        ▼
   FastAPI Server (server.py)
   ┌────┴────────────────────┐
   │  /api/chat/stream (SSE) │
   │  /api/chat (POST)       │
   │  /voice/ws (WebSocket)  │
   └────┬────────────────────┘
        │
        ├── Web Chat ──► Chat Handler (chat_handler.py)
        │                    │
        │                    ├──► semantic_search + hybrid_search
        │                    ├──► Prompt Builder (Felicia persona)
        │                    ├──► LLM Streaming (HuggingFace)
        │                    └──► Fallback CTA Detection
        │
        └── Voice Call ──► Voice Handler (voice_handler.py)
                             │
                             ├──► voice_search (boosts voice_script chunks)
                             │    + hybrid_search
                             ├──► Voice System Prompt (2 sentences max)
                             ├──► Non-streaming LLM (150 tokens)
                             └──► TTS Cleaning (strip markdown)
        │
        ▼
   RAG Search Layer (search.py)
        │
        ├── semantic_search (pgvector cosine, top 5)
        ├── hybrid_search (semantic 70% + keyword 30%, top 3)
        └── voice_search (cosine + voice_script source boost)
               │
               ▼
        PostgreSQL + pgvector
        (~120+ chunks, 1024-dim BAAI/bge-m3 embeddings)
        (source: knowledge_base | voice_script | chatbot_qr)
```

---

## File-by-File Breakdown

### Root — RAG Pipeline

| File | Purpose |
|------|---------|
| `config.py` | DB connection pool (psycopg2, max 10), embedding provider config, Google Sheets client, GOOGLE_SHEET_ID |
| `models.py` | Dataclasses: `ChunkRecord` (with sheet_name, source, metadata), `SearchResult`, `ChangeLogEntry` |
| `embedding.py` | Dual-provider embeddings: Voyage AI (API, 1024-dim) or local BAAI/bge-m3 (CPU) |
| `chunk_builder.py` | Voice-optimized builders for all 14 sheets. Router + per-sheet builder functions producing speakable ChunkRecords |
| `setup_db.py` | One-time DB init: creates tables with source/sheet_name/metadata columns + pgvector indexes |
| `ingest.py` | Sheet-first: Google Sheets → chunk_builder → JSON backup → embed → pgvector (clean slate) |
| `search.py` | semantic_search + hybrid_search (unchanged) + voice_search (new, boosts voice_script source) |
| `sync.py` | Google Sheets Change Log → rebuild via chunk_builder → re-embed → upsert. Handles Promotions status changes |
| `requirements.txt` | All Python deps |

### chatbot/ — Chatbot Server + Widget

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app. Routes: `/api/chat`, `/api/chat/stream` (SSE), `/api/chat/reset`, `/api/health`, `/voice/*`. Voice WebSocket routes through `handle_voice_message` |
| `llm.py` | HuggingFace `InferenceClient` wrapper. Streaming generator with 3x retry (429/503 backoff). Also `generate_response_sync()` for voice |
| `prompt_builder.py` | System prompt: persona "Felicia" (real staff member vibe), voice/TTS rules, tone guide, knowledge-only answers, de-escalation. Expanded knowledge scope for new topics |
| `chat_handler.py` | Core orchestrator: sanitize input → concurrent RAG retrieval → deduplicate/filter (≥0.55 relevance) → build prompt → stream LLM → detect fallback CTAs → persist to session |
| `conversation.py` | Thread-safe in-memory session store. 30-min TTL, 10 turns max, auto-trim, cleanup every 5 min |
| `fallback.py` | Regex-based CTA detection: no-confidence deflection, booking intent, custom events, other-location queries |
| `voice_handler.py` | Twilio ConversationRelay handler. Uses voice_search() for retrieval, non-streaming LLM (max 150 tokens), TTS text cleaning, voice-specific system prompt |

### chatbot/static/ — Frontend

| File | Purpose |
|------|---------|
| `index.html` | Dev test page: health check, API docs links, test query chips, cURL examples |
| `widget.html` | Minimal iframe wrapper for embedding the chat widget |
| `widget.js` | Self-contained chat widget (~350 lines). Floating toggle button, SSE streaming, session management (sessionStorage), lightweight markdown rendering, XSS-safe |
| `widget.css` | Dark theme styling. Neon pink (#F00C74) + green (#39FF14). Responsive. Animated open/close + typing indicator |

### Data

| File | Purpose |
|------|---------|
| `data/knowledge_base.json` | Auto-generated JSON backup of all chunks (no embeddings). Created by `ingest.py` from Google Sheets |
| `credentials/google_service_account.json` | Google Sheets API auth |

---

## Database Schema

```sql
-- PostgreSQL 16 + pgvector extension

-- Main knowledge base (~120+ rows)
knowledge_chunks (
    id           TEXT PRIMARY KEY,           -- e.g. "scb_jump_003"
    category     TEXT,                       -- e.g. "Pricing"
    subcategory  TEXT,                       -- e.g. "Jump Passes"
    location     TEXT DEFAULT 'Scarborough',
    question     TEXT,                       -- e.g. "How much does it cost to jump?"
    answer       TEXT,                       -- Full speakable answer text
    tags         TEXT[],                     -- e.g. ["pricing", "jump"]
    embedding    vector(1024),              -- BAAI/bge-m3 embedding
    sheet_name   TEXT DEFAULT '',            -- Source sheet name
    source       TEXT DEFAULT 'knowledge_base', -- 'knowledge_base' | 'voice_script' | 'chatbot_qr'
    metadata     JSONB DEFAULT '{}',        -- Extra structured data
    created_at   TIMESTAMP,
    updated_at   TIMESTAMP
)
-- Indexes: category, subcategory, GIN(tags), IVFFlat(embedding cosine), source, sheet_name

-- Version tracking (1 row)
sync_state (id INTEGER PRIMARY KEY, last_version TEXT, last_synced_at TIMESTAMP)

-- Audit log
sync_history (id SERIAL, change_id TEXT, chunk_id TEXT, change_type TEXT, field_changed TEXT, synced_at TIMESTAMP)
```

---

## Data Flow: User Message → Response

### Web Chat Path
1. **Input** — User types in widget
2. **Server** — FastAPI receives via POST or SSE
3. **Sanitize** — Strip whitespace, truncate to 500 chars
4. **RAG Retrieval** — Concurrent: `semantic_search(top 5)` + `hybrid_search(top 3)` via asyncio thread pool
5. **Deduplicate** — Merge results, keep first occurrence per chunk_id, filter ≥ 0.55 relevance, cap at 5
6. **Prompt Build** — System prompt (Felicia persona) + RAG context with relevance % + conversation history + user message
7. **LLM Stream** — HuggingFace Llama 3.1 8B, temp=0.3, tokens streamed via async queue bridge
8. **Fallback CTAs** — Post-LLM: append booking/contact/location CTAs based on regex intent detection
9. **Persist** — Store user message + full response in session (30-min TTL, 10 turns)
10. **Response** — SSE events: `{token: "..."}` per token, then `{done: true, sources: [...]}`

### Voice Call Path
1. **Input** — Caller speaks, Twilio ASR transcribes
2. **WebSocket** — `/voice/ws` receives transcribed text
3. **Sanitize** — Strip whitespace, truncate to 500 chars
4. **RAG Retrieval** — Concurrent: `voice_search(top 5)` (boosts voice_script chunks) + `hybrid_search(top 3)`
5. **Deduplicate** — Merge, keep first per chunk_id, cap at 5
6. **Prompt Build** — Voice system prompt (2 sentences max, no markdown) + RAG context + history
7. **LLM (Non-streaming)** — Same HuggingFace API, max_tokens=150, blocking call
8. **TTS Cleaning** — Strip markdown, bullets, code fences
9. **Persist** — Store in shared session store (uses call_sid as session ID)
10. **Response** — Single text message back to Twilio → ElevenLabs TTS → caller hears speech

---

## Source-Aware Search

Chunks have a `source` field:
- `knowledge_base` — Standard KB chunks (jump prices, birthday parties, FAQs, etc.)
- `voice_script` — Phone call scripts optimized for spoken delivery
- `chatbot_qr` — Chatbot quick replies for common intents

`voice_search()` in search.py applies a 0.1 similarity boost to `voice_script` chunks, ensuring phone-optimised scripts surface first when relevant. The web chatbot uses `semantic_search` + `hybrid_search` as before — it naturally benefits from richer KB since all source types are in the same table.

---

## Chunking Strategy

All chunks are **voice-optimized**: complete, speakable sentences or paragraphs. No abbreviations, no tabular shorthand. Each chunk should sound natural when spoken aloud by "Felicia" on a phone call.

14 sheets are chunked via `chunk_builder.py`:

| Sheet | chunk_id prefix | Source Tag |
|-------|----------------|------------|
| Location Info | scb_contact_, scb_hours_, scb_links_ | knowledge_base |
| Jump Prices | scb_jump_, scb_socks_ | knowledge_base |
| Go Karting | scb_gokart_ | knowledge_base |
| Special Programs | scb_toddler_, scb_glow_, scb_special_ | knowledge_base |
| Attractions | scb_attr_ | knowledge_base |
| Birthday Parties | scb_bday_, scb_bday_addons_ | knowledge_base |
| Group Bookings | scb_group_, scb_corporate_, scb_school_, scb_fundraise_, scb_facility_, scb_rooms_ | knowledge_base |
| Aero Camp | scb_camp_ | knowledge_base |
| Passes | scb_passes_ | knowledge_base |
| Promotions | scb_promo_ (Active only) | knowledge_base |
| Park Rules | scb_rules_ | knowledge_base |
| FAQs | scb_faq_ | knowledge_base |
| Voice Call Scripts | scb_voice_ | voice_script |
| Chatbot Quick Replies | scb_qr_ | chatbot_qr |

Skipped sheets: Change Log, Unanswered Questions.

---

## Key Configuration (.env)

| Variable | Value | Purpose |
|----------|-------|---------|
| `EMBEDDING_PROVIDER` | `local` | Uses BAAI/bge-m3 on CPU (1024-dim) |
| `HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | LLM for chat responses |
| `HF_TOKEN` | (set) | HuggingFace API auth |
| `GOOGLE_SHEET_ID` | (set) | Updated KB spreadsheet with 14 data sheets |
| `CHUNK_RELEVANCE_THRESHOLD` | `0.55` | Min similarity to include in context |
| `MAX_CONTEXT_CHUNKS` | `5` | Max RAG results sent to LLM |
| `MAX_CONVERSATION_TURNS` | `30` | Session history depth |
| `SYNC_ON_STARTUP` | `false` | Auto-sync Google Sheets on boot |
| `BASE_URL` | `https://aerosportsscb.share.zrok.io` | Public URL (zrok tunnel) |

---

## Chatbot Persona

- **Name:** Felicia (acts as real AeroSports staff, not a bot)
- **Tone:** Warm but efficient, uses contractions and natural fillers ("No worries," "Gotcha," "Yeah")
- **Rules:** Only answers from RAG context, never invents prices, deflects unknowns to phone/email
- **Voice-optimized:** No markdown, phonetic prices ("nineteen ninety plus tax"), spelled-out URLs
- **De-escalation:** Listen → validate feelings → redirect to facts
- **Knowledge scope:** Jump pricing, go karting, individual attractions, birthday packages, group/corporate/school/fundraising events, facility rentals, Aero Camp, passes, active promotions, park rules, special programs, FAQs

---

## Running

```bash
# Bootstrap database (creates tables + new columns)
python setup_db.py

# Full ingest from Google Sheets (sheet-first)
python ingest.py

# Fallback: ingest from local JSON backup
python ingest.py --from-json

# Incremental sync from Change Log
python sync.py

# Start chatbot server
uvicorn chatbot.server:app --host 0.0.0.0 --port 8000 --reload

# With Google Sheets sync on startup
SYNC_ON_STARTUP=true uvicorn chatbot.server:app --port 8000
```

---

## Twilio Voice Integration

**Flow:** Caller dials → Twilio webhook (`POST /voice/inbound`) → returns TwiML with `<ConversationRelay>` → Twilio opens WebSocket (`wss://.../voice/ws`) → real-time loop: Twilio ASR (speech→text) → WebSocket → `handle_voice_message` → voice_search + hybrid_search → voice system prompt → non-streaming LLM (150 tokens) → TTS cleaning → tokens back → Twilio ElevenLabs TTS (text→speech)

**Three endpoints:**
1. `POST /voice/inbound` — Returns TwiML XML configuring ConversationRelay with ElevenLabs TTS, welcome greeting as "Felicia"
2. `WS /voice/ws` — Real-time WebSocket: receives transcribed speech, calls `handle_voice_message` from `voice_handler.py`, sends clean reply back
3. `POST /voice/action` — End-of-call handler, says goodbye and hangs up

**Voice-specific adaptations (voice_handler.py):**
- Uses `voice_search()` which boosts voice_script source chunks
- Shorter token budget (150 vs 1024)
- TTS cleaning: strips markdown, bullets, code fences
- Voice-specific system prompt (2 sentences max, no markdown)
- Shared session store with web chat (uses `call_sid` as session ID)

---

## Current State

- **Branch:** `airesponse`
- **Architecture:** Sheet-first (Google Sheets → chunk_builder → JSON backup → pgvector)
- **~120+ chunks** across 14 sheets (up from 93 in the original JSON)
- **Source-aware retrieval:** voice_search() boosts voice_script chunks for phone calls
- **Voice WebSocket:** correctly routes through handle_voice_message (voice-specific prompt, TTS cleaning, 150 tokens)
- **`data/knowledge_base.json`** is auto-generated by ingest.py, not hand-crafted
