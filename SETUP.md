# AeroSports RAG Pipeline + Chatbot — Setup Guide

A complete RAG (Retrieval-Augmented Generation) pipeline and conversational chatbot for AeroSports Scarborough. Includes semantic search over a PostgreSQL/pgvector knowledge base, a streaming FastAPI chatbot server, an embeddable web widget, and Twilio voice integration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Clone & Environment Setup](#clone--environment-setup)
4. [PostgreSQL + pgvector Setup](#postgresql--pgvector-setup)
5. [Environment Variables (.env)](#environment-variables-env)
6. [Install Python Dependencies](#install-python-dependencies)
7. [Initialize the Database](#initialize-the-database)
8. [Ingest the Knowledge Base](#ingest-the-knowledge-base)
9. [Google Sheets Sync (Optional)](#google-sheets-sync-optional)
10. [Run the Chatbot Server](#run-the-chatbot-server)
11. [Twilio Voice Integration (Optional)](#twilio-voice-integration-optional)
12. [Project Structure](#project-structure)
13. [API Reference](#api-reference)
14. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
User (Web Widget / Twilio Voice)
        │
        ▼
FastAPI Server (chatbot/server.py)
        │
        ├── RAG Pipeline
        │     ├── Semantic Search  (pgvector cosine similarity)
        │     └── Hybrid Search    (semantic 70% + keyword 30%)
        │
        ├── LLM (Llama 3.1 8B via HuggingFace Inference API)
        │
        └── PostgreSQL 16 + pgvector
              └── knowledge_chunks table (93 chunks, 1024-dim embeddings)
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 + pgvector |
| Embeddings | Voyage AI (API) or Sentence-Transformers (local/CPU) |
| LLM | Llama 3.1 8B Instruct via HuggingFace Inference API |
| Web Server | FastAPI + uvicorn + SSE-Starlette |
| Frontend | Vanilla JS embeddable chat widget |
| Voice | Twilio ConversationRelay (WebSocket) |
| Sync | Google Sheets → gspread → pgvector |

---

## Prerequisites

Ensure the following are installed before proceeding:

- **Python 3.11+** — `python3 --version`
- **PostgreSQL 16** with the **pgvector** extension
- **pip** or a Python package manager

**API Keys / Accounts Required:**

| Service | Required? | Purpose |
|---|---|---|
| HuggingFace | **Yes** | Llama 3.1 8B LLM inference |
| Voyage AI | Optional | Cloud embeddings (skip if using local) |
| Google Cloud | Optional | Sync knowledge base from Google Sheets |
| Twilio | Optional | Voice call integration |

---

## Clone & Environment Setup

```bash
git clone <your-repo-url>
cd rag_pipeline

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat     # Windows (CMD)
# venv\Scripts\Activate.ps1     # Windows (PowerShell)
```

---

## PostgreSQL + pgvector Setup

### macOS (Homebrew)

```bash
# Install PostgreSQL 16
brew install postgresql@16
brew services start postgresql@16

# Add to PATH if needed
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Install pgvector extension
brew install pgvector
```

### Ubuntu / Debian

```bash
# Install PostgreSQL 16
sudo apt update
sudo apt install -y postgresql-16 postgresql-client-16

# Install pgvector (build from source or use apt)
sudo apt install -y postgresql-16-pgvector

# Start the service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Create the Database

```bash
# Log into PostgreSQL as superuser
psql -U postgres

# Inside psql:
CREATE DATABASE aerosports_rag;
\c aerosports_rag
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

Verify the extension is available:

```bash
psql -U postgres -d aerosports_rag -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

Expected output: `vector`

---

## Environment Variables (.env)

Copy the template below and save it as `.env` in the project root. Fill in all required values.

```dotenv
# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aerosports_rag
DB_USER=postgres
DB_PASSWORD=your_postgres_password

# ─────────────────────────────────────────────
# EMBEDDINGS
# Choose: "local" (no API key needed) or "voyage" (requires VOYAGE_API_KEY)
# ─────────────────────────────────────────────
EMBEDDING_PROVIDER=local

# If EMBEDDING_PROVIDER=local, set the model name:
LOCAL_MODEL_NAME=BAAI/bge-m3
# Options: BAAI/bge-m3 (1024-dim), all-MiniLM-L6-v2 (384-dim)

# If EMBEDDING_PROVIDER=voyage, set your Voyage AI API key:
# VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ─────────────────────────────────────────────
# HUGGINGFACE (Required for LLM)
# Get your token at: https://huggingface.co/settings/tokens
# ─────────────────────────────────────────────
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LLM model (default: Llama 3.1 8B Instruct)
# HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
# HF_PROVIDER=                  # Leave blank for auto-routing

# ─────────────────────────────────────────────
# RAG SETTINGS
# ─────────────────────────────────────────────
DEFAULT_TOP_K=5
HYBRID_SEMANTIC_WEIGHT=0.7
HYBRID_KEYWORD_WEIGHT=0.3
MAX_CONTEXT_CHUNKS=5

# ─────────────────────────────────────────────
# CHATBOT SESSION SETTINGS
# ─────────────────────────────────────────────
MAX_CONVERSATION_TURNS=10
SESSION_TIMEOUT=1800            # seconds (30 minutes)

# ─────────────────────────────────────────────
# GOOGLE SHEETS SYNC (Optional)
# Required only if syncing knowledge base from Google Sheets
# ─────────────────────────────────────────────
# GOOGLE_SHEET_ID=your_google_sheet_id
# GOOGLE_CREDENTIALS_PATH=credentials/google_service_account.json
# CHANGE_LOG_SHEET=Change Log
# VERSION_CELL=M1
SYNC_ON_STARTUP=false           # Set to true to sync on every server start

# ─────────────────────────────────────────────
# TWILIO VOICE (Optional)
# Required only for voice call integration
# ─────────────────────────────────────────────
# TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TWILIO_AUTH_TOKEN=your_auth_token
# TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
# BASE_URL=https://your-ngrok-url.ngrok-free.app

# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────
CHATBOT_CORS_ORIGINS=*          # Restrict in production: https://yourdomain.com
```

> **Security Note:** Never commit `.env` to version control. It is already listed in `.gitignore`.

---

## Install Python Dependencies

With your virtual environment active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first run will download the embedding model (BAAI/bge-m3 is ~2GB) if using local embeddings. This is a one-time download cached in `~/.cache/huggingface/`.

---

## Initialize the Database

Run this once to create all required tables and indexes:

```bash
python setup_db.py
```

This creates:

- `knowledge_chunks` — Main vector store with pgvector embedding column
- `sync_state` — Tracks last synced Google Sheets version
- `sync_history` — Audit log of sync operations
- Indexes: GIN index on tags, IVFFlat index on embedding column

---

## Ingest the Knowledge Base

Load the 93 knowledge chunks from `data/aerosports_scb_knowledge_base.json` into PostgreSQL:

```bash
python ingest.py
```

This will:
1. Read and validate all chunks from the JSON file
2. Generate embeddings for each chunk (question + answer)
3. Upsert everything into the `knowledge_chunks` table

Options:

```bash
python ingest.py --dry-run          # Preview without writing to DB
python ingest.py --path /path/to/other.json   # Use a different knowledge base file
```

Expected output:

```
Loaded 93 chunks from data/aerosports_scb_knowledge_base.json
Generating embeddings...
100%|████████████████████| 93/93 [00:12<00:00]
Upserting 93 chunks into database...
Done. 93 chunks ingested successfully.
```

Verify ingestion:

```bash
psql -U postgres -d aerosports_rag -c "SELECT COUNT(*) FROM knowledge_chunks;"
```

Expected: `93`

---

## Google Sheets Sync (Optional)

This syncs changes from a Google Sheets "Change Log" tab into the vector database. Skip this section if you don't use Google Sheets for knowledge management.

### 1. Set Up a Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use an existing one)
3. Enable the **Google Sheets API** and **Google Drive API**
4. Go to **IAM & Admin → Service Accounts → Create Service Account**
5. Grant the role: **Editor** (or a custom role with Sheets read access)
6. Click the service account → **Keys → Add Key → Create New Key → JSON**
7. Download the JSON file

### 2. Configure Access

```bash
mkdir -p credentials
mv ~/Downloads/your-service-account-key.json credentials/google_service_account.json
```

Share your Google Sheet with the service account email (found inside the JSON file as `client_email`).

### 3. Set Environment Variables

In your `.env`:

```dotenv
GOOGLE_SHEET_ID=your_google_sheet_id_from_url
GOOGLE_CREDENTIALS_PATH=credentials/google_service_account.json
CHANGE_LOG_SHEET=Change Log
VERSION_CELL=M1
```

### 4. Run a Sync

```bash
python sync.py              # Normal sync (only new changes)
python sync.py --force      # Force re-sync all rows
python sync.py --dry-run    # Preview changes without writing
```

---

## Run the Chatbot Server

### Development

```bash
uvicorn chatbot.server:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
uvicorn chatbot.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Google Sheets sync on startup

```bash
SYNC_ON_STARTUP=true uvicorn chatbot.server:app --host 0.0.0.0 --port 8000
```

### Access the App

| URL | Description |
|---|---|
| `http://localhost:8000` | Test page with live chat widget |
| `http://localhost:8000/widget` | Embeddable chat widget (iframe) |
| `http://localhost:8000/docs` | Swagger UI (interactive API docs) |
| `http://localhost:8000/api/health` | Health check endpoint |

### Test the Search Directly

```bash
# Interactive mode
python search.py

# Single query
python search.py "how much does it cost to jump"
```

---

## Twilio Voice Integration (Optional)

This enables inbound phone calls to be handled by the AI chatbot.

### 1. Get Twilio Credentials

1. Sign up at [twilio.com](https://www.twilio.com)
2. From the Twilio Console, copy your **Account SID** and **Auth Token**
3. Purchase or use an existing phone number

### 2. Expose Your Local Server

Twilio needs a public URL to reach your server. Use [ngrok](https://ngrok.com/):

```bash
# Install ngrok
brew install ngrok                  # macOS
# or download from https://ngrok.com/download

# Start tunnel
ngrok http 8000
```

Copy the generated HTTPS URL (e.g., `https://2235-24-50-57-202.ngrok-free.app`).

### 3. Configure Environment Variables

```dotenv
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
BASE_URL=https://your-ngrok-url.ngrok-free.app
```

### 4. Configure Twilio Webhook

1. In the Twilio Console, go to **Phone Numbers → Manage → Active Numbers**
2. Click your phone number
3. Under **Voice Configuration → A call comes in**, set:
   - Webhook: `https://your-ngrok-url.ngrok-free.app/voice/inbound`
   - HTTP Method: `POST`
4. Save

### 5. Test

Call your Twilio phone number. The server handles the call via:
- `POST /voice/inbound` — Returns TwiML to start ConversationRelay
- `WebSocket /voice/ws` — Handles real-time audio and AI responses

---

## Project Structure

```
rag_pipeline/
├── .env                        # Environment variables (never commit)
├── requirements.txt            # Python dependencies
├── config.py                   # DB pool, embedding config, Google Sheets client
├── models.py                   # ChunkRecord, SearchResult, ChangeLogEntry dataclasses
├── embedding.py                # Embedding generation (Voyage AI or local)
├── search.py                   # Semantic and hybrid search over pgvector
├── ingest.py                   # One-time knowledge base ingestion
├── sync.py                     # Google Sheets → pgvector incremental sync
├── setup_db.py                 # Database schema initialization
│
├── chatbot/
│   ├── server.py               # FastAPI app and all route definitions
│   ├── chat_handler.py         # RAG → LLM pipeline, streaming orchestration
│   ├── llm.py                  # HuggingFace InferenceClient wrapper
│   ├── prompt_builder.py       # System prompt + RAG context + history builder
│   ├── conversation.py         # In-memory session store (TTL, turn limits)
│   ├── fallback.py             # CTA detection (booking links, phone, email)
│   ├── voice_handler.py        # Twilio voice call orchestration
│   └── static/
│       ├── index.html          # Test page
│       ├── widget.html         # Standalone embeddable widget page
│       ├── widget.js           # Chat widget (SSE streaming, session handling)
│       └── widget.css          # Widget styling
│
├── data/
│   └── aerosports_scb_knowledge_base.json   # 93 knowledge chunks
│
└── credentials/
    └── google_service_account.json          # Google Cloud credentials (never commit)
```

---

## API Reference

### POST /api/chat

Non-streaming chat request.

**Request:**
```json
{
  "message": "How much does it cost to jump?",
  "session_id": "optional-session-uuid"
}
```

**Response:**
```json
{
  "response": "A jump pass is $15 +tax per person for 1 hour...",
  "session_id": "uuid",
  "sources": ["scb_faq_140", "scb_faq_141"]
}
```

### GET /api/chat/stream

Streaming chat via Server-Sent Events (SSE).

**Query Parameters:**
- `message` — The user's message
- `session_id` — (Optional) Session UUID for conversation history

**Events:**
- `data: {"token": "word"}` — Streaming token
- `data: {"done": true, "sources": [...]}` — Stream complete

**Example (JavaScript):**
```javascript
const evtSource = new EventSource(
  `/api/chat/stream?message=How+much+to+jump&session_id=${sessionId}`
);
evtSource.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.token) appendToChat(data.token);
  if (data.done) evtSource.close();
};
```

### POST /api/chat/reset

Clear conversation history for a session.

**Request:** `{ "session_id": "uuid" }`

### GET /api/health

Returns server health status.

**Response:**
```json
{
  "status": "ok",
  "db": "connected",
  "hf_token": "configured"
}
```

---

## Troubleshooting

### `could not connect to server` (PostgreSQL)

```bash
# Check if PostgreSQL is running
brew services list | grep postgresql      # macOS
sudo systemctl status postgresql          # Linux

# Start if stopped
brew services start postgresql@16         # macOS
sudo systemctl start postgresql           # Linux
```

### `extension "vector" does not exist`

The pgvector extension must be installed and enabled per database:

```bash
# Reinstall pgvector (macOS)
brew reinstall pgvector

# Enable in the database
psql -U postgres -d aerosports_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Local embedding model is slow

By default, sentence-transformers runs on CPU. This is expected for BAAI/bge-m3. To speed up ingestion:

- Switch to Voyage AI: set `EMBEDDING_PROVIDER=voyage` and add your `VOYAGE_API_KEY`
- Use a smaller local model: `LOCAL_MODEL_NAME=all-MiniLM-L6-v2` (384-dim, much faster)

Note: if you change the embedding model after ingestion, you must re-ingest all chunks since embedding dimensions will differ.

### `401 Unauthorized` from HuggingFace

Your `HF_TOKEN` is missing or invalid. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and ensure the model `meta-llama/Llama-3.1-8B-Instruct` is accessible (requires accepting the model's license agreement on HuggingFace).

### Twilio webhook not reaching server

- Ensure ngrok is running and `BASE_URL` in `.env` matches the current ngrok URL (ngrok URLs change on restart unless you have a paid plan)
- Verify the webhook URL in Twilio Console is set to `{BASE_URL}/voice/inbound`
- Check server logs for incoming requests

### Session history lost after server restart

By design, conversation history is stored in-memory and does not persist across restarts. For persistent sessions in production, replace the in-memory store in [chatbot/conversation.py](chatbot/conversation.py) with a Redis-backed store.

### `ModuleNotFoundError` on import

Ensure your virtual environment is activated:

```bash
source venv/bin/activate
python -c "import fastapi; print('OK')"
```

---

## Quick Start Checklist

```
[ ] Python 3.11+ installed
[ ] PostgreSQL 16 running with pgvector extension
[ ] Database "aerosports_rag" created
[ ] .env file created and filled in
[ ] HF_TOKEN set (required)
[ ] Virtual environment created and activated
[ ] pip install -r requirements.txt completed
[ ] python setup_db.py run successfully
[ ] python ingest.py run successfully (93 chunks ingested)
[ ] uvicorn chatbot.server:app --port 8000 running
[ ] http://localhost:8000 loads the test page
[ ] Chat widget responds to a test message
```
