# CLAUDE.md — AeroSports Scarborough RAG + Chatbot

## What This Is
Monorepo with two components:
1. **rag/** — RAG pipeline: pgvector knowledge base (93 chunks), Google Sheets sync
2. **chatbot/** — Customer-facing chatbot: FastAPI + Llama (HuggingFace) + streaming widget

## Tech Stack
- Python 3.11+, PostgreSQL 16 + pgvector
- FastAPI + uvicorn + SSE for chatbot server
- HuggingFace Inference Providers API (Llama 3.1 8B Instruct) via huggingface_hub
- Voyage AI embeddings (1024 dim) or sentence-transformers (384 dim)
- gspread + google-auth for Google Sheets sync
- Vanilla HTML/CSS/JS for embeddable chat widget

## Structure
```
aerosports-rag/
├── CLAUDE.md
├── .env
├── requirements.txt
├── rag/                          # Existing RAG pipeline
│   ├── __init__.py
│   ├── config.py, models.py, embedding.py
│   ├── setup_db.py, ingest.py, search.py, sync.py
├── chatbot/                      # NEW: chatbot server + widget
│   ├── __init__.py
│   ├── server.py                 # FastAPI app (POST /api/chat, SSE /api/chat/stream)
│   ├── llm.py                    # HuggingFace InferenceClient wrapper + streaming
│   ├── prompt_builder.py         # System prompt + RAG context injection
│   ├── chat_handler.py           # Orchestrator: search → prompt → LLM → stream
│   ├── conversation.py           # In-memory session store (30 min TTL, 10 turns max)
│   ├── fallback.py               # CTA detection (booking links, phone, email, upsells)
│   └── static/                   # Frontend widget
│       ├── index.html, widget.html, widget.js, widget.css
└── data/
    └── aerosports_scb_knowledge_base.json
```

## Coding Conventions
- Type hints everywhere, dataclasses for models
- Context managers for DB, async where possible in chatbot
- Logging module (not print)
- python-dotenv for all secrets
- chatbot/ imports from rag/ (from rag.search import semantic_search)

## Chatbot Flow
1. User sends message → POST /api/chat or SSE /api/chat/stream
2. Embed query → semantic_search + hybrid_search pgvector (top 5)
3. Build prompt: system prompt + RAG context + conversation history + user message
4. Stream Llama response via HuggingFace InferenceClient
5. Detect fallback CTAs (booking link, phone, email)
6. Store in conversation history (in-memory, per session_id)

## LLM Config
- Model: meta-llama/Llama-3.1-8B-Instruct via HuggingFace Inference Providers
- Temperature: 0.3 (factual), max_tokens: 1024
- Streaming: always (SSE for frontend, generator for API)
- Retry: 3 attempts with exponential backoff on 429/503

## Brand
- Primary: #F00C74 (neon pink), Accent: #39FF14 (neon green)
- Chatbot name: AeroBot
- Tone: professional, friendly, concise (2-4 sentences default)
- All prices +tax

## Business Context
- Single location: AeroSports Scarborough, Ontario
- Phone: 289-454-5555, Email: events.scb@aerosportsparks.ca
- Hours: Sun-Thur 10am-8pm, Fri-Sat 10am-10pm
- Website: https://www.aerosportsparks.ca

## Running
```bash
# RAG
python -m rag.setup_db && python -m rag.ingest && python -m rag.search

# Chatbot
uvicorn chatbot.server:app --host 0.0.0.0 --port 8000 --reload

# Both with sync on startup
SYNC_ON_STARTUP=true uvicorn chatbot.server:app --port 8000
```


## To run
```bash
# Install new deps
pip install fastapi uvicorn[standard] sse-starlette huggingface-hub pydantic

# Start server
uvicorn chatbot.server:app --host 0.0.0.0 --port 8000 --reload

# Open in browser
open http://localhost:8000
```