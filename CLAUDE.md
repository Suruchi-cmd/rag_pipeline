# CLAUDE.md — AeroSports Scarborough RAG Project

## What This Is
RAG pipeline for AeroSports Scarborough's customer chatbot. Ingests structured JSON into PostgreSQL/pgvector, syncs changes from Google Sheets.

## Tech Stack
Python 3.11+, PostgreSQL 16 + pgvector, Voyage AI or sentence-transformers, gspread, psycopg2-binary

## Structure
```
aerosports-rag/
├── CLAUDE.md
├── .env
├── requirements.txt
├── config.py, models.py, embedding.py
├── setup_db.py, ingest.py, search.py, sync.py
├── data/aerosports_scb_knowledge_base.json
└── credentials/google_service_account.json
```

## Coding Conventions
- Type hints everywhere
- Dataclasses for ChunkRecord, SearchResult, ChangeLogEntry
- Context managers for DB ops
- Batch embeddings (128/batch), tqdm progress bars
- Logging module (not print), python-dotenv for secrets
- Transaction per sync change with rollback

## Database
- DB: aerosports_rag
- Tables: knowledge_chunks (93 rows), sync_state, sync_history
- Embedding: vector(1024) Voyage / vector(384) local
- Indexes: btree category/subcategory, GIN tags, ivfflat embedding

## Sync Design (simplified)
- Google Sheet Change Log has `chunk_id` column — no mapping file needed
- Version cell M1 checked on startup → skip if unchanged
- Each change row has chunk_id, change_type, old/new values
- FAQ updates: read new Q+A directly from FAQs sheet
- Pricing updates: read all rows with that chunk_id, rebuild text

## Chunk IDs
- Data chunks: scb_contact_001 through scb_passes_016
- FAQs: scb_faq_100–191 (100s General, 110s Go Kart, 120s Group, 140s Birthday, 170s Camp)
- Future locations: oak_*, ldn_*

## Business Context
- Single location: AeroSports Scarborough, Ontario
- Chatbot: professional & friendly
- Fallbacks: booking links, phone (289-454-5555), email, upsell
- All prices +tax