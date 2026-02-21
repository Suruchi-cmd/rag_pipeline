"""
AeroBot FastAPI server.

Endpoints
---------
POST /api/chat          — non-streaming, full response JSON
GET  /api/chat/stream   — SSE streaming (EventSource-compatible)
POST /api/chat/reset    — clear a session's conversation history
GET  /api/health        — health check (DB + env)

Static
------
GET /          → chatbot/static/index.html  (test page)
GET /widget    → chatbot/static/widget.html (embeddable iframe page)
GET /static/*  → chatbot/static/           (CSS / JS assets)

Run
---
    uvicorn chatbot.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

# Load .env before any other local imports so all env vars are available.
load_dotenv()

# Ensure repo root is importable (config.py, search.py, etc.).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402  (loads DB pool, embedding config, etc.)
from fastapi import FastAPI, Form, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from chatbot.chat_handler import handle_message  # noqa: E402
from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.voice_handler import handle_voice_message  # noqa: E402
from twilio.twiml.voice_response import Gather, VoiceResponse  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


async def _session_cleanup_loop() -> None:
    """Purge expired sessions every 5 minutes."""
    while True:
        await asyncio.sleep(5 * 60)
        n = conversation_store.cleanup_expired()
        if n:
            logger.info("Cleaned up %d expired session(s)", n)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AeroBot server starting…")

    # Verify database connectivity
    try:
        pool = config.get_db_pool()
        conn = pool.getconn()
        pool.putconn(conn)
        logger.info("Database connection verified")
    except Exception as exc:
        logger.warning("Database connection failed on startup: %s", exc)

    # Optional: run Google Sheets sync on startup
    if os.environ.get("SYNC_ON_STARTUP", "").lower() == "true":
        try:
            import sync as sync_module  # noqa: PLC0415

            logger.info("Running knowledge-base sync…")
            await asyncio.to_thread(sync_module.sync)
            logger.info("Sync complete")
        except Exception as exc:
            logger.error("Sync failed: %s", exc)

    cleanup_task = asyncio.create_task(_session_cleanup_loop())

    yield  # ← server runs here

    cleanup_task.cancel()
    config.close_db_pool()
    logger.info("AeroBot server stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AeroBot API",
    description="Customer chatbot for AeroSports Scarborough",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restrict origins in production via CHATBOT_CORS_ORIGINS env var.
_cors_origins_raw = os.environ.get("CHATBOT_CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Static files (CSS / JS)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Routes — static pages
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/widget", include_in_schema=False)
async def widget():
    return FileResponse(os.path.join(_STATIC_DIR, "widget.html"))


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def chat_endpoint(body: ChatRequest):
    """
    Non-streaming chat endpoint.

    Returns the full response once the LLM is done.
    Useful for testing and non-streaming clients.
    """
    session_id = body.session_id or str(uuid.uuid4())
    full_response = ""
    sources: list = []

    async for item in handle_message(session_id, body.message):
        if item["type"] == "token":
            full_response += item["content"]
        elif item["type"] == "done":
            sources = item["sources"]

    return {
        "response": full_response,
        "session_id": session_id,
        "sources": sources,
    }


@app.get("/api/chat/stream")
async def chat_stream(
    message: str = Query(..., max_length=500, description="User message"),
    session_id: Optional[str] = Query(None, description="Session ID (omit to start new session)"),
):
    """
    SSE streaming chat endpoint.

    Each SSE event carries JSON:
        {"token": "<text>", "done": false, "session_id": "<id>"}

    The final event has done=true and includes sources:
        {"token": "", "done": true, "session_id": "<id>", "sources": [...]}
    """
    sid = session_id or str(uuid.uuid4())

    async def _generator():
        try:
            async for item in handle_message(sid, message):
                if item["type"] == "token":
                    payload = json.dumps(
                        {"token": item["content"], "done": False, "session_id": sid}
                    )
                    yield {"data": payload}
                elif item["type"] == "done":
                    payload = json.dumps(
                        {
                            "token": "",
                            "done": True,
                            "session_id": sid,
                            "sources": item["sources"],
                        }
                    )
                    yield {"data": payload}
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            payload = json.dumps(
                {"token": "", "done": True, "session_id": sid, "error": str(exc), "sources": []}
            )
            yield {"data": payload}

    return EventSourceResponse(_generator())


@app.post("/api/chat/reset")
async def reset_session(body: ResetRequest):
    """Clear conversation history for the given session."""
    conversation_store.clear(body.session_id)
    return {"status": "ok", "session_id": body.session_id}


@app.get("/api/health")
async def health():
    """Health check — reports DB connectivity and whether HF_TOKEN is configured."""
    db_ok = False
    try:
        pool = config.get_db_pool()
        conn = pool.getconn()
        pool.putconn(conn)
        db_ok = True
    except Exception:
        pass

    hf_ok = bool(os.environ.get("HF_TOKEN"))

    status = "ok" if (db_ok and hf_ok) else "degraded"
    return JSONResponse(
        content={"status": status, "db": db_ok, "hf_token_configured": hf_ok},
        status_code=200 if status == "ok" else 503,
    )


# ---------------------------------------------------------------------------
# Voice routes (Twilio)
# ---------------------------------------------------------------------------

_BASE_URL = os.environ.get("BASE_URL", "")

_WELCOME_MESSAGE = (
    "Hi! You've reached AeroBot at AeroSports Scarborough. "
    "How can I help you today?"
)
_VOICE_FALLBACK_MESSAGE = (
    "I didn't catch that. "
    "You can also reach us directly at 289-454-5555."
)


def _twiml_response(text: str, action_url: str) -> Response:
    """Speak *text* then open a <Gather> to capture the next speech input."""
    vr = VoiceResponse()
    gather = Gather(input="speech", action=action_url, timeout=5, speechTimeout="auto")
    gather.say(text)
    vr.append(gather)
    # If the caller says nothing, Gather falls through to this fallback Say.
    vr.say(_VOICE_FALLBACK_MESSAGE)
    return Response(content=str(vr), media_type="text/xml")


@app.post("/voice/inbound", tags=["voice"])
async def voice_inbound(
    CallSid: str = Form(...),
    SpeechResult: str = Form(default=""),
    From: str = Form(default=""),
):
    """
    Twilio webhook for inbound voice calls.

    Configure in Twilio Console → Phone Numbers → your number →
    "A call comes in" → Webhook → POST → {BASE_URL}/voice/inbound
    """
    action_url = f"{_BASE_URL}/voice/inbound"

    if not SpeechResult.strip():
        # First POST from Twilio (no speech yet) — greet the caller.
        logger.info("New inbound call: %s from %s", CallSid, From)
        return _twiml_response(_WELCOME_MESSAGE, action_url)

    # Subsequent turns — route speech through the RAG + LLM pipeline.
    logger.info("[%s] Caller said: %s", CallSid, SpeechResult)
    reply = await handle_voice_message(call_sid=CallSid, user_text=SpeechResult.strip())
    logger.info("[%s] AeroBot replied: %s", CallSid, reply)
    return _twiml_response(reply, action_url)
