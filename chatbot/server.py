"""
AeroBot Twilio voice-call server.

Endpoints
---------
POST /voice/inbound  — Twilio webhook; returns ConversationRelay TwiML
POST /voice/action   — Twilio session-end webhook
WS   /voice/ws       — ConversationRelay WebSocket (ASR ↔ LLM ↔ TTS)
GET  /api/health     — health check

The RAG retrieval lives in the separate `core/rag/` service at $RAG_API_URL.

Run
---
    uvicorn chatbot.server:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before any other local imports so all env vars are available.
load_dotenv()

# Ensure repo root is importable for src.utils.* used by voice_handler.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import db  # noqa: E402

from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from chatbot.conversation import conversation_store  # noqa: E402
from chatbot.voice_handler import (  # noqa: E402
    build_end_decision_from_definite,
    check_booking_capture_trigger,
    check_end_keywords,
    classify_turn_for_end,
    clean_for_tts,
    close_session_logger,
    get_session_logger,
    prepare_voice_stream,
    stream_voice_tokens,
)
from chatbot.llm import _FALLBACK_MSG  # noqa: E402
from twilio.twiml.voice_response import VoiceResponse  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.environ.get("SESSION_LOG_DIR", "logs"),
)


class _CallSidFilter(logging.Filter):
    """Only pass log records whose formatted message contains this call_sid."""

    def __init__(self, call_sid: str) -> None:
        super().__init__()
        self._call_sid = call_sid

    def filter(self, record: logging.LogRecord) -> bool:
        return self._call_sid in record.getMessage()


def _open_session_log(call_sid: str) -> logging.FileHandler:
    os.makedirs(_LOG_DIR, exist_ok=True)
    path = os.path.join(_LOG_DIR, f"{call_sid}.log")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    handler.addFilter(_CallSidFilter(call_sid))
    logging.getLogger().addHandler(handler)
    logger.info("[%s] Session log opened → %s", call_sid, path)
    return handler


def _close_session_log(call_sid: str, handler: logging.FileHandler) -> None:
    logger.info("[%s] Session log closed", call_sid)
    logging.getLogger().removeHandler(handler)
    handler.close()


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
    logger.info("AeroBot voice server starting…")
    try:
        db.init_db()
        logger.info("SQLite call log DB initialized")
    except Exception as exc:
        logger.warning("db.init_db() failed: %s", exc)

    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    yield
    cleanup_task.cancel()
    logger.info("AeroBot voice server stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AeroBot Voice API",
    description="Twilio voice-call backend for AeroSports Scarborough",
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


@app.get("/api/health")
async def health():
    """Liveness check — used by uptime monitors and Twilio status checks."""
    return JSONResponse(content={"status": "ok"}, status_code=200)


# ---------------------------------------------------------------------------
# Voice routes (Twilio)
# ---------------------------------------------------------------------------

_BASE_URL = os.environ.get("BASE_URL", "").rstrip(
    "/"
)  # e.g. https://abc.ngrok-free.app


@app.post("/voice/inbound", tags=["voice"])
@app.post("/voice/inbound/", include_in_schema=False)
async def voice_inbound(CallSid: str = Form(...), From: str = Form(default="")):
    """
    Twilio webhook for inbound voice calls — returns ConversationRelay TwiML.

    Configure in Twilio Console → Phone Numbers → your number →
    "A call comes in" → Webhook → POST → {BASE_URL}/voice/inbound
    """
    logger.info("New inbound call: %s from %s", CallSid, From)
    ws_host = _BASE_URL.replace("https://", "").replace("http://", "")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect action="{_BASE_URL}/voice/action">
    <ConversationRelay url="wss://{ws_host}/voice/ws"
                    language="en-CA"
                    transcriptionProvider="deepgram"
                    ttsProvider="ElevenLabs"
                    voice="uYXf8XasLslADfZ2MB4u"
                       welcomeGreeting="Thank you for calling AeroSports Scarborough, this is Maya. How can I help you?"
                       dtmfDetection="true"
                       interruptByDtmf="false"
                       interruptSensitivity="medium">
        <Transcription hints="AeroSports, Aerosports, Aero Sports, Scarborough, Birchmount, aerosportsparks, Aero Slam, birthday, Aero Camp, Ninja Maze, Ninja Warrior, 360 Bicycle, Donut Slide, Carpet Slide, go kart, go karting, dual kart, Mini Track, Main Track, Premium Jump Pass, VIP Jump Pass, All Day Pass, 30 Day Pass, 90 Day Pass, Annual Pass, Premium Party, VIP Party, Ultimate Party, Toddler Time, PA Day Camp, March Break Camp, availability, Special Needs Program, Birds Eye Party Arena,jump tower, slam basketball, dodgeball, rock walls, waiver" />
    </ConversationRelay>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.post("/voice/action", tags=["voice"])
async def voice_action():
    """Called by Twilio when the ConversationRelay session ends."""
    vr = VoiceResponse()
    vr.hangup()
    return Response(content=str(vr), media_type="text/xml")


# ---------------------------------------------------------------------------
# Voice WebSocket session state
# ---------------------------------------------------------------------------
# Each active call gets an entry keyed by callSid.
#   "conversation" — managed by conversation_store (shared w/ web chat)
#   "current_task" — the asyncio.Task running the LLM stream (or None)
#
# We track current_task here (not in conversation_store) because it is
# ephemeral async state that only matters for the life of the WebSocket.
# ---------------------------------------------------------------------------

_voice_sessions: dict[str, dict] = {}

# The welcome greeting is spoken by Twilio TTS when the call connects.
# Twilio's ASR is already active at that point, so it can pick up the bot's
# own voice and send it back as a PROMPT.  We skip any prompt that looks like
# an echo of the greeting so it doesn't trigger an LLM response.
_WELCOME_GREETING = (
    "Thank you for calling AeroSports Scarborough, this is Maya. How can I help you?"
)
_WELCOME_ECHO_PREFIX = "thank you for calling aerosports"


async def _send_canned_to_twilio(ws: WebSocket, text: str) -> None:
    """
    Send a canned (non-LLM) text response to Twilio ConversationRelay as a single turn.

    Used by the booking-change capture state machine where we bypass the LLM
    entirely and speak deterministic canned responses instead.
    """
    await ws.send_text(json.dumps({"type": "text", "token": text, "last": False}))
    await ws.send_text(json.dumps({"type": "text", "token": "", "last": True}))


def _log_message(call_sid: str, db_call_id: int | None, role: str, content: str) -> None:
    """db.log_message wrapped in try/except so a logging failure never breaks a turn."""
    if db_call_id is None:
        return
    try:
        db.log_message(db_call_id, role, content)
    except Exception as exc:
        logger.error("[%s] db.log_message (%s) failed: %s", call_sid, role, exc)


async def _run_capture_step(
    websocket: WebSocket,
    call_sid: str,
    session: dict,
    user_text: str,
    next_mode: str,
    canned: str,
    log_msg: str,
    capture_field: str | None = None,
) -> None:
    """
    Common booking-capture transition: optionally store the field, advance the
    state machine, log, send the canned reply, and persist both turns.
    """
    if capture_field is not None:
        session["capture_data"][capture_field] = user_text
    session["capture_mode"] = next_mode
    db_call_id = session.get("db_call_id")
    logger.info("[%s] %s", call_sid, log_msg)
    await _send_canned_to_twilio(websocket, canned)
    conversation_store.add(call_sid, "user", user_text)
    conversation_store.add(call_sid, "assistant", canned)
    _log_message(call_sid, db_call_id, "assistant", canned)


async def _stream_llm_to_twilio(
    ws: WebSocket,
    call_sid: str,
    messages: list[dict],
    segments: list[str],
    t_silence_end: float = 0.0,
) -> None:
    """
    Stream LLM tokens to Twilio via the WebSocket.

    Sends each token as a Twilio ConversationRelay text message:
        {"type": "text", "token": "<text>", "last": false}

    When the stream finishes, sends the end-of-turn marker:
        {"type": "text", "token": "", "last": true}

    Appends each token to the shared `segments` list so the caller can
    access whatever was generated even if the task is cancelled mid-stream.

    Raises asyncio.CancelledError if the task is cancelled mid-stream.
    """
    try:
        first_token = True
        async for token in stream_voice_tokens(messages):
            if first_token:
                first_token = False
                if t_silence_end:
                    ttfr_ms = (time.perf_counter() - t_silence_end) * 1000
                    logger.info(
                        "[%s] LATENCY silence_to_first_reply=%.0fms", call_sid, ttfr_ms
                    )
            # Each token is sent immediately so Twilio's TTS can start
            # speaking while the LLM is still generating.
            await ws.send_text(
                json.dumps({"type": "text", "token": token, "last": False})
            )
            segments.append(token)

        # Stream completed normally — send end-of-turn marker so Twilio
        # knows the assistant is done speaking.
        await ws.send_text(json.dumps({"type": "text", "token": "", "last": True}))

    except asyncio.CancelledError:
        # Interrupt arrived — stop streaming, do NOT send "last": true.
        # segments already contains whatever was sent before cancellation.
        logger.info("[%s] LLM stream cancelled (user interrupted)", call_sid)
        raise


@app.websocket("/voice/ws")
async def voice_ws(websocket: WebSocket):
    """
    WebSocket handler for Twilio ConversationRelay.

    Receives transcribed speech from Twilio, streams LLM tokens back in
    real time. Twilio handles ASR (speech-to-text) and TTS (text-to-speech).

    Token streaming flow:
    1. User speaks → Twilio transcribes → sends {"type": "prompt"} over WS
    2. We run RAG search, build the prompt, then stream Ollama tokens
    3. Each token is sent as {"type": "text", "token": "...", "last": false}
    4. After the last token: {"type": "text", "token": "", "last": true}

    Interrupt flow:
    1. User starts speaking mid-response → Twilio sends {"type": "interrupt"}
    2. We cancel the LLM streaming task immediately
    3. Conversation history records only what Twilio actually spoke aloud
       (using utteranceUntilInterrupt from the interrupt message)
    """
    await websocket.accept()
    call_sid: str | None = None
    _session_log_handler: logging.FileHandler | None = None

    try:
        async for raw in websocket.iter_text():
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ----------------------------------------------------------
            # SETUP — Twilio sends this once when the WS connects
            # ----------------------------------------------------------
            if msg_type == "setup":
                call_sid = msg.get("callSid", str(uuid.uuid4()))
                caller_from = msg.get("from", "unknown")
                try:
                    db_call_id = db.start_call(caller_from)
                except Exception as exc:
                    logger.error("[%s] db.start_call failed: %s", call_sid, exc)
                    db_call_id = None
                _voice_sessions[call_sid] = {
                    "current_task": None,
                    "db_call_id": db_call_id,
                    "phone_number": caller_from,
                    "capture_mode": "none",  # "none" | "name" | "phone" | "details" | "done"
                    "capture_data": {"name": "", "phone": "", "details": ""},
                    "capture_triggered_on": "",
                }
                _session_log_handler = _open_session_log(call_sid)
                logger.info(
                    "[%s] ConversationRelay connected from %s",
                    call_sid,
                    msg.get("from"),
                )

            # ----------------------------------------------------------
            # PROMPT — user finished speaking, we generate a response
            # ----------------------------------------------------------
            elif msg_type == "prompt":
                # t_silence_end marks when Twilio detected customer silence
                # and sent us the transcribed prompt — start of our latency clock.
                t_silence_end = time.perf_counter()

                user_text = msg.get("voicePrompt", "").strip()
                if not user_text or not call_sid:
                    continue

                # Skip prompts that are Twilio ASR echoing our own welcome
                # greeting back at us (ASR is active while TTS plays).
                if user_text.lower().startswith(_WELCOME_ECHO_PREFIX):
                    logger.info(
                        "[%s] Skipping echo of welcome greeting: %s",
                        call_sid,
                        user_text[:80],
                    )
                    continue

                logger.info("[%s] User said: %s", call_sid, user_text)
                session = _voice_sessions.get(call_sid, {})

                # Cancel any in-flight generation from a previous turn
                prev_task = session.get("current_task")
                if prev_task and not prev_task.done():
                    prev_task.cancel()

                # ------------------------------------------------------
                # Booking-change capture state machine
                # ------------------------------------------------------
                # If capture is already in progress OR the user's message
                # triggers it, we bypass the LLM entirely and run the
                # deterministic state machine instead.
                # ------------------------------------------------------
                capture_mode = session.get("capture_mode", "none")
                db_call_id = session.get("db_call_id")

                # Log the user turn to the DB no matter what path we take.
                _log_message(call_sid, db_call_id, "user", user_text)

                # --- New capture trigger: enter the state machine ---
                if capture_mode == "none" and check_booking_capture_trigger(user_text):
                    session["capture_triggered_on"] = user_text
                    await _run_capture_step(
                        websocket, call_sid, session, user_text,
                        next_mode="name",
                        canned="Sure, I can take down some details so my manager can give you a call back. Can I get your name please?",
                        log_msg="Booking capture TRIGGERED. State → name",
                    )
                    continue

                # --- Advance the state machine ---
                if capture_mode == "name":
                    await _run_capture_step(
                        websocket, call_sid, session, user_text,
                        next_mode="phone",
                        canned="Thanks. And what's the best phone number to reach you at?",
                        log_msg="Captured name. State → phone",
                        capture_field="name",
                    )
                    continue

                if capture_mode == "phone":
                    await _run_capture_step(
                        websocket, call_sid, session, user_text,
                        next_mode="details",
                        canned="Got it. And what would you like to change about your booking?",
                        log_msg="Captured phone. State → details",
                        capture_field="phone",
                    )
                    continue

                if capture_mode == "details":
                    # Persist the booking-change row *before* the canned reply
                    # so save_booking_change also flags the call with needs_human=1
                    # while the caller is still listening.
                    session["capture_data"]["details"] = user_text
                    if db_call_id is not None:
                        cd = session["capture_data"]
                        try:
                            db.save_booking_change(
                                db_call_id, cd["name"], cd["phone"], cd["details"]
                            )
                            logger.info("[%s] Booking change saved to DB", call_sid)
                        except Exception as exc:
                            logger.error(
                                "[%s] db.save_booking_change failed: %s", call_sid, exc
                            )
                    await _run_capture_step(
                        websocket, call_sid, session, user_text,
                        next_mode="done",
                        canned="Perfect, I've got all that. My manager will give you a call back as soon as possible. Is there anything else I can help you with today?",
                        log_msg="Captured details. State → done. Handing back to LLM.",
                        # capture_field intentionally None — already set above
                    )
                    continue

                # ------------------------------------------------------
                # End of capture state machine. If we reach here, capture is
                # either "none" (normal conversation) or "done" (post-capture,
                # normal conversation resumes). Fall through to LLM streaming.
                # ------------------------------------------------------

                # Phase 1: RAG search + prompt building (not cancellable —
                # it's fast and we need the result before streaming).
                try:
                    messages = await prepare_voice_stream(call_sid, user_text)
                except Exception as exc:
                    logger.error("[%s] prepare_voice_stream failed: %s", call_sid, exc)
                    await websocket.send_text(
                        json.dumps(
                            {"type": "text", "token": _FALLBACK_MSG, "last": True}
                        )
                    )
                    conversation_store.add(call_sid, "assistant", _FALLBACK_MSG)
                    continue

                # Phase 2: Stream LLM tokens to Twilio inside an asyncio.Task
                # so it can be cancelled if the user interrupts.
                # The segments list is shared so we can access partial output
                # even if the task is cancelled mid-stream.
                segments: list[str] = []

                async def _run_stream(
                    ws: WebSocket,
                    sid: str,
                    msgs: list[dict],
                    segs: list[str],
                    t_silence: float,
                ):
                    try:
                        await _stream_llm_to_twilio(ws, sid, msgs, segs, t_silence)
                        # Normal completion — clean for TTS and save
                        raw_joined = "".join(segs)
                        full_reply = clean_for_tts(raw_joined)
                        if full_reply.strip():
                            conversation_store.add(sid, "assistant", full_reply)
                            logger.info(
                                "[%s] Assistant (full): %s", sid, full_reply[:200]
                            )
                        # Steps 5 & 6 — log LLM output and final TTS text
                        _pl = get_session_logger(sid)
                        if _pl is not None:
                            _pl.log_llm_response(raw_joined)
                            _pl.log_final_response(full_reply)

                        # --- DB log assistant turn + end-call detection ---
                        _sess = _voice_sessions.get(sid, {})
                        _db_call_id = _sess.get("db_call_id")

                        if full_reply.strip():
                            _log_message(sid, _db_call_id, "assistant", full_reply)

                        # Keyword pre-filter avoids the classifier LLM hop on
                        # obvious goodbye turns; "maybe" hits run the classifier.
                        _kw_result = check_end_keywords(user_text)
                        _end_decision = None
                        if _kw_result == "definite":
                            _end_decision = build_end_decision_from_definite(
                                user_text, full_reply
                            )
                            logger.info("[%s] End-call keyword DEFINITE match", sid)
                        elif _kw_result == "maybe":
                            logger.info(
                                "[%s] End-call keyword MAYBE — running classifier", sid
                            )
                            _end_decision = await classify_turn_for_end(
                                user_text, full_reply
                            )

                        if _end_decision is not None and _db_call_id is not None:
                            _summary = _end_decision["summary"]
                            _needs_human = _end_decision["needs_human"]
                            _flag_reason = _end_decision["flag_reason"] or None

                            try:
                                db.end_call(
                                    _db_call_id, _summary, _needs_human, _flag_reason
                                )
                            except Exception as exc:
                                logger.error("[%s] db.end_call failed: %s", sid, exc)

                            if _needs_human:
                                # TODO: hook up email when a `mailer` module exists.
                                logger.info(
                                    "[%s] needs_human=True — flagged in DB; email "
                                    "alert skipped (mailer module not wired up)", sid
                                )

                            try:
                                _handoff = json.dumps(
                                    {
                                        "reasonCode": "bot-ended-call",
                                        "reason": _summary,
                                        "needs_human": _needs_human,
                                    }
                                )
                                await ws.send_text(
                                    json.dumps(
                                        {"type": "end", "handoffData": _handoff}
                                    )
                                )
                                logger.info(
                                    "[%s] Sent end-session message to Twilio", sid
                                )
                            except Exception as exc:
                                logger.error(
                                    "[%s] Failed to send end message: %s", sid, exc
                                )

                    except asyncio.CancelledError:
                        # Save whatever was generated before cancellation so
                        # history doesn't have an orphan user turn.
                        partial = clean_for_tts("".join(segs))
                        if partial.strip():
                            conversation_store.add(sid, "assistant", partial)
                            logger.info(
                                "[%s] Assistant (partial/cancelled): %s",
                                sid,
                                partial[:200],
                            )
                            _db_call_id = _voice_sessions.get(sid, {}).get("db_call_id")
                            _log_message(
                                sid, _db_call_id, "assistant", partial + " [INTERRUPTED]"
                            )
                        _pl = get_session_logger(sid)
                        if _pl is not None:
                            _pl.log_llm_response("".join(segs) + " [INTERRUPTED]")
                            _pl.log_final_response(partial + " [INTERRUPTED]")
                    except Exception as exc:
                        logger.error("[%s] Stream error: %s", sid, exc)
                        _pl = get_session_logger(sid)
                        if _pl is not None:
                            _pl.log_error(f"Stream error: {exc}", exc)
                        try:
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "text",
                                        "token": _FALLBACK_MSG,
                                        "last": True,
                                    }
                                )
                            )
                            conversation_store.add(sid, "assistant", _FALLBACK_MSG)
                        except Exception:
                            pass

                task = asyncio.create_task(
                    _run_stream(websocket, call_sid, messages, segments, t_silence_end)
                )
                session["current_task"] = task

            # ----------------------------------------------------------
            # INTERRUPT — user started speaking, cancel LLM generation
            # ----------------------------------------------------------
            elif msg_type == "interrupt":
                spoken_fragment = msg.get("utteranceUntilInterrupt", "")
                logger.info(
                    "[%s] User interrupted. Spoken so far: %s",
                    call_sid,
                    spoken_fragment[:200],
                )

                if call_sid:
                    session = _voice_sessions.get(call_sid, {})
                    current_task = session.get("current_task")

                    # Cancel the in-flight LLM streaming task
                    if current_task and not current_task.done():
                        current_task.cancel()
                        # Wait for cancellation to complete cleanly
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass

                    # Replace whatever the CancelledError handler saved
                    # with what Twilio actually spoke aloud.  This keeps
                    # conversation history accurate — the LLM sees what
                    # the user actually heard, not the full generation.
                    if spoken_fragment.strip():
                        conversation_store.replace_last_assistant(
                            call_sid, spoken_fragment.strip()
                        )
                        logger.info(
                            "[%s] Assistant (truncated to spoken): %s",
                            call_sid,
                            spoken_fragment.strip()[:200],
                        )

            # ----------------------------------------------------------
            # DTMF — keypad press
            # ----------------------------------------------------------
            elif msg_type == "dtmf":
                logger.info("[%s] DTMF: %s", call_sid, msg.get("digit"))

    except WebSocketDisconnect:
        logger.info("[%s] ConversationRelay disconnected", call_sid)
    except Exception as exc:
        logger.error("[%s] ConversationRelay error: %s", call_sid, exc)
    finally:
        # Clean up session state and conversation history
        if call_sid:
            session = _voice_sessions.pop(call_sid, {})
            current_task = session.get("current_task")
            if current_task and not current_task.done():
                current_task.cancel()
            _db_call_id = session.get("db_call_id")
            if _db_call_id is not None:
                try:
                    _row = db.get_call(_db_call_id)
                    if _row and not _row.get("ended_at"):
                        db.end_call(_db_call_id, "Call disconnected", False, None)
                        logger.info("[%s] Finalized abandoned call in DB", call_sid)
                except Exception as exc:
                    logger.error("[%s] Cleanup db.end_call failed: %s", call_sid, exc)
            conversation_store.clear(call_sid)
        if _session_log_handler and call_sid:
            _close_session_log(call_sid, _session_log_handler)
        if call_sid:
            close_session_logger(call_sid)
