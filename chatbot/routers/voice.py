"""
Twilio ConversationRelay voice routes.

Endpoints
---------
POST /voice/inbound  — Twilio webhook; returns ConversationRelay TwiML
POST /voice/action   — Twilio session-end webhook
WS   /voice/ws       — ConversationRelay WebSocket (ASR ↔ LLM ↔ TTS)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid

from fastapi import APIRouter, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlmodel import Session
from twilio.twiml.voice_response import VoiceResponse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chatbot.config import settings
from chatbot.conversation import conversation_store
from chatbot.llm import _FALLBACK_MSG, release_session_client
from chatbot.voice_handler import (
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
from api.events import broadcaster
from database.repository import (
    add_message,
    add_rag_retrieval,
    create_call,
    end_call,
    get_call_by_id,
    save_booking_change,
    update_avg_turn_ms,
)
from database.session import engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

# ── Logging helpers ────────────────────────────────────────────────────────────

_LOG_DIR = os.path.join(_REPO_ROOT, settings.SESSION_LOG_DIR)


class _CallSidFilter(logging.Filter):
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


# ── DB helpers (sync SQLite — fast enough to call from async context) ──────────

def _db_log_message(
    call_id: int | None,
    role: str,
    content: str,
    turn_number: int,
    was_interrupted: bool = False,
) -> None:
    if call_id is None:
        return
    try:
        with Session(engine) as s:
            add_message(s, call_id, role, content, turn_number, was_interrupted)
    except Exception as exc:
        logger.error("db log_message failed: %s", exc)


def _db_store_rag(
    call_id: int | None,
    turn_number: int,
    original_query: str,
    rewritten_query: str,
    rag_docs: list[dict],
    was_skipped: bool,
) -> None:
    if call_id is None:
        return
    try:
        with Session(engine) as s:
            add_rag_retrieval(
                s, call_id, turn_number,
                original_query, rewritten_query, rag_docs, was_skipped,
            )
    except Exception as exc:
        logger.error("db store_rag failed: %s", exc)


def _db_end_call(
    call_id: int | None,
    summary: str,
    needs_human: bool,
    flag_reason: str | None,
    status: str = "completed",
) -> None:
    if call_id is None:
        return
    try:
        with Session(engine) as s:
            end_call(s, call_id, summary, needs_human, flag_reason, status)
    except Exception as exc:
        logger.error("db end_call failed: %s", exc)


def _fire_classification(call_id: int | None) -> None:
    if call_id is None:
        return
    asyncio.create_task(_classify_bg(call_id))


async def _classify_bg(call_id: int) -> None:
    # Small delay so DB writes and session cleanup finish before we touch Ollama.
    await asyncio.sleep(3)
    try:
        from chatbot.classifier import classify_and_store
        await classify_and_store(call_id)
    except Exception as exc:
        logger.error("Background classification failed for call %d: %s", call_id, exc)


# ── Session cleanup (exported so main.py lifespan can schedule it) ─────────────

async def session_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(settings.SESSION_CLEANUP_INTERVAL)
        n = conversation_store.cleanup_expired()
        if n:
            logger.info("Cleaned up %d expired session(s)", n)


# ── In-flight voice sessions ───────────────────────────────────────────────────

_voice_sessions: dict[str, dict] = {}

_WELCOME_GREETING = settings.welcome_greeting
_WELCOME_ECHO_PREFIX = settings.WELCOME_ECHO_PREFIX


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/voice/inbound")
@router.post("/voice/inbound/", include_in_schema=False)
async def voice_inbound(CallSid: str = Form(...), From: str = Form(default="")):
    """Twilio webhook — returns ConversationRelay TwiML."""
    logger.info("New inbound call: %s from %s", CallSid, From)
    base = settings.BASE_URL.rstrip("/")
    ws_host = base.replace("https://", "").replace("http://", "")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect action="{base}/voice/action">
    <ConversationRelay url="wss://{ws_host}/voice/ws"
                       language="{settings.TWILIO_LANGUAGE}"
                       transcriptionProvider="{settings.TWILIO_ASR_PROVIDER}"
                       speechModel="{settings.TWILIO_SPEECH_MODEL}"
                       ttsProvider="{settings.TWILIO_TTS_PROVIDER}"
                       voice="{settings.TWILIO_VOICE_ID}"
                       welcomeGreeting="{settings.welcome_greeting}"
                       dtmfDetection="true"
                       interruptible="speech"
                       interruptSensitivity="{settings.TWILIO_INTERRUPT_SENSITIVITY}"
                       hints="{settings.TWILIO_ASR_HINTS}" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")
    
@router.post("/voice/action")
async def voice_action():
    """Called by Twilio when the ConversationRelay session ends."""
    vr = VoiceResponse()
    vr.hangup()
    return Response(content=str(vr), media_type="text/xml")


# ── WebSocket helpers ──────────────────────────────────────────────────────────

async def _send_canned(ws: WebSocket, text: str) -> None:
    await ws.send_text(json.dumps({"type": "text", "token": text, "last": False}))
    await ws.send_text(json.dumps({"type": "text", "token": "", "last": True}))


async def _run_capture_step(
    ws: WebSocket,
    call_sid: str,
    session: dict,
    user_text: str,
    next_mode: str,
    canned: str,
    log_msg: str,
    capture_field: str | None = None,
) -> None:
    if capture_field is not None:
        session["capture_data"][capture_field] = user_text
    session["capture_mode"] = next_mode
    call_id = session.get("call_id")
    turn_num = session.get("turn_number", 0)
    logger.info("[%s] %s", call_sid, log_msg)
    await _send_canned(ws, canned)
    conversation_store.add(call_sid, "user", user_text)
    conversation_store.add(call_sid, "assistant", canned)
    _db_log_message(call_id, "assistant", canned, turn_num)


async def _stream_llm_to_twilio(
    ws: WebSocket,
    call_sid: str,
    messages: list[dict],
    segments: list[str],
    t_silence_end: float = 0.0,
    ttfr_holder: list[float] | None = None,
) -> None:
    try:
        first_token = True
        async for token in stream_voice_tokens(call_sid, messages):
            if first_token:
                first_token = False
                if t_silence_end:
                    ttfr_ms = (time.perf_counter() - t_silence_end) * 1000
                    logger.info("[%s] LATENCY silence_to_first_reply=%.0fms", call_sid, ttfr_ms)
                    if ttfr_holder is not None:
                        ttfr_holder.append(ttfr_ms)
            await ws.send_text(json.dumps({"type": "text", "token": token, "last": False}))
            segments.append(token)

        await ws.send_text(json.dumps({"type": "text", "token": "", "last": True}))

    except asyncio.CancelledError:
        logger.info("[%s] LLM stream cancelled (user interrupted)", call_sid)
        raise


# ── WebSocket handler ──────────────────────────────────────────────────────────

@router.websocket("/voice/ws")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    call_sid: str | None = None
    _session_log_handler: logging.FileHandler | None = None

    try:
        async for raw in websocket.iter_text():
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── SETUP ──────────────────────────────────────────────────────────
            if msg_type == "setup":
                call_sid = msg.get("callSid", str(uuid.uuid4()))
                caller_from = msg.get("from", "unknown")
                call_id: int | None = None
                try:
                    with Session(engine) as s:
                        call = create_call(s, call_sid, caller_from)
                        call_id = call.id
                except Exception as exc:
                    logger.error("[%s] create_call failed: %s", call_sid, exc)

                _voice_sessions[call_sid] = {
                    "current_task": None,
                    "call_id": call_id,
                    "phone_number": caller_from,
                    "capture_mode": "none",
                    "capture_data": {"name": "", "phone": "", "details": ""},
                    "capture_triggered_on": "",
                    "turn_number": 0,
                    "turn_count": 0,
                    "total_turn_ms": 0.0,
                }
                _session_log_handler = _open_session_log(call_sid)
                logger.info("[%s] ConversationRelay connected from %s", call_sid, caller_from)
                await broadcaster.broadcast("call_started", {
                    "call_id": call_id,
                    "call_sid": call_sid,
                    "phone_number": caller_from,
                })

            # ── PROMPT ─────────────────────────────────────────────────────────
            elif msg_type == "prompt":
                t_silence_end = time.perf_counter()
                user_text = msg.get("voicePrompt", "").strip()
                if not user_text or not call_sid:
                    continue

                if user_text.lower().startswith(_WELCOME_ECHO_PREFIX):
                    logger.info("[%s] Skipping echo of welcome greeting", call_sid)
                    continue

                logger.info("[%s] User said: %s", call_sid, user_text)
                session = _voice_sessions.get(call_sid, {})

                # Increment turn counter and cancel any in-flight task
                session["turn_number"] = session.get("turn_number", 0) + 1
                turn_num: int = session["turn_number"]
                call_id = session.get("call_id")

                prev_task = session.get("current_task")
                if prev_task and not prev_task.done():
                    prev_task.cancel()

                capture_mode = session.get("capture_mode", "none")

                # Log user turn to DB regardless of which path we take
                _db_log_message(call_id, "user", user_text, turn_num)

                # ── Booking-capture state machine ───────────────────────────────
                if capture_mode == "none" and check_booking_capture_trigger(user_text):
                    session["capture_triggered_on"] = user_text
                    await _run_capture_step(
                        websocket, call_sid, session, user_text,
                        next_mode="name",
                        canned="Sure, I can take down some details so my manager can give you a call back. Can I get your name please?",
                        log_msg="Booking capture TRIGGERED. State → name",
                    )
                    continue

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
                    session["capture_data"]["details"] = user_text
                    if call_id is not None:
                        cd = session["capture_data"]
                        try:
                            with Session(engine) as s:
                                save_booking_change(
                                    s, call_id, cd["name"], cd["phone"], cd["details"]
                                )
                            logger.info("[%s] Booking change saved to DB", call_sid)
                        except Exception as exc:
                            logger.error("[%s] save_booking_change failed: %s", call_sid, exc)
                    await _run_capture_step(
                        websocket, call_sid, session, user_text,
                        next_mode="done",
                        canned="Perfect, I've got all that. My manager will give you a call back as soon as possible. Is there anything else I can help you with today?",
                        log_msg="Captured details. State → done.",
                    )
                    continue

                # ── LLM path ────────────────────────────────────────────────────
                try:
                    messages, rag_docs, rewritten_query, rag_skipped = await prepare_voice_stream(
                        call_sid, user_text
                    )
                except Exception as exc:
                    logger.error("[%s] prepare_voice_stream failed: %s", call_sid, exc)
                    await websocket.send_text(
                        json.dumps({"type": "text", "token": _FALLBACK_MSG, "last": True})
                    )
                    conversation_store.add(call_sid, "assistant", _FALLBACK_MSG)
                    continue

                _db_store_rag(call_id, turn_num, user_text, rewritten_query, rag_docs, rag_skipped)

                segments: list[str] = []

                async def _run_stream(
                    ws: WebSocket,
                    sid: str,
                    msgs: list[dict],
                    segs: list[str],
                    t_silence: float,
                    _turn_num: int = turn_num,
                    _call_id: int | None = call_id,
                    _user_text: str = user_text,
                ):
                    _ttfr_holder: list[float] = []
                    try:
                        await _stream_llm_to_twilio(
                            ws, sid, msgs, segs, t_silence, _ttfr_holder
                        )

                        raw_joined = "".join(segs)
                        full_reply = clean_for_tts(raw_joined)
                        if full_reply.strip():
                            conversation_store.add(sid, "assistant", full_reply)
                            logger.info("[%s] Assistant (full): %s", sid, full_reply[:200])

                        _pl = get_session_logger(sid)
                        if _pl is not None:
                            _pl.log_llm_response(raw_joined)
                            _pl.log_final_response(full_reply)

                        if full_reply.strip():
                            _db_log_message(_call_id, "assistant", full_reply, _turn_num)

                        # End-of-call detection (NOT included in per-turn timing)
                        _kw = check_end_keywords(_user_text)
                        _end = None
                        if _kw == "definite":
                            _end = build_end_decision_from_definite(_user_text, full_reply)
                            logger.info("[%s] End-call keyword DEFINITE", sid)
                        elif _kw == "maybe":
                            logger.info("[%s] End-call keyword MAYBE — classifying", sid)
                            _end = await classify_turn_for_end(sid, _user_text, full_reply)

                        if _end is not None:
                            _db_end_call(
                                _call_id,
                                _end["summary"],
                                _end["needs_human"],
                                _end["flag_reason"] or None,
                            )
                            _fire_classification(_call_id)
                            await broadcaster.broadcast("call_ended", {
                                "call_id": _call_id,
                                "call_sid": sid,
                                "status": "completed",
                                "needs_human": _end["needs_human"],
                                "summary": _end["summary"],
                            })
                            try:
                                await ws.send_text(json.dumps({
                                    "type": "end",
                                    "handoffData": json.dumps({
                                        "reasonCode": "bot-ended-call",
                                        "reason": _end["summary"],
                                        "needs_human": _end["needs_human"],
                                    }),
                                }))
                            except Exception as exc:
                                logger.error("[%s] Failed to send end message: %s", sid, exc)

                    except asyncio.CancelledError:
                        partial = clean_for_tts("".join(segs))
                        if partial.strip():
                            conversation_store.add(sid, "assistant", partial)
                            _db_log_message(_call_id, "assistant", partial, _turn_num, was_interrupted=True)
                            logger.info("[%s] Assistant (partial/cancelled): %s", sid, partial[:200])
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
                                json.dumps({"type": "text", "token": _FALLBACK_MSG, "last": True})
                            )
                            conversation_store.add(sid, "assistant", _FALLBACK_MSG)
                        except Exception:
                            pass

                    finally:
                        _sess = _voice_sessions.get(sid)
                        if _sess is not None and _ttfr_holder:
                            _sess["turn_count"] = _sess.get("turn_count", 0) + 1
                            _sess["total_turn_ms"] = (
                                _sess.get("total_turn_ms", 0.0) + _ttfr_holder[0]
                            )

                task = asyncio.create_task(
                    _run_stream(websocket, call_sid, messages, segments, t_silence_end)
                )
                session["current_task"] = task

            # ── INTERRUPT ──────────────────────────────────────────────────────
            elif msg_type == "interrupt":
                spoken_fragment = msg.get("utteranceUntilInterrupt", "")
                logger.info("[%s] User interrupted. Spoken: %s", call_sid, spoken_fragment[:200])

                if call_sid:
                    session = _voice_sessions.get(call_sid, {})
                    current_task = session.get("current_task")
                    if current_task and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                    if spoken_fragment.strip():
                        conversation_store.replace_last_assistant(call_sid, spoken_fragment.strip())

            # ── DTMF ───────────────────────────────────────────────────────────
            elif msg_type == "dtmf":
                logger.info("[%s] DTMF: %s", call_sid, msg.get("digit"))

    except WebSocketDisconnect:
        logger.info("[%s] ConversationRelay disconnected", call_sid)
    except Exception as exc:
        logger.error("[%s] ConversationRelay error: %s", call_sid, exc)
    finally:
        if call_sid:
            session = _voice_sessions.pop(call_sid, {})
            _turns = session.get("turn_count", 0)
            _avg_ms = (session.get("total_turn_ms", 0.0) / _turns) if _turns else None
            if _avg_ms is not None:
                logger.info(
                    "[%s] Call summary — %d request(s), avg %.2fs per request",
                    call_sid, _turns, _avg_ms / 1000,
                )
                _call_id_for_avg = session.get("call_id")
                if _call_id_for_avg is not None:
                    try:
                        with Session(engine) as s:
                            update_avg_turn_ms(s, _call_id_for_avg, _avg_ms)
                    except Exception as exc:
                        logger.error("[%s] update_avg_turn_ms failed: %s", call_sid, exc)
            current_task = session.get("current_task")
            if current_task and not current_task.done():
                current_task.cancel()
            call_id = session.get("call_id")
            if call_id is not None:
                try:
                    with Session(engine) as s:
                        call = get_call_by_id(s, call_id)
                        if call and not call.ended_at:
                            end_call(s, call_id, "Call disconnected", False, None, status="abandoned")
                            logger.info("[%s] Finalised abandoned call", call_sid)
                            _fire_classification(call_id)
                            await broadcaster.broadcast("call_ended", {
                                "call_id": call_id,
                                "call_sid": call_sid,
                                "status": "abandoned",
                                "needs_human": False,
                                "summary": "Call disconnected",
                            })
                except Exception as exc:
                    logger.error("[%s] Cleanup end_call failed: %s", call_sid, exc)
            conversation_store.clear(call_sid)
        if _session_log_handler and call_sid:
            _close_session_log(call_sid, _session_log_handler)
        if call_sid:
            close_session_logger(call_sid)
            release_session_client(call_sid)
