# Migration: Twilio ConversationRelay → Kyutai STT + TTS

## Overview

**What this migration does:**  
Replaces Twilio's managed Deepgram STT and ElevenLabs TTS with Kyutai's open-source
streaming models (`kyutai/stt-1b-en_fr` and Kyutai TTS), running locally on your GPU
server. The LLM (Ollama phi4), RAG pipeline, conversation store, DB logging, and
classifier are **completely unchanged**.

**Why Kyutai:**
- Built-in semantic VAD on the STT model — no separate VAD library needed
- Streaming STT: word-by-word transcription in real time
- Streaming TTS: audio starts before the full text is ready
- Runs on the same GPU server as Ollama
- MIT/Apache licensed, free

---

## Architecture: Before vs After

### Before (ConversationRelay)
```
Twilio ──── POST /voice/inbound ────────────── returns ConversationRelay TwiML
              │
              │  ConversationRelay manages:
              │    STT = Deepgram nova-3 (cloud)
              │    TTS = ElevenLabs (cloud)
              │
        WS /voice/ws  (text in, text out)
              │
        voice_handler.py → Ollama phi4 → pgvector
```

### After (Media Streams + Kyutai)
```
Twilio ──── POST /voice/inbound ────────────── returns Media Streams TwiML
              │
              │  You manage:
              │    STT = Kyutai STT 1B (local, GPU server)
              │    TTS = Kyutai TTS (local, GPU server)
              │
        WS /voice/stream  (raw µ-law audio in, raw µ-law audio out)
              │
              ├── audio_utils.py (format conversion)
              ├── kyutai_stt.py ──── ws://192.168.50.150:8090
              ├── voice_handler.py → Ollama phi4 → pgvector  ← UNCHANGED
              └── kyutai_tts.py ──── ws://192.168.50.150:8089
```

### Twilio Media Streams Protocol (replaces ConversationRelay)

**Twilio → Your server (inbound audio):**
```json
{"event": "start",  "start":  {"streamSid": "MZ...", "callSid": "CA..."}}
{"event": "media",  "media":  {"track": "inbound", "payload": "<base64-mulaw-8kHz>"}}
{"event": "stop"}
```

**Your server → Twilio (outbound audio):**
```json
{"event": "media",  "streamSid": "MZ...", "media": {"payload": "<base64-mulaw-8kHz>"}}
{"event": "clear",  "streamSid": "MZ..."}   // interruption: flush Twilio's audio buffer
{"event": "mark",   "streamSid": "MZ...", "mark": {"name": "turn-done"}}
```

### Kyutai STT Protocol (MessagePack over WebSocket)

**Server:** `ws://192.168.50.150:8090/api/asr-streaming`  
**Auth header:** `kyutai-api-key: public_token`

```
On connect:  server sends  {"type": "Ready"}
You send:    {"type": "Audio", "pcm": [float32 samples @ 24kHz]}   ← 1920 samples = 80ms
             {"type": "Marker", "id": 42}                           ← optional sync
Server sends: {"type": "Word",  "text": "hello"}                   ← transcription
              {"type": "Step",  "prs": [p0, p1, p2, ...]}          ← prs[2] = pause prob
              {"type": "Marker","id": 42}                           ← marker echo
```

**VAD:** `prs[2] > 0.6` for 3+ consecutive frames = user finished speaking → trigger LLM

**Important:** Send 1 second of silence (24000 zero-samples) immediately after connect.
This is required by the model for proper initialization.

### Kyutai TTS Protocol (MessagePack over WebSocket)

**Server:** `ws://192.168.50.150:8089/api/tts_streaming?voice=<voice>&cfg_alpha=1.5&format=PcmMessagePack`  
**Auth header:** `kyutai-api-key: public_token`

```
On connect:  server sends  {"type": "Ready"}
You send:    {"type": "Text", "text": "word "}     ← one word at a time, with space
             {"type": "Eos"}                        ← end of utterance
Server sends: {"type": "Audio", "pcm": [float32 samples @ 24kHz]}
```

One TTS connection per bot turn. Close it when audio stream ends.

### Audio Format Bridge

```
Twilio:  µ-law  8000 Hz  8-bit    160 bytes/frame  (20ms)
Kyutai:  PCM   24000 Hz  float32  1920 samples     (80ms)

Twilio → Kyutai:
  base64_decode → µ-law bytes
  audioop.ulaw2lin → PCM int16 @ 8kHz
  audioop.ratecv → PCM int16 @ 24kHz  (3x upsample)
  / 32768.0 → float32

Kyutai → Twilio:
  * 32767 → int16
  audioop.ratecv → PCM int16 @ 8kHz  (1/3 downsample)
  audioop.lin2ulaw → µ-law bytes
  base64_encode → payload string
```

---

## Files: What Changes vs What Stays

| File | Action | Notes |
|------|--------|-------|
| `chatbot/routers/voice.py` | **Full rewrite** | TwiML + new WS handler |
| `config.py` | **Add settings** | STT/TTS URLs, voice, API key |
| `requirements.txt` | **Add deps** | msgpack, websockets, scipy |
| `chatbot/audio_utils.py` | **New file** | µ-law ↔ float32 conversion |
| `chatbot/kyutai_stt.py` | **New file** | STT WebSocket client |
| `chatbot/kyutai_tts.py` | **New file** | TTS WebSocket client |
| `chatbot/voice_handler.py` | **Unchanged** | All RAG + LLM logic stays |
| `chatbot/llm.py` | **Unchanged** | Ollama client stays |
| `chatbot/conversation.py` | **Unchanged** | Session store stays |
| `chatbot/vector_store.py` | **Unchanged** | pgvector stays |
| `chatbot/classifier.py` | **Unchanged** | End-of-call logic stays |
| `chatbot/summarizer.py` | **Unchanged** | Call summary stays |
| All database code | **Unchanged** | DB models, repo, session stay |

---

## Phase 1: GPU Server Setup (on 192.168.50.150)

> Run everything below on the Linux GPU server, not your Mac.

### Hardware Requirements

| Component | Minimum VRAM |
|-----------|-------------|
| Kyutai STT 1B (EN/FR, has built-in VAD) | 2.5 GB |
| Kyutai TTS | 5.3 GB |
| Ollama phi4 (already running) | ~8 GB |
| **Total** | **~16 GB recommended** |

If VRAM is tight: use `kyutai/stt-1b-en_fr` (not 2.6B) and keep Ollama's `keep_alive=-1`.

### 1.1 Install Rust (if not installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version  # verify
```

### 1.2 Install the moshi-server crate

The Kyutai Rust server handles both STT and TTS. It wraps the Python model
under the hood, so Python + CUDA deps are also needed.

```bash
# Install moshi Python package first (required by the Rust server)
pip install moshi>=0.2.6

# Install the Rust server binary (CUDA build)
cargo install --features cuda moshi-server

# Verify
moshi-server --version
```

If the install fails, try forcing a reinstall:
```bash
cargo uninstall moshi-server
cargo install --features cuda moshi-server
```

### 1.3 Clone the delayed-streams-modeling repo (for configs)

```bash
cd ~
git clone https://github.com/kyutai-labs/delayed-streams-modeling.git
cd delayed-streams-modeling
```

This repo contains the TOML config files for the STT and TTS servers.

### 1.4 Set your Hugging Face token

The model weights are gated on HuggingFace. Create a token at
https://huggingface.co/settings/tokens (read access is enough).

```bash
export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
# Add to ~/.bashrc or ~/.zshrc to persist
```

Accept the model license at:
- https://huggingface.co/kyutai/stt-1b-en_fr
- https://huggingface.co/kyutai/tts-1b-en (or whichever TTS model you choose)

### 1.5 Start the STT server

```bash
cd ~/delayed-streams-modeling

# For the 1B English+French model (recommended — has semantic VAD, lower VRAM)
moshi-server worker --config configs/config-stt-en_fr-hf.toml

# Alternative: 2.6B English-only model (higher accuracy, more VRAM, 2.5s latency)
# moshi-server worker --config configs/config-stt-en-hf.toml
```

Default port: **8090**. The server prints `Listening on 0.0.0.0:8090` when ready.  
First run downloads the model weights (~2-5 GB). Subsequent starts are fast.

To run as background service:
```bash
nohup moshi-server worker --config configs/config-stt-en_fr-hf.toml > /var/log/kyutai-stt.log 2>&1 &
echo $! > /var/run/kyutai-stt.pid
```

### 1.6 Start the TTS server

The TTS server needs the Unmute install script because it uses Python under the hood:

```bash
# Clone unmute for the TTS startup script
cd ~
git clone https://github.com/kyutai-labs/unmute.git
cd unmute

# Run the TTS startup script (handles Python + Rust deps together)
bash dockerless/start_tts.sh
```

Default port: **8089**.

Alternatively, if you already have moshi-server installed:
```bash
cd ~/delayed-streams-modeling
moshi-server worker --config configs/config-tts.toml
```

### 1.7 Choose a TTS voice

Voices are hosted at https://huggingface.co/kyutai/tts-voices

Browse available voices:
```bash
python3 -c "
from huggingface_hub import list_repo_files
for f in list_repo_files('kyutai/tts-voices'):
    print(f)
"
```

Example voice names (use in `KYUTAI_TTS_VOICE` env var):
```
expresso/ex03-neutral_normal      # neutral female
expresso/ex01-happy_surprise      # expressive female
expresso/ex04-calm_soft           # calm male
```

### 1.8 Verify both servers are running

```bash
# Quick WebSocket ping for STT (requires wscat: npm install -g wscat)
wscat -c ws://localhost:8090/api/asr-streaming \
      -H "kyutai-api-key: public_token" \
      --wait 2
# Should print: {"type": "Ready"} then disconnect

# Check ports are listening
ss -tlnp | grep -E '8089|8090'
```

From your Mac (replace with actual server IP):
```bash
wscat -c ws://192.168.50.150:8090/api/asr-streaming \
      -H "kyutai-api-key: public_token" \
      --wait 2
```

If the ports aren't reachable from outside: open them in the server's firewall:
```bash
sudo ufw allow 8089/tcp
sudo ufw allow 8090/tcp
```

---

## Phase 2: Project Code Changes

### 2.1 requirements.txt — add dependencies

Add these lines to `requirements.txt`:

```
# Kyutai STT + TTS bridge
msgpack>=1.1.0          # Kyutai server wire format
websockets>=13.0        # client-side WebSocket connections to Kyutai servers
scipy>=1.13.0           # audio resampling (8kHz ↔ 24kHz)
numpy>=1.26.0           # already present, ensure it stays
```

> **Python 3.13+ note:** `audioop` was removed in Python 3.13. If you're on 3.13,
> also add `audioop-lts>=0.2.1`. On 3.11/3.12, `audioop` is in the stdlib.

Then install:
```bash
pip install -r requirements.txt
```

### 2.2 .env — add new variables

Add to your `.env` file:

```env
# Kyutai STT server (on GPU server)
KYUTAI_STT_URL=ws://192.168.50.150:8090/api/asr-streaming
KYUTAI_API_KEY=public_token

# Kyutai TTS server (on GPU server)
KYUTAI_TTS_URL=ws://192.168.50.150:8089/api/tts_streaming
KYUTAI_TTS_VOICE=expresso/ex03-neutral_normal

# VAD tuning (prs[2] threshold for detecting end of user speech)
KYUTAI_VAD_THRESHOLD=0.6
KYUTAI_VAD_PAUSE_FRAMES=3
```

### 2.3 config.py — add settings

Add these fields to the `Settings` class in `config.py`, inside the `class Settings(BaseSettings):` block.
Place them after the existing Twilio settings section:

```python
# ── Kyutai STT + TTS (local models, replace Twilio ConversationRelay) ───────
KYUTAI_STT_URL: str = Field(
    default="ws://192.168.50.150:8090/api/asr-streaming",
    description="Kyutai STT Rust server WebSocket URL.",
)
KYUTAI_TTS_URL: str = Field(
    default="ws://192.168.50.150:8089/api/tts_streaming",
    description="Kyutai TTS Rust server WebSocket URL.",
)
KYUTAI_TTS_VOICE: str = Field(
    default="expresso/ex03-neutral_normal",
    description="Voice name from kyutai/tts-voices HuggingFace repo.",
)
KYUTAI_API_KEY: str = Field(
    default="public_token",
    description="API key for Kyutai servers (public_token for self-hosted).",
)
KYUTAI_VAD_THRESHOLD: float = Field(
    default=0.6,
    description="STT prs[2] score above which a frame counts as a pause.",
)
KYUTAI_VAD_PAUSE_FRAMES: int = Field(
    default=3,
    description="Consecutive pause frames required to trigger end-of-utterance.",
)
```

### 2.4 New file: chatbot/audio_utils.py

Create this file at `chatbot/audio_utils.py`:

```python
"""
Audio format conversion: Twilio Media Streams ↔ Kyutai STT/TTS.

Twilio:  µ-law  8000 Hz  8-bit    160 bytes/frame   20ms
Kyutai:  PCM   24000 Hz  float32  1920 samples/frame 80ms
"""

from __future__ import annotations

import audioop
import base64

import numpy as np

TWILIO_SAMPLE_RATE = 8_000
KYUTAI_SAMPLE_RATE = 24_000
TWILIO_FRAME_BYTES = 160       # 20ms of µ-law at 8kHz
KYUTAI_FRAME_SAMPLES = 1_920   # 80ms of PCM at 24kHz (= 4 Twilio frames)

# audioop.ratecv state — None means start fresh; we pass state through for
# continuous streams. For one-shot conversions, always pass None.


def mulaw_payload_to_float32(payload: str) -> np.ndarray:
    """
    Decode a base64-encoded Twilio µ-law payload to float32 PCM at 24kHz.

    payload  — the 'payload' field from a Twilio media event
    returns  — float32 array, shape (N,), values in [-1.0, 1.0]
    """
    mulaw_bytes = base64.b64decode(payload)
    # µ-law → PCM16 @ 8kHz
    pcm16_8k: bytes = audioop.ulaw2lin(mulaw_bytes, 2)
    # Resample 8kHz → 24kHz (3× upsample)
    pcm16_24k, _ = audioop.ratecv(pcm16_8k, 2, 1, TWILIO_SAMPLE_RATE, KYUTAI_SAMPLE_RATE, None)
    # PCM16 int → float32 [-1, 1]
    samples = np.frombuffer(pcm16_24k, dtype=np.int16).astype(np.float32) / 32_768.0
    return samples


def float32_to_mulaw_payload(samples: np.ndarray) -> str:
    """
    Encode float32 PCM at 24kHz to a base64-encoded Twilio µ-law payload.

    samples  — float32 array, values in [-1.0, 1.0]
    returns  — base64 string to use as Twilio media payload
    """
    # float32 → int16
    pcm16_24k = (np.clip(samples, -1.0, 1.0) * 32_767).astype(np.int16).tobytes()
    # Resample 24kHz → 8kHz (1/3 downsample)
    pcm16_8k, _ = audioop.ratecv(pcm16_24k, 2, 1, KYUTAI_SAMPLE_RATE, TWILIO_SAMPLE_RATE, None)
    # PCM16 → µ-law
    mulaw_bytes: bytes = audioop.lin2ulaw(pcm16_8k, 2)
    return base64.b64encode(mulaw_bytes).decode()


def silence_float32(duration_s: float, sample_rate: int = KYUTAI_SAMPLE_RATE) -> np.ndarray:
    """Return a float32 array of zeros (silence) of the given duration."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def chunk_samples(samples: np.ndarray, frame_size: int = KYUTAI_FRAME_SAMPLES):
    """Yield fixed-size frames from a sample array, zero-padding the last frame."""
    for i in range(0, len(samples), frame_size):
        frame = samples[i : i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        yield frame
```

### 2.5 New file: chatbot/kyutai_stt.py

Create this file at `chatbot/kyutai_stt.py`:

```python
"""
Kyutai STT client — persistent WebSocket connection to the STT Rust server.

Usage per call:
    stt = KyutaiSTT(url, api_key)
    await stt.connect()          # sends 1s silence init burst
    await stt.send_audio(arr)    # call for every incoming audio frame
    async for event in stt.events():
        if event["type"] == "Word":   ...
        if event["type"] == "VAD":    ...  # event["score"] = prs[2]
    await stt.close()
"""

from __future__ import annotations

import asyncio
import logging

import msgpack
import numpy as np
import websockets
import websockets.exceptions

from chatbot.audio_utils import KYUTAI_FRAME_SAMPLES, chunk_samples, silence_float32

logger = logging.getLogger(__name__)

_SILENCE_INIT_S = 1.0   # Required by the model before real audio


class KyutaiSTT:
    """Streaming STT client. One instance per call, kept alive for call duration."""

    def __init__(self, url: str, api_key: str = "public_token") -> None:
        self._url = url
        self._api_key = api_key
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Open WebSocket, wait for Ready, send silence init burst."""
        self._ws = await websockets.connect(
            self._url,
            additional_headers={"kyutai-api-key": self._api_key},
            max_size=2**22,  # 4 MB — large audio payloads
        )
        # Wait for Ready
        raw = await self._ws.recv()
        msg = msgpack.unpackb(raw, raw=False)
        if msg.get("type") != "Ready":
            raise RuntimeError(f"Kyutai STT: expected Ready, got {msg}")
        logger.info("Kyutai STT connected and ready")

        # Start background task that reads server responses into queue
        self._recv_task = asyncio.create_task(self._recv_loop())

        # Send 1 second of silence (model initialisation requirement)
        await self._send_pcm(silence_float32(_SILENCE_INIT_S))

    async def _send_pcm(self, samples: np.ndarray) -> None:
        """Send float32 samples to the STT server in 80ms frames."""
        if self._ws is None:
            return
        for frame in chunk_samples(samples, KYUTAI_FRAME_SAMPLES):
            payload = msgpack.packb({"type": "Audio", "pcm": frame.tolist()})
            await self._ws.send(payload)

    async def send_audio(self, samples: np.ndarray) -> None:
        """
        Public API: forward an audio chunk (float32 @ 24kHz) to the STT server.
        Call this for every inbound frame from Twilio (after format conversion).
        """
        await self._send_pcm(samples)

    async def _recv_loop(self) -> None:
        """Background loop: read server messages and enqueue parsed events."""
        try:
            async for raw in self._ws:
                msg = msgpack.unpackb(raw, raw=False)
                t = msg.get("type")
                if t == "Word":
                    word = (msg.get("text") or "").strip()
                    if word:
                        await self._event_queue.put({"type": "Word", "text": word})
                elif t == "Step":
                    prs = msg.get("prs") or []
                    if len(prs) > 2:
                        await self._event_queue.put({"type": "VAD", "score": float(prs[2])})
                elif t == "Error":
                    logger.error("Kyutai STT server error: %s", msg.get("message"))
                # Marker, EndWord, Ready — ignored
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.error("Kyutai STT recv_loop error: %s", exc)
        finally:
            await self._event_queue.put({"type": "_closed"})

    async def events(self):
        """
        Async generator — yields STT events:
            {"type": "Word", "text": "hello"}
            {"type": "VAD",  "score": 0.73}   # prs[2] from Step messages
        Stops when the connection closes.
        """
        while True:
            event = await self._event_queue.get()
            if event["type"] == "_closed":
                return
            yield event

    async def close(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        self._ws = None
```

### 2.6 New file: chatbot/kyutai_tts.py

Create this file at `chatbot/kyutai_tts.py`:

```python
"""
Kyutai TTS client — one connection per bot utterance.

Usage per turn:
    tts = KyutaiTTS(url, voice, api_key)
    await tts.connect()
    await tts.send_text("Hello ")       # send words with trailing space
    await tts.send_text("world. ")
    await tts.send_eos()                # end of utterance
    async for chunk in tts.audio_chunks():
        # chunk is float32 ndarray @ 24kHz
        ...
    await tts.close()
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse

import msgpack
import numpy as np
import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)


class KyutaiTTS:
    """
    Streaming TTS client. Create one instance per bot turn; close after the
    turn's audio is fully received. Do not reuse across turns.
    """

    def __init__(
        self,
        base_url: str,
        voice: str,
        api_key: str = "public_token",
        cfg_alpha: float = 1.5,
    ) -> None:
        params = urllib.parse.urlencode(
            {"voice": voice, "cfg_alpha": cfg_alpha, "format": "PcmMessagePack"}
        )
        self._url = f"{base_url}?{params}"
        self._api_key = api_key
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self._url,
            additional_headers={"kyutai-api-key": self._api_key},
            max_size=2**22,
        )
        raw = await self._ws.recv()
        msg = msgpack.unpackb(raw, raw=False)
        if msg.get("type") != "Ready":
            raise RuntimeError(f"Kyutai TTS: expected Ready, got {msg}")
        logger.debug("Kyutai TTS connected and ready")

    async def send_text(self, text: str) -> None:
        """Send one word (or small chunk) of text. Include trailing space."""
        if not text.strip():
            return
        payload = msgpack.packb({"type": "Text", "text": text})
        await self._ws.send(payload)

    async def send_eos(self) -> None:
        """Signal end of utterance — server will flush remaining audio."""
        payload = msgpack.packb({"type": "Eos"})
        await self._ws.send(payload)

    async def audio_chunks(self):
        """
        Async generator — yields float32 numpy arrays (24kHz PCM) as they
        arrive from the TTS server. Stops when the server closes the connection.
        """
        try:
            async for raw in self._ws:
                msg = msgpack.unpackb(raw, raw=False)
                if msg.get("type") == "Audio":
                    yield np.array(msg["pcm"], dtype=np.float32)
                elif msg.get("type") == "Error":
                    logger.error("Kyutai TTS server error: %s", msg.get("message"))
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.error("Kyutai TTS audio_chunks error: %s", exc)

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
        self._ws = None
```

### 2.7 Full replacement: chatbot/routers/voice.py

Replace the entire contents of `chatbot/routers/voice.py` with the following.
Note: all DB helpers (`_db_log_message`, `_db_store_rag`, `_db_end_call`, etc.),
`session_cleanup_loop`, and the `voice_action` endpoint are **identical** to the
original — only the inbound TwiML and the WebSocket handler change.

```python
"""
Twilio Media Streams voice routes.

Endpoints
---------
POST /voice/inbound  — Twilio webhook; returns Media Streams TwiML
POST /voice/action   — Twilio session-end webhook
WS   /voice/stream   — bidirectional audio bridge (replaces ConversationRelay)

Audio flow
----------
Twilio → µ-law 8kHz → audio_utils → float32 24kHz → Kyutai STT
Kyutai STT → Word + VAD → voice_handler (RAG + LLM) → text tokens
text tokens → Kyutai TTS → float32 24kHz → audio_utils → µ-law 8kHz → Twilio
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid

from fastapi import APIRouter, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlmodel import Session

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chatbot.audio_utils import float32_to_mulaw_payload, mulaw_payload_to_float32
from chatbot.config import settings
from chatbot.conversation import conversation_store
from chatbot.kyutai_stt import KyutaiSTT
from chatbot.kyutai_tts import KyutaiTTS
from chatbot.llm import _FALLBACK_MSG, release_session_client
from chatbot.voice_handler import (
    build_end_decision_from_definite,
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
    update_avg_turn_ms,
)
from database.session import engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

# ── Logging helpers (unchanged) ────────────────────────────────────────────────

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


# ── DB helpers (unchanged) ─────────────────────────────────────────────────────

def _db_log_message(
    call_id: int | None, role: str, content: str, turn_number: int,
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
    call_id: int | None, turn_number: int, original_query: str,
    rewritten_query: str, rag_docs: list[dict], was_skipped: bool,
) -> None:
    if call_id is None:
        return
    try:
        with Session(engine) as s:
            add_rag_retrieval(
                s, call_id, turn_number, original_query, rewritten_query,
                rag_docs, was_skipped,
            )
    except Exception as exc:
        logger.error("db store_rag failed: %s", exc)


def _db_end_call(
    call_id: int | None, summary: str, needs_human: bool,
    flag_reason: str | None, status: str = "completed",
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


FOLLOWUP_EMAIL_CATEGORIES = {"Birthday Parties"}


async def _classify_bg(call_id: int) -> None:
    await asyncio.sleep(3)
    matched: list[str] = []
    try:
        from chatbot.classifier import classify_and_store
        matched = await classify_and_store(call_id)
    except Exception as exc:
        logger.error("Background classification failed for call %d: %s", call_id, exc)
    try:
        await _maybe_send_followup(call_id, matched)
    except Exception as exc:
        logger.error("Follow-up email check failed for call %d: %s", call_id, exc)


async def _maybe_send_followup(call_id: int, matched_categories: list[str]) -> None:
    from chatbot.summarizer import summarize_call
    from database.repository import get_messages
    from src.email_service import send_followup_email

    triggered_category = next(
        (c for c in matched_categories if c in FOLLOWUP_EMAIL_CATEGORIES), None
    )
    with Session(engine) as s:
        call = get_call_by_id(s, call_id)
        messages_rows = get_messages(s, call_id) if call else []

    if call is None:
        return
    if not call.needs_human and triggered_category is None:
        return

    reason = call.flag_reason or "needs human" if call.needs_human else f"category: {triggered_category}"
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages_rows]
    summary = await summarize_call(msg_dicts)
    await asyncio.to_thread(
        send_followup_email, call.call_sid, call.phone_number, summary, reason,
    )


# ── Session cleanup (unchanged) ────────────────────────────────────────────────

async def session_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(settings.SESSION_CLEANUP_INTERVAL)
        n = conversation_store.cleanup_expired()
        if n:
            logger.info("Cleaned up %d expired session(s)", n)


# ── In-flight voice sessions ───────────────────────────────────────────────────

_voice_sessions: dict[str, dict] = {}


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/voice/inbound")
@router.post("/voice/inbound/", include_in_schema=False)
async def voice_inbound(CallSid: str = Form(...), From: str = Form(default="")):
    """Twilio webhook — returns Media Streams TwiML (replaces ConversationRelay)."""
    logger.info("New inbound call: %s from %s", CallSid, From)
    base = settings.BASE_URL.rstrip("/")
    ws_host = base.replace("https://", "").replace("http://", "")
    # track="both_tracks" enables bidirectional audio over WebSocket
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        f'  <Connect action="{base}/voice/action">\n'
        f'    <Stream url="wss://{ws_host}/voice/stream" track="both_tracks" />\n'
        "  </Connect>\n"
        "</Response>"
    )
    return Response(content=twiml, media_type="text/xml")


@router.post("/voice/action")
async def voice_action():
    """Called by Twilio when the Media Streams session ends."""
    from twilio.twiml.voice_response import VoiceResponse
    vr = VoiceResponse()
    vr.hangup()
    return Response(content=str(vr), media_type="text/xml")


# ── WebSocket handler ──────────────────────────────────────────────────────────

@router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket handler.

    Three concurrent tasks run for the lifetime of the call:
      1. main_loop     — reads Twilio audio frames, converts, forwards to STT
      2. stt_loop      — reads STT events (words + VAD), triggers LLM on pause
      3. audio_writer  — reads from out_queue, sends µ-law audio to Twilio
    """
    await websocket.accept()

    call_sid: str | None = None
    stream_sid: str | None = None
    call_id: int | None = None
    _session_log_handler: logging.FileHandler | None = None

    # Outbound audio queue (producer: TTS task; consumer: audio_writer task)
    out_queue: asyncio.Queue[str | None] = asyncio.Queue()

    # Per-call state
    transcript_words: list[str] = []
    bot_speaking = False
    consecutive_pause_frames = 0
    current_task: asyncio.Task | None = None
    turn_number = 0
    turn_count = 0
    total_turn_ms = 0.0
    t_silence_end = 0.0

    stt: KyutaiSTT | None = None

    VAD_THRESHOLD = settings.KYUTAI_VAD_THRESHOLD
    VAD_PAUSE_FRAMES = settings.KYUTAI_VAD_PAUSE_FRAMES

    # ── helpers ────────────────────────────────────────────────────────────────

    async def send_to_twilio(payload_b64: str) -> None:
        """Send a µ-law audio payload to Twilio."""
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload_b64},
        }))

    async def clear_twilio_buffer() -> None:
        """Tell Twilio to discard buffered outbound audio (interruption)."""
        await websocket.send_text(json.dumps({
            "event": "clear",
            "streamSid": stream_sid,
        }))

    async def drain_out_queue() -> None:
        """Discard any audio frames already in the outbound queue."""
        while not out_queue.empty():
            try:
                out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── audio_writer task ──────────────────────────────────────────────────────

    async def audio_writer() -> None:
        """Consume µ-law payloads from out_queue and send to Twilio."""
        while True:
            payload = await out_queue.get()
            if payload is None:
                return
            try:
                await send_to_twilio(payload)
            except Exception as exc:
                logger.error("[%s] audio_writer send error: %s", call_sid, exc)

    # ── per-turn LLM + TTS task ────────────────────────────────────────────────

    async def process_turn(user_text: str) -> None:
        """
        Full per-turn pipeline: RAG → LLM stream → TTS → audio to Twilio.
        Cancellation-safe: cleans up TTS connection if interrupted.
        """
        nonlocal bot_speaking, turn_number, turn_count, total_turn_ms

        turn_number += 1
        turn_num = turn_number
        t_start = time.perf_counter()

        _db_log_message(call_id, "user", user_text, turn_num)

        # RAG + message assembly
        try:
            messages, rag_docs, rewritten_query, rag_skipped = await prepare_voice_stream(
                call_sid, user_text
            )
        except Exception as exc:
            logger.error("[%s] prepare_voice_stream failed: %s", call_sid, exc)
            # Synthesize fallback via TTS
            await _synthesize_and_queue(settings.fallback_message)
            conversation_store.add(call_sid, "assistant", settings.fallback_message)
            return

        _db_store_rag(call_id, turn_num, user_text, rewritten_query, rag_docs, rag_skipped)

        segments: list[str] = []
        tts_conn: KyutaiTTS | None = None
        ttfr_ms: float | None = None

        try:
            bot_speaking = True

            tts_conn = KyutaiTTS(
                settings.KYUTAI_TTS_URL,
                settings.KYUTAI_TTS_VOICE,
                settings.KYUTAI_API_KEY,
            )
            await tts_conn.connect()

            async def feed_tts() -> None:
                """Stream LLM tokens to TTS server word by word."""
                nonlocal ttfr_ms
                first_token = True
                async for token in stream_voice_tokens(call_sid, messages):
                    if first_token:
                        first_token = False
                        ttfr_ms = (time.perf_counter() - t_silence_end) * 1000
                        logger.info(
                            "[%s] LATENCY silence_to_first_reply=%.0fms",
                            call_sid, ttfr_ms,
                        )
                    segments.append(token)
                    # Send word by word for minimum TTS latency
                    words = token.split()
                    for word in words:
                        await tts_conn.send_text(word + " ")
                await tts_conn.send_eos()

            async def recv_tts_audio() -> None:
                """Read TTS audio chunks and push to Twilio output queue."""
                async for chunk in tts_conn.audio_chunks():
                    payload = float32_to_mulaw_payload(chunk)
                    await out_queue.put(payload)

            await asyncio.gather(feed_tts(), recv_tts_audio())

            # Persist full reply
            raw_reply = "".join(segments)
            full_reply = clean_for_tts(raw_reply)
            if full_reply.strip():
                conversation_store.add(call_sid, "assistant", full_reply)
                _db_log_message(call_id, "assistant", full_reply, turn_num)
                logger.info("[%s] Assistant: %s", call_sid, full_reply[:200])

            pl = get_session_logger(call_sid)
            if pl:
                pl.log_llm_response(raw_reply)
                pl.log_final_response(full_reply)

            # End-of-call detection
            kw = check_end_keywords(user_text)
            end_decision = None
            if kw == "definite":
                end_decision = build_end_decision_from_definite(user_text, full_reply)
                logger.info("[%s] End-call keyword DEFINITE", call_sid)
            elif kw == "maybe":
                logger.info("[%s] End-call keyword MAYBE — classifying", call_sid)
                end_decision = await classify_turn_for_end(call_sid, user_text, full_reply)

            if end_decision is not None:
                _db_end_call(
                    call_id, end_decision["summary"],
                    end_decision["needs_human"], end_decision["flag_reason"] or None,
                )
                _fire_classification(call_id)
                await broadcaster.broadcast("call_ended", {
                    "call_id": call_id, "call_sid": call_sid,
                    "status": "completed",
                    "needs_human": end_decision["needs_human"],
                    "summary": end_decision["summary"],
                })

        except asyncio.CancelledError:
            partial = clean_for_tts("".join(segments))
            if partial.strip():
                conversation_store.add(call_sid, "assistant", partial)
                _db_log_message(call_id, "assistant", partial, turn_num, was_interrupted=True)
                logger.info("[%s] Assistant (interrupted): %s", call_sid, partial[:200])
            pl = get_session_logger(call_sid)
            if pl:
                pl.log_llm_response("".join(segments) + " [INTERRUPTED]")
                pl.log_final_response(partial + " [INTERRUPTED]")
            raise

        except Exception as exc:
            logger.error("[%s] process_turn error: %s", call_sid, exc)
            pl = get_session_logger(call_sid)
            if pl:
                pl.log_error(f"process_turn error: {exc}", exc)
            await _synthesize_and_queue(settings.fallback_message)
            conversation_store.add(call_sid, "assistant", settings.fallback_message)

        finally:
            bot_speaking = False
            if tts_conn:
                await tts_conn.close()
            if ttfr_ms is not None:
                turn_count += 1
                total_turn_ms += ttfr_ms

    async def _synthesize_and_queue(text: str) -> None:
        """Synthesize text directly via TTS and put audio in out_queue."""
        tts_conn = KyutaiTTS(
            settings.KYUTAI_TTS_URL, settings.KYUTAI_TTS_VOICE, settings.KYUTAI_API_KEY,
        )
        try:
            await tts_conn.connect()
            words = text.split()
            for word in words:
                await tts_conn.send_text(word + " ")
            await tts_conn.send_eos()
            async for chunk in tts_conn.audio_chunks():
                payload = float32_to_mulaw_payload(chunk)
                await out_queue.put(payload)
        except Exception as exc:
            logger.error("[%s] _synthesize_and_queue failed: %s", call_sid, exc)
        finally:
            await tts_conn.close()

    # ── STT event loop ─────────────────────────────────────────────────────────

    async def stt_loop() -> None:
        """
        Read Kyutai STT events (Words + VAD scores).
        When VAD detects a sustained pause, trigger process_turn().
        When VAD detects speech while bot is talking, trigger interruption.
        """
        nonlocal transcript_words, consecutive_pause_frames, bot_speaking
        nonlocal current_task, t_silence_end

        user_has_spoken = False

        async for event in stt.events():
            etype = event["type"]

            if etype == "Word":
                word = event["text"]
                transcript_words.append(word)
                consecutive_pause_frames = 0
                user_has_spoken = True
                logger.debug("[%s] STT word: %s", call_sid, word)

            elif etype == "VAD":
                score = event["score"]

                if score < VAD_THRESHOLD:
                    consecutive_pause_frames = 0
                    continue

                # Score above threshold — user paused or bot is misread
                if bot_speaking and user_has_spoken:
                    # User spoke over bot — interrupt
                    logger.info("[%s] Interruption detected (VAD=%.2f)", call_sid, score)
                    if current_task and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                    await drain_out_queue()
                    await clear_twilio_buffer()
                    bot_speaking = False

                consecutive_pause_frames += 1

                if consecutive_pause_frames >= VAD_PAUSE_FRAMES and transcript_words:
                    # Sustained pause — user finished speaking
                    user_text = " ".join(transcript_words).strip()
                    transcript_words.clear()
                    consecutive_pause_frames = 0
                    user_has_spoken = False
                    t_silence_end = time.perf_counter()

                    logger.info("[%s] User said: %s", call_sid, user_text)

                    if current_task and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass

                    current_task = asyncio.create_task(process_turn(user_text))

    # ── Main Twilio frame loop ─────────────────────────────────────────────────

    writer_task: asyncio.Task | None = None
    stt_task: asyncio.Task | None = None

    try:
        async for raw in websocket.iter_text():
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                logger.debug("Twilio Media Streams connected")

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid") or str(uuid.uuid4())
                caller_from = msg["start"].get("customParameters", {}).get("From", "unknown")

                # DB: create call record
                try:
                    with Session(engine) as s:
                        call = create_call(s, call_sid, caller_from)
                        call_id = call.id
                except Exception as exc:
                    logger.error("[%s] create_call failed: %s", call_sid, exc)

                _voice_sessions[call_sid] = {"call_id": call_id, "phone_number": caller_from}
                _session_log_handler = _open_session_log(call_sid)
                logger.info("[%s] Media Streams started from %s", call_sid, caller_from)

                await broadcaster.broadcast("call_started", {
                    "call_id": call_id, "call_sid": call_sid, "phone_number": caller_from,
                })

                # Connect to Kyutai STT server
                stt = KyutaiSTT(settings.KYUTAI_STT_URL, settings.KYUTAI_API_KEY)
                await stt.connect()

                # Start background tasks
                writer_task = asyncio.create_task(audio_writer())
                stt_task = asyncio.create_task(stt_loop())

                # Play welcome greeting via TTS immediately
                asyncio.create_task(_synthesize_and_queue(settings.welcome_greeting))

            elif event == "media":
                if stt is None:
                    continue
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue
                payload = msg["media"]["payload"]
                try:
                    float32_samples = mulaw_payload_to_float32(payload)
                    await stt.send_audio(float32_samples)
                except Exception as exc:
                    logger.error("[%s] Audio conversion error: %s", call_sid, exc)

            elif event == "stop":
                logger.info("[%s] Twilio stream stopped", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("[%s] Twilio WebSocket disconnected", call_sid)
    except Exception as exc:
        logger.error("[%s] voice_stream error: %s", call_sid, exc)
    finally:
        # Cancel in-flight LLM+TTS task
        if current_task and not current_task.done():
            current_task.cancel()

        # Shut down background tasks
        await out_queue.put(None)  # signal audio_writer to stop
        if writer_task:
            writer_task.cancel()
        if stt_task:
            stt_task.cancel()

        # Close Kyutai STT
        if stt:
            await stt.close()

        # Finalise DB call record if not already ended
        if call_id is not None:
            try:
                with Session(engine) as s:
                    call = get_call_by_id(s, call_id)
                    if call and not call.ended_at:
                        end_call(s, call_id, "Call disconnected", False, None, status="abandoned")
                        logger.info("[%s] Finalised abandoned call", call_sid)
                        _fire_classification(call_id)
                        await broadcaster.broadcast("call_ended", {
                            "call_id": call_id, "call_sid": call_sid,
                            "status": "abandoned", "needs_human": False,
                            "summary": "Call disconnected",
                        })
            except Exception as exc:
                logger.error("[%s] Cleanup end_call failed: %s", call_sid, exc)

        # Avg latency logging
        if turn_count > 0:
            avg_ms = total_turn_ms / turn_count
            logger.info(
                "[%s] Call summary — %d turn(s), avg %.2fs to first reply",
                call_sid, turn_count, avg_ms / 1000,
            )
            if call_id is not None:
                try:
                    with Session(engine) as s:
                        update_avg_turn_ms(s, call_id, avg_ms)
                except Exception as exc:
                    logger.error("[%s] update_avg_turn_ms failed: %s", call_sid, exc)

        # Cleanup session state
        if call_sid:
            _voice_sessions.pop(call_sid, None)
            conversation_store.clear(call_sid)
            if _session_log_handler:
                _close_session_log(call_sid, _session_log_handler)
            close_session_logger(call_sid)
            release_session_client(call_sid)
```

---

## Phase 3: Testing

### 3.1 Test STT server in isolation

```bash
# From the delayed-streams-modeling repo on the GPU server:
uv run scripts/stt_from_mic_rust_server.py
# Speak — should print transcription words in terminal

# Or from a file:
uv run scripts/stt_from_file_rust_server.py audio/bria.mp3
```

### 3.2 Test TTS server in isolation

```bash
echo "Hello, how can I help you?" | python scripts/tts_rust_server.py - -
# Should play audio through speakers
```

### 3.3 Test audio_utils conversion round-trip

Create `test_audio_utils.py` anywhere in the project:

```python
import base64, audioop, numpy as np
from chatbot.audio_utils import mulaw_payload_to_float32, float32_to_mulaw_payload

# Simulate a Twilio frame of silence (160 bytes µ-law)
silence_mulaw = audioop.lin2ulaw(bytes(320), 2)   # 160 samples of PCM16 silence
payload_in = base64.b64encode(silence_mulaw).decode()

float32_arr = mulaw_payload_to_float32(payload_in)
print(f"float32 shape: {float32_arr.shape}")        # expect (60,) approx
print(f"max amplitude: {float32_arr.max():.4f}")    # expect ~0.0

payload_out = float32_to_mulaw_payload(float32_arr)
print(f"output payload length: {len(payload_out)}")  # non-zero string
print("Round-trip OK")
```

```bash
python test_audio_utils.py
```

### 3.4 End-to-end call test

1. Start your FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

2. Expose to the internet (ngrok/zrok):
   ```bash
   ngrok http 8080
   # or: zrok share public http://localhost:8080
   ```

3. Update `BASE_URL` in `.env` to the public URL.

4. Configure Twilio:
   - Voice webhook → `POST https://your-url/voice/inbound`
   - No changes needed to `/voice/action` URL.

5. Call your Twilio number. You should hear the welcome greeting synthesized
   by Kyutai TTS.

### 3.5 Latency benchmarks to expect

| Step | Expected latency |
|------|-----------------|
| STT first word | ~500–800ms after user stops speaking (1B model) |
| RAG retrieval | ~50–100ms (pgvector, unchanged) |
| LLM first token | ~200–500ms (phi4 on GPU, unchanged) |
| TTS first audio | ~150–300ms after first text token |
| **Total time-to-first-audio** | **~1–2s** from end of user speech |

Compare to ConversationRelay baseline: ~600ms–1.2s (Deepgram + ElevenLabs, cloud).

---

## Troubleshooting

### STT server won't start: CUDA error
```
Error: no CUDA device found
```
Solution: ensure `nvidia-smi` works, CUDA 12.1+ is installed, and you built
with `--features cuda`:
```bash
cargo install --features cuda moshi-server
```

### "Expected Ready, got ..." on STT connect
The server is still loading the model. It can take 30–60 seconds on first
start (downloading weights). Check `journalctl` or the server log.

### audioop ImportError on Python 3.13
```
ModuleNotFoundError: No module named 'audioop'
```
```bash
pip install audioop-lts
```
Then replace `import audioop` with `import audioop_lts as audioop` in
`chatbot/audio_utils.py`.

### No audio reaching Twilio (caller hears nothing)
1. Check `out_queue` — add `logger.debug` to `audio_writer` to confirm payloads arrive.
2. Verify `stream_sid` is set before any `send_to_twilio` calls.
3. Check Twilio Media Streams console for any errors.
4. Confirm TwiML uses `track="both_tracks"`.

### Transcription never triggers (VAD never fires)
1. The STT 1B model may need more audio context. Try lowering `KYUTAI_VAD_THRESHOLD`
   to `0.5` or increasing `KYUTAI_VAD_PAUSE_FRAMES` to `5`.
2. Verify inbound audio reaches `stt.send_audio()` — check for conversion errors.
3. Test the STT server directly with `stt_from_file_rust_server.py`.

### Interruption not working (bot keeps talking)
Confirm `track="both_tracks"` in TwiML — without it, Twilio won't send
inbound audio while the bot is speaking. Also verify the `clear` event
is being sent by checking server logs.

### Very high latency (>3s to first audio)
- Check GPU utilisation: `nvidia-smi dmon` — STT/TTS/Ollama may be competing.
- Move STT to a separate GPU or use the 1B model (lower VRAM, faster).
- Reduce `KYUTAI_VAD_PAUSE_FRAMES` to trigger sooner (trade-off: more false triggers).

### "websockets.exceptions.InvalidURI" on connect
Ensure `KYUTAI_STT_URL` starts with `ws://` (not `http://`). The TTS URL
also needs `ws://`.

---

## Quick Reference: Key Environment Variables

| Variable | Example | Purpose |
|----------|---------|---------|
| `KYUTAI_STT_URL` | `ws://192.168.50.150:8090/api/asr-streaming` | STT server WebSocket |
| `KYUTAI_TTS_URL` | `ws://192.168.50.150:8089/api/tts_streaming` | TTS server WebSocket |
| `KYUTAI_TTS_VOICE` | `expresso/ex03-neutral_normal` | Voice from kyutai/tts-voices |
| `KYUTAI_API_KEY` | `public_token` | Auth header (keep as-is for self-hosted) |
| `KYUTAI_VAD_THRESHOLD` | `0.6` | prs[2] score for pause detection |
| `KYUTAI_VAD_PAUSE_FRAMES` | `3` | Consecutive frames to confirm pause |
| `BASE_URL` | `https://abc.ngrok.io` | Public URL (unchanged from before) |

---

## What Is NOT Changed

All of the following are completely unchanged and work exactly as before:

- `chatbot/voice_handler.py` — RAG pipeline, LLM streaming, TTS text cleaning
- `chatbot/llm.py` — Ollama round-robin client
- `chatbot/conversation.py` — in-memory session store
- `chatbot/vector_store.py` — pgvector retrieval
- `chatbot/classifier.py` — end-of-call LLM classifier
- `chatbot/summarizer.py` — post-call summarizer
- `chatbot/prompt_loader.py` / `prompt_defaults.py` — prompt management
- All database models, repositories, and session management
- The `/api/events` broadcaster (SSE dashboard events)
- The `/chat` endpoint (if present)
- Everything in `core/rag/`
