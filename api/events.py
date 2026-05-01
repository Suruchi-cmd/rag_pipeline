"""
Server-Sent Events broadcaster.

One shared EventBroadcaster instance pushes JSON payloads to all connected
frontend clients. Events are fired from the voice router when calls start/end.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["events"])


class EventBroadcaster:
    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[str]] = []

    async def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    async def broadcast(self, event_type: str, data: dict) -> None:
        if not self._queues:
            return
        payload = json.dumps({
            "type": event_type,
            "data": data,
            "ts": datetime.utcnow().isoformat(),
        })
        dead: list[asyncio.Queue[str]] = []
        for q in self._queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)
        if dead:
            logger.debug("Dropped %d slow SSE subscriber(s)", len(dead))


# Singleton used by voice router and the SSE endpoint
broadcaster = EventBroadcaster()


@router.get("/api/events")
async def sse_stream(request: Request):
    """
    SSE stream — clients connect once and receive real-time call events.
    Sends a keepalive comment every 20 s to prevent proxy timeouts.
    """
    async def generate():
        q = await broadcaster.subscribe()
        logger.debug("SSE client connected (%d total)", len(broadcaster._queues))
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            broadcaster.unsubscribe(q)
            logger.debug("SSE client disconnected (%d remaining)", len(broadcaster._queues))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
