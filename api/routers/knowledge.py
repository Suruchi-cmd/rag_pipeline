from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from chatbot.vector_store import vector_store
from database.repository import (
    create_chunk,
    delete_chunk,
    get_chunk,
    list_chunks,
    update_chunk,
)
from database.session import engine, get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class ChunkCreate(BaseModel):
    name: str
    content: str


class ChunkUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None


@router.get("/chunks")
def api_list_chunks(session: Session = Depends(get_session)):
    return list_chunks(session)


@router.get("/chunks/{chunk_id}")
def api_get_chunk(chunk_id: int, session: Session = Depends(get_session)):
    chunk = get_chunk(session, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk


@router.post("/chunks", status_code=201)
def api_create_chunk(body: ChunkCreate, session: Session = Depends(get_session)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    if not body.content.strip():
        raise HTTPException(status_code=422, detail="Content cannot be empty")
    existing = [c for c in list_chunks(session) if c.name.lower() == name.lower()]
    if existing:
        raise HTTPException(status_code=409, detail="Chunk with this name already exists")
    return create_chunk(session, name, body.content)


@router.put("/chunks/{chunk_id}")
def api_update_chunk(
    chunk_id: int,
    body: ChunkUpdate,
    session: Session = Depends(get_session),
):
    if get_chunk(session, chunk_id) is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    updated = update_chunk(session, chunk_id, body.name, body.content)
    if updated is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return updated


@router.delete("/chunks/{chunk_id}", status_code=204)
def api_delete_chunk(chunk_id: int, session: Session = Depends(get_session)):
    if get_chunk(session, chunk_id) is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    delete_chunk(session, chunk_id)


@router.post("/resync")
async def api_resync(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    chunks = list_chunks(session)
    background_tasks.add_task(_resync_background)
    return {
        "queued": len(chunks),
        "message": f"Re-vectorizing {len(chunks)} chunks in background",
    }


async def _resync_background() -> None:
    from sqlmodel import Session as DBSession

    with DBSession(engine) as s:
        chunks = list_chunks(s)
    try:
        if not vector_store._ready:
            await asyncio.to_thread(vector_store.initialize)
        await asyncio.to_thread(vector_store.resync, chunks)
    except Exception as exc:
        logger.error("Background resync failed: %s", exc)


@router.get("/health")
def api_knowledge_health():
    return {"ready": vector_store._ready}
