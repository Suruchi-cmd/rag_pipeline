from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from database.repository import (
    create_category,
    delete_category,
    list_calls_for_resync,
    list_categories,
)
from database.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/categories", tags=["categories"])


class CategoryCreate(BaseModel):
    name: str


@router.get("/")
def api_list_categories(session: Session = Depends(get_session)):
    return list_categories(session)


@router.post("/", status_code=201)
def api_create_category(body: CategoryCreate, session: Session = Depends(get_session)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    existing = [c for c in list_categories(session) if c.name.lower() == name.lower()]
    if existing:
        raise HTTPException(status_code=409, detail="Category already exists")
    return create_category(session, name)


@router.delete("/{category_id}", status_code=204)
def api_delete_category(category_id: int, session: Session = Depends(get_session)):
    if not delete_category(session, category_id):
        raise HTTPException(status_code=404, detail="Category not found")


@router.post("/resync")
async def api_resync(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    calls = list_calls_for_resync(session)
    call_ids = [c.id for c in calls]
    background_tasks.add_task(_resync_all, call_ids)
    return {"queued": len(call_ids), "message": f"Reclassifying {len(call_ids)} calls in background"}


async def _resync_all(call_ids: list[int]) -> None:
    from chatbot.classifier import classify_and_store
    for call_id in call_ids:
        try:
            await classify_and_store(call_id)
        except Exception as exc:
            logger.error("Resync failed for call %d: %s", call_id, exc)
