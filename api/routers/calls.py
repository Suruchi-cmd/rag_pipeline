from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlmodel import Session

from database.repository import (
    batch_get_categories,
    delete_call,
    delete_calls,
    get_booking_change,
    get_call_by_id,
    get_call_categories,
    get_call_stats,
    get_messages,
    get_rag_retrievals,
    list_calls,
)
from database.session import get_session

router = APIRouter(prefix="/api/calls", tags=["calls"])


@router.get("/stats")
def api_stats(session: Session = Depends(get_session)):
    return get_call_stats(session)


@router.get("/")
def api_list_calls(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    needs_human: Optional[bool] = Query(None),
    status: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    calls = list_calls(session, offset=offset, limit=limit, needs_human=needs_human, status=status)
    call_ids = [c.id for c in calls if c.id is not None]
    categories_by_call = batch_get_categories(session, call_ids)
    return [
        {**c.model_dump(), "categories": categories_by_call.get(c.id, [])}
        for c in calls
    ]


@router.post("/bulk-delete")
def api_bulk_delete_calls(
    payload: dict = Body(...),
    session: Session = Depends(get_session),
):
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise HTTPException(status_code=400, detail="ids must be a list of integers")
    deleted = delete_calls(session, ids)
    return {"deleted": deleted}


@router.delete("/{call_id}")
def api_delete_call(call_id: int, session: Session = Depends(get_session)):
    if not delete_call(session, call_id):
        raise HTTPException(status_code=404, detail="Call not found")
    return {"deleted": 1}


@router.get("/{call_id}")
def api_get_call(call_id: int, session: Session = Depends(get_session)):
    call = get_call_by_id(session, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    messages = get_messages(session, call_id)
    rag = get_rag_retrievals(session, call_id)
    booking = get_booking_change(session, call_id)
    categories = get_call_categories(session, call_id)
    return {
        "call": call,
        "messages": messages,
        "rag_retrievals": [
            {**r.model_dump(), "retrieved_chunks": json.loads(r.retrieved_chunks)}
            for r in rag
        ],
        "booking_change": booking,
        "categories": categories,
    }


@router.get("/{call_id}/messages")
def api_get_messages(call_id: int, session: Session = Depends(get_session)):
    if get_call_by_id(session, call_id) is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return get_messages(session, call_id)


@router.get("/{call_id}/rag")
def api_get_rag(call_id: int, session: Session = Depends(get_session)):
    if get_call_by_id(session, call_id) is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return [
        {**r.model_dump(), "retrieved_chunks": json.loads(r.retrieved_chunks)}
        for r in get_rag_retrievals(session, call_id)
    ]


@router.get("/{call_id}/booking-change")
def api_get_booking_change(call_id: int, session: Session = Depends(get_session)):
    if get_call_by_id(session, call_id) is None:
        raise HTTPException(status_code=404, detail="Call not found")
    bc = get_booking_change(session, call_id)
    if bc is None:
        raise HTTPException(status_code=404, detail="No booking change for this call")
    return bc
