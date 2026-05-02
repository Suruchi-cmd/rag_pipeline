from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlmodel import Session, func, select

from database.models import (
    BookingChange,
    Call,
    CallClassification,
    Category,
    KnowledgeChunk,
    Message,
    Prompt,
    PromptVersion,
    RAGRetrieval,
)

logger = logging.getLogger(__name__)


# ── Calls ─────────────────────────────────────────────────────────────────────

def create_call(session: Session, call_sid: str, phone_number: str) -> Call:
    call = Call(call_sid=call_sid, phone_number=phone_number, started_at=datetime.utcnow())
    session.add(call)
    session.commit()
    session.refresh(call)
    return call


def get_call_by_id(session: Session, call_id: int) -> Optional[Call]:
    return session.get(Call, call_id)


def get_call_by_sid(session: Session, call_sid: str) -> Optional[Call]:
    return session.exec(select(Call).where(Call.call_sid == call_sid)).first()


def end_call(
    session: Session,
    call_id: int,
    summary: str,
    needs_human: bool,
    flag_reason: Optional[str],
    status: str = "completed",
) -> None:
    call = session.get(Call, call_id)
    if call is None:
        return
    call.ended_at = datetime.utcnow()
    call.summary = summary
    call.needs_human = needs_human or call.needs_human
    call.flag_reason = flag_reason or call.flag_reason
    call.status = status
    call.total_turns = session.exec(
        select(func.count(Message.id))
        .where(Message.call_id == call_id, Message.role == "user")
    ).one()
    session.add(call)
    session.commit()


def update_avg_turn_ms(session: Session, call_id: int, avg_turn_ms: float) -> None:
    call = session.get(Call, call_id)
    if call is None:
        return
    call.avg_turn_ms = avg_turn_ms
    session.add(call)
    session.commit()


def list_calls(
    session: Session,
    offset: int = 0,
    limit: int = 50,
    needs_human: Optional[bool] = None,
    status: Optional[str] = None,
) -> list[Call]:
    stmt = select(Call).order_by(Call.started_at.desc()).offset(offset).limit(limit)
    if needs_human is not None:
        stmt = stmt.where(Call.needs_human == needs_human)
    if status is not None:
        stmt = stmt.where(Call.status == status)
    return list(session.exec(stmt).all())


def get_call_stats(session: Session) -> dict:
    total = session.exec(select(func.count(Call.id))).one()
    flagged = session.exec(
        select(func.count(Call.id)).where(Call.needs_human == True)  # noqa: E712
    ).one()
    active = session.exec(
        select(func.count(Call.id)).where(Call.status == "active")
    ).one()
    return {"total_calls": total, "needs_human": flagged, "active_calls": active}


# ── Messages ──────────────────────────────────────────────────────────────────

def add_message(
    session: Session,
    call_id: int,
    role: str,
    content: str,
    turn_number: int,
    was_interrupted: bool = False,
) -> Message:
    msg = Message(
        call_id=call_id,
        role=role,
        content=content,
        turn_number=turn_number,
        was_interrupted=was_interrupted,
        created_at=datetime.utcnow(),
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return msg


def get_messages(session: Session, call_id: int) -> list[Message]:
    return list(
        session.exec(
            select(Message)
            .where(Message.call_id == call_id)
            .order_by(Message.turn_number, Message.id)
        ).all()
    )


# ── RAG Retrievals ────────────────────────────────────────────────────────────

def add_rag_retrieval(
    session: Session,
    call_id: int,
    turn_number: int,
    original_query: str,
    rewritten_query: str,
    retrieved_chunks: list[dict],
    was_skipped: bool = False,
) -> RAGRetrieval:
    row = RAGRetrieval(
        call_id=call_id,
        turn_number=turn_number,
        original_query=original_query,
        rewritten_query=rewritten_query,
        retrieved_chunks=json.dumps(retrieved_chunks),
        was_skipped=was_skipped,
        created_at=datetime.utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_rag_retrievals(session: Session, call_id: int) -> list[RAGRetrieval]:
    return list(
        session.exec(
            select(RAGRetrieval)
            .where(RAGRetrieval.call_id == call_id)
            .order_by(RAGRetrieval.turn_number)
        ).all()
    )


# ── Booking Changes ───────────────────────────────────────────────────────────

def save_booking_change(
    session: Session,
    call_id: int,
    caller_name: str,
    caller_phone: str,
    change_details: str,
) -> BookingChange:
    bc = BookingChange(
        call_id=call_id,
        caller_name=caller_name,
        caller_phone=caller_phone,
        change_details=change_details,
        created_at=datetime.utcnow(),
    )
    session.add(bc)
    call = session.get(Call, call_id)
    if call:
        call.needs_human = True
        call.flag_reason = call.flag_reason or "booking change requested"
        session.add(call)
    session.commit()
    session.refresh(bc)
    return bc


def get_booking_change(session: Session, call_id: int) -> Optional[BookingChange]:
    return session.exec(
        select(BookingChange).where(BookingChange.call_id == call_id)
    ).first()


# ── Categories ────────────────────────────────────────────────────────────────

def list_categories(session: Session) -> list[Category]:
    return list(session.exec(select(Category).order_by(Category.name)).all())


def create_category(session: Session, name: str) -> Category:
    cat = Category(name=name.strip(), created_at=datetime.utcnow())
    session.add(cat)
    session.commit()
    session.refresh(cat)
    return cat


def delete_category(session: Session, category_id: int) -> bool:
    cat = session.get(Category, category_id)
    if cat is None:
        return False
    # Remove classifications that reference this category
    for row in session.exec(
        select(CallClassification).where(CallClassification.category_id == category_id)
    ).all():
        session.delete(row)
    session.delete(cat)
    session.commit()
    return True


# ── Call classifications ───────────────────────────────────────────────────────

def upsert_call_classifications(
    session: Session, call_id: int, category_ids: list[int]
) -> None:
    for row in session.exec(
        select(CallClassification).where(CallClassification.call_id == call_id)
    ).all():
        session.delete(row)
    for cat_id in category_ids:
        session.add(CallClassification(call_id=call_id, category_id=cat_id))
    session.commit()


def get_call_categories(session: Session, call_id: int) -> list[str]:
    rows = session.exec(
        select(CallClassification).where(CallClassification.call_id == call_id)
    ).all()
    if not rows:
        return []
    cat_ids = [r.category_id for r in rows]
    cats = session.exec(select(Category).where(Category.id.in_(cat_ids))).all()  # type: ignore[attr-defined]
    return sorted(c.name for c in cats)


def batch_get_categories(session: Session, call_ids: list[int]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {cid: [] for cid in call_ids}
    if not call_ids:
        return result
    classifications = session.exec(
        select(CallClassification).where(CallClassification.call_id.in_(call_ids))  # type: ignore[attr-defined]
    ).all()
    if not classifications:
        return result
    cat_ids = list({c.category_id for c in classifications})
    cats = session.exec(select(Category).where(Category.id.in_(cat_ids))).all()  # type: ignore[attr-defined]
    cat_map = {c.id: c.name for c in cats}
    for clf in classifications:
        if clf.call_id in result and clf.category_id in cat_map:
            result[clf.call_id].append(cat_map[clf.category_id])
    for cid in result:
        result[cid].sort()
    return result


def list_calls_for_resync(session: Session) -> list[Call]:
    return list(
        session.exec(
            select(Call).where(Call.status.in_(["completed", "abandoned"]))  # type: ignore[attr-defined]
        ).all()
    )


# ── Knowledge Chunks ───────────────────────────────────────────────────────────

def list_chunks(session: Session) -> list[KnowledgeChunk]:
    return list(session.exec(select(KnowledgeChunk).order_by(KnowledgeChunk.name)).all())


def get_chunk(session: Session, chunk_id: int) -> Optional[KnowledgeChunk]:
    return session.get(KnowledgeChunk, chunk_id)


def create_chunk(session: Session, name: str, content: str) -> KnowledgeChunk:
    now = datetime.utcnow()
    chunk = KnowledgeChunk(name=name.strip(), content=content, created_at=now, updated_at=now)
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk


def update_chunk(
    session: Session, chunk_id: int, name: Optional[str], content: Optional[str]
) -> Optional[KnowledgeChunk]:
    chunk = session.get(KnowledgeChunk, chunk_id)
    if chunk is None:
        return None
    if name is not None:
        chunk.name = name.strip()
    if content is not None:
        chunk.content = content
    chunk.updated_at = datetime.utcnow()
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk


def delete_chunk(session: Session, chunk_id: int) -> bool:
    chunk = session.get(KnowledgeChunk, chunk_id)
    if chunk is None:
        return False
    session.delete(chunk)
    session.commit()
    return True


# ── Prompts ───────────────────────────────────────────────────────────────────

def list_prompts(session: Session) -> list[Prompt]:
    return list(session.exec(select(Prompt).order_by(Prompt.id)).all())


def get_prompt(session: Session, prompt_id: int) -> Optional[Prompt]:
    return session.get(Prompt, prompt_id)


def get_prompt_by_slug(session: Session, slug: str) -> Optional[Prompt]:
    return session.exec(select(Prompt).where(Prompt.slug == slug)).first()


def create_prompt(
    session: Session,
    slug: str,
    name: str,
    description: Optional[str],
    initial_content: str,
) -> Prompt:
    now = datetime.utcnow()
    prompt = Prompt(
        slug=slug.strip(),
        name=name.strip(),
        description=(description or None),
        created_at=now,
        updated_at=now,
    )
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    version = PromptVersion(
        prompt_id=prompt.id,
        version_no=1,
        content=initial_content,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(version)
    session.commit()
    return prompt


def update_prompt_meta(
    session: Session,
    prompt_id: int,
    name: Optional[str],
    description: Optional[str],
) -> Optional[Prompt]:
    prompt = session.get(Prompt, prompt_id)
    if prompt is None:
        return None
    if name is not None:
        prompt.name = name.strip()
    if description is not None:
        prompt.description = description or None
    prompt.updated_at = datetime.utcnow()
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt


def delete_prompt(session: Session, prompt_id: int) -> bool:
    prompt = session.get(Prompt, prompt_id)
    if prompt is None:
        return False
    for v in session.exec(
        select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
    ).all():
        session.delete(v)
    session.delete(prompt)
    session.commit()
    return True


def list_prompt_versions(session: Session, prompt_id: int) -> list[PromptVersion]:
    return list(
        session.exec(
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt_id)
            .order_by(PromptVersion.version_no.desc())  # type: ignore[attr-defined]
        ).all()
    )


def get_prompt_version(session: Session, version_id: int) -> Optional[PromptVersion]:
    return session.get(PromptVersion, version_id)


def get_active_version(session: Session, prompt_id: int) -> Optional[PromptVersion]:
    return session.exec(
        select(PromptVersion)
        .where(PromptVersion.prompt_id == prompt_id, PromptVersion.is_active == True)  # noqa: E712
    ).first()


def get_active_prompt_content(session: Session, slug: str) -> Optional[str]:
    prompt = get_prompt_by_slug(session, slug)
    if prompt is None:
        return None
    version = get_active_version(session, prompt.id)
    return version.content if version else None


def add_prompt_version(
    session: Session,
    prompt_id: int,
    content: str,
    activate: bool = False,
    label: Optional[str] = None,
) -> Optional[PromptVersion]:
    prompt = session.get(Prompt, prompt_id)
    if prompt is None:
        return None
    last = session.exec(
        select(PromptVersion)
        .where(PromptVersion.prompt_id == prompt_id)
        .order_by(PromptVersion.version_no.desc())  # type: ignore[attr-defined]
    ).first()
    next_no = (last.version_no + 1) if last else 1
    now = datetime.utcnow()
    version = PromptVersion(
        prompt_id=prompt_id,
        version_no=next_no,
        label=(label.strip() or None) if label else None,
        content=content,
        is_active=False,
        created_at=now,
        updated_at=now,
    )
    session.add(version)
    session.commit()
    session.refresh(version)
    if activate:
        set_active_version(session, prompt_id, version.id)
        session.refresh(version)
    prompt.updated_at = now
    session.add(prompt)
    session.commit()
    return version


def update_prompt_version(
    session: Session,
    version_id: int,
    content: Optional[str] = None,
    label: Optional[str] = None,
    set_label: bool = False,
) -> Optional[PromptVersion]:
    version = session.get(PromptVersion, version_id)
    if version is None:
        return None
    if content is not None:
        version.content = content
    if set_label:
        version.label = (label.strip() or None) if label else None
    version.updated_at = datetime.utcnow()
    session.add(version)
    prompt = session.get(Prompt, version.prompt_id)
    if prompt is not None:
        prompt.updated_at = version.updated_at
        session.add(prompt)
    session.commit()
    session.refresh(version)
    return version


def set_active_version(
    session: Session,
    prompt_id: int,
    version_id: int,
) -> Optional[PromptVersion]:
    target = session.get(PromptVersion, version_id)
    if target is None or target.prompt_id != prompt_id:
        return None
    for v in session.exec(
        select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
    ).all():
        new_state = v.id == version_id
        if v.is_active != new_state:
            v.is_active = new_state
            session.add(v)
    prompt = session.get(Prompt, prompt_id)
    if prompt is not None:
        prompt.updated_at = datetime.utcnow()
        session.add(prompt)
    session.commit()
    session.refresh(target)
    return target


def delete_prompt_version(
    session: Session,
    version_id: int,
) -> tuple[bool, str]:
    version = session.get(PromptVersion, version_id)
    if version is None:
        return False, "not_found"
    if version.is_active:
        return False, "active_version"
    remaining = session.exec(
        select(func.count(PromptVersion.id)).where(
            PromptVersion.prompt_id == version.prompt_id
        )
    ).one()
    if remaining <= 1:
        return False, "last_version"
    session.delete(version)
    session.commit()
    return True, "ok"
