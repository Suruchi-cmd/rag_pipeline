from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from chatbot.prompt_loader import invalidate_prompt
from database.repository import (
    add_prompt_version,
    create_prompt,
    delete_prompt,
    delete_prompt_version,
    get_prompt,
    get_prompt_by_slug,
    get_prompt_version,
    list_prompt_versions,
    list_prompts,
    set_active_version,
    update_prompt_meta,
    update_prompt_version,
)
from database.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/prompts", tags=["prompts"])

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class PromptCreate(BaseModel):
    slug: str
    name: str
    description: Optional[str] = None
    content: str


class PromptUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class VersionCreate(BaseModel):
    content: str
    activate: bool = False
    label: Optional[str] = None


class VersionUpdate(BaseModel):
    content: Optional[str] = None
    label: Optional[str] = None


def _serialize_prompt(prompt, versions) -> dict:
    active = next((v for v in versions if v.is_active), None)
    return {
        "id": prompt.id,
        "slug": prompt.slug,
        "name": prompt.name,
        "description": prompt.description,
        "created_at": prompt.created_at,
        "updated_at": prompt.updated_at,
        "active_version_id": active.id if active else None,
        "active_version_no": active.version_no if active else None,
        "active_version_label": active.label if active else None,
        "version_count": len(versions),
    }


@router.get("")
def api_list_prompts(session: Session = Depends(get_session)):
    out = []
    for prompt in list_prompts(session):
        versions = list_prompt_versions(session, prompt.id)
        out.append(_serialize_prompt(prompt, versions))
    return out


@router.get("/{prompt_id}")
def api_get_prompt(prompt_id: int, session: Session = Depends(get_session)):
    prompt = get_prompt(session, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    versions = list_prompt_versions(session, prompt_id)
    payload = _serialize_prompt(prompt, versions)
    payload["versions"] = [v.dict() for v in versions]
    return payload


@router.post("", status_code=201)
def api_create_prompt(body: PromptCreate, session: Session = Depends(get_session)):
    slug = body.slug.strip().lower()
    name = body.name.strip()
    content = body.content
    if not slug or not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=422,
            detail="Slug must be lowercase letters, numbers, underscores or hyphens",
        )
    if not name:
        raise HTTPException(status_code=422, detail="Name is required")
    if not content.strip():
        raise HTTPException(status_code=422, detail="Content is required")
    if get_prompt_by_slug(session, slug) is not None:
        raise HTTPException(status_code=409, detail="Prompt with this slug already exists")
    prompt = create_prompt(session, slug, name, body.description, content)
    versions = list_prompt_versions(session, prompt.id)
    invalidate_prompt(slug)
    return _serialize_prompt(prompt, versions)


@router.put("/{prompt_id}")
def api_update_prompt(
    prompt_id: int,
    body: PromptUpdate,
    session: Session = Depends(get_session),
):
    prompt = get_prompt(session, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    updated = update_prompt_meta(session, prompt_id, body.name, body.description)
    versions = list_prompt_versions(session, prompt_id)
    return _serialize_prompt(updated, versions)


@router.delete("/{prompt_id}", status_code=204)
def api_delete_prompt(prompt_id: int, session: Session = Depends(get_session)):
    prompt = get_prompt(session, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    slug = prompt.slug
    delete_prompt(session, prompt_id)
    invalidate_prompt(slug)


@router.get("/{prompt_id}/versions")
def api_list_versions(prompt_id: int, session: Session = Depends(get_session)):
    if get_prompt(session, prompt_id) is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return list_prompt_versions(session, prompt_id)


@router.post("/{prompt_id}/versions", status_code=201)
def api_create_version(
    prompt_id: int,
    body: VersionCreate,
    session: Session = Depends(get_session),
):
    prompt = get_prompt(session, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if not body.content.strip():
        raise HTTPException(status_code=422, detail="Content is required")
    version = add_prompt_version(
        session,
        prompt_id,
        body.content,
        activate=body.activate,
        label=body.label,
    )
    if version is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if body.activate:
        invalidate_prompt(prompt.slug)
    return version


@router.put("/{prompt_id}/versions/{version_id}")
def api_update_version(
    prompt_id: int,
    version_id: int,
    body: VersionUpdate,
    session: Session = Depends(get_session),
):
    prompt = get_prompt(session, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    version = get_prompt_version(session, version_id)
    if version is None or version.prompt_id != prompt_id:
        raise HTTPException(status_code=404, detail="Version not found")
    fields = body.dict(exclude_unset=True)
    if "content" in fields and not (body.content or "").strip():
        raise HTTPException(status_code=422, detail="Content cannot be empty")
    updated = update_prompt_version(
        session,
        version_id,
        content=body.content if "content" in fields else None,
        label=body.label,
        set_label="label" in fields,
    )
    if updated and updated.is_active:
        invalidate_prompt(prompt.slug)
    return updated


@router.post("/{prompt_id}/versions/{version_id}/activate")
def api_activate_version(
    prompt_id: int,
    version_id: int,
    session: Session = Depends(get_session),
):
    prompt = get_prompt(session, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    version = set_active_version(session, prompt_id, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    invalidate_prompt(prompt.slug)
    return version


@router.delete("/{prompt_id}/versions/{version_id}", status_code=204)
def api_delete_version(
    prompt_id: int,
    version_id: int,
    session: Session = Depends(get_session),
):
    if get_prompt(session, prompt_id) is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    version = get_prompt_version(session, version_id)
    if version is None or version.prompt_id != prompt_id:
        raise HTTPException(status_code=404, detail="Version not found")
    ok, reason = delete_prompt_version(session, version_id)
    if not ok:
        if reason == "active_version":
            raise HTTPException(
                status_code=409,
                detail="Cannot delete the active version — activate another first",
            )
        if reason == "last_version":
            raise HTTPException(
                status_code=409,
                detail="Cannot delete the only version — delete the prompt instead",
            )
        raise HTTPException(status_code=404, detail="Version not found")
