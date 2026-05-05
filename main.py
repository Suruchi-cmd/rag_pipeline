"""
AeroBot — API + voice server entry point.

Usage
-----
    cd frontend && npm run build   # build SPA into frontend/dist
    python main.py                 # serves API + frontend on port 3232
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.events import router as events_router
from api.routers.calls import router as calls_router
from api.routers.categories import router as categories_router
from api.routers.knowledge import router as knowledge_router
from api.routers.prompts import router as prompts_router
from chatbot.llm import warmup_models
from chatbot.routers.voice import router as voice_router
from chatbot.routers.voice import session_cleanup_loop
from chatbot.vector_store import vector_store
from config import settings
from database.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AeroBot starting…")
    init_db()
    logger.info("Database initialised")
    await asyncio.to_thread(vector_store.initialize)
    # Pre-load Ollama voice models so the first inbound call doesn't pay
    # the cold-start cost. Detached so startup isn't blocked if Ollama is slow.
    asyncio.create_task(warmup_models())
    cleanup_task = asyncio.create_task(session_cleanup_loop())
    yield
    cleanup_task.cancel()
    logger.info("AeroBot stopped")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AeroBot API",
    description="Twilio voice server + frontend API for AeroSports Scarborough",
    version="2.0.0",
    lifespan=lifespan,
)

_cors_origins = settings.cors_origins_list or ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API & voice routes (registered before the SPA catch-all) ──────────────────

app.include_router(voice_router)
app.include_router(calls_router)
app.include_router(categories_router)
app.include_router(knowledge_router)
app.include_router(prompts_router)
app.include_router(events_router)


@app.get("/api/health", tags=["health"])
async def health():
    return {"status": "ok"}


# ── Frontend (SPA) ────────────────────────────────────────────────────────────

_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="assets",
    )

    _INDEX_HTML = _FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "voice/")):
            raise HTTPException(status_code=404)
        candidate = (_FRONTEND_DIST / full_path).resolve()
        if (
            full_path
            and candidate.is_file()
            and _FRONTEND_DIST.resolve() in candidate.parents
        ):
            return FileResponse(candidate)
        return FileResponse(_INDEX_HTML)
else:
    logger.warning("frontend/dist not found — run `npm run build` to enable SPA serving")


# ── Dev runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3232,
        reload=False,
        log_level="info",
    )
