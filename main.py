"""
AeroBot — unified entry point.

Serves the Twilio voice server, the REST API, and the built Vue frontend
all from one process on port 3232.

Usage
-----
    python main.py

Frontend
--------
    cd frontend && npm run build   # build once (or run.sh does it automatically)
    # then just: python main.py
    # dashboard: http://localhost:3232
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.events import router as events_router
from api.routers.calls import router as calls_router
from api.routers.categories import router as categories_router
from chatbot.routers.voice import router as voice_router
from chatbot.routers.voice import session_cleanup_loop
from config import settings
from database.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AeroBot starting…")
    init_db()
    logger.info("Database initialised")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── API & voice routes (registered before the SPA catch-all) ──────────────────

app.include_router(voice_router)
app.include_router(calls_router)
app.include_router(categories_router)
app.include_router(events_router)


@app.get("/api/health", tags=["health"])
async def health():
    return {"status": "ok"}


# ── Frontend static files ──────────────────────────────────────────────────────

_assets_dir = FRONTEND_DIST / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")
    logger.info("Serving frontend from %s", FRONTEND_DIST)
else:
    logger.warning("Frontend not built — run: cd frontend && npm run build")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    f = FRONTEND_DIST / "favicon.ico"
    return FileResponse(f) if f.exists() else JSONResponse({}, status_code=404)


# SPA catch-all — must stay LAST so all /api/* and /voice/* routes match first
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {"detail": "Frontend not built. Run: cd frontend && npm run build"},
        status_code=503,
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3232,
        reload=False,
        log_level="info",
    )
