import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# Allow `python src/main.py` (script mode) in addition to `python -m src.main`
# and uvicorn "src.main:app". When run as a script, sys.path[0] is `src/`
# itself, so `from src.config import ...` fails — prepend its parent.
if __package__ in (None, ""):
    _PKG_PARENT = str(Path(__file__).resolve().parent.parent)
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from llama_index.core import SimpleDirectoryReader
from sqlalchemy import text
from sqlmodel import SQLModel, create_engine

from src.config import settings
from src.history.routes import history_router
from src.llm_config.dependencies import get_llm_config_service
from src.llm_config.routes import llm_config_router
from src.rag.repositories import RAGRepository
from src.rag.routes import rag_router
from src.schemas import APIInfoResponse, HealthCheckResponse


def setup_logging():
    """Configure logging to save to timestamped files."""
    log_dir = Path("logs/server_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


logger = setup_logging()


def wait_for_database(
    max_attempts: int | None = None, delay: float | None = None
) -> None:
    """Poll the DB until it accepts connections (covers pgvector container boot)."""
    max_attempts = max_attempts if max_attempts is not None else settings.DB_WAIT_ATTEMPTS
    delay = delay if delay is not None else settings.DB_WAIT_DELAY
    engine = create_engine(settings.database_url, echo=False)
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database reachable (attempt %d)", attempt)
            return
        except Exception as e:
            last_err = e
            logger.info(
                "Database not ready (attempt %d/%d): %s", attempt, max_attempts, e
            )
            time.sleep(delay)
    raise RuntimeError(f"Database unreachable after {max_attempts} attempts: {last_err}")


def ensure_pgvector_extension() -> None:
    engine = create_engine(settings.database_url, echo=False)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector extension ensured")


def create_app_tables() -> None:
    """Create SQLModel tables (history + llm_config). pgvector table is
    handled by PGVectorStore lazily on first write."""
    # Import models so SQLModel.metadata knows about them
    from src.history.models import QueryHistory, SourceDocumentHistory  # noqa: F401
    from src.llm_config.models import LLMConfig  # noqa: F401

    engine = create_engine(settings.database_url, echo=False)
    SQLModel.metadata.create_all(engine)
    logger.info("Application tables ensured")


def maybe_auto_ingest(rag_repo: RAGRepository) -> None:
    """If vector table empty and the enriched-data folder has markdown, ingest it.

    DATA_FOLDER points at `parser/output/enriched/` — the voicebot/RAG-optimised
    output of the enrichment pipeline. If it is empty the operator must run the
    parser with `--enrich` (or POST /rag/resync) before queries can return hits.
    """
    if not settings.AUTO_INGEST:
        logger.info("AUTO_INGEST disabled; skipping bootstrap ingest")
        return

    if rag_repo.get_document_count() > 0:
        logger.info("Vector store non-empty; skipping bootstrap ingest")
        return

    data_folder = Path(settings.DATA_FOLDER)
    md_files = list(data_folder.rglob("*.md")) if data_folder.exists() else []
    if not md_files:
        logger.warning(
            "No enriched markdown files in %s; skipping bootstrap ingest. "
            "Run `python run_parsers.py --enrich` (or POST /rag/resync) to generate them.",
            data_folder,
        )
        return

    logger.info(
        "Bootstrap ingest: %d enriched markdown files from %s",
        len(md_files),
        data_folder,
    )
    try:
        documents = SimpleDirectoryReader(
            input_files=[str(f) for f in md_files]
        ).load_data()
        ok = rag_repo.index_documents(documents)
        logger.info("Bootstrap ingest %s", "succeeded" if ok else "failed")
    except Exception as e:
        logger.error("Bootstrap ingest errored: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Ultimate Advisor API bootstrap...")

    wait_for_database()
    ensure_pgvector_extension()
    create_app_tables()

    llm_config_service = get_llm_config_service()
    llm_config = llm_config_service.bootstrap()
    logger.info(
        "LLMConfig active: %s @ %s", llm_config.chat_model, llm_config.ollama_base_url
    )

    rag_repo = RAGRepository(llm_config=llm_config)
    app.state.rag_repo = rag_repo
    logger.info("RAGRepository initialized and attached to app.state")

    maybe_auto_ingest(rag_repo)

    logger.info("Bootstrap complete — API ready")
    yield
    logger.info("Ultimate Advisor API shutting down")


app = FastAPI(
    title="Ultimate Advisor API",
    description="A RAG-based chat system with runtime-configurable LLM",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Response: {response.status_code} for {request.method} {request.url.path} "
        f"- {process_time:.3f}s"
    )
    return response


app.include_router(history_router)
app.include_router(rag_router)
app.include_router(llm_config_router)


@app.get("/files/download/{filename}")
async def download_file(filename: str):
    if not re.match(r"^[a-zA-Z0-9._-]+$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = Path("data") / filename

    try:
        file_path = file_path.resolve()
        data_dir = Path("data").resolve()
        file_path.relative_to(data_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied") from None

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"Serving file download: {filename}")
    return FileResponse(
        path=file_path, filename=filename, media_type="application/octet-stream"
    )


static_path = Path("files")
if static_path.exists() and static_path.is_dir():
    app.mount("/files", StaticFiles(directory="files"), name="files")
    logger.info("Static files mounted at /files")


frontend_dist_path = Path("frontend/dist")
if frontend_dist_path.exists() and frontend_dist_path.is_dir():
    app.mount(
        "/assets", StaticFiles(directory=frontend_dist_path / "assets"), name="assets"
    )
    logger.info("Frontend assets mounted at /assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if (
            full_path.startswith("api/")
            or full_path.startswith("docs")
            or full_path.startswith("redoc")
            or full_path.startswith("files/")
        ):
            raise HTTPException(status_code=404, detail="Not found")

        file_path = frontend_dist_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        index_file = frontend_dist_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend not found")

    logger.info("SPA routing configured for frontend")
else:
    logger.warning("Frontend dist directory not found - frontend will not be served")


@app.exception_handler(500)
async def internal_server_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": str(exc) if app.debug else None,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found.",
            "path": str(request.url.path),
        },
    )


@app.get("/", response_model=APIInfoResponse)
async def root() -> APIInfoResponse:
    return APIInfoResponse(
        message="Welcome to Ultimate Advisor API",
        description="A RAG-based chat system with runtime-configurable LLM",
        version="1.1.0",
        endpoints={
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "rag": "/rag",
            "resync": "/rag/resync",
            "resync_raw": "/rag/resync-raw",
            "history": "/history",
            "settings": "/settings/llm",
        },
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(
        status="healthy", service="Ultimate Advisor API", version="1.1.0"
    )


if __name__ == "__main__":
    import uvicorn

    # Reload mode needs an import string rather than the app object so the
    # reloader subprocess can re-import on file changes.
    uvicorn.run(
        "src.main:app" if settings.RELOAD else app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1,
        log_level=settings.LOG_LEVEL,
    )
