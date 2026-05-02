import logging
from datetime import datetime

from sqlmodel import Session, SQLModel, create_engine, select

logger = logging.getLogger(__name__)

engine = create_engine(
    "sqlite:///aerobot.db",
    echo=False,
    connect_args={"check_same_thread": False},
)

_DEFAULT_CATEGORIES = [
    "Birthday Parties",
    "GO Karting",
    "Bazooka Ball",
    "Trampoline",
    "General Inquiries",
]


def init_db() -> None:
    from database.models import (  # noqa: F401
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
    SQLModel.metadata.create_all(engine)
    _ensure_prompt_version_label_column()
    _seed_categories()
    _seed_prompts()


def _ensure_prompt_version_label_column() -> None:
    with engine.connect() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(prompt_versions)").fetchall()
        names = {row[1] for row in cols}
        if "label" not in names:
            conn.exec_driver_sql("ALTER TABLE prompt_versions ADD COLUMN label VARCHAR")
            conn.commit()
            logger.info("Added prompt_versions.label column")


def _seed_categories() -> None:
    from database.models import Category
    with Session(engine) as session:
        if session.exec(select(Category)).first() is not None:
            return
        for name in _DEFAULT_CATEGORIES:
            session.add(Category(name=name, created_at=datetime.utcnow()))
        session.commit()
        logger.info("Seeded %d default categories", len(_DEFAULT_CATEGORIES))


def _seed_prompts() -> None:
    from chatbot.prompt_defaults import DEFAULT_PROMPTS
    from database.models import Prompt, PromptVersion

    with Session(engine) as session:
        existing = {p.slug for p in session.exec(select(Prompt)).all()}
        now = datetime.utcnow()
        seeded = 0
        for spec in DEFAULT_PROMPTS:
            if spec["slug"] in existing:
                continue
            prompt = Prompt(
                slug=spec["slug"],
                name=spec["name"],
                description=spec.get("description"),
                created_at=now,
                updated_at=now,
            )
            session.add(prompt)
            session.commit()
            session.refresh(prompt)
            session.add(
                PromptVersion(
                    prompt_id=prompt.id,
                    version_no=1,
                    content=spec["content"],
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
            seeded += 1
        if seeded:
            logger.info("Seeded %d default prompts", seeded)


def get_session():
    with Session(engine) as session:
        yield session
