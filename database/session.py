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
        BookingChange, Call, CallClassification, Category, Message, RAGRetrieval,
    )
    SQLModel.metadata.create_all(engine)
    _seed_categories()


def _seed_categories() -> None:
    from database.models import Category
    with Session(engine) as session:
        if session.exec(select(Category)).first() is not None:
            return
        for name in _DEFAULT_CATEGORIES:
            session.add(Category(name=name, created_at=datetime.utcnow()))
        session.commit()
        logger.info("Seeded %d default categories", len(_DEFAULT_CATEGORIES))


def get_session():
    with Session(engine) as session:
        yield session
