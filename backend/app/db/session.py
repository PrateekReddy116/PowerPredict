from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, future=True)
    return _SessionLocal


def reset_engine() -> None:
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_db() -> Generator[Session, None, None]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()
