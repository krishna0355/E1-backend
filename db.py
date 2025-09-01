# backend/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
import os

# Place DB file beside this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'e1.db')}"

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=False,       # set True if you want verbose SQL logs
    future=True
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,
)

# Base model class
Base = declarative_base()

# Context manager for session usage
@contextmanager
def db_session():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
