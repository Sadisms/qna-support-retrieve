from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session as SQLAlchemySession

from app.core.config import get_config


config = get_config()
engine = create_engine(config.database_url)

Session = sessionmaker(bind=engine)


def get_db() -> Generator[SQLAlchemySession, None, None]:
    db = Session()
    try:
        yield db
    finally:
        db.close()


from contextlib import contextmanager

@contextmanager
def get_db_context() -> Generator[SQLAlchemySession, None, None]:
    db = Session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
