import os

from pydantic import BaseModel


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config(BaseModel):
    ollama_url: str
    ollama_model: str
    database_url: str
    qdrant_url: str
    qdrant_collection_name: str


def get_config() -> Config:
    return Config(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma2:2b"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./qa_support.db"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "qa_support")
    )
