import os

from pydantic import BaseModel


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config(BaseModel):
    # LLM settings
    llm_provider: str = "ollama"  # "ollama" or "openai"
    
    # Ollama settings
    ollama_url: str
    ollama_model: str
    
    # OpenAI settings
    openai_api_key: str | None = None
    openai_model: str = "gpt-4"
    openai_proxy_url: str | None = None
    
    # Other settings
    database_url: str
    qdrant_url: str
    qdrant_collection_name: str
    api_token: str
    workspace_tokens: dict[str, str] = {}


def get_config() -> Config:
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise ValueError("API_TOKEN environment variable is required")
    
    workspace_tokens = {}
    workspace_tokens_env = os.getenv("WORKSPACE_TOKENS", "")
    if workspace_tokens_env:
        for pair in workspace_tokens_env.split(","):
            if ":" in pair:
                ws_id, token = pair.strip().split(":", 1)
                workspace_tokens[ws_id] = token
    
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    # Проверяем, что если выбран OpenAI, то есть API ключ
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if llm_provider == "openai" and not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required when LLM_PROVIDER=openai")
    
    return Config(
        llm_provider=llm_provider,
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma2:2b"),
        openai_api_key=openai_api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
        openai_proxy_url=os.getenv("OPENAI_PROXY_URL"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./qa_support.db"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "qa_support"),
        api_token=api_token,
        workspace_tokens=workspace_tokens
    )
