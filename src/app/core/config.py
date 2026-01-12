from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    APP_ENV: str = "local"

    # Qdrant
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_VECTOR_DIM: int = 1024

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    # Memory / context control
    REDIS_TTL_SECONDS: int = 60 * 60 * 24 * 7  
    MEMORY_MAX_MESSAGES: int = 12  
    MEMORY_SUMMARY_MAX_CHARS: int = 2000  

    # LLM / embeddings
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL_NAME: str = "gpt-4.1-mini"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Retrieval
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_DENSE_K: int = 25 
    RETRIEVAL_LEXICAL_K: int = 25 

    # Which LLM backend to use: "openai" or "ollama"
    LLM_PROVIDER: str = "ollama"

    # Ollama specific settings
    OLLAMA_BASE_URL: str = "https://ollama.com/api"
    OLLAMA_API_KEY: str = "1e6bb9b1d55b4e99b6438e1d3dca0410.hGkY_0jDkg7BEZ7TfuRJALin"
    OLLAMA_MODEL_NAME: str = "gpt-oss:120b"

settings = Settings()
