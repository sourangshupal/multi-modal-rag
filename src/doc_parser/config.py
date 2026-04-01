"""Configuration management using pydantic-settings."""
from __future__ import annotations

import logging

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Parser backend
    parser_backend: str = "cloud"  # "cloud" | "ollama"
    z_ai_api_key: SecretStr | None = None
    log_level: str = "INFO"
    output_dir: str = "./output"
    config_yaml_path: str = "config.yaml"

    # OpenAI
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_llm_model: str = "gpt-4o"

    # Embedding (provider-agnostic)
    embedding_provider: str = "openai"  # "openai" | "gemini" | "qwen"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 2048
    gemini_api_key: SecretStr | None = None
    qwen_embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"

    # Reranker
    reranker_backend: str = "qwen"  # "jina" | "openai" | "bge" | "qwen"
    reranker_top_n: int = 5
    jina_api_key: SecretStr | None = None

    # Feature flags
    # When True, enrich_chunks() is called during ingestion:
    #   - image chunks: the region is cropped and stored as image_base64 for direct
    #     visual embedding — no LLM call is made for images.
    #   - table, formula, and algorithm chunks: a text description is generated via
    #     the configured LLM (openai_llm_model) to improve retrieval quality.
    image_caption_enabled: bool = True

    # Captioning tuning
    table_max_tokens: int = 2000
    table_max_input_chars: int = 12_000
    image_max_tokens: int = 800
    table_use_vision: bool = False

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Logging
    log_json: bool = False

    @model_validator(mode="after")
    def _validate_backend(self) -> Settings:
        """Enforce backend-specific constraints and auto-set config path."""
        if self.parser_backend == "cloud":
            if self.z_ai_api_key is None:
                raise ValueError(
                    "Z_AI_API_KEY is required when PARSER_BACKEND=cloud"
                )
        elif self.parser_backend == "ollama":
            if self.config_yaml_path == "config.yaml":
                self.config_yaml_path = "ollama/config.yaml"
        else:
            raise ValueError(
                f"PARSER_BACKEND must be 'cloud' or 'ollama', got: {self.parser_backend!r}"
            )
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with the given level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
