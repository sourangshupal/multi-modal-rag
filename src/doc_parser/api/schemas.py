"""Pydantic request/response models for the doc-parser RAG API."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(..., description="Natural-language query string.")
    top_k: int = Field(20, ge=1, le=200, description="Candidate count from Qdrant.")
    top_n: int | None = Field(
        None, ge=1, description="Results after reranking (defaults to settings.reranker_top_n)."
    )
    filter_modality: str | None = Field(
        None, description='Restrict to modality: "text" | "image" | "table" | "formula".'
    )
    rerank: bool = Field(True, description="If False, return raw Qdrant results without reranking.")


class IngestRequest(BaseModel):
    """Request body for POST /ingest (JSON path-based variant)."""

    file_path: str = Field(..., description="Absolute or relative path to the document file (PDF or image).")
    collection: str | None = Field(None, description="Override collection name from settings.")
    overwrite: bool = Field(False, description="If True, recreate the collection before ingesting.")
    max_chunk_tokens: int = Field(512, ge=64, le=4096, description="Max tokens per text chunk.")
    caption: bool = Field(True, description="Run GPT-4o captioning on image chunks.")


# ── Response models ────────────────────────────────────────────────────────────


class ChunkResult(BaseModel):
    """A single retrieved and (optionally) reranked document chunk."""

    chunk_id: str
    text: str
    source_file: str
    page: int
    modality: str
    element_types: list[str]
    bbox: list[float] | None
    is_atomic: bool
    caption: str | None
    rerank_score: float | None
    image_base64: str | None = None


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str
    backend: str
    total_candidates: int
    results: list[ChunkResult]
    latency_ms: float


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    source_file: str
    collection: str
    chunks_upserted: int
    modality_counts: dict[str, int]
    latency_ms: float


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    qdrant: str
    openai: str
    reranker_backend: str


class CollectionsResponse(BaseModel):
    """Response body for GET /collections."""

    collections: list[str]


class GenerateRequest(BaseModel):
    """Request body for POST /generate."""

    query: str = Field(..., description="Natural-language question to answer.")
    top_k: int = Field(20, ge=1, le=200, description="Candidate count from Qdrant.")
    top_n: int | None = Field(None, ge=1, description="Context chunks after reranking.")
    filter_modality: str | None = Field(
        None, description='"text"|"image"|"table"|"formula" or null for all.'
    )
    rerank: bool = Field(True, description="If False, use raw Qdrant results.")
    system_prompt: str | None = Field(None, description="Override default RAG system prompt.")
    max_tokens: int = Field(1024, ge=64, le=4096, description="Max tokens in LLM response.")


class GenerateResponse(BaseModel):
    """Response body for POST /generate."""

    query: str
    answer: str
    sources: list[ChunkResult]
    total_candidates: int
    latency_ms: float


class DeleteCollectionResponse(BaseModel):
    """Response body for DELETE /collections/{name}."""

    collection: str
    deleted: bool
    message: str
