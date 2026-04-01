"""Unit tests for API Pydantic schemas — no API key or network required."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from doc_parser.api.schemas import (
    ChunkResult,
    CollectionsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

# ── SearchRequest ──────────────────────────────────────────────────────────────


def test_search_request_defaults() -> None:
    req = SearchRequest(query="transformer attention")
    assert req.top_k == 20
    assert req.top_n is None
    assert req.filter_modality is None
    assert req.rerank is True


def test_search_request_custom_values() -> None:
    req = SearchRequest(query="bar chart", top_k=5, top_n=3, filter_modality="image", rerank=False)
    assert req.top_k == 5
    assert req.top_n == 3
    assert req.filter_modality == "image"
    assert req.rerank is False


def test_search_request_requires_query() -> None:
    with pytest.raises(ValidationError):
        SearchRequest()  # type: ignore[call-arg]


def test_search_request_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=201)


# ── IngestRequest ──────────────────────────────────────────────────────────────


def test_ingest_request_defaults() -> None:
    req = IngestRequest(file_path="/tmp/test.pdf")
    assert req.collection is None
    assert req.overwrite is False
    assert req.max_chunk_tokens == 512
    assert req.caption is True


def test_ingest_request_custom() -> None:
    req = IngestRequest(file_path="/tmp/doc.pdf", collection="my_col", overwrite=True, max_chunk_tokens=256, caption=False)
    assert req.collection == "my_col"
    assert req.overwrite is True
    assert req.max_chunk_tokens == 256
    assert req.caption is False


def test_ingest_request_token_bounds() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(file_path="/tmp/x.pdf", max_chunk_tokens=32)
    with pytest.raises(ValidationError):
        IngestRequest(file_path="/tmp/x.pdf", max_chunk_tokens=5000)


# ── ChunkResult ────────────────────────────────────────────────────────────────


def test_chunk_result_full() -> None:
    chunk = ChunkResult(
        chunk_id="doc_1_0",
        text="Hello world",
        source_file="doc.pdf",
        page=1,
        modality="text",
        element_types=["paragraph"],
        bbox=[0.1, 0.2, 0.8, 0.5],
        is_atomic=False,
        caption=None,
        rerank_score=8.5,
    )
    assert chunk.image_base64 is None
    assert chunk.rerank_score == 8.5


def test_chunk_result_image() -> None:
    chunk = ChunkResult(
        chunk_id="doc_1_1",
        text="A bar chart",
        source_file="doc.pdf",
        page=2,
        modality="image",
        element_types=["figure"],
        bbox=None,
        is_atomic=True,
        caption="Bar chart showing accuracy",
        rerank_score=None,
        image_base64="abc123==",
    )
    assert chunk.caption == "Bar chart showing accuracy"
    assert chunk.image_base64 == "abc123=="


# ── SearchResponse ─────────────────────────────────────────────────────────────


def test_search_response() -> None:
    resp = SearchResponse(
        query="attention mechanism",
        backend="openai",
        total_candidates=20,
        results=[],
        latency_ms=123.4,
    )
    assert resp.total_candidates == 20
    assert resp.results == []


# ── IngestResponse ─────────────────────────────────────────────────────────────


def test_ingest_response() -> None:
    resp = IngestResponse(
        source_file="/tmp/paper.pdf",
        collection="documents",
        chunks_upserted=42,
        modality_counts={"text": 35, "image": 5, "table": 2},
        latency_ms=4500.1,
    )
    assert resp.chunks_upserted == 42
    assert resp.modality_counts["image"] == 5


# ── HealthResponse ─────────────────────────────────────────────────────────────


def test_health_response_ok() -> None:
    resp = HealthResponse(status="ok", qdrant="ok", openai="ok", reranker_backend="openai")
    assert resp.status == "ok"


def test_health_response_degraded() -> None:
    resp = HealthResponse(
        status="degraded",
        qdrant="error: connection refused",
        openai="ok",
        reranker_backend="bge",
    )
    assert resp.status == "degraded"
    assert "connection refused" in resp.qdrant


# ── CollectionsResponse ────────────────────────────────────────────────────────


def test_collections_response() -> None:
    resp = CollectionsResponse(collections=["documents", "papers"])
    assert len(resp.collections) == 2
    assert "papers" in resp.collections


def test_collections_response_empty() -> None:
    resp = CollectionsResponse(collections=[])
    assert resp.collections == []
