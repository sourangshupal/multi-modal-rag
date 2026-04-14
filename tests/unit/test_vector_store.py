"""Unit tests for QdrantDocumentStore (mocked Qdrant client)."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ── fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class _FakeChunk:
    text: str = "hello world"
    chunk_id: str = "doc.pdf_1_0"
    page: int = 1
    element_types: list = field(default_factory=lambda: ["paragraph"])
    bbox: list | None = None
    source_file: str = "doc.pdf"
    is_atomic: bool = False
    modality: str = "text"
    image_base64: str | None = None
    caption: str | None = None


def _make_settings(collection: str = "test_col") -> MagicMock:
    settings = MagicMock()
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_api_key = None
    settings.qdrant_collection_name = collection
    settings.embedding_dimensions = 4
    settings.embedding_model = "text-embedding-3-large"
    return settings


def _make_sparse_vector(indices: list[int] | None = None, values: list[float] | None = None):
    from qdrant_client.models import SparseVector

    return SparseVector(
        indices=indices or [1, 2],
        values=values or [0.5, 0.5],
    )


# ── create_collection ─────────────────────────────────────────────────────────


class TestCreateCollection:
    @pytest.mark.asyncio
    async def test_creates_collection_when_not_exists(self):
        """create_collection should call create_collection on the client."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings(collection="test_col")
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = "test_col"

        mock_client = AsyncMock()
        other_col = MagicMock()
        other_col.name = "other_col"
        collections_response = MagicMock()
        collections_response.collections = [other_col]
        mock_client.get_collections = AsyncMock(return_value=collections_response)
        mock_client.create_collection = AsyncMock()
        store._client = mock_client

        await store.create_collection(overwrite=False)
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_creation_when_collection_exists_no_overwrite(self):
        """create_collection with overwrite=False must not recreate existing collection."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings(collection="existing_col")
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = "existing_col"

        mock_client = AsyncMock()
        existing = MagicMock()
        existing.name = "existing_col"
        collections_response = MagicMock()
        collections_response.collections = [existing]
        mock_client.get_collections = AsyncMock(return_value=collections_response)
        mock_client.create_collection = AsyncMock()
        store._client = mock_client

        await store.create_collection(overwrite=False)
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_overwrites_existing_collection_when_flag_set(self):
        """create_collection with overwrite=True must delete then recreate."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings(collection="existing_col")
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = "existing_col"

        mock_client = AsyncMock()
        existing = MagicMock()
        existing.name = "existing_col"
        collections_response = MagicMock()
        collections_response.collections = [existing]
        mock_client.get_collections = AsyncMock(return_value=collections_response)
        mock_client.delete_collection = AsyncMock()
        mock_client.create_collection = AsyncMock()
        store._client = mock_client

        await store.create_collection(overwrite=True)
        mock_client.delete_collection.assert_called_once_with("existing_col")
        mock_client.create_collection.assert_called_once()


# ── upsert_chunks ─────────────────────────────────────────────────────────────


class TestUpsertChunks:
    @pytest.mark.asyncio
    async def test_upserts_all_chunks_in_one_batch(self):
        """Five chunks with batch_size=10 → single upsert call."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings()
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = settings.qdrant_collection_name

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        store._client = mock_client

        chunks = [_FakeChunk(chunk_id=f"doc.pdf_1_{i}") for i in range(5)]
        dense = [[0.1, 0.2, 0.3, 0.4]] * 5
        sparse = [_make_sparse_vector()] * 5

        count = await store.upsert_chunks(chunks, dense, sparse, batch_size=10)
        assert count == 5
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upserts_in_multiple_batches(self):
        """Seven chunks with batch_size=3 → ceil(7/3)=3 upsert calls."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings()
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = settings.qdrant_collection_name

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        store._client = mock_client

        chunks = [_FakeChunk(chunk_id=f"doc.pdf_1_{i}") for i in range(7)]
        dense = [[0.1, 0.2, 0.3, 0.4]] * 7
        sparse = [_make_sparse_vector()] * 7

        count = await store.upsert_chunks(chunks, dense, sparse, batch_size=3)
        assert count == 7
        assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_on_length_mismatch(self):
        """upsert_chunks must raise ValueError when lists have different lengths."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings()
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = settings.qdrant_collection_name
        store._client = AsyncMock()

        chunks = [_FakeChunk()]
        dense = [[0.1, 0.2]]  # length mismatch
        sparse = [_make_sparse_vector(), _make_sparse_vector()]  # length mismatch

        with pytest.raises(ValueError, match="Length mismatch"):
            await store.upsert_chunks(chunks, dense, sparse)

    @pytest.mark.asyncio
    async def test_payload_includes_all_fields(self):
        """The upserted point payload must include all schema fields."""
        from doc_parser.ingestion.vector_store import QdrantDocumentStore

        settings = _make_settings()
        store = QdrantDocumentStore.__new__(QdrantDocumentStore)
        store._settings = settings
        store._collection = settings.qdrant_collection_name

        captured_calls: list = []

        async def fake_upsert(collection_name, points):
            captured_calls.extend(points)

        mock_client = AsyncMock()
        mock_client.upsert = fake_upsert
        store._client = mock_client

        chunk = _FakeChunk(text="test text", modality="text")
        await store.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], [_make_sparse_vector()])

        assert len(captured_calls) == 1
        payload = captured_calls[0].payload
        required_keys = {
            "text", "chunk_id", "source_file", "page",
            "element_types", "bbox", "is_atomic", "modality",
            "image_base64", "caption",
        }
        assert required_keys.issubset(payload.keys())
        assert payload["modality"] == "text"
