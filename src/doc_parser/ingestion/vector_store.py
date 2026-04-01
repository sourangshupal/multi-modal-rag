"""Qdrant vector store wrapper for hybrid dense + sparse document retrieval."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from doc_parser.ingestion.embedder import BaseEmbedder, compute_sparse_vectors

if TYPE_CHECKING:
    from doc_parser.chunker import Chunk
    from doc_parser.config import Settings

logger = logging.getLogger(__name__)


class QdrantDocumentStore:
    """Async wrapper around Qdrant for hybrid-search document ingestion and retrieval.

    Uses two named vector spaces:
    - ``multimodal_dense``: dense embeddings for both text and image chunks (COSINE distance)
    - ``bm25_sparse``: BM25 sparse vectors from fastembed
    """

    def __init__(self, settings: "Settings") -> None:
        """Initialise the store from application settings.

        Args:
            settings: Application settings containing Qdrant URL, API key, and
                collection name.
        """
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )
        self._client = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)
        self._collection = settings.qdrant_collection_name
        self._settings = settings

    async def create_collection(self, overwrite: bool = False) -> None:
        """Create the Qdrant collection with hybrid vector config.

        If the collection already exists and ``overwrite`` is False, this is a
        no-op. If ``overwrite`` is True, the existing collection is deleted first.

        Args:
            overwrite: When True, delete and recreate the collection.
        """
        response = await self._client.get_collections()
        existing = {c.name for c in response.collections}

        if self._collection in existing:
            if not overwrite:
                logger.info("Collection '%s' already exists — skipping creation", self._collection)
                return
            logger.info("Deleting existing collection '%s'", self._collection)
            await self._client.delete_collection(self._collection)

        logger.info("Creating collection '%s'", self._collection)
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "multimodal_dense": VectorParams(
                    size=self._settings.embedding_dimensions,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "bm25_sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection by name.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            True if the collection existed and was deleted, False if not found.
        """
        response = await self._client.get_collections()
        existing = {c.name for c in response.collections}
        if collection_name not in existing:
            return False
        await self._client.delete_collection(collection_name)
        logger.info("Deleted collection '%s'", collection_name)
        return True

    async def upsert_chunks(
        self,
        chunks: list["Chunk"],
        dense_embeddings: list[list[float]],
        sparse_vectors: list[SparseVector],
        batch_size: int = 64,
    ) -> int:
        """Upsert chunk embeddings into Qdrant in batches.

        Args:
            chunks: Document chunks (same order as embeddings).
            dense_embeddings: Dense float vectors, one per chunk.
            sparse_vectors: BM25 sparse vectors, one per chunk.
            batch_size: Number of points per upsert request.

        Returns:
            Total number of points upserted.
        """
        if len(chunks) != len(dense_embeddings) or len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"Length mismatch: chunks={len(chunks)}, "
                f"dense={len(dense_embeddings)}, sparse={len(sparse_vectors)}"
            )

        points: list[PointStruct] = []
        for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_vectors):
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "element_types": chunk.element_types,
                "bbox": chunk.bbox,
                "is_atomic": chunk.is_atomic,
                "modality": chunk.modality,
                "image_base64": chunk.image_base64,
                "caption": chunk.caption,
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector={"multimodal_dense": dense, "bm25_sparse": sparse},
                    payload=payload,
                )
            )

        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )
            total += len(batch)
            logger.debug("Upserted batch %d–%d (%d total)", i, i + len(batch), total)

        logger.info("Upserted %d points to collection '%s'", total, self._collection)
        return total

    async def search(
        self,
        query_text: str,
        embedder: "BaseEmbedder",
        settings: "Settings",
        top_k: int = 10,
        filter_modality: str | None = None,
    ) -> list[dict]:
        """Hybrid dense + sparse search with RRF fusion.

        Args:
            query_text: Natural-language query string.
            embedder: Configured BaseEmbedder for query embedding.
            settings: Application settings.
            top_k: Number of results to return.
            filter_modality: If set, restrict results to this modality
                ("text", "image", "table", "formula").

        Returns:
            List of payload dicts from matching points, ordered by relevance.
        """
        # Embed the query using the configured provider
        query_dense = (await embedder.embed([query_text]))[0]
        query_sparse = compute_sparse_vectors([query_text])[0]

        query_filter = None
        if filter_modality is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            query_filter = Filter(
                must=[FieldCondition(key="modality", match=MatchValue(value=filter_modality))]
            )

        results = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=query_dense, using="multimodal_dense", limit=top_k * 2),
                Prefetch(query=query_sparse, using="bm25_sparse", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        return [point.payload for point in results.points]
