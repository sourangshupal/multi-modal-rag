"""GET /health and GET /collections endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from loguru import logger
from openai import AsyncOpenAI

from doc_parser.api.dependencies import get_openai_client, get_store
from doc_parser.api.schemas import CollectionsResponse, DeleteCollectionResponse, HealthResponse
from doc_parser.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Ping Qdrant and OpenAI to verify connectivity."""
    settings = get_settings()
    store = get_store()
    client: AsyncOpenAI = get_openai_client()
    reranker_backend = settings.reranker_backend

    # Check Qdrant
    qdrant_status: str
    try:
        await store._client.get_collections()
        qdrant_status = "ok"
    except Exception as exc:
        logger.warning("Qdrant health check failed: {}", exc)
        qdrant_status = f"error: {exc}"

    # Check OpenAI embeddings connectivity (skipped for non-OpenAI providers)
    openai_status: str
    if settings.embedding_provider.lower() == "openai":
        try:
            await client.embeddings.create(
                model=settings.embedding_model,
                input=["ping"],
                dimensions=8,
            )
            openai_status = "ok"
        except Exception as exc:
            logger.warning("OpenAI health check failed: {}", exc)
            openai_status = f"error: {exc}"
    else:
        openai_status = f"skipped (embedding_provider={settings.embedding_provider})"

    overall = "ok" if qdrant_status == "ok" and openai_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        openai=openai_status,
        reranker_backend=reranker_backend,
    )


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections() -> CollectionsResponse:
    """List all Qdrant collection names."""
    store = get_store()
    response = await store._client.get_collections()
    names = [c.name for c in response.collections]
    return CollectionsResponse(collections=names)


@router.delete("/collections/{collection_name}", response_model=DeleteCollectionResponse)
async def delete_collection(collection_name: str) -> DeleteCollectionResponse:
    """Permanently delete a Qdrant collection by name.

    This is irreversible — re-ingestion is required to rebuild.
    Returns 200 with deleted=False if the collection does not exist.
    """
    store = get_store()
    deleted = await store.delete_collection(collection_name)
    return DeleteCollectionResponse(
        collection=collection_name,
        deleted=deleted,
        message="Collection deleted." if deleted else "Collection not found.",
    )
