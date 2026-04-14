"""Embedder: dense text embeddings (OpenAI/Gemini) + BM25 sparse vectors (feature hashing)."""
from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from qdrant_client.models import SparseVector

if TYPE_CHECKING:
    from doc_parser.chunker import Chunk
    from doc_parser.config import Settings

logger = logging.getLogger(__name__)

# Number of hash buckets for the sparse feature space.
# 2^17 = 131072 — large enough to minimise collisions, small enough for Qdrant.
_BM25_N_FEATURES: int = 2**17


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser (alphanumeric tokens only).

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


async def embed_texts(
    texts: list[str],
    client: AsyncOpenAI,
    model: str = "text-embedding-3-large",
    dimensions: int = 3072,
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed texts using the OpenAI embeddings API.

    Empty strings are replaced with "[empty]" because the API rejects blank inputs.
    Results are returned in the same order as the input list.

    Args:
        texts: Input texts to embed.
        client: Authenticated AsyncOpenAI client.
        model: Embedding model name.
        dimensions: Output dimensionality (SDK v2 supports truncation).
        batch_size: Maximum texts per API request (OpenAI limit is 2048 inputs).

    Returns:
        List of float vectors, one per input text, in input order.
    """
    # Sanitise: OpenAI rejects empty strings
    sanitised = [t if t.strip() else "[empty]" for t in texts]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(sanitised), batch_size):
        batch = sanitised[i : i + batch_size]
        response = await client.embeddings.create(
            model=model,
            input=batch,
            dimensions=dimensions,
        )
        # API guarantees order is preserved
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings


def compute_sparse_vectors(
    texts: list[str],
    n_features: int = _BM25_N_FEATURES,
) -> list[SparseVector]:
    """Compute TF-weighted sparse vectors using the feature-hashing trick.

    Maps each unique term to a stable integer bucket via Python's built-in hash
    function, using normalised term frequency as the weight. This produces
    consistent indices for both documents and queries without requiring a
    pre-built vocabulary, making it suitable for streaming ingestion.

    Qdrant's ``bm25_sparse`` vector space accepts arbitrary sparse vectors, so
    feature-hashed TF vectors serve as a lightweight BM25 proxy that supports
    keyword-boosted retrieval via RRF fusion.

    Args:
        texts: Input texts to encode.
        n_features: Hash space size (default 2^17 ≈ 131k buckets).

    Returns:
        List of SparseVector objects (sorted by index) ready for Qdrant upsert.
    """
    vectors: list[SparseVector] = []

    for text in texts:
        tokens = _tokenize(text)
        if not tokens:
            vectors.append(SparseVector(indices=[], values=[]))
            continue

        tf = Counter(tokens)
        total_terms = len(tokens)

        # Map terms to hash buckets; last-write wins on rare collisions
        bucket_weights: dict[int, float] = {}
        for term, count in tf.items():
            idx = abs(hash(term)) % n_features
            bucket_weights[idx] = count / total_terms  # normalised TF

        # Sort by index (Qdrant expects sorted sparse vectors)
        sorted_items = sorted(bucket_weights.items())
        indices = [i for i, _ in sorted_items]
        values = [v for _, v in sorted_items]

        vectors.append(SparseVector(indices=indices, values=values))

    return vectors


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one float vector per text, in input order."""


class OpenAIEmbedder(BaseEmbedder):
    """Embedder backed by the OpenAI embeddings API."""

    def __init__(self, settings: Settings) -> None:
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts, self._client, self._model, self._dimensions)


class GeminiEmbedder(BaseEmbedder):
    """Embedder backed by the Google Gemini embeddings API."""

    _MODEL = "gemini-embedding-2-preview"

    def __init__(self, settings: Settings) -> None:
        if settings.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY must be set when EMBEDDING_PROVIDER=gemini.")
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required. Install with: uv pip install 'doc-parser[gemini]'"
            ) from exc
        self._client = _genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        sanitised = [t if t.strip() else "[empty]" for t in texts]
        result = self._client.models.embed_content(model=self._MODEL, contents=sanitised)
        return [e.values for e in result.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)


_PROVIDERS: dict[str, type[BaseEmbedder]] = {"openai": OpenAIEmbedder, "gemini": GeminiEmbedder}


def get_embedder(settings: Settings) -> BaseEmbedder:
    """Return the configured embedder instance.

    Args:
        settings: Application settings containing provider and model config.

    Returns:
        A BaseEmbedder instance for the configured provider.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider = settings.embedding_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider: {provider!r}. Choose from: {list(_PROVIDERS)}"
        )
    logger.info("Initialising embedding provider: %s", provider)
    return _PROVIDERS[provider](settings)  # type: ignore[call-arg]


async def embed_chunks(
    chunks: list[Chunk],
    embedder: BaseEmbedder,
    settings: Settings,
) -> tuple[list[list[float]], list[SparseVector]]:
    """Embed all chunks with both dense and sparse encodings.

    Args:
        chunks: Document chunks (text already populated, including image captions).
        embedder: Configured BaseEmbedder instance.
        settings: Application settings (kept for call-site compatibility).

    Returns:
        Tuple of (dense_embeddings, sparse_vectors), each in chunk order.
    """
    texts = [c.text for c in chunks]

    logger.info("Embedding %d chunks (dense + sparse)", len(chunks))

    dense = await embedder.embed(texts)
    sparse = compute_sparse_vectors(texts)

    return dense, sparse
