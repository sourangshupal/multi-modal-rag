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
    dimensions: int = 2048,
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

    def __init__(self, settings: "Settings") -> None:
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts, self._client, self._model, self._dimensions)


class GeminiEmbedder(BaseEmbedder):
    """Embedder backed by the Google Gemini embeddings API."""

    _MODEL = "gemini-embedding-2-preview"

    def __init__(self, settings: "Settings") -> None:
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


class QwenVLEmbedder(BaseEmbedder):
    """Embedder backed by Qwen3-VL-Embedding-2B running in-process via HuggingFace.

    Supports both text embedding and direct image embedding (bypassing the caption
    step for image chunks).  Image embedding uses the ``embed_images()`` method, which
    is NOT on the ``BaseEmbedder`` ABC — callers should check ``hasattr(embedder,
    "embed_images")`` before routing image chunks here.

    Requires: ``pip install transformers>=4.51.0 torch>=2.7.0 Pillow``
    (available via ``uv pip install 'doc-parser[qwen]'``)

    Memory footprint: ~8–12 GB (bfloat16 weights).
    Device priority: CUDA > MPS > CPU.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "QwenVLEmbedder requires transformers and torch. "
                "Install with: uv pip install 'doc-parser[qwen]'"
            ) from exc

        self._model_name = model_name
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logger.info("Loading %s on device=%s", model_name, device)

        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model.eval()

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts synchronously using EOS-token pooling.

        Args:
            texts: Input texts to embed.

        Returns:
            L2-normalised float embeddings, one per input text.
        """
        import torch

        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # EOS pooling: use the last token's hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state[:, -1, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().float().tolist()

    def _embed_images_sync(self, images_b64: list[str]) -> list[list[float]]:
        """Embed a list of base64-encoded images synchronously using EOS-token pooling.

        Args:
            images_b64: Base64-encoded PNG/JPEG strings, one per image.

        Returns:
            L2-normalised float embeddings, one per input image.
        """
        import base64
        import io

        import torch
        from PIL import Image

        images = [
            Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            for b64 in images_b64
        ]
        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().float().tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts by offloading the sync worker to a thread-pool executor.

        Args:
            texts: Input texts to embed.

        Returns:
            L2-normalised float embeddings, one per input text.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_texts_sync, texts)

    async def embed_images(self, images_b64: list[str]) -> list[list[float]]:
        """Embed images directly from base64 pixel data (no caption needed).

        This method is NOT on the ``BaseEmbedder`` ABC.  Callers must check
        ``hasattr(embedder, "embed_images")`` before routing image chunks here.

        Args:
            images_b64: Base64-encoded PNG/JPEG strings, one per image.

        Returns:
            L2-normalised float embeddings, one per input image.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_images_sync, images_b64)


_PROVIDERS: dict[str, type[BaseEmbedder]] = {
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
    "qwen": lambda s: QwenVLEmbedder(s.qwen_embedding_model),  # type: ignore[dict-item]
}


def get_embedder(settings: "Settings") -> BaseEmbedder:
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
    return _PROVIDERS[provider](settings)


async def embed_chunks(
    chunks: list["Chunk"],
    embedder: "BaseEmbedder",
    settings: "Settings",
) -> tuple[list[list[float]], list[SparseVector]]:
    """Embed all chunks with both dense and sparse encodings.

    When the embedder supports direct image embedding (i.e. it has an
    ``embed_images`` method — only ``QwenVLEmbedder`` at present), image chunks
    with ``image_base64`` populated are sent to ``embed_images()`` so that pixel
    content drives the vector rather than an AI caption.  All other chunks
    (text, table, formula, algorithm, and image chunks whose embedder does not
    support ``embed_images``) are routed to the standard ``embed()`` path.

    Args:
        chunks: Document chunks (text already populated, including image captions).
        embedder: Configured BaseEmbedder instance.
        settings: Application settings (kept for call-site compatibility).

    Returns:
        Tuple of (dense_embeddings, sparse_vectors), each in chunk order.
        The sparse vectors are always BM25 feature-hash vectors computed from
        ``chunk.text`` (or an empty string for raw-image chunks).
    """
    logger.info("Embedding %d chunks (dense + sparse)", len(chunks))

    dense: list[list[float] | None] = [None] * len(chunks)

    # Route image chunks to embed_images() when the embedder supports it
    image_indices = [
        i
        for i, c in enumerate(chunks)
        if c.modality == "image" and c.image_base64 and hasattr(embedder, "embed_images")
    ]
    text_indices = [i for i in range(len(chunks)) if i not in set(image_indices)]

    # Embed text (and caption-only image) chunks
    if text_indices:
        texts = [chunks[i].text or "[empty]" for i in text_indices]
        text_embeddings = await embedder.embed(texts)
        for i, emb in zip(text_indices, text_embeddings):
            dense[i] = emb

    # Embed image chunks directly via pixel content
    if image_indices:
        images = [chunks[i].image_base64 for i in image_indices]
        image_embeddings = await embedder.embed_images(images)  # type: ignore[attr-defined]
        for i, emb in zip(image_indices, image_embeddings):
            dense[i] = emb

    # Sparse: BM25 from text for all chunks; image chunks with no text get empty vector
    sparse = compute_sparse_vectors([c.text or "" for c in chunks])

    return dense, sparse  # type: ignore[return-value]
