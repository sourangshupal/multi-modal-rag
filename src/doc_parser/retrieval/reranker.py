"""Re-ranker backends for post-retrieval relevance scoring.

Pipeline position:
    QdrantDocumentStore.search() → top-k candidates
        ↓
    BaseReranker.rerank(query, candidates)  ← this module
        ↓
    LLM generation

Supported backends (controlled by ``RERANKER_BACKEND`` env var):
    - ``openai``  – GPT-4o-mini as async cross-encoder (default, no extra deps)
    - ``jina``    – Jina Reranker M0 cloud API (multimodal, needs JINA_API_KEY)
    - ``bge``     – BAAI/bge-reranker-v2-minicpm-layerwise (local, fast, text-only)
    - ``qwen``    – Qwen3-VL-Reranker-2B (local, multimodal, heavier)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from doc_parser.config import Settings

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseReranker(ABC):
    """Abstract re-ranker interface.

    All concrete backends must implement :meth:`rerank`.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank *candidates* against *query*, returning the top-n most relevant.

        Args:
            query: The user's natural-language query.
            candidates: Payload dicts returned by ``QdrantDocumentStore.search()``.
                Each dict contains at minimum ``"text"``, ``"modality"``, and
                optionally ``"image_base64"`` for image chunks.
            top_n: Maximum number of results to return (highest score first).

        Returns:
            Up to *top_n* candidate dicts, sorted by relevance (best first).
            Each dict is the original payload extended with a ``"rerank_score"`` key.
        """


# ── OpenAI backend ────────────────────────────────────────────────────────────


class OpenAIReranker(BaseReranker):
    """Re-rank using GPT-4o-mini as an async cross-encoder.

    Scores each (query, chunk) pair via a short prompt, firing all candidates
    in parallel with ``asyncio.gather``.  Image chunks pass ``image_base64``
    inline as a vision message.  Text-only chunks use a text-only message.

    Cost: ~$0.03–0.10 per re-rank call (20 candidates).
    Latency: ~800ms–2s (parallel async).
    """

    _SCORE_PROMPT = (
        "Rate the relevance of the following document to the query on a scale of 1 to 10. "
        "Reply with ONLY the integer score (e.g. '7'), nothing else.\n\n"
        "Query: {query}\n\nDocument: {text}"
    )

    def __init__(self, settings: Settings) -> None:
        api_key = (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )
        self._client = AsyncOpenAI(api_key=api_key, base_url=settings.openai_base_url)
        self._model = "gpt-4o-mini"

    async def _score_one(self, query: str, candidate: dict[str, Any]) -> float:
        """Return a relevance score in [1, 10] for one candidate."""
        text = candidate.get("text") or ""
        image_b64 = candidate.get("image_base64")
        modality = candidate.get("modality", "text")

        if modality == "image" and image_b64:
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Rate the relevance of the following image (and its caption) "
                                f"to the query on a scale of 1 to 10. "
                                f"Reply with ONLY the integer score.\n\nQuery: {query}"
                                + (f"\n\nCaption: {text}" if text else "")
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ]
        else:
            prompt = self._SCORE_PROMPT.format(query=query, text=text[:2000])
            messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=4,
            )
            raw = (response.choices[0].message.content or "").strip()
            return float(raw)
        except (ValueError, IndexError):
            logger.warning("Could not parse score from OpenAI response: %r", raw)
            return 0.0
        except Exception as exc:
            logger.error("OpenAI scoring failed for chunk: %s", exc)
            return 0.0

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Score all candidates in parallel then return the top-n."""
        scores = await asyncio.gather(
            *[self._score_one(query, c) for c in candidates]
        )
        scored = [
            {**c, "rerank_score": score} for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]


# ── Jina backend ──────────────────────────────────────────────────────────────


class JinaReranker(BaseReranker):
    """Re-rank using the Jina Reranker M0 cloud API (multimodal).

    Accepts text + base64 images in a single API call.  Image chunks pass
    ``image_base64`` directly; text chunks send text only.

    Requires ``JINA_API_KEY`` in environment.
    Cost: ~$0.01–0.02 per re-rank call.
    Latency: ~500ms–2s.
    """

    _API_URL = "https://api.jina.ai/v1/rerank"
    _MODEL = "jina-reranker-m0"

    def __init__(self, settings: Settings) -> None:
        if settings.jina_api_key is None:
            raise ValueError(
                "JINA_API_KEY must be set when RERANKER_BACKEND=jina. "
                "Sign up at https://jina.ai to get a free API key."
            )
        self._api_key = settings.jina_api_key.get_secret_value()

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Call Jina rerank API and return re-ordered candidates."""
        documents: list[dict[str, Any]] = []
        for c in candidates:
            text = c.get("text") or ""
            image_b64 = c.get("image_base64")
            modality = c.get("modality", "text")

            if modality == "image" and image_b64:
                documents.append({"text": text, "images": [image_b64]})
            else:
                documents.append({"text": text})

        payload = {
            "model": self._MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self._API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        reranked: list[dict[str, Any]] = []
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            reranked.append({**candidates[idx], "rerank_score": score})

        return reranked


# ── BGE backend ───────────────────────────────────────────────────────────────


class BGEReranker(BaseReranker):
    """Re-rank using BAAI/bge-reranker-v2-minicpm-layerwise (local, text-only).

    Extremely fast on Apple Silicon CPU (~50–100ms for 20 candidates).
    Relies on captions as image representations — no raw pixel input.
    Runs synchronously in a thread-pool to avoid blocking the event loop.

    Requires: ``pip install FlagEmbedding>=1.3.0``
    Cost: free (local).
    """

    _MODEL_NAME = "BAAI/bge-reranker-v2-minicpm-layerwise"
    _CUTOFF_LAYERS = [28]

    def __init__(self, settings: Settings) -> None:  # noqa: ARG002
        try:
            from FlagEmbedding import LayerWiseFlagLLMReranker  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "BGE reranker requires FlagEmbedding. "
                "Install with: uv pip install 'doc-parser[bge]'"
            ) from exc

        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading BGE reranker on device=%s", device)
        self._reranker = LayerWiseFlagLLMReranker(
            self._MODEL_NAME,
            use_fp16=True,
            device=device,
        )

    def _compute_scores_sync(self, pairs: list[list[str]]) -> list[float]:
        """Run the synchronous BGE scorer (called in thread pool)."""
        return self._reranker.compute_score(pairs, cutoff_layers=self._CUTOFF_LAYERS)

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Score candidates via BGE (offloaded to thread pool) and return top-n."""
        pairs = [[query, (c.get("text") or "")[:2000]] for c in candidates]

        loop = asyncio.get_running_loop()
        scores: list[float] = await loop.run_in_executor(
            None, self._compute_scores_sync, pairs
        )

        scored = [
            {**c, "rerank_score": float(score)} for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]


# ── Qwen VL backend ───────────────────────────────────────────────────────────


class QwenVLReranker(BaseReranker):
    """Re-rank using Qwen3-VL-Reranker-2B (local, multimodal).

    Ranked #1 on MMEB-V2.  Can consume ``image_base64`` directly.
    Requires ~8–12 GB RAM.  Uses Apple Silicon MPS if available.

    Requires: ``pip install transformers>=4.51.0 torch>=2.7.0``
    Cost: free (local).
    Latency: ~400–800ms on Apple Silicon MPS; ~1–2s on CPU.
    """

    _MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"

    def __init__(self, settings: Settings) -> None:  # noqa: ARG002
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen VL reranker requires transformers and torch. "
                "Install with: uv pip install 'doc-parser[qwen]'"
            ) from exc

        import torch

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading Qwen3-VL-Reranker-2B on device=%s", self._device)

        self._processor = AutoProcessor.from_pretrained(self._MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._MODEL_NAME,
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
        ).to(self._device)
        self._model.eval()

    def _score_one_sync(self, query: str, candidate: dict[str, Any]) -> float:
        """Score a single (query, candidate) pair synchronously."""
        import base64
        import io

        import torch
        from PIL import Image

        text = candidate.get("text") or ""
        image_b64 = candidate.get("image_base64")
        modality = candidate.get("modality", "text")

        if modality == "image" and image_b64:
            img_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            inputs = self._processor(
                text=[[query, text]],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)
        else:
            inputs = self._processor(
                text=[[query, text[:2000]]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
        return float(logits[0].item())

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Score candidates via Qwen VL (offloaded to thread pool) and return top-n."""
        loop = asyncio.get_running_loop()

        async def score_one(c: dict[str, Any]) -> float:
            return await loop.run_in_executor(None, self._score_one_sync, query, c)

        scores = await asyncio.gather(*[score_one(c) for c in candidates])
        scored = [
            {**c, "rerank_score": float(score)} for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]


# ── Factory ───────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[BaseReranker]] = {
    "openai": OpenAIReranker,
    "jina": JinaReranker,
    "bge": BGEReranker,
    "qwen": QwenVLReranker,
}


def get_reranker(settings: Settings) -> BaseReranker:
    """Instantiate and return the configured re-ranker backend.

    Args:
        settings: Application settings.  ``settings.reranker_backend`` must be
            one of ``"openai"``, ``"jina"``, ``"bge"``, or ``"qwen"``.

    Returns:
        A :class:`BaseReranker` instance ready to call :meth:`~BaseReranker.rerank`.

    Raises:
        ValueError: If ``reranker_backend`` is not a recognised backend name.
    """
    backend = settings.reranker_backend.lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown reranker backend: {backend!r}. "
            f"Choose from: {list(_BACKENDS)}"
        )
    logger.info("Initialising reranker backend: %s", backend)
    return _BACKENDS[backend](settings)
