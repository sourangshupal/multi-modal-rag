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
    """Re-rank using Qwen3-VL-Reranker-2B (local, multimodal, GPU-accelerated).

    Implements the official scoring pattern from
    https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B :

      1. Format ``(query, doc)`` as a chat prompt with a yes/no instruction.
      2. Forward pass through the backbone (``Qwen3VLModel``) to get the last
         hidden state of the final input token.
      3. Project that hidden state through a binary linear layer whose weight
         is ``lm_head[yes_id] - lm_head[no_id]`` to obtain a scalar logit
         equivalent to ``logit("yes") − logit("no")`` from the LM head.
      4. Sigmoid → relevance probability in ``(0, 1)``.

    This is mathematically the same as taking ``softmax([logit_yes, logit_no])[0]``
    over a generative LM's output — the standard generative-reranker score.

    Memory: ~5 GB in bfloat16 (Qwen3-VL-Reranker-2B is a 2B-param model + ViT).
    Latency: ~150–400 ms per (query, doc) pair on an L40S, depending on doc size
    and whether the doc contains an image.
    """

    _MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
    _MAX_LENGTH = 8192
    _MAX_DOC_TEXT_CHARS = 4000  # ~1000 tokens — bounds prompt growth from huge chunks
    _DEFAULT_INSTRUCTION = (
        "Given a search query, retrieve relevant candidates that answer the query."
    )
    _SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    def __init__(self, settings: Settings) -> None:  # noqa: ARG002
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen VL reranker requires transformers>=4.57 and torch. "
                "Install with: uv pip install 'doc-parser[qwen]'"
            ) from exc

        # Device + dtype: prefer CUDA + bf16 (Ampere/Ada/Hopper), fall back to
        # MPS (fp16) on Apple Silicon, then CPU (fp32). bf16 matches the model's
        # native dtype and avoids overflow in the LM head projection.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._dtype = torch.float16
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32
        logger.info(
            "Loading %s on device=%s dtype=%s",
            self._MODEL_NAME,
            self._device,
            self._dtype,
        )

        # Load the full ConditionalGeneration model first so we can copy
        # lm_head weights into the binary score head, then keep only the
        # backbone (lm.model) for inference.
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            self._MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        ).to(self._device)
        lm.eval()

        # padding_side="left" is required by the upstream wrapper: the score is
        # read from the *last* token of each sequence, so left-padding ensures
        # padded positions never become the "last token".
        self._processor = AutoProcessor.from_pretrained(
            self._MODEL_NAME,
            trust_remote_code=True,
            padding_side="left",
        )

        # Build the binary linear scorer: weight = lm_head[yes_id] - lm_head[no_id].
        # Applied to the backbone's last-token hidden state, this yields a scalar
        # equal to (logit_yes - logit_no), whose sigmoid is P(yes | input).
        token_yes_id = self._processor.tokenizer.get_vocab()["yes"]
        token_no_id = self._processor.tokenizer.get_vocab()["no"]
        lm_head_w = lm.lm_head.weight.data  # [vocab, hidden]
        weight_diff = lm_head_w[token_yes_id] - lm_head_w[token_no_id]  # [hidden]
        hidden_size = weight_diff.shape[0]
        self._score_linear = torch.nn.Linear(hidden_size, 1, bias=False)
        with torch.no_grad():
            self._score_linear.weight[0] = weight_diff
        self._score_linear = self._score_linear.to(self._device, dtype=self._dtype)
        self._score_linear.eval()

        # Drop the LM head — we don't need it for inference and freeing the
        # reference lets the allocator reclaim ~600 MB of vocab projection.
        self._backbone = lm.model
        self._backbone.eval()
        del lm
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_messages(
        self,
        query: str,
        candidate: dict[str, Any],
        instruction: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Build the chat-message structure for one (query, doc) pair.

        Mirrors the official ``format_mm_instruction`` from the model card:

            system: "Judge whether the Document meets the requirements ... yes / no."
            user:   <Instruct>: {instruction}
                    <Query>: {query_text}
                    \\n<Document>: [optional image] {doc_text}

        Returns the message list (for ``apply_chat_template``) and the list of
        PIL images embedded in the doc, which the processor needs as ``images=``.
        """
        import base64
        import io

        from PIL import Image

        instruct = instruction or self._DEFAULT_INSTRUCTION
        text = (candidate.get("text") or "").strip()
        image_b64 = candidate.get("image_base64")
        modality = candidate.get("modality", "text")

        # Document content: text + optional image. Order matches upstream
        # (prefix label, then image, then text body).
        doc_content: list[dict[str, Any]] = [{"type": "text", "text": "\n<Document>:"}]
        doc_images: list[Any] = []
        if modality == "image" and image_b64:
            pil = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            doc_content.append({"type": "image", "image": pil})
            doc_images.append(pil)
        if text:
            doc_content.append({"type": "text", "text": text[: self._MAX_DOC_TEXT_CHARS]})
        if not text and not doc_images:
            doc_content.append({"type": "text", "text": "NULL"})

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<Instruct>: " + instruct},
                    {"type": "text", "text": "<Query>:"},
                    {"type": "text", "text": query or "NULL"},
                    *doc_content,
                ],
            },
        ]
        return messages, doc_images

    def _score_one_sync(self, query: str, candidate: dict[str, Any]) -> float:
        """Score a single (query, candidate) pair synchronously on the GPU."""
        import torch

        messages, images = self._build_messages(query, candidate)

        # apply_chat_template renders the message structure into the model's
        # tokenizer-ready text, substituting <|vision_start|><|image_pad|><|vision_end|>
        # for each {"type": "image", ...} block. add_generation_prompt=True appends
        # the assistant turn header so the *next* token is where we read the score.
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._MAX_LENGTH,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._backbone(**inputs)
            last_hidden = outputs.last_hidden_state[:, -1, :]  # [1, hidden]
            logit = self._score_linear(last_hidden)  # [1, 1]
            score = torch.sigmoid(logit).squeeze(-1)  # [1]

        return float(score.float().cpu().item())

    def _score_all_sync(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[float]:
        """Score every candidate sequentially in a single thread-pool task.

        Per-pair forward passes (matching upstream) avoid the variable-size
        image-batching headaches; running them in one executor call avoids
        ``asyncio.gather`` spawning N threads that would all serialize at the
        single CUDA stream and just thrash the GIL.
        """
        return [self._score_one_sync(query, c) for c in candidates]

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Score candidates via Qwen3-VL-Reranker and return the top-n."""
        if not candidates:
            return []
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, self._score_all_sync, query, candidates
        )
        scored = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores, strict=True)
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
