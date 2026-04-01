"""Unit tests for re-ranker backends (all external calls mocked)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_settings(
    backend: str = "openai",
    top_n: int = 5,
    openai_key: str = "sk-test",
    jina_key: str | None = None,
) -> MagicMock:
    settings = MagicMock()
    settings.reranker_backend = backend
    settings.reranker_top_n = top_n
    # openai_api_key is a SecretStr-like mock
    openai_secret = MagicMock()
    openai_secret.get_secret_value.return_value = openai_key
    settings.openai_api_key = openai_secret
    settings.openai_base_url = None
    # jina_api_key
    if jina_key is not None:
        jina_secret = MagicMock()
        jina_secret.get_secret_value.return_value = jina_key
        settings.jina_api_key = jina_secret
    else:
        settings.jina_api_key = None
    return settings


def _make_candidates(n: int = 3) -> list[dict]:
    return [
        {
            "text": f"This is document {i} about transformers.",
            "chunk_id": f"doc_{i}",
            "modality": "text",
            "image_base64": None,
            "source_file": "paper.pdf",
            "page": i,
        }
        for i in range(n)
    ]


def _make_image_candidate() -> dict:
    return {
        "text": "A bar chart showing accuracy vs recall.",
        "chunk_id": "doc_img",
        "modality": "image",
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "source_file": "paper.pdf",
        "page": 5,
    }


# ── get_reranker factory ──────────────────────────────────────────────────────


class TestGetReranker:
    def test_returns_openai_reranker(self):
        from doc_parser.retrieval.reranker import OpenAIReranker, get_reranker

        settings = _make_settings(backend="openai")
        reranker = get_reranker(settings)
        assert isinstance(reranker, OpenAIReranker)

    def test_returns_jina_reranker(self):
        from doc_parser.retrieval.reranker import JinaReranker, get_reranker

        settings = _make_settings(backend="jina", jina_key="jina_test_key")
        reranker = get_reranker(settings)
        assert isinstance(reranker, JinaReranker)

    def test_raises_on_unknown_backend(self):
        from doc_parser.retrieval.reranker import get_reranker

        settings = _make_settings(backend="unknown_backend")
        with pytest.raises(ValueError, match="Unknown reranker backend"):
            get_reranker(settings)

    def test_jina_raises_without_api_key(self):
        from doc_parser.retrieval.reranker import JinaReranker

        settings = _make_settings(backend="jina")  # jina_key=None
        with pytest.raises(ValueError, match="JINA_API_KEY"):
            JinaReranker(settings)


# ── OpenAI backend ────────────────────────────────────────────────────────────


class TestOpenAIReranker:
    @pytest.mark.asyncio
    async def test_rerank_returns_top_n(self):
        """rerank() should return at most top_n items."""
        from doc_parser.retrieval.reranker import OpenAIReranker

        settings = _make_settings(backend="openai")

        mock_choice = MagicMock()
        mock_choice.message.content = "8"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("doc_parser.retrieval.reranker.AsyncOpenAI") as mock_cls:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_openai

            reranker = OpenAIReranker(settings)
            candidates = _make_candidates(n=6)
            result = await reranker.rerank("attention mechanism", candidates, top_n=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_adds_score_field(self):
        """Each result must include 'rerank_score'."""
        from doc_parser.retrieval.reranker import OpenAIReranker

        settings = _make_settings(backend="openai")

        mock_choice = MagicMock()
        mock_choice.message.content = "7"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("doc_parser.retrieval.reranker.AsyncOpenAI") as mock_cls:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_openai

            reranker = OpenAIReranker(settings)
            candidates = _make_candidates(n=2)
            result = await reranker.rerank("query", candidates, top_n=5)

        for item in result:
            assert "rerank_score" in item
            assert item["rerank_score"] == 7.0

    @pytest.mark.asyncio
    async def test_rerank_sorted_descending(self):
        """Results must be sorted highest score first."""
        from doc_parser.retrieval.reranker import OpenAIReranker

        settings = _make_settings(backend="openai")
        scores = ["3", "9", "5"]

        mock_responses = []
        for s in scores:
            choice = MagicMock()
            choice.message.content = s
            resp = MagicMock()
            resp.choices = [choice]
            mock_responses.append(resp)

        with patch("doc_parser.retrieval.reranker.AsyncOpenAI") as mock_cls:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(side_effect=mock_responses)
            mock_cls.return_value = mock_openai

            reranker = OpenAIReranker(settings)
            candidates = _make_candidates(n=3)
            result = await reranker.rerank("query", candidates, top_n=3)

        rerank_scores = [r["rerank_score"] for r in result]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_image_chunk_uses_vision_message(self):
        """Image chunks must trigger a vision message (content is a list)."""
        from doc_parser.retrieval.reranker import OpenAIReranker

        settings = _make_settings(backend="openai")

        captured_messages: list = []

        async def fake_create(**kwargs):
            captured_messages.append(kwargs["messages"])
            choice = MagicMock()
            choice.message.content = "8"
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        with patch("doc_parser.retrieval.reranker.AsyncOpenAI") as mock_cls:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = fake_create
            mock_cls.return_value = mock_openai

            reranker = OpenAIReranker(settings)
            img = _make_image_candidate()
            await reranker.rerank("bar chart accuracy", [img], top_n=1)

        assert len(captured_messages) == 1
        msg = captured_messages[0][0]
        # Vision message content is a list
        assert isinstance(msg["content"], list)
        types = [part["type"] for part in msg["content"]]
        assert "image_url" in types

    @pytest.mark.asyncio
    async def test_unparseable_score_defaults_to_zero(self):
        """If OpenAI returns non-numeric text, score should fall back to 0."""
        from doc_parser.retrieval.reranker import OpenAIReranker

        settings = _make_settings(backend="openai")

        mock_choice = MagicMock()
        mock_choice.message.content = "not-a-number"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("doc_parser.retrieval.reranker.AsyncOpenAI") as mock_cls:
            mock_openai = AsyncMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_openai

            reranker = OpenAIReranker(settings)
            candidates = _make_candidates(n=1)
            result = await reranker.rerank("query", candidates, top_n=1)

        assert result[0]["rerank_score"] == 0.0


# ── Jina backend ──────────────────────────────────────────────────────────────


class TestJinaReranker:
    @pytest.mark.asyncio
    async def test_rerank_calls_jina_api(self):
        """rerank() should POST to Jina API and return re-ordered candidates."""
        from doc_parser.retrieval.reranker import JinaReranker

        settings = _make_settings(backend="jina", jina_key="jina_test_key")

        api_response = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.72},
                {"index": 2, "relevance_score": 0.50},
            ]
        }

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = api_response
        mock_http_response.raise_for_status = MagicMock()

        with patch("doc_parser.retrieval.reranker.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.post = AsyncMock(return_value=mock_http_response)
            mock_cls.return_value = mock_client_instance

            reranker = JinaReranker(settings)
            candidates = _make_candidates(n=3)
            result = await reranker.rerank("accuracy results", candidates, top_n=3)

        assert len(result) == 3
        assert result[0]["rerank_score"] == pytest.approx(0.95)
        assert result[1]["rerank_score"] == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_image_candidate_includes_images_field(self):
        """Image chunks must appear with an 'images' key in the Jina payload."""
        from doc_parser.retrieval.reranker import JinaReranker

        settings = _make_settings(backend="jina", jina_key="jina_test_key")

        captured_payloads: list = []

        api_response = {"results": [{"index": 0, "relevance_score": 0.9}]}
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = api_response
        mock_http_response.raise_for_status = MagicMock()

        async def fake_post(url, json, headers):
            captured_payloads.append(json)
            return mock_http_response

        with patch("doc_parser.retrieval.reranker.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.post = fake_post
            mock_cls.return_value = mock_client_instance

            reranker = JinaReranker(settings)
            img = _make_image_candidate()
            await reranker.rerank("bar chart", [img], top_n=1)

        payload = captured_payloads[0]
        doc = payload["documents"][0]
        assert "images" in doc
        assert isinstance(doc["images"], list)
        assert len(doc["images"]) == 1

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_payload(self):
        """The result dicts must contain all original candidate fields."""
        from doc_parser.retrieval.reranker import JinaReranker

        settings = _make_settings(backend="jina", jina_key="jina_test_key")

        api_response = {"results": [{"index": 0, "relevance_score": 0.8}]}
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = api_response
        mock_http_response.raise_for_status = MagicMock()

        with patch("doc_parser.retrieval.reranker.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.post = AsyncMock(return_value=mock_http_response)
            mock_cls.return_value = mock_client_instance

            reranker = JinaReranker(settings)
            candidates = _make_candidates(n=1)
            result = await reranker.rerank("query", candidates, top_n=1)

        assert result[0]["chunk_id"] == "doc_0"
        assert result[0]["source_file"] == "paper.pdf"
        assert "rerank_score" in result[0]


# ── BGE backend ───────────────────────────────────────────────────────────────


class TestBGEReranker:
    @pytest.mark.asyncio
    async def test_rerank_returns_top_n(self):
        """BGE reranker should return at most top_n results."""
        from doc_parser.retrieval.reranker import BGEReranker

        _make_settings(backend="bge")

        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]

        with (
            patch("doc_parser.retrieval.reranker.BGEReranker.__init__", return_value=None),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop_instance = MagicMock()
            mock_loop_instance.run_in_executor = AsyncMock(
                return_value=[0.9, 0.3, 0.7, 0.1, 0.5]
            )
            mock_loop.return_value = mock_loop_instance

            reranker = BGEReranker.__new__(BGEReranker)
            reranker._reranker = mock_reranker
            candidates = _make_candidates(n=5)
            result = await reranker.rerank("query", candidates, top_n=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_sorted_descending(self):
        """BGE results must be sorted highest score first."""
        from doc_parser.retrieval.reranker import BGEReranker

        mock_reranker = MagicMock()
        scores = [0.2, 0.8, 0.5]

        with (
            patch("doc_parser.retrieval.reranker.BGEReranker.__init__", return_value=None),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop_instance = MagicMock()
            mock_loop_instance.run_in_executor = AsyncMock(return_value=scores)
            mock_loop.return_value = mock_loop_instance

            reranker = BGEReranker.__new__(BGEReranker)
            reranker._reranker = mock_reranker
            candidates = _make_candidates(n=3)
            result = await reranker.rerank("query", candidates, top_n=3)

        rerank_scores = [r["rerank_score"] for r in result]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    def test_init_raises_without_flag_embedding(self):
        """BGEReranker init must raise ImportError when FlagEmbedding is missing."""
        from doc_parser.retrieval.reranker import BGEReranker

        settings = _make_settings(backend="bge")

        with patch.dict("sys.modules", {"FlagEmbedding": None}):
            with pytest.raises((ImportError, TypeError)):
                BGEReranker(settings)


# ── Qwen VL backend ───────────────────────────────────────────────────────────


class TestQwenVLReranker:
    @pytest.mark.asyncio
    async def test_rerank_returns_top_n(self):
        """Qwen VL reranker should return at most top_n results."""
        from doc_parser.retrieval.reranker import QwenVLReranker

        scores = [2.1, 0.5, 1.8, -0.3, 0.9]

        with (
            patch("doc_parser.retrieval.reranker.QwenVLReranker.__init__", return_value=None),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            # gather runs score_one coroutines; simulate via run_in_executor
            call_count = 0

            async def fake_run_in_executor(executor, func, *args):
                nonlocal call_count
                score = scores[call_count]
                call_count += 1
                return score

            mock_loop_instance = MagicMock()
            mock_loop_instance.run_in_executor = fake_run_in_executor
            mock_loop.return_value = mock_loop_instance

            reranker = QwenVLReranker.__new__(QwenVLReranker)
            reranker._device = "cpu"
            reranker._processor = MagicMock()
            reranker._model = MagicMock()
            candidates = _make_candidates(n=5)
            result = await reranker.rerank("query", candidates, top_n=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_sorted_descending(self):
        """Qwen VL results must be sorted highest score first."""
        from doc_parser.retrieval.reranker import QwenVLReranker

        scores = [0.5, 2.5, 1.0]
        call_count = 0

        async def fake_run_in_executor(executor, func, *args):
            nonlocal call_count
            score = scores[call_count]
            call_count += 1
            return score

        with (
            patch("doc_parser.retrieval.reranker.QwenVLReranker.__init__", return_value=None),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop_instance = MagicMock()
            mock_loop_instance.run_in_executor = fake_run_in_executor
            mock_loop.return_value = mock_loop_instance

            reranker = QwenVLReranker.__new__(QwenVLReranker)
            reranker._device = "cpu"
            reranker._processor = MagicMock()
            reranker._model = MagicMock()
            candidates = _make_candidates(n=3)
            result = await reranker.rerank("query", candidates, top_n=3)

        rerank_scores = [r["rerank_score"] for r in result]
        assert rerank_scores == sorted(rerank_scores, reverse=True)
