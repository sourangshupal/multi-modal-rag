"""Unit tests for the generate route — multimodal context building logic.

All external calls (Qdrant, OpenAI, embedder, reranker) are mocked.
Tests focus on the user_content construction logic for image, table,
and text-only candidate sets.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ── Helpers ──────────────────────────────────────────────────────────────────

_TINY_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _make_text_candidate(page: int = 1, text: str = "Some text content.") -> dict:
    return {
        "chunk_id": f"chunk_text_{page}",
        "text": text,
        "caption": None,
        "modality": "text",
        "page": page,
        "image_base64": None,
        "source_file": "paper.pdf",
        "element_types": ["text"],
        "bbox": None,
        "is_atomic": False,
        "rerank_score": 0.9,
    }


def _make_image_candidate(page: int = 2, b64: str | None = _TINY_B64) -> dict:
    return {
        "chunk_id": f"chunk_image_{page}",
        "text": "Figure 1: Overview of the architecture.",
        "caption": None,
        "modality": "image",
        "page": page,
        "image_base64": b64,
        "source_file": "paper.pdf",
        "element_types": ["image"],
        "bbox": [100, 100, 500, 400],
        "is_atomic": True,
        "rerank_score": 0.85,
    }


def _make_table_candidate(page: int = 3, b64: str | None = _TINY_B64) -> dict:
    return {
        "chunk_id": f"chunk_table_{page}",
        "text": "Comparison of model performance across benchmarks.",
        "caption": "| Model | Acc | F1 |\n|---|---|---|\n| BERT | 92 | 91 |",
        "modality": "table",
        "page": page,
        "image_base64": b64,
        "source_file": "paper.pdf",
        "element_types": ["table"],
        "bbox": [50, 200, 800, 600],
        "is_atomic": True,
        "rerank_score": 0.88,
    }


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.openai_llm_model = "gpt-4o"
    s.reranker_top_n = 5
    return s


def _make_completion(content: str = "The answer is 42.") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ── Fixture: patch all FastAPI dependencies ────────────────────────────────

@pytest.fixture()
def patched_deps(request):
    """Patch all FastAPI dependency singletons and return a dict of mocks."""
    candidates = getattr(request, "param", [_make_text_candidate()])

    store_mock = MagicMock()
    store_mock.search = AsyncMock(return_value=candidates)

    embedder_mock = MagicMock()

    reranker_mock = MagicMock()
    # reranker just returns same candidates with rerank_score set
    reranked = [{**c, "rerank_score": 0.9} for c in candidates]
    reranker_mock.rerank = AsyncMock(return_value=reranked)

    openai_client_mock = MagicMock()
    completion = _make_completion()
    openai_client_mock.chat = MagicMock()
    openai_client_mock.chat.completions = MagicMock()
    openai_client_mock.chat.completions.create = AsyncMock(return_value=completion)

    settings_mock = _make_settings()

    patches = [
        patch("doc_parser.api.routes.generate.get_store", return_value=store_mock),
        patch("doc_parser.api.routes.generate.get_embedder_dep", return_value=embedder_mock),
        patch("doc_parser.api.routes.generate.get_reranker_dep", return_value=reranker_mock),
        patch("doc_parser.api.routes.generate.get_openai_client", return_value=openai_client_mock),
        patch("doc_parser.api.routes.generate.get_settings", return_value=settings_mock),
    ]

    for p in patches:
        p.start()
    yield {
        "store": store_mock,
        "embedder": embedder_mock,
        "reranker": reranker_mock,
        "client": openai_client_mock,
        "settings": settings_mock,
        "patches": patches,
    }
    for p in patches:
        p.stop()


# ── Helpers to extract user_content from actual LLM call ──────────────────

async def _call_generate(candidates: list[dict], query: str = "What is the result?") -> dict:
    """Drive the generate() handler and return the captured create() kwargs."""
    from doc_parser.api.routes.generate import generate
    from doc_parser.api.schemas import GenerateRequest

    store_mock = MagicMock()
    store_mock.search = AsyncMock(return_value=candidates)

    reranker_mock = MagicMock()
    reranked = [{**c, "rerank_score": 0.9} for c in candidates]
    reranker_mock.rerank = AsyncMock(return_value=reranked)

    openai_client_mock = MagicMock()
    completion = _make_completion()
    openai_client_mock.chat.completions.create = AsyncMock(return_value=completion)

    settings_mock = _make_settings()

    with (
        patch("doc_parser.api.routes.generate.get_store", return_value=store_mock),
        patch("doc_parser.api.routes.generate.get_embedder_dep", return_value=MagicMock()),
        patch("doc_parser.api.routes.generate.get_reranker_dep", return_value=reranker_mock),
        patch("doc_parser.api.routes.generate.get_openai_client", return_value=openai_client_mock),
        patch("doc_parser.api.routes.generate.get_settings", return_value=settings_mock),
    ):
        req = GenerateRequest(query=query)
        await generate(req)

    # Return the kwargs passed to create()
    return openai_client_mock.chat.completions.create.call_args


# ── Test: image chunk with image_base64 → list user_content with image_url ──

class TestImageChunkMultimodal:
    @pytest.mark.asyncio
    async def test_image_chunk_with_b64_produces_list_user_content(self):
        """An image candidate with image_base64 must yield list user_content containing image_url blocks."""
        candidates = [_make_image_candidate(page=2, b64=_TINY_B64)]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        user_content = user_msg["content"]

        assert isinstance(user_content, list), "Expected list user_content for image chunk with b64"
        types = [block["type"] for block in user_content]
        assert "image_url" in types, "Expected image_url block in user_content"

    @pytest.mark.asyncio
    async def test_image_chunk_with_b64_contains_correct_data_url(self):
        """The image_url block must use the data:image/png;base64,... scheme."""
        candidates = [_make_image_candidate(page=2, b64=_TINY_B64)]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        image_blocks = [
            b for b in user_msg["content"]
            if b.get("type") == "image_url"
        ]
        assert len(image_blocks) == 1
        url = image_blocks[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), f"Unexpected URL prefix: {url[:40]}"
        assert _TINY_B64 in url

    @pytest.mark.asyncio
    async def test_image_chunk_without_b64_goes_to_text_context(self):
        """An image candidate without image_base64 should be treated as plain text (no image_url)."""
        candidates = [_make_image_candidate(page=2, b64=None)]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        user_content = user_msg["content"]

        # With no visuals, user_content should be a plain string
        assert isinstance(user_content, str), "Expected plain string for image chunk without b64"
        assert "image_url" not in str(user_content)


# ── Test: table chunk with image_base64 → text context + image_url blocks ──

class TestTableChunkMultimodal:
    @pytest.mark.asyncio
    async def test_table_chunk_with_b64_produces_list_user_content(self):
        """A table candidate with image_base64 must yield list user_content."""
        candidates = [_make_table_candidate(page=3, b64=_TINY_B64)]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        user_content = user_msg["content"]

        assert isinstance(user_content, list), "Expected list user_content for table chunk with b64"

    @pytest.mark.asyncio
    async def test_table_chunk_with_b64_has_both_text_and_image_url(self):
        """Table with b64 must appear in both the text preamble and as an image_url block."""
        candidates = [_make_table_candidate(page=3, b64=_TINY_B64)]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        content_list = user_msg["content"]

        # Should have a text preamble block containing the table markdown
        text_blocks = [b for b in content_list if b.get("type") == "text"]
        image_blocks = [b for b in content_list if b.get("type") == "image_url"]

        assert len(image_blocks) == 1, "Expected exactly one image_url block for the table"

        # The table markdown must appear in one of the text blocks (preamble)
        all_text = " ".join(b["text"] for b in text_blocks)
        assert "BERT" in all_text or "Comparison" in all_text, (
            "Table text/caption should appear in the preamble text block"
        )

    @pytest.mark.asyncio
    async def test_table_chunk_without_b64_still_passes_markdown_as_text(self):
        """Table without image_base64 must still pass the full markdown as plain text context."""
        candidates = [_make_table_candidate(page=3, b64=None)]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        user_content = user_msg["content"]

        # No visuals → plain string
        assert isinstance(user_content, str), "Expected plain string for table chunk without b64"
        # Both the summary and the markdown table caption should be present
        assert "Comparison" in user_content
        assert "BERT" in user_content

    @pytest.mark.asyncio
    async def test_table_chunk_without_caption_uses_text_as_fallback(self):
        """A table candidate with empty caption should still include the text summary."""
        candidate = _make_table_candidate(page=3, b64=None)
        candidate["caption"] = ""  # no caption
        candidate["text"] = "Summary only text."

        call_args = await _call_generate([candidate])

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Summary only text." in user_msg["content"]


# ── Test: text-only result set → plain string user_content ─────────────────

class TestTextOnlyMultimodal:
    @pytest.mark.asyncio
    async def test_text_only_candidates_produce_plain_string(self):
        """When no image/table b64 is present, user_content must be a plain str."""
        candidates = [_make_text_candidate(page=1), _make_text_candidate(page=2, text="More text.")]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        user_content = user_msg["content"]

        assert isinstance(user_content, str), "Text-only set should produce plain string"

    @pytest.mark.asyncio
    async def test_text_only_content_includes_context_and_question(self):
        """Plain string user_content must include the context prefix and the query."""
        candidates = [_make_text_candidate(page=1, text="Alpha bravo charlie.")]
        call_args = await _call_generate(candidates, query="What is Alpha?")

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        content = user_msg["content"]

        assert "Context:" in content
        assert "Alpha bravo charlie." in content
        assert "What is Alpha?" in content

    @pytest.mark.asyncio
    async def test_mixed_text_and_image_no_b64_is_plain_string(self):
        """Image chunk without b64 + text chunk → plain string (no multimodal overhead)."""
        candidates = [
            _make_text_candidate(page=1),
            _make_image_candidate(page=2, b64=None),  # no b64 → falls into text branch
        ]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert isinstance(user_msg["content"], str)


# ── Test: mixed modalities (text + image + table) ──────────────────────────

class TestMixedModalities:
    @pytest.mark.asyncio
    async def test_mixed_modalities_produce_list_user_content(self):
        """Mix of text + table-with-b64 should produce list user_content."""
        candidates = [
            _make_text_candidate(page=1),
            _make_table_candidate(page=2, b64=_TINY_B64),
        ]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert isinstance(user_msg["content"], list)

    @pytest.mark.asyncio
    async def test_text_candidate_appears_in_preamble_not_as_image_url(self):
        """A plain text candidate must appear in the preamble text, not as an image_url block."""
        candidates = [
            _make_text_candidate(page=1, text="Important finding here."),
            _make_image_candidate(page=2, b64=_TINY_B64),
        ]
        call_args = await _call_generate(candidates)

        messages = call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        content_list = user_msg["content"]

        # Text blocks (preamble)
        text_blocks = [b for b in content_list if b.get("type") == "text"]
        preamble_text = " ".join(b["text"] for b in text_blocks)

        assert "Important finding here." in preamble_text

        # Only one image_url block (the image chunk)
        image_blocks = [b for b in content_list if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
