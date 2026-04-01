"""Unit tests for embedder.embed_texts() and compute_sparse_vectors()."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ── embed_texts ───────────────────────────────────────────────────────────────


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_returns_one_vector_per_text(self):
        from doc_parser.ingestion.embedder import embed_texts

        async def fake_create(model, input, dimensions):  # noqa: A002
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * dimensions) for _ in input]
            )

        client = AsyncMock()
        client.embeddings.create = fake_create

        result = await embed_texts(["hello", "world"], client, model="m", dimensions=4)
        assert len(result) == 2
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_preserves_input_order_across_batches(self):
        """Batching must not reorder embeddings."""
        from doc_parser.ingestion.embedder import embed_texts

        texts = [f"text_{i}" for i in range(7)]
        call_counter = [0]

        async def fake_create(model, input, dimensions):  # noqa: A002
            batch_idx = call_counter[0]
            call_counter[0] += 1
            return MagicMock(
                data=[
                    MagicMock(embedding=[float(batch_idx * 100 + j)] * dimensions)
                    for j in range(len(input))
                ]
            )

        client = AsyncMock()
        client.embeddings.create = fake_create

        result = await embed_texts(texts, client, model="m", dimensions=2, batch_size=3)
        assert len(result) == 7
        # Batch 0 → indices 0..2 start with 0.0; batch 1 → indices 3..5 start with 100.0
        assert result[0][0] == 0.0
        assert result[3][0] == 100.0
        assert result[6][0] == 200.0

    @pytest.mark.asyncio
    async def test_empty_string_replaced_with_placeholder(self):
        """OpenAI rejects empty strings; embedder must replace with '[empty]'."""
        from doc_parser.ingestion.embedder import embed_texts

        captured_inputs: list[list[str]] = []

        async def fake_create(model, input, dimensions):  # noqa: A002
            captured_inputs.append(list(input))
            return MagicMock(data=[MagicMock(embedding=[0.0] * dimensions)])

        client = AsyncMock()
        client.embeddings.create = fake_create

        await embed_texts([""], client, model="m", dimensions=2)
        assert captured_inputs[0][0] == "[empty]"

    @pytest.mark.asyncio
    async def test_whitespace_only_string_replaced_with_placeholder(self):
        """Whitespace-only strings are also treated as empty."""
        from doc_parser.ingestion.embedder import embed_texts

        captured_inputs: list[list[str]] = []

        async def fake_create(model, input, dimensions):  # noqa: A002
            captured_inputs.append(list(input))
            return MagicMock(data=[MagicMock(embedding=[0.0] * dimensions)])

        client = AsyncMock()
        client.embeddings.create = fake_create

        await embed_texts(["   "], client, model="m", dimensions=2)
        assert captured_inputs[0][0] == "[empty]"

    @pytest.mark.asyncio
    async def test_batching_calls_api_correct_number_of_times(self):
        from doc_parser.ingestion.embedder import embed_texts

        texts = ["t"] * 10
        call_count = [0]

        async def fake_create(model, input, dimensions):  # noqa: A002
            call_count[0] += 1
            return MagicMock(
                data=[MagicMock(embedding=[0.0] * dimensions) for _ in input]
            )

        client = AsyncMock()
        client.embeddings.create = fake_create

        await embed_texts(texts, client, model="m", dimensions=2, batch_size=3)
        # ceil(10 / 3) = 4 batches
        assert call_count[0] == 4


# ── compute_sparse_vectors ───────────────────────────────────────────────────


class TestComputeSparseVectors:
    def test_returns_one_sparse_vector_per_text(self):
        """compute_sparse_vectors must return the same number of vectors as inputs."""
        from doc_parser.ingestion.embedder import compute_sparse_vectors

        texts = ["hello world", "machine learning", "retrieval augmented generation"]
        result = compute_sparse_vectors(texts)
        assert len(result) == len(texts)

    def test_sparse_vector_has_non_empty_indices_and_values(self):
        """Non-empty text produces a SparseVector with non-empty indices and values."""
        from qdrant_client.models import SparseVector

        from doc_parser.ingestion.embedder import compute_sparse_vectors

        result = compute_sparse_vectors(["the quick brown fox"])
        assert isinstance(result[0], SparseVector)
        assert len(result[0].indices) > 0
        assert len(result[0].values) > 0

    def test_empty_text_produces_empty_sparse_vector(self):
        """Empty string produces a SparseVector with no entries."""
        from doc_parser.ingestion.embedder import compute_sparse_vectors

        result = compute_sparse_vectors([""])
        assert result[0].indices == []
        assert result[0].values == []

    def test_indices_are_sorted(self):
        """Indices in each SparseVector must be in ascending order."""
        from doc_parser.ingestion.embedder import compute_sparse_vectors

        texts = ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"]
        result = compute_sparse_vectors(texts)
        indices = result[0].indices
        assert indices == sorted(indices)

    def test_same_text_produces_same_indices(self):
        """compute_sparse_vectors is deterministic (no randomness)."""
        from doc_parser.ingestion.embedder import compute_sparse_vectors

        text = "deterministic hashing should produce the same output every time"
        r1 = compute_sparse_vectors([text])
        r2 = compute_sparse_vectors([text])
        assert r1[0].indices == r2[0].indices
        assert r1[0].values == r2[0].values

    def test_values_sum_to_approximately_one(self):
        """Normalised TF weights should sum to 1.0 (one value per unique term)."""
        from doc_parser.ingestion.embedder import compute_sparse_vectors

        # "a b c" — 3 unique tokens, each with frequency 1/3
        result = compute_sparse_vectors(["a b c"])
        assert abs(sum(result[0].values) - 1.0) < 1e-9

    def test_repeated_term_increases_weight(self):
        """A term repeated more often should produce a higher weight than a rare term."""

        from doc_parser.ingestion.embedder import compute_sparse_vectors

        text = "apple apple apple banana"
        result = compute_sparse_vectors([text])

        # Reconstruct the index for "apple" and "banana"
        apple_idx = abs(hash("apple")) % (2**17)
        banana_idx = abs(hash("banana")) % (2**17)

        idx_to_val = dict(zip(result[0].indices, result[0].values, strict=True))
        assert idx_to_val[apple_idx] > idx_to_val[banana_idx]


# ── OpenAIEmbedder ────────────────────────────────────────────────────────────


class TestOpenAIEmbedder:
    @pytest.mark.asyncio
    async def test_embed_delegates_to_embed_texts_with_correct_model_and_dims(self):
        """OpenAIEmbedder.embed() should call embed_texts with the configured model and dims."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from doc_parser.ingestion.embedder import OpenAIEmbedder

        settings = MagicMock()
        settings.openai_api_key = None
        settings.embedding_model = "text-embedding-3-small"
        settings.embedding_dimensions = 1536

        fake_result = [[0.1] * 1536, [0.2] * 1536]
        with patch("doc_parser.ingestion.embedder.AsyncOpenAI", return_value=MagicMock()):
            with patch(
                "doc_parser.ingestion.embedder.embed_texts", new=AsyncMock(return_value=fake_result)
            ) as mock_et:
                embedder = OpenAIEmbedder(settings)
                result = await embedder.embed(["hello", "world"])

        mock_et.assert_called_once_with(["hello", "world"], embedder._client, "text-embedding-3-small", 1536)
        assert result == fake_result


# ── GeminiEmbedder ────────────────────────────────────────────────────────────


class TestGeminiEmbedder:
    def test_init_raises_value_error_when_api_key_missing(self):
        """GeminiEmbedder should raise ValueError when gemini_api_key is None."""
        from unittest.mock import MagicMock

        from doc_parser.ingestion.embedder import GeminiEmbedder

        settings = MagicMock()
        settings.gemini_api_key = None

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiEmbedder(settings)

    def test_init_raises_import_error_when_google_genai_not_installed(self):
        """GeminiEmbedder should raise ImportError with install hint when google-genai missing."""
        from unittest.mock import MagicMock

        from doc_parser.ingestion.embedder import GeminiEmbedder

        settings = MagicMock()
        settings.gemini_api_key = MagicMock()
        settings.gemini_api_key.get_secret_value.return_value = "fake-key"

        # Hide google.genai from import
        with pytest.raises(ImportError, match="google-genai"):
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "google.genai" or name == "google":
                    raise ImportError("No module named 'google'")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = mock_import
            try:
                GeminiEmbedder(settings)
            finally:
                builtins.__import__ = real_import

    @pytest.mark.asyncio
    async def test_embed_calls_run_in_executor(self):
        """GeminiEmbedder.embed() should offload sync work to an executor."""
        from unittest.mock import MagicMock, patch

        from doc_parser.ingestion.embedder import GeminiEmbedder

        settings = MagicMock()
        settings.gemini_api_key = MagicMock()
        settings.gemini_api_key.get_secret_value.return_value = "fake-key"

        fake_vectors = [[0.1, 0.2], [0.3, 0.4]]

        mock_genai_module = MagicMock()
        mock_genai_module.Client.return_value = MagicMock()

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai_module}):
            embedder = GeminiEmbedder.__new__(GeminiEmbedder)
            embedder._client = MagicMock()

            async def fake_executor(executor, fn, *args):
                return fake_vectors

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = fake_executor
                result = await embedder.embed(["a", "b"])

        assert result == fake_vectors

    def test_embed_sync_sanitises_empty_strings(self):
        """_embed_sync should replace empty strings with '[empty]' before calling the API."""
        from unittest.mock import MagicMock

        from doc_parser.ingestion.embedder import GeminiEmbedder

        embedder = GeminiEmbedder.__new__(GeminiEmbedder)

        captured: list[list[str]] = []

        def fake_embed_content(model, contents):
            captured.append(list(contents))
            return MagicMock(embeddings=[MagicMock(values=[0.0, 0.0])])

        embedder._client = MagicMock()
        embedder._client.models.embed_content = fake_embed_content

        embedder._embed_sync(["", "hello", "   "])
        assert captured[0][0] == "[empty]"
        assert captured[0][1] == "hello"
        assert captured[0][2] == "[empty]"


# ── get_embedder ──────────────────────────────────────────────────────────────


class TestGetEmbedder:
    def test_returns_openai_embedder_for_openai_provider(self):
        """get_embedder returns OpenAIEmbedder when embedding_provider='openai'."""
        from unittest.mock import MagicMock, patch

        from doc_parser.ingestion.embedder import OpenAIEmbedder, get_embedder

        settings = MagicMock()
        settings.embedding_provider = "openai"
        settings.openai_api_key = None
        settings.embedding_model = "text-embedding-3-large"
        settings.embedding_dimensions = 3072

        with patch("doc_parser.ingestion.embedder.AsyncOpenAI", return_value=MagicMock()):
            result = get_embedder(settings)
        assert isinstance(result, OpenAIEmbedder)

    def test_returns_gemini_embedder_for_gemini_provider(self):
        """get_embedder returns GeminiEmbedder when embedding_provider='gemini' and key is set."""
        from unittest.mock import MagicMock, patch

        from doc_parser.ingestion.embedder import GeminiEmbedder, get_embedder

        settings = MagicMock()
        settings.embedding_provider = "gemini"
        settings.gemini_api_key = MagicMock()
        settings.gemini_api_key.get_secret_value.return_value = "fake-key"

        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            result = get_embedder(settings)

        assert isinstance(result, GeminiEmbedder)

    def test_raises_value_error_for_unknown_provider(self):
        """get_embedder raises ValueError for an unrecognised provider name."""
        from unittest.mock import MagicMock

        from doc_parser.ingestion.embedder import get_embedder

        settings = MagicMock()
        settings.embedding_provider = "cohere"

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder(settings)

    def test_returns_qwen_embedder_for_qwen_provider(self):
        """get_embedder returns QwenVLEmbedder when embedding_provider='qwen'."""
        from unittest.mock import MagicMock, patch

        from doc_parser.ingestion.embedder import QwenVLEmbedder, get_embedder

        settings = MagicMock()
        settings.embedding_provider = "qwen"
        settings.qwen_embedding_model = "Qwen/Qwen3-VL-Embedding-2B"

        mock_model = MagicMock()
        mock_model.eval.return_value = None

        with (
            patch("transformers.AutoModel") as mock_auto_model,
            patch("transformers.AutoProcessor") as mock_auto_processor,
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_processor.from_pretrained.return_value = MagicMock()

            # Patch the imports inside __init__
            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(
                        cuda=MagicMock(is_available=lambda: False),
                        backends=MagicMock(mps=MagicMock(is_available=lambda: False)),
                        bfloat16=MagicMock(),
                    ),
                    "transformers": MagicMock(
                        AutoModel=MagicMock(
                            from_pretrained=MagicMock(return_value=mock_model)
                        ),
                        AutoProcessor=MagicMock(
                            from_pretrained=MagicMock(return_value=MagicMock())
                        ),
                    ),
                },
            ):
                result = get_embedder(settings)

        assert isinstance(result, QwenVLEmbedder)


# ── QwenVLEmbedder ────────────────────────────────────────────────────────────


def _make_qwen_embedder(model_name: str = "Qwen/Qwen3-VL-Embedding-2B"):
    """Create a QwenVLEmbedder with mocked HuggingFace model and processor."""
    from doc_parser.ingestion.embedder import QwenVLEmbedder

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.device = "cpu"

    embedder = QwenVLEmbedder.__new__(QwenVLEmbedder)
    embedder._model_name = model_name
    embedder._model = mock_model
    embedder._processor = MagicMock()
    return embedder


class TestQwenVLEmbedder:
    def test_init_raises_import_error_when_transformers_not_installed(self):
        """QwenVLEmbedder should raise ImportError with install hint when transformers missing."""
        import builtins

        from doc_parser.ingestion.embedder import QwenVLEmbedder

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="QwenVLEmbedder requires transformers"):
                QwenVLEmbedder("Qwen/Qwen3-VL-Embedding-2B")
        finally:
            builtins.__import__ = real_import

    @pytest.mark.asyncio
    async def test_embed_calls_run_in_executor(self):
        """QwenVLEmbedder.embed() should offload to thread-pool executor."""
        from unittest.mock import patch

        embedder = _make_qwen_embedder()
        fake_vectors = [[0.1, 0.2], [0.3, 0.4]]

        async def fake_executor(executor, fn, *args):
            return fake_vectors

        with patch("doc_parser.ingestion.embedder.asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = fake_executor
            result = await embedder.embed(["hello", "world"])

        assert result == fake_vectors

    @pytest.mark.asyncio
    async def test_embed_images_calls_run_in_executor(self):
        """QwenVLEmbedder.embed_images() should offload to thread-pool executor."""
        from unittest.mock import patch

        embedder = _make_qwen_embedder()
        fake_vectors = [[0.5, 0.6]]

        async def fake_executor(executor, fn, *args):
            return fake_vectors

        with patch("doc_parser.ingestion.embedder.asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = fake_executor
            result = await embedder.embed_images(["base64data"])

        assert result == fake_vectors

    def test_has_embed_images_method(self):
        """QwenVLEmbedder must expose embed_images so embed_chunks routing works."""
        embedder = _make_qwen_embedder()
        assert hasattr(embedder, "embed_images")
        assert callable(embedder.embed_images)


# ── embed_chunks routing ──────────────────────────────────────────────────────


def _make_chunk(
    modality: str = "text",
    text: str = "hello",
    image_base64: str | None = None,
) -> object:
    """Build a minimal Chunk-like object for routing tests."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeChunk:
        text: str
        modality: str
        image_base64: str | None = field(default=None)
        chunk_id: str = "test_0_0"
        page: int = 0
        element_types: list = field(default_factory=list)
        bbox: list | None = None
        source_file: str = "test.pdf"
        is_atomic: bool = False
        caption: str | None = None

    return FakeChunk(text=text, modality=modality, image_base64=image_base64)


class TestEmbedChunksRouting:
    @pytest.mark.asyncio
    async def test_image_chunks_routed_to_embed_images_when_supported(self):
        """embed_chunks routes image chunks with image_base64 to embed_images()."""
        from unittest.mock import AsyncMock, MagicMock

        from doc_parser.ingestion.embedder import embed_chunks

        img_chunk = _make_chunk(modality="image", text="a figure", image_base64="abc123")
        txt_chunk = _make_chunk(modality="text", text="some text")

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])
        embedder.embed_images = AsyncMock(return_value=[[0.9, 0.8]])
        # hasattr check: MagicMock has embed_images by default since we set it above

        settings = MagicMock()
        dense, sparse = await embed_chunks([img_chunk, txt_chunk], embedder, settings)

        # embed_images called with the image chunk's base64
        embedder.embed_images.assert_called_once_with(["abc123"])
        # embed called with the text chunk's text
        embedder.embed.assert_called_once_with(["some text"])
        assert dense[0] == [0.9, 0.8]
        assert dense[1] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_text_chunks_routed_to_embed(self):
        """embed_chunks sends text/table/formula chunks to embed()."""
        from unittest.mock import AsyncMock, MagicMock

        from doc_parser.ingestion.embedder import embed_chunks

        chunks = [
            _make_chunk(modality="text", text="paragraph text"),
            _make_chunk(modality="table", text="| col1 | col2 |"),
            _make_chunk(modality="formula", text="E = mc^2"),
        ]

        embedder = MagicMock()
        embedder.embed = AsyncMock(
            return_value=[[0.1] * 3, [0.2] * 3, [0.3] * 3]
        )
        # No embed_images on this embedder (simulate OpenAI path)
        del embedder.embed_images

        settings = MagicMock()
        dense, sparse = await embed_chunks(chunks, embedder, settings)

        embedder.embed.assert_called_once_with(
            ["paragraph text", "| col1 | col2 |", "E = mc^2"]
        )
        assert len(dense) == 3
        assert len(sparse) == 3

    @pytest.mark.asyncio
    async def test_image_chunks_fall_back_to_text_when_embed_images_absent(self):
        """Image chunks fall back to embed() when embedder lacks embed_images (OpenAI/Gemini)."""
        from unittest.mock import AsyncMock, MagicMock

        from doc_parser.ingestion.embedder import embed_chunks

        img_chunk = _make_chunk(modality="image", text="caption text", image_base64="abc123")

        embedder = MagicMock(spec=["embed"])  # spec limits to only 'embed'
        embedder.embed = AsyncMock(return_value=[[0.5, 0.5]])

        settings = MagicMock()
        dense, sparse = await embed_chunks([img_chunk], embedder, settings)

        # Falls back to embed() with caption text
        embedder.embed.assert_called_once_with(["caption text"])
        assert dense[0] == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_image_chunks_without_image_base64_go_to_text_path(self):
        """Image chunks without image_base64 are embedded via text even if embed_images exists."""
        from unittest.mock import AsyncMock, MagicMock

        from doc_parser.ingestion.embedder import embed_chunks

        img_chunk = _make_chunk(modality="image", text="figure caption", image_base64=None)

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.3, 0.3]])
        embedder.embed_images = AsyncMock(return_value=[[0.9, 0.9]])

        settings = MagicMock()
        dense, sparse = await embed_chunks([img_chunk], embedder, settings)

        embedder.embed.assert_called_once_with(["figure caption"])
        embedder.embed_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_sparse_vectors_always_computed_from_text(self):
        """Sparse vectors are computed from chunk.text for all chunks regardless of modality."""
        from unittest.mock import AsyncMock, MagicMock

        from doc_parser.ingestion.embedder import embed_chunks

        chunks = [
            _make_chunk(modality="image", text="caption", image_base64="b64"),
            _make_chunk(modality="text", text="hello world"),
        ]

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1] * 2])
        embedder.embed_images = AsyncMock(return_value=[[0.9] * 2])

        settings = MagicMock()
        _, sparse = await embed_chunks(chunks, embedder, settings)

        assert len(sparse) == 2
        # Image chunk "caption" should have non-empty sparse vector
        assert len(sparse[0].indices) > 0
        # Text chunk "hello world" should also have non-empty sparse vector
        assert len(sparse[1].indices) > 0
