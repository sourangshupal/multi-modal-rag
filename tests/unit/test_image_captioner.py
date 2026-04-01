"""Unit tests for image_captioner — table JSON parsing, validation, and crop helpers."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from doc_parser.chunker import Chunk  # noqa: E402
from doc_parser.ingestion.image_captioner import (  # noqa: E402
    _crop_image_chunk,
    _parse_table_json_response,
    _validate_table_extraction,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(text: str = "", modality: str = "text", page: int = 1) -> Chunk:
    return Chunk(
        text=text,
        chunk_id="test_chunk",
        page=page,
        element_types=["text"],
        bbox=None,
        source_file="test.pdf",
        is_atomic=False,
        modality=modality,
    )


# ── _parse_table_json_response ───────────────────────────────────────────────


class TestParseTableJsonResponse:
    def test_valid_json_with_all_fields(self):
        raw_ocr = "col1 col2\nval1 val2"
        json_str = json.dumps({
            "num_columns": 2,
            "num_rows": 1,
            "markdown_table": "| col1 | col2 |\n|---|---|\n| val1 | val2 |",
            "summary": "A table comparing col1 and col2 values.",
        })
        caption, text = _parse_table_json_response(raw_ocr, json_str)
        assert "col1" in caption
        assert "col2" in caption
        assert "| val1 | val2 |" in caption
        assert "comparing" in text

    def test_malformed_json_falls_back_to_raw(self):
        raw_ocr = "original table text"
        caption, text = _parse_table_json_response(raw_ocr, "not json at all")
        assert caption == raw_ocr
        assert text == raw_ocr

    def test_empty_fields_fall_back_to_raw(self):
        raw_ocr = "original table text"
        json_str = json.dumps({
            "num_columns": 0,
            "num_rows": 0,
            "markdown_table": "",
            "summary": "",
        })
        caption, text = _parse_table_json_response(raw_ocr, json_str)
        assert caption == raw_ocr
        assert text == raw_ocr

    def test_missing_markdown_uses_raw_for_caption(self):
        raw_ocr = "original table text"
        json_str = json.dumps({
            "num_columns": 2,
            "num_rows": 1,
            "summary": "A summary.",
        })
        caption, text = _parse_table_json_response(raw_ocr, json_str)
        assert caption == raw_ocr  # no markdown_table key
        assert text == "A summary."

    def test_missing_summary_uses_raw_for_text(self):
        raw_ocr = "original table text"
        json_str = json.dumps({
            "num_columns": 2,
            "num_rows": 1,
            "markdown_table": "| a | b |\n|---|---|\n| 1 | 2 |",
        })
        caption, text = _parse_table_json_response(raw_ocr, json_str)
        assert "| a | b |" in caption
        assert text == raw_ocr  # no summary key

    def test_none_json_str_falls_back(self):
        raw_ocr = "original"
        caption, text = _parse_table_json_response(raw_ocr, None)
        assert caption == raw_ocr
        assert text == raw_ocr


# ── _validate_table_extraction ───────────────────────────────────────────────


class TestValidateTableExtraction:
    def test_matching_counts_passes(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
        assert _validate_table_extraction("raw", 3, 2, md) is True

    def test_missing_rows_fails(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert _validate_table_extraction("raw", 5, 2, md) is False

    def test_empty_markdown_passes(self):
        assert _validate_table_extraction("raw", 5, 2, "") is True

    def test_zero_reported_rows_passes(self):
        assert _validate_table_extraction("raw", 0, 2, "| A |\n|---|\n| 1 |") is True

    def test_extra_rows_within_tolerance(self):
        md = "| A |\n|---|\n| 1 |\n| 2 |\n| 3 |"
        assert _validate_table_extraction("raw", 3, 1, md) is True

    def test_significantly_extra_rows_fails(self):
        # 10 data rows reported but only 2 in markdown
        md = "| A |\n|---|\n| 1 |\n| 2 |"
        assert _validate_table_extraction("raw", 10, 1, md) is False


# ── _crop_image_chunk ────────────────────────────────────────────────────────


class TestCropImageChunk:
    """Tests for _crop_image_chunk — no real PDF needed (mocked pdf_page_to_image)."""

    def _make_image_chunk(self, bbox=(100, 100, 500, 400), page: int = 1) -> Chunk:
        return Chunk(
            text="Figure 1: Architecture",
            chunk_id="img_chunk_1",
            page=page,
            element_types=["image"],
            bbox=bbox,
            source_file="test.pdf",
            is_atomic=True,
            modality="image",
        )

    def _make_test_page_image(self, width: int = 1000, height: int = 1400):
        """Return a mock PIL Image-like object."""
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        return img

    def test_returns_base64_string_on_valid_crop(self):
        chunk = self._make_image_chunk(bbox=(0, 0, 500, 500))
        mock_img = self._make_test_page_image()
        with patch(
            "doc_parser.ingestion.image_captioner.pdf_page_to_image",
            return_value=mock_img,
        ):
            result = _crop_image_chunk(chunk, Path("fake.pdf"))
        assert result is not None
        # Must be a valid base64 string
        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_returns_none_for_tiny_crop(self):
        # bbox that maps to less than _MIN_CROP_SIZE_PX in both dimensions
        chunk = self._make_image_chunk(bbox=(0, 0, 1, 1))  # 1x1 pixel at 1000px page
        mock_img = self._make_test_page_image(width=1000, height=1000)
        with patch(
            "doc_parser.ingestion.image_captioner.pdf_page_to_image",
            return_value=mock_img,
        ):
            result = _crop_image_chunk(chunk, Path("fake.pdf"))
        assert result is None

    def test_returns_none_on_exception(self):
        chunk = self._make_image_chunk()
        with patch(
            "doc_parser.ingestion.image_captioner.pdf_page_to_image",
            side_effect=RuntimeError("PDF read error"),
        ):
            result = _crop_image_chunk(chunk, Path("fake.pdf"))
        assert result is None
