"""Multimodal chunk enricher: generates structured descriptions for tables,
formulas, and algorithms via GPT-4o to improve embedding quality for retrieval.

Image/figure chunks are NOT sent to GPT-4o — instead, the PDF region is cropped and
stored as ``chunk.image_base64`` for direct visual embedding (e.g. Qwen3-VL-Embedding).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

from doc_parser.chunker import Chunk
from doc_parser.utils.pdf_utils import pdf_page_to_image

logger = logging.getLogger(__name__)

# Minimum crop dimension in pixels; smaller regions are likely detection noise
_MIN_CROP_SIZE_PX: int = 50

# Table input/output limits
_TABLE_MAX_INPUT_CHARS: int = 12_000
_TABLE_MAX_TOKENS: int = 2000

# ── Prompts ───────────────────────────────────────────────────────────────────

_TABLE_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given a table from a research document, you MUST respond with valid JSON only — no text outside the JSON object.

Think step by step:
1. Count the number of columns (including row-label columns).
2. Count the number of data rows (excluding the header row).
3. Reproduce the COMPLETE table in markdown format with | delimiters. Include EVERY row and EVERY column — do not summarise, skip, or truncate any data. Use exact values from the original.
4. Write a 1-2 sentence semantic summary of what the table shows, for search indexing.

Respond in this exact JSON schema:
{
  "num_columns": <integer>,
  "num_rows": <integer, excluding header>,
  "markdown_table": "<complete markdown table with | delimiters — ALL rows, ALL columns, exact values>",
  "summary": "<1-2 sentence description of what this table shows, measures, or compares>"
}

Rules:
- For merged or spanning cells, repeat the value across all affected columns/rows.
- For empty cells, use "-" as a placeholder.
- Escape any pipe characters within cell values as \\|.
- Do not round, paraphrase, or abbreviate any numbers or text.\
"""

_FORMULA_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given a mathematical formula or equation in LaTeX, respond in EXACTLY this format:

SUMMARY: <One sentence in plain English: what the formula computes or represents, its domain (e.g. probability, optimisation, signal processing), and where it typically appears.>
DETAIL: <Define each symbol or variable. List key properties and what the formula is used for.>

Use precise mathematical language but prefer plain English where equivalent.\
"""

_ALGORITHM_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given pseudocode or an algorithm from a research paper, respond in EXACTLY this format:

SUMMARY: <One paragraph describing what the algorithm does, its purpose, and the problem it solves.>
DETAIL: <Cover: (1) inputs and outputs, (2) main steps or phases, (3) time and space complexity if determinable, (4) notable design decisions or properties.>

Use the variable names and terminology from the algorithm itself.\
"""


# ── Response parsers ──────────────────────────────────────────────────────────

def _parse_text_response(raw_original: str, enriched: str) -> tuple[str, str]:
    """Return (original_raw_for_caption, enriched_text_for_embedding)."""
    return raw_original, enriched.strip() if enriched.strip() else raw_original


def _parse_table_json_response(raw_ocr: str, json_str: str) -> tuple[str, str]:
    """Parse structured table JSON response.

    Returns:
        (caption, text) where caption = full markdown table (for generation LLM)
        and text = semantic summary (for embedding/retrieval).
        Falls back to raw OCR on parse failure.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Table JSON parse failed, falling back to raw OCR")
        return raw_ocr, raw_ocr

    markdown_table = data.get("markdown_table", "")
    summary = data.get("summary", "")

    if not markdown_table and not summary:
        return raw_ocr, raw_ocr

    # caption carries the full table for the generation LLM
    caption = markdown_table if markdown_table else raw_ocr
    # text carries the summary for embedding/retrieval
    text = summary if summary else raw_ocr

    return caption, text


def _validate_table_extraction(
    raw_ocr: str,
    num_rows_reported: int,
    num_columns_reported: int,
    markdown_table: str,
) -> bool:
    """Check if extracted table dimensions roughly match expectations.

    Returns True if valid, False if suspicious (mismatch > 30%).
    """
    if not markdown_table or num_rows_reported <= 0:
        return True  # nothing to validate against

    # Count non-separator, non-empty rows in the extracted markdown
    md_lines = [
        ln for ln in markdown_table.strip().splitlines()
        if ln.strip() and not re.match(r"^\s*\|[\s\-:|]+\|\s*$", ln)
    ]
    # First line is header, rest are data rows
    actual_data_rows = max(0, len(md_lines) - 1)

    if num_rows_reported == 0:
        return True

    row_ratio = actual_data_rows / num_rows_reported
    if row_ratio < 0.7 or row_ratio > 1.5:
        logger.warning(
            "Table validation: reported %d rows but markdown has %d data rows (ratio=%.2f)",
            num_rows_reported, actual_data_rows, row_ratio,
        )
        return False

    return True


# ── Per-modality enrichment helpers ──────────────────────────────────────────

def _crop_image_chunk(chunk: Chunk, pdf_path: Path) -> str | None:
    """Crop the PDF region for an image chunk and return a base64-encoded PNG string.

    Returns the base64 string on success, or ``None`` if the crop is too small
    (likely a detection artefact) or if an error occurs.
    """
    try:
        page_img = pdf_page_to_image(pdf_path, chunk.page - 1, dpi=150)
        w, h = page_img.size

        bbox = chunk.bbox  # normalised 0–1000 coords
        x1 = int(bbox[0] * w / 1000)
        y1 = int(bbox[1] * h / 1000)
        x2 = int(bbox[2] * w / 1000)
        y2 = int(bbox[3] * h / 1000)

        crop = page_img.crop((x1, y1, x2, y2))
        if crop.size[0] < _MIN_CROP_SIZE_PX or crop.size[1] < _MIN_CROP_SIZE_PX:
            logger.debug(
                "Skipping tiny crop (%dx%d) for chunk %s",
                crop.size[0], crop.size[1], chunk.chunk_id,
            )
            return None

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    except Exception:
        logger.warning("PDF crop failed for chunk %s", chunk.chunk_id, exc_info=True)
        return None


async def _enrich_table_single(
    chunk: Chunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    pdf_path: Path | None = None,
) -> None:
    """Generate a structured table extraction with full markdown reproduction.

    Uses JSON mode with chain-of-thought to extract the complete table,
    then stores the markdown table in caption (for generation) and the
    semantic summary in text (for embedding/retrieval).
    """
    async with semaphore:
        try:
            raw = chunk.text
            if len(raw) > _TABLE_MAX_INPUT_CHARS:
                table_text = raw[:_TABLE_MAX_INPUT_CHARS] + "\n...[truncated]"
                logger.warning(
                    "Table chunk %s exceeds %d chars (%d), truncating input",
                    chunk.chunk_id, _TABLE_MAX_INPUT_CHARS, len(raw),
                )
            else:
                table_text = raw

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is a table from a research document:\n\n{table_text}"
                        ),
                    },
                ],
                max_tokens=_TABLE_MAX_TOKENS,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            json_str = (response.choices[0].message.content or "").strip()
            caption, text = _parse_table_json_response(raw, json_str)

            # Validate extraction completeness
            try:
                data = json.loads(json_str)
                num_rows = data.get("num_rows", 0)
                num_cols = data.get("num_columns", 0)
                md_table = data.get("markdown_table", "")

                if not _validate_table_extraction(raw, num_rows, num_cols, md_table):
                    logger.info(
                        "Table validation failed for %s, retrying with correction",
                        chunk.chunk_id,
                    )
                    caption, text = await _retry_table_extraction(
                        raw, table_text, num_rows, client, model, semaphore,
                    )
            except (json.JSONDecodeError, TypeError):
                pass  # already fell back in _parse_table_json_response

            chunk.caption = caption
            chunk.text = text

            logger.debug("Enriched table chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning("Table enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)


async def _retry_table_extraction(
    raw_ocr: str,
    table_text: str,
    prev_num_rows: int,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """Retry table extraction once with an explicit correction prompt."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Here is a table from a research document:\n\n{table_text}\n\n"
                        f"IMPORTANT: A previous extraction reported {prev_num_rows} rows "
                        f"but the markdown output was incomplete. Please carefully count "
                        f"ALL rows and reproduce the COMPLETE table. Do not skip any rows."
                    ),
                },
            ],
            max_tokens=_TABLE_MAX_TOKENS,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        json_str = (response.choices[0].message.content or "").strip()
        return _parse_table_json_response(raw_ocr, json_str)
    except Exception:
        logger.warning("Table retry also failed, using raw OCR")
        return raw_ocr, raw_ocr


async def _enrich_formula_single(
    chunk: Chunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
) -> None:
    """Generate a verbal formula description in-place."""
    async with semaphore:
        try:
            raw = chunk.text

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _FORMULA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is a formula from a research document:\n\n{raw}\n\n"
                            "Provide a verbal description for document retrieval."
                        ),
                    },
                ],
                max_tokens=350,
                temperature=0.0,
            )

            enriched = (response.choices[0].message.content or "").strip()
            chunk.caption, chunk.text = _parse_text_response(raw, enriched)

            logger.debug("Enriched formula chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning("Formula enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)


async def _enrich_algorithm_single(
    chunk: Chunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
) -> None:
    """Generate a semantic algorithm description in-place."""
    async with semaphore:
        try:
            raw = chunk.text

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _ALGORITHM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is an algorithm from a research document:\n\n{raw}\n\n"
                            "Provide a semantic description for document retrieval."
                        ),
                    },
                ],
                max_tokens=450,
                temperature=0.0,
            )

            enriched = (response.choices[0].message.content or "").strip()
            chunk.caption, chunk.text = _parse_text_response(raw, enriched)

            logger.debug("Enriched algorithm chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning(
                "Algorithm enrichment failed for chunk %s", chunk.chunk_id, exc_info=True
            )


# ── Public API ────────────────────────────────────────────────────────────────

async def enrich_chunks(
    chunks: list[Chunk],
    pdf_path: Path,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    max_concurrent: int = 5,
) -> list[Chunk]:
    """Enrich non-text chunks for improved retrieval quality.

    Dispatches by modality:
    - image   → PDF region is cropped and stored as ``chunk.image_base64`` (base64 PNG)
                for direct visual embedding (e.g. Qwen3-VL-Embedding).  No LLM call is
                made; ``chunk.caption`` is set to ``None`` and ``chunk.text`` is preserved
                from the chunker (e.g. figure title) or falls back to ``"[figure]"``.
    - table   → complete markdown table reproduction + semantic summary (JSON mode, GPT-4o)
    - formula → SUMMARY / DETAIL verbal description (text call, GPT-4o)
    - algorithm → SUMMARY / DETAIL semantic description (text call, GPT-4o)
    - text    → unchanged

    For tables, ``chunk.caption`` holds the full markdown table (for the generation
    LLM) and ``chunk.text`` holds the semantic summary (for embedding/retrieval).

    Args:
        chunks: All chunks from the document (mixed modalities).
        pdf_path: Path to the source PDF (used for image crop rendering).
        client: Authenticated AsyncOpenAI client.
        model: OpenAI model to use for enrichment (default "gpt-4o").
        max_concurrent: Max concurrent API calls (shared semaphore across all modalities).

    Returns:
        The same list mutated in-place with enriched ``text``, ``caption``, and
        ``image_base64`` fields.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    counts: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        if chunk.modality == "image":
            counts["image"] += 1
            if chunk.bbox is not None:
                b64 = _crop_image_chunk(chunk, pdf_path)
                if b64 is not None:
                    chunk.image_base64 = b64
                    chunk.caption = None
                    if not chunk.text:
                        chunk.text = "[figure]"
                    logger.debug("Cropped image chunk %s for visual embedding", chunk.chunk_id)
                else:
                    # Crop too small or failed — treat as placeholder
                    chunk.caption = None
                    chunk.text = chunk.text or "[figure]"
            else:
                logger.debug("Image chunk %s has no bbox; setting text='[figure]'", chunk.chunk_id)
                chunk.caption = None
                chunk.text = chunk.text or "[figure]"
        elif chunk.modality == "table":
            # Crop the table region for visual context — generation LLM can see the
            # actual table layout (merged cells, multi-level headers) alongside the
            # extracted markdown. Mirrors the image chunk crop pattern.
            if chunk.bbox is not None:
                b64 = _crop_image_chunk(chunk, pdf_path)
                if b64 is not None:
                    chunk.image_base64 = b64
            tasks.append(
                _enrich_table_single(chunk, client, semaphore, model, pdf_path=pdf_path)
            )
            counts["table"] += 1
        elif chunk.modality == "formula":
            tasks.append(_enrich_formula_single(chunk, client, semaphore, model))
            counts["formula"] += 1
        elif chunk.modality == "algorithm":
            tasks.append(_enrich_algorithm_single(chunk, client, semaphore, model))
            counts["algorithm"] += 1

    if not tasks:
        return chunks

    await asyncio.gather(*tasks)

    logger.info(
        "Processed %d image (cropped, no LLM) / %d table / %d formula / %d algorithm chunks from %s",
        counts["image"], counts["table"], counts["formula"], counts["algorithm"],
        pdf_path.name,
    )
    return chunks


# Backward-compatibility alias (ingest.py and any external callers)
async def enrich_image_chunks(
    chunks: list[Chunk],
    pdf_path: Path,
    client: AsyncOpenAI,
    max_concurrent: int = 5,
) -> list[Chunk]:
    """Deprecated alias for enrich_chunks — kept for backward compatibility."""
    return await enrich_chunks(
        chunks, pdf_path=pdf_path, client=client, max_concurrent=max_concurrent
    )
