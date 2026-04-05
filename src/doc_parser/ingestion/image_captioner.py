"""Multimodal chunk enricher: generates structured descriptions for images, tables,
formulas, and algorithms via GPT-4o to improve embedding quality for retrieval."""
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
_IMAGE_MAX_TOKENS: int = 800

# ── Prompts ───────────────────────────────────────────────────────────────────

_IMAGE_SYSTEM_PROMPT = """\
You are a scientific figure analysis assistant for a document retrieval system.

First, classify the figure into one of these types:
CHART — bar charts, line graphs, scatter plots, pie charts, heatmaps
DIAGRAM — flowcharts, architecture diagrams, block diagrams, network diagrams
PHOTO — photographs, microscopy images, medical scans
SCREENSHOT — UI screenshots, code screenshots, terminal output
OTHER — any figure that does not fit the above categories

Then analyze the figure and respond in EXACTLY this format with no extra text:

TYPE: <CHART | DIAGRAM | PHOTO | SCREENSHOT | OTHER>
CAPTION: <1-2 sentence description of what the figure shows overall — for semantic search.>
DETAIL:
- For CHART: describe chart type, all axis labels, data series names, key data points, and the main trend or comparison.
- For DIAGRAM: describe all components, their labels, connections, and the overall flow or hierarchy.
- For PHOTO: describe the subject, setting, notable features, and any annotations or labels.
- For SCREENSHOT: describe the UI elements, visible text, layout, and what operation is shown.
- For OTHER: describe the key visual components, their arrangement, and purpose.
STRUCTURE: <Grouping and containment relationships — which components belong to which group or module. Use dashes for sub-items.>

Be specific and technical. Reference labels, numbers, and text visible in the figure. Do not invent information not visible in the figure.\
"""

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

def _parse_image_response(text: str) -> tuple[str, str]:
    """Return (short_caption, full_structured_text) from a GPT-4o image response."""
    caption = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("CAPTION:"):
            caption = stripped[len("CAPTION:"):].strip()
            break
    if not caption:
        caption = text.strip()[:200]
    return caption, text.strip()


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


def _get_surrounding_context(chunks: list[Chunk], idx: int, max_chars: int = 400) -> str:
    """Extract text from adjacent text chunks for document context.

    Looks up to 2 positions before and after the target chunk,
    within the same or adjacent pages.
    """
    target = chunks[idx]
    parts: list[str] = []

    # Look backward
    for i in range(max(0, idx - 2), idx):
        c = chunks[i]
        if c.modality == "text" and abs(c.page - target.page) <= 1:
            parts.append(c.text[:max_chars])

    # Look forward
    for i in range(idx + 1, min(len(chunks), idx + 3)):
        c = chunks[i]
        if c.modality == "text" and abs(c.page - target.page) <= 1:
            parts.append(c.text[:max_chars])

    combined = " ... ".join(parts)
    return combined[:max_chars * 2] if combined else ""


# ── Per-modality enrichment helpers ──────────────────────────────────────────

def _crop_chunk_to_base64(
    pdf_path: Path,
    chunk: Chunk,
) -> str | None:
    """Crop the PDF region for a chunk and return a base64-encoded PNG.

    Returns None when bbox is missing or the crop is below the minimum size
    threshold. Does not raise — callers should treat None as no image available.
    """
    if chunk.bbox is None:
        return None
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


async def _enrich_image_single(
    chunk: Chunk,
    pdf_path: Path,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    surrounding_context: str = "",
) -> None:
    """Crop the PDF region and generate a structured image description in-place."""
    async with semaphore:
        try:
            b64 = _crop_chunk_to_base64(pdf_path, chunk)
            if b64 is None:
                logger.debug("Skipping tiny/missing crop for image chunk %s", chunk.chunk_id)
                chunk.text = "[figure]"
                return

            user_content: list[dict] = []
            if surrounding_context:
                user_content.append({
                    "type": "text",
                    "text": (
                        f"Surrounding document context (use for reference):\n"
                        f"{surrounding_context}"
                    ),
                })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=_IMAGE_MAX_TOKENS,
                temperature=0.0,
            )

            raw_response = (response.choices[0].message.content or "").strip()
            caption, full_text = _parse_image_response(raw_response)

            chunk.caption = caption
            chunk.text = full_text
            chunk.image_base64 = b64

            logger.debug("Enriched image chunk %s: %s", chunk.chunk_id, caption[:80])

        except Exception:
            logger.warning("Image enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)
            chunk.text = "[figure]"


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

            # Also crop and store the visual for multimodal generation
            if pdf_path is not None and chunk.bbox is not None:
                try:
                    chunk.image_base64 = _crop_chunk_to_base64(pdf_path, chunk)
                except Exception:
                    logger.warning(
                        "Table crop failed for chunk %s", chunk.chunk_id, exc_info=True
                    )

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
    pdf_path: Path | None = None,
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

            # Also crop and store the visual for multimodal generation
            if pdf_path is not None and chunk.bbox is not None:
                try:
                    chunk.image_base64 = _crop_chunk_to_base64(pdf_path, chunk)
                except Exception:
                    logger.warning(
                        "Formula crop failed for chunk %s", chunk.chunk_id, exc_info=True
                    )

            logger.debug("Enriched formula chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning("Formula enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)


async def _enrich_algorithm_single(
    chunk: Chunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    pdf_path: Path | None = None,
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

            # Also crop and store the visual for multimodal generation
            if pdf_path is not None and chunk.bbox is not None:
                try:
                    chunk.image_base64 = _crop_chunk_to_base64(pdf_path, chunk)
                except Exception:
                    logger.warning(
                        "Algorithm crop failed for chunk %s", chunk.chunk_id, exc_info=True
                    )

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
    """Enrich all non-text chunks with GPT-4o generated structured descriptions.

    Dispatches by modality:
    - image   → structured TYPE / CAPTION / DETAIL / STRUCTURE description (vision call
                with surrounding document context for better understanding)
    - table   → complete markdown table reproduction + semantic summary (JSON mode)
    - formula → SUMMARY / DETAIL verbal description (text call)
    - algorithm → SUMMARY / DETAIL semantic description (text call)
    - text    → unchanged

    For tables, ``chunk.caption`` holds the full markdown table (for the generation
    LLM) and ``chunk.text`` holds the semantic summary (for embedding/retrieval).
    For images, ``chunk.caption`` holds the short 1-2 sentence caption line.

    Args:
        chunks: All chunks from the document (mixed modalities).
        pdf_path: Path to the source PDF (used for image crop rendering).
        client: Authenticated AsyncOpenAI client.
        model: OpenAI model to use for enrichment (default "gpt-4o").
        max_concurrent: Max concurrent API calls (shared semaphore across all modalities).

    Returns:
        The same list mutated in-place with enriched ``text`` and ``caption`` fields.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    counts: dict[str, int] = defaultdict(int)

    for idx, chunk in enumerate(chunks):
        if chunk.modality == "image":
            if chunk.bbox is not None:
                context = _get_surrounding_context(chunks, idx)
                tasks.append(
                    _enrich_image_single(
                        chunk, pdf_path, client, semaphore, model,
                        surrounding_context=context,
                    )
                )
                counts["image"] += 1
            else:
                logger.debug("Image chunk %s has no bbox; setting text='[figure]'", chunk.chunk_id)
                chunk.text = "[figure]"
        elif chunk.modality == "table":
            tasks.append(
                _enrich_table_single(chunk, client, semaphore, model, pdf_path=pdf_path)
            )
            counts["table"] += 1
        elif chunk.modality == "formula":
            tasks.append(
                _enrich_formula_single(chunk, client, semaphore, model, pdf_path=pdf_path)
            )
            counts["formula"] += 1
        elif chunk.modality == "algorithm":
            tasks.append(
                _enrich_algorithm_single(chunk, client, semaphore, model, pdf_path=pdf_path)
            )
            counts["algorithm"] += 1

    if not tasks:
        return chunks

    await asyncio.gather(*tasks)

    logger.info(
        "Enriched %d image / %d table / %d formula / %d algorithm chunks from %s",
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
