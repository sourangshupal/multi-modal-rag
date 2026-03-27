"""Multimodal chunk enricher: generates structured descriptions for images, tables,
formulas, and algorithms via GPT-4o to improve embedding quality for retrieval."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

from doc_parser.chunker import Chunk
from doc_parser.utils.pdf_utils import pdf_page_to_image

logger = logging.getLogger(__name__)

# Minimum crop dimension in pixels; smaller regions are likely detection noise
_MIN_CROP_SIZE_PX: int = 50

# ── Prompts ───────────────────────────────────────────────────────────────────

_IMAGE_SYSTEM_PROMPT = """\
You are a scientific figure analysis assistant for a document retrieval system.
Analyze the figure and respond in EXACTLY this format with no extra text:

CAPTION: <1-2 sentence description of what the figure shows overall — for semantic search.>
FLOW: <Numbered step-by-step description of the process or sequence depicted. If no sequence exists, describe the key visual components instead. Each step on a new line.>
STRUCTURE: <Grouping and containment relationships — which components belong to which group or module. Use dashes for sub-items.>

Be specific and technical. Do not invent information not visible in the figure.\
"""

_TABLE_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given a table from a research document, respond in EXACTLY this format:

SUMMARY: <One paragraph describing what this table shows, what it measures or compares, and its significance in the document.>
DETAIL: <2-5 bullet points listing the key takeaways, notable values, trends, or comparisons visible in the data.>

Be specific about column names, row labels, and numbers present. Do not invent values.\
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


# ── Per-modality enrichment helpers ──────────────────────────────────────────

async def _enrich_image_single(
    chunk: Chunk,
    pdf_path: Path,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
) -> None:
    """Crop the PDF region and generate a structured image description in-place."""
    async with semaphore:
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
                chunk.text = "[figure]"
                return

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ],
                    },
                ],
                max_tokens=512,
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
) -> None:
    """Generate a semantic table summary in-place."""
    async with semaphore:
        try:
            raw = chunk.text
            # Guard against very large tables
            table_text = raw[:3000] + "\n...[truncated]" if len(raw) > 3000 else raw

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is a table from a research document:\n\n{table_text}\n\n"
                            "Provide a semantic description for document retrieval."
                        ),
                    },
                ],
                max_tokens=400,
                temperature=0.0,
            )

            enriched = (response.choices[0].message.content or "").strip()
            chunk.caption, chunk.text = _parse_text_response(raw, enriched)

            logger.debug("Enriched table chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning("Table enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)


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
    """Enrich all non-text chunks with GPT-4o generated structured descriptions.

    Dispatches by modality:
    - image   → structured CAPTION / FLOW / STRUCTURE description (vision call)
    - table   → SUMMARY / DETAIL semantic description (text call)
    - formula → SUMMARY / DETAIL verbal description (text call)
    - algorithm → SUMMARY / DETAIL semantic description (text call)
    - text    → unchanged

    For each modality the enriched text replaces ``chunk.text`` (which is what
    gets embedded) while the original raw content is preserved in ``chunk.caption``.
    For images, ``chunk.caption`` holds only the short 1-2 sentence caption line.

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

    for chunk in chunks:
        if chunk.modality == "image":
            if chunk.bbox is not None:
                tasks.append(_enrich_image_single(chunk, pdf_path, client, semaphore, model))
                counts["image"] += 1
            else:
                logger.debug("Image chunk %s has no bbox; setting text='[figure]'", chunk.chunk_id)
                chunk.text = "[figure]"
        elif chunk.modality == "table":
            tasks.append(_enrich_table_single(chunk, client, semaphore, model))
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
