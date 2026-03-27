"""POST /ingest endpoint — file upload and JSON-path variants."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from qdrant_client.models import SparseVector

from doc_parser.api.dependencies import get_embedder_dep, get_openai_client, get_store
from doc_parser.api.schemas import IngestRequest, IngestResponse
from doc_parser.chunker import Chunk, document_aware_chunking
from doc_parser.config import get_settings
from doc_parser.ingestion.embedder import embed_chunks
from doc_parser.ingestion.image_captioner import enrich_chunks
from doc_parser.pipeline import DocumentParser

_CHUNKS_OUTPUT_DIR = Path("data/chunks")

router = APIRouter()


def _save_chunks_to_disk(
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list[SparseVector],
    source_name: str,
) -> None:
    """Save enriched chunks + embeddings to data/chunks/<stem>.json for inspection.

    Never raises — a write failure is logged as a warning but does not abort ingestion.
    image_base64 is excluded (binary noise; not useful for enrichment review).
    """
    try:
        _CHUNKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(source_name).stem
        out_path = _CHUNKS_OUTPUT_DIR / f"{stem}.json"

        records = []
        for chunk, d_emb, s_emb in zip(chunks, dense, sparse):
            records.append({
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "modality": chunk.modality,
                "element_types": chunk.element_types,
                "is_atomic": chunk.is_atomic,
                "bbox": chunk.bbox,
                "source_file": chunk.source_file,
                "text": chunk.text,
                "caption": chunk.caption,
                "dense_embedding": d_emb,
                "sparse_embedding": {
                    "indices": s_emb.indices,
                    "values": s_emb.values,
                },
            })

        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        logger.info("Saved {} chunks to {}", len(records), out_path)
    except Exception:
        logger.warning("Failed to save chunks debug file for {}", source_name, exc_info=True)


async def _run_ingest(
    pdf_path: Path,
    collection_override: str | None,
    overwrite: bool,
    max_chunk_tokens: int,
    caption: bool,
    display_name: str | None = None,
) -> IngestResponse:
    """Core ingest logic shared by both endpoint variants.

    Args:
        pdf_path: Actual file path used for I/O (may be a temp file).
        display_name: Human-readable filename stored in chunk metadata and the
            response (e.g. the original upload filename). Falls back to
            ``pdf_path.name`` when not provided.
    """
    settings = get_settings()
    client = get_openai_client()
    embedder = get_embedder_dep()
    store = get_store()

    # Use the original filename for all metadata; pdf_path is only for I/O
    source_name = display_name or pdf_path.name

    # Override collection name when requested
    if collection_override:
        store._collection = collection_override
    collection = store._collection

    t0 = time.perf_counter()

    # 1. Parse PDF (synchronous SDK call — offload to thread pool)
    try:
        parser = DocumentParser()
        loop = asyncio.get_running_loop()
        parse_result = await loop.run_in_executor(None, parser.parse_file, pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Parsing failed for {}: {}", source_name, exc)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}") from exc

    # 2. Chunk — single pass over all pages so section headings and figure captions
    #    are linked correctly even across page boundaries.
    chunks = document_aware_chunking(
        [(page.page_num, page.elements) for page in parse_result.pages],
        source_file=source_name,
        max_chunk_tokens=max_chunk_tokens,
    )
    logger.info("Chunked {} → {} chunks", source_name, len(chunks))

    # 3. Caption image chunks (if enabled)
    if caption and settings.image_caption_enabled:
        chunks = await enrich_chunks(
            chunks,
            pdf_path=pdf_path,
            client=client,
            model=settings.openai_llm_model,
        )

    # 4. Embed
    dense, sparse = await embed_chunks(chunks, embedder, settings)

    # 4b. Persist chunks + embeddings to disk for inspection
    _save_chunks_to_disk(chunks, dense, sparse, source_name)

    # 5. Ensure collection exists then upsert
    await store.create_collection(overwrite=overwrite)
    upserted = await store.upsert_chunks(chunks, dense, sparse)

    latency_ms = (time.perf_counter() - t0) * 1000
    modality_counts = dict(Counter(c.modality for c in chunks))

    return IngestResponse(
        source_file=source_name,
        collection=collection,
        chunks_upserted=upserted,
        modality_counts=modality_counts,
        latency_ms=round(latency_ms, 2),
    )


_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"})


@router.post("/file", response_model=IngestResponse, summary="Ingest document via file upload")
async def ingest_file(
    file: UploadFile = File(..., description="Document file to ingest (PDF or image)."),
    collection: str | None = Form(None, description="Override collection name. Leave blank to use the default from QDRANT_COLLECTION_NAME env var.", example=None),
    overwrite: bool = Form(False, description="Recreate collection before ingesting."),
    max_chunk_tokens: int = Form(512, ge=64, le=4096, description="Max tokens per chunk."),
    caption: bool = Form(True, description="Run GPT-4o captioning on image chunks."),
) -> IngestResponse:
    """Upload a PDF or image file and ingest it into the vector store."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if not file.filename or suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    # Save upload to a temp file (preserve original suffix for parser)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return await _run_ingest(
            tmp_path, collection, overwrite, max_chunk_tokens, caption,
            display_name=file.filename,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("", response_model=IngestResponse, summary="Ingest document by file path")
async def ingest_by_path(req: IngestRequest) -> IngestResponse:
    """Ingest a document referenced by its local file path."""
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return await _run_ingest(file_path, req.collection, req.overwrite, req.max_chunk_tokens, req.caption)
