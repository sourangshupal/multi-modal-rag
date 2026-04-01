"""POST /generate endpoint — full RAG in one call."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from loguru import logger

from doc_parser.api.dependencies import (
    get_embedder_dep,
    get_openai_client,
    get_reranker_dep,
    get_store,
)
from doc_parser.api.schemas import ChunkResult, GenerateRequest, GenerateResponse
from doc_parser.config import get_settings

router = APIRouter()

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say \"I don't have enough information to answer this.\" "
    "Cite the source page numbers when possible."
)


@router.post("", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Retrieve relevant chunks and generate an answer with GPT-4o.

    1. Embed query → hybrid search in Qdrant.
    2. Optionally rerank candidates.
    3. Build context string from top-n chunks.
    4. Call GPT-4o and return answer + source chunks.
    """
    settings = get_settings()
    store = get_store()
    embedder = get_embedder_dep()
    reranker = get_reranker_dep()
    client = get_openai_client()

    top_n = req.top_n if req.top_n is not None else settings.reranker_top_n

    t0 = time.perf_counter()

    try:
        candidates = await store.search(
            query_text=req.query,
            embedder=embedder,
            settings=settings,
            top_k=req.top_k,
            filter_modality=req.filter_modality,
        )
    except Exception as exc:
        logger.exception("Search failed: {}", exc)
        raise HTTPException(status_code=502, detail=f"Vector store search failed: {exc}") from exc

    total_candidates = len(candidates)

    if req.rerank and candidates:
        try:
            candidates = await reranker.rerank(req.query, candidates, top_n=top_n)
        except Exception as exc:
            logger.exception("Reranking failed: {}", exc)
            raise HTTPException(status_code=502, detail=f"Reranking failed: {exc}") from exc
    else:
        for c in candidates:
            c.setdefault("rerank_score", None)
        candidates = candidates[:top_n]

    # Build modality-aware context.
    # - Text/formula/algorithm chunks: plain text in a single context string.
    # - Image chunks with base64: injected as image_url blocks so the VLM sees pixels.
    # - Table chunks: full markdown text + table image (if available) for maximum fidelity.
    text_context_parts: list[str] = []
    visual_parts: list[dict] = []

    for c in candidates:
        page = c.get("page", "?")
        modality = c.get("modality", "text")
        b64 = c.get("image_base64")

        if modality == "table":
            caption = c.get("caption") or ""
            summary = c.get("text") or ""
            if caption and summary:
                table_text = f"{summary}\n\nFull table data:\n{caption}"
            else:
                table_text = caption or summary
            text_context_parts.append(f"[page {page}, table]\n{table_text}")
            if b64:
                visual_parts.append({"type": "text", "text": f"[page {page}, table image:]"})
                visual_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        elif modality == "image" and b64:
            label = c.get("text") or "[figure]"
            visual_parts.append({"type": "text", "text": f"[page {page}, figure: {label}]"})
            visual_parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )

        else:
            text = c.get("text", "") or c.get("caption") or ""
            text_context_parts.append(f"[page {page}] {text}")

    text_context = "\n\n".join(text_context_parts)
    system_prompt = req.system_prompt or _DEFAULT_SYSTEM_PROMPT

    # Use multimodal message content when visuals are present; fall back to
    # plain string for text-only documents (avoids unnecessary overhead).
    if visual_parts:
        preamble = f"Context:\n{text_context}" if text_context else "Context: (see images/tables below)"
        user_content: str | list[dict] = (
            [{"type": "text", "text": preamble}]
            + visual_parts
            + [{"type": "text", "text": f"\nQuestion: {req.query}"}]
        )
    else:
        user_content = f"Context:\n{text_context}\n\nQuestion: {req.query}"

    try:
        completion = await client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=req.max_tokens,
            temperature=0.0,
        )
        answer = completion.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("LLM generation failed: {}", exc)
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    sources = [
        ChunkResult(
            chunk_id=c.get("chunk_id", ""),
            text=c.get("text", ""),
            source_file=c.get("source_file", ""),
            page=c.get("page", 0),
            modality=c.get("modality", "text"),
            element_types=c.get("element_types", []),
            bbox=c.get("bbox"),
            is_atomic=c.get("is_atomic", False),
            caption=c.get("caption"),
            rerank_score=c.get("rerank_score"),
        )
        for c in candidates
    ]

    return GenerateResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        total_candidates=total_candidates,
        latency_ms=round(latency_ms, 2),
    )
