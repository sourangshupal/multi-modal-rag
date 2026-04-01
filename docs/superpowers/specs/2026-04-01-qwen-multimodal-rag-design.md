# Qwen3 VL Full-Stack Multimodal RAG — Design Spec

**Date:** 2026-04-01
**Branch:** `qwen`
**Status:** Approved

---

## Problem

The current pipeline depends on three paid OpenAI cloud services:

| Service | Usage | Cost driver |
|---|---|---|
| GPT-4o | Image, table, formula, algorithm captioning during ingestion | Per-token, vision calls expensive |
| text-embedding-3-large | Text-only dense embeddings (3072-dim) | Per-token |
| GPT-4o-mini cross-encoder | Reranking retrieved candidates | Per-token |

Additionally, image chunks are embedded as text captions (lossy), not as images directly. This loses visual information.

---

## Goal

Replace all three with the open-source Qwen3 VL family:

1. **Remove image captioning entirely** — embed images directly as vectors via `Qwen3-VL-Embedding-2B`
2. **Keep table/formula/algo captioning** but switch the LLM to `Qwen3-VL-8B-Instruct` on RunPod serverless
3. **Replace text embeddings** with `Qwen3-VL-Embedding-2B` (unified multimodal vector space, 2048-dim)
4. **Enable Qwen reranker** — `QwenVLReranker` already implemented, just activate it

Zero new cloud API dependencies after this change.

---

## Model Selection

| Model | Role | Hosting | VRAM |
|---|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | Dense embedding (text + image) | Local in-process (L4/L40s) | ~4–5 GB |
| `Qwen/Qwen3-VL-Reranker-2B` | Reranking | Local in-process (already done) | ~4–5 GB |
| `Qwen/Qwen3-VL-8B-Instruct` | Captioning + generation | RunPod serverless vLLM | ~20–24 GB |

**Total local GPU budget (L4, 24GB):**

| Model | VRAM | Phase |
|---|---|---|
| GLM-OCR (Ollama) | ~4–6 GB | Ingestion parse |
| PP-DocLayout-V3-L | ~0.5–1.5 GB | Ingestion parse |
| Qwen3-VL-Embedding-2B | ~4–5 GB | Embed + query |
| Qwen3-VL-Reranker-2B | ~4–5 GB | Query rerank |
| **Peak (pre-warmed)** | **~13–18 GB** | All loaded |

L4 (24GB): 6–11 GB headroom ✅
L40s (48GB): 30+ GB headroom ✅ (recommended for multi-user)

---

## Architecture

### Ingestion Pipeline (After)

```
PDF
 └─ GLM-OCR + PP-DocLayout-V3 (parse)
     └─ structure_aware_chunking()
         └─ enrich_chunks() [image_captioner.py]
             ├─ image/figure → crop PNG → set chunk.image_base64 (NO LLM call)
             ├─ table        → crop PNG → Qwen3-VL-8B-Instruct/RunPod → chunk.text = markdown table
             ├─ formula      → crop PNG → Qwen3-VL-8B-Instruct/RunPod → chunk.text = verbal description
             ├─ algorithm    → crop PNG → Qwen3-VL-8B-Instruct/RunPod → chunk.text = verbal description
             └─ text         → no change
         └─ embed_chunks() [embedder.py]
             ├─ image chunks  → QwenVLEmbedder.embed_images(image_base64) → 2048-dim vector
             └─ text chunks   → QwenVLEmbedder.embed(texts)              → 2048-dim vector
         └─ Qdrant upsert (multimodal_dense: 2048-dim + bm25_sparse)
```

### Query Pipeline (After)

```
User query (text)
 └─ QwenVLEmbedder.embed([query])        → 2048-dim vector
 └─ Qdrant hybrid search (RRF)           → top-k candidates
 └─ QwenVLReranker.rerank(query, cands)  → image chunks scored via image_base64
 └─ Top-n results returned
```

---

## Component Changes

### `image_captioner.py`
- **Remove** `_enrich_image_single()` function
- **Keep** PDF crop logic (`pdf_page_to_image()`) — populates `chunk.image_base64` for embedder + reranker
- **Keep** `_enrich_table_single()`, `_enrich_formula_single()`, `_enrich_algorithm_single()` — unchanged code, LLM endpoint switches via env vars
- Image chunks: `chunk.text` = figure title from chunker, or `"[figure]"` fallback. `chunk.caption = None`.

### `embedder.py`
- **Add** `QwenVLEmbedder(BaseEmbedder)` — in-process HuggingFace model, same pattern as `QwenVLReranker`
- **Add** `embed_images(images_b64: list[str]) -> list[list[float]]` method (not on ABC)
- **Modify** `embed_chunks()` — split chunks by modality; image chunks with `image_base64` go to `embed_images()`, all others go to `embed(texts)`
- **Add** `"qwen"` to `_PROVIDERS` factory dict

### `vector_store.py`
- **Rename** `text_dense` → `multimodal_dense` everywhere
- **Update** dimension from `3072` → `settings.embedding_dimensions` (default now 2048)

### `config.py`
- **Add** `qwen_embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"`
- **Change** `embedding_dimensions` default: `3072` → `2048`

### `pyproject.toml`
- **Merge** `[qwen]` extra: add `qwen-vl-utils>=0.0.14`, `Pillow>=11.0`, `accelerate>=1.13`

### `docker-compose.yml`
- **Add** env vars to `app` service: `EMBEDDING_PROVIDER=qwen`, `RERANKER_BACKEND=qwen`, `QWEN_EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS=2048`

### `.env.example`
- **Update** to show RunPod endpoint pattern and Qwen vars

---

## Configuration

```dotenv
# RunPod serverless (replaces OpenAI for captioning + generation)
OPENAI_API_KEY=<runpod_api_key>
OPENAI_BASE_URL=https://api.runpod.ai/v2/<endpoint-id>/openai/v1
OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# Local Qwen embedding (in-process)
EMBEDDING_PROVIDER=qwen
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DIMENSIONS=2048

# Local Qwen reranker (already implemented)
RERANKER_BACKEND=qwen
```

`OPENAI_BASE_URL` + `OPENAI_API_KEY` now point to RunPod. The existing `AsyncOpenAI` client in `image_captioner.py` requires zero code changes — only env var changes.

---

## Qdrant Migration

The collection must be recreated because:
- Vector name changes: `text_dense` → `multimodal_dense`
- Dimension changes: 3072 → 2048

Run with `--overwrite` flag on first ingest:
```bash
python scripts/ingest.py paper.pdf --overwrite
```

---

## License Note

`Qwen3-VL-Embedding-2B` is trained on MS MARCO (non-commercial dataset). Released under Apache 2.0 but commercial use should be verified with Alibaba/Qwen team before production deployment.

---

## Verification

```bash
# 1. Unit tests (no GPU)
uv run pytest tests/unit/ -v

# 2. Parse sanity check
uv run python ollama/test_parse.py data/raw/paper.pdf

# 3. Full ingestion
python scripts/ingest.py paper.pdf --overwrite
# Expect: 2048-dim vectors, image chunks have image_base64, caption=null

# 4. Search
python scripts/search.py "query"
# Expect: results with rerank_score from QwenVLReranker

# 5. Qdrant dashboard: http://localhost:6333/dashboard
# Collection → multimodal_dense: 2048-dim COSINE
```
