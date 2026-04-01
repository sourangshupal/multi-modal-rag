# Future Improvements — Qwen Multimodal RAG

Tracked improvements identified during the Qwen3 VL migration. Ordered by impact within each category.

---

## Retrieval Quality

### 1. Row-level sub-chunking for large tables
**Problem:** A table with 40+ rows is stored as a single atomic chunk. A question about row 35 retrieves the whole table, but the LLM may still miss the specific value because the context is too dense.

**Fix:** During ingestion, split large tables into sub-chunks of N rows each, where each sub-chunk inherits the column headers. Index each row-group as a separate Qdrant point. A specific-value query then hits the right row-group directly.

**Files:** `src/doc_parser/chunker.py`, `src/doc_parser/ingestion/image_captioner.py`

**Effort:** ~1 day

---

### 2. Context window expansion at generation time
**Problem:** A retrieved chunk is missing the section title, introductory paragraph, or column-definition text that lives in adjacent chunks. The LLM answers without full context.

**Fix:** After reranking, fetch ±1 adjacent chunks from Qdrant by `(source_file, page, chunk_id)` and prepend/append them to the context passed to the generation LLM. Keep the retrieved chunk as the primary source, neighbors as supporting context.

**Files:** `src/doc_parser/api/routes/generate.py`, `src/doc_parser/ingestion/vector_store.py` (add `get_neighbors()` method)

**Effort:** ~2 hours

---

### 3. HyDE — Hypothetical Document Embeddings
**Problem:** User queries use different vocabulary than the document. "What is the recall score for model X?" may not match a table summary that says "performance metrics across benchmarks."

**Fix:** Before embedding the query, ask the LLM to generate a short hypothetical answer ("a technical document excerpt that would answer this question"), then embed that instead of the raw query. This bridges the query-to-document vocabulary gap.

**Files:** `src/doc_parser/api/routes/generate.py` (add HyDE pre-step), `src/doc_parser/api/schemas.py` (add `use_hyde: bool = False` to `GenerateRequest`)

**Effort:** ~2 hours

---

### 4. Drop LLM captioning for tables — use OCR text directly
**Problem:** `_enrich_table_single()` calls the RunPod LLM for every table chunk, which is the main ingestion bottleneck. GLM-OCR already extracts table content as structured markdown. The LLM caption adds semantic summarisation but often not enough to justify the latency and cost.

**Fix:** Skip `_enrich_table_single()` for tables. Use `chunk.text` (GLM-OCR output) directly for embedding + retrieval. The table image (`chunk.image_base64`) is already passed to the generation LLM, so visual context is preserved.

**Trade-off:** Slightly weaker semantic retrieval for tables (no summary), but zero LLM calls during ingestion for table chunks.

**Files:** `src/doc_parser/ingestion/image_captioner.py`

**Effort:** ~30 minutes

---

### 5. Extend direct image embedding to tables and formulas
**Problem:** Tables and formulas still go through the text-embedding path (embed their caption/summary as text). The Qwen3-VL-Embedding-2B model can embed any image directly — tables and formula screenshots could be embedded visually just like figures.

**Fix:** In `embed_chunks()`, route table/formula chunks with `image_base64` to `embedder.embed_images()` the same way image chunks are routed today. This unifies all visual content in the same embedding space and removes the text-summary dependency for retrieval.

**Trade-off:** Sparse BM25 vectors become empty for these chunks, reducing keyword-search quality. Hybrid search relies more heavily on the dense vector.

**Files:** `src/doc_parser/ingestion/embedder.py` (`embed_chunks()` routing condition)

**Effort:** ~1 hour

---

## Model / Inference

### 6. Fix QwenVLReranker device selection — add CUDA support
**Problem:** `QwenVLReranker.__init__` only checks for MPS (Apple Silicon), defaulting to CPU on CUDA-capable machines (L4/L40s). The embedder correctly checks CUDA first. On a GPU server, both models should land on CUDA.

**Fix:** Add `cuda` check before `mps` in `QwenVLReranker`:
```python
if torch.cuda.is_available():
    self._device = "cuda"
elif torch.backends.mps.is_available():
    self._device = "mps"
else:
    self._device = "cpu"
```

**Files:** `src/doc_parser/retrieval/reranker.py` (lines ~310–315)

**Effort:** 5 minutes

---

### 7. Batch size limit in QwenVLEmbedder
**Problem:** `_embed_texts_sync()` and `_embed_images_sync()` pass the entire input list to the processor at once. For a document with many chunks, all tensors materialise simultaneously. On an L4 with 24 GB this could cause OOM during large-batch ingestion.

**Fix:** Add a `batch_size: int = 32` parameter (matching `OpenAIEmbedder`'s batching pattern). Process chunks in batches, concatenate results.

**Files:** `src/doc_parser/ingestion/embedder.py` (`QwenVLEmbedder._embed_texts_sync`, `_embed_images_sync`)

**Effort:** ~1 hour

---

### 8. RunPod min_workers=1 to eliminate cold start
**Problem:** RunPod serverless spins down when idle. An 8B model takes 30–90 seconds to load on a cold pod. Sporadic ingestion jobs (one document every few hours) pay this penalty every time.

**Fix:** Set `min_workers=1` in the RunPod endpoint configuration. Keeps one pod warm at all times. Small idle cost but eliminates cold start entirely for table/formula captioning and generation.

**Where:** RunPod dashboard → Serverless endpoint settings (not a code change)

**Effort:** 2 minutes

---

### 9. Verify Qwen3-VL-Embedding image-only processor call
**Problem:** `_embed_images_sync()` calls `self._processor(images=images, ...)` without a `text=` argument. Some multimodal processors require a paired text prompt even for image-only inference. If the processor raises, a stub text list (`text=[""] * len(images)`) is needed.

**Fix:** Test with the real model on first deployment. If it raises, add:
```python
inputs = self._processor(
    text=[""] * len(images),
    images=images,
    return_tensors="pt",
    padding=True,
)
```

**Files:** `src/doc_parser/ingestion/embedder.py` (`QwenVLEmbedder._embed_images_sync`)

**Effort:** 15 minutes (verify + fix if needed)

---

## Infrastructure

### 10. Pre-warm all local models at startup
**Problem:** On container restart, `QwenVLEmbedder` and `QwenVLReranker` are loaded lazily on first request. The first query after restart is slow.

**Fix:** Extend `scripts/pre_warm.py` (currently warms GLM-OCR and PP-DocLayout-V3) to also trigger the embedder and reranker singletons so they load at startup.

**Files:** `scripts/pre_warm.py`, `src/doc_parser/api/dependencies.py`

**Effort:** ~1 hour

---

## Notes

- Items 1–3 (row sub-chunking, context expansion, HyDE) have the highest impact on specific-value queries (the main failure mode identified).
- Item 6 (reranker CUDA) should be fixed before first L4/L40s deployment — easy win.
- Items 4 and 5 (drop table captioning, extend image embedding) can be enabled together as a "zero LLM ingestion" mode via a single env flag.
