# Qwen Branch — Testing Guide

Complete reference for testing the `qwen` branch end-to-end.

---

## Stack Overview

| Component | Model / Service | Runs Where |
|---|---|---|
| **OCR / Parsing** | GLM-OCR (GLM-4V) | Z.AI MaaS cloud API |
| **Layout detection** | PP-DocLayout-V3-L | Ollama (local, only in `ollama` mode) |
| **Embedding** | Qwen3-VL-Embedding-2B | Local — in-process HuggingFace |
| **Reranker** | Qwen3-VL-Reranker-2B | Local — in-process HuggingFace |
| **LLM (captions + generation)** | Qwen3-VL-8B-Instruct | RunPod serverless (OpenAI-compat API) |
| **Vector DB** | Qdrant | Local Docker |

---

## 1. API Keys Required

### Mandatory

| Key | Where to get it | Used for |
|---|---|---|
| `Z_AI_API_KEY` | [z.ai](https://z.ai) console | GLM-OCR document parsing (cloud mode) |
| `OPENAI_API_KEY` | RunPod console → your serverless endpoint | Table/formula captioning + RAG generation |
| `OPENAI_BASE_URL` | RunPod console → endpoint URL | Points to your Qwen3-VL-8B-Instruct endpoint |

### Optional

| Key | When needed |
|---|---|
| `JINA_API_KEY` | Only if you override `RERANKER_BACKEND=jina` for comparison testing |

> **Note:** The local Qwen embedding and reranking models load from HuggingFace automatically on first run. No API key needed — just internet access for the initial download (~4–5 GB each). After that they run entirely offline.

---

## 2. RunPod Endpoint Setup

The LLM (`Qwen3-VL-8B-Instruct`) runs on RunPod serverless. Before testing:

1. Log in to [runpod.io](https://runpod.io)
2. Create a serverless endpoint with `Qwen/Qwen3-VL-8B-Instruct` via vLLM
3. Set `min_workers=1` to avoid cold-start delays (30–90 s otherwise)
4. Copy the endpoint ID — your `OPENAI_BASE_URL` will be:
   ```
   https://api.runpod.ai/v2/<endpoint-id>/openai/v1
   ```
5. Copy your RunPod API key → this is your `OPENAI_API_KEY`

---

## 3. Environment Setup

### 3.1 Confirm branch

```bash
git branch --show-current   # must output: qwen
```

### 3.2 Install dependencies

```bash
uv venv --python 3.12
source .venv/bin/activate

# Base + dev tools
uv pip install -e ".[dev]"

# Qwen local models (embedding + reranker)
uv pip install -e ".[qwen]"
```

> The `[qwen]` extra installs `transformers`, `torch`, `accelerate`, and `Pillow`.
> First load of each model downloads ~4–5 GB from HuggingFace.

### 3.3 Create `.env`

```bash
cp .env.example .env
```

Fill in `.env`:

```dotenv
# ── Parser ─────────────────────────────────────────────────────
PARSER_BACKEND=cloud
Z_AI_API_KEY=<your_z_ai_key>

# ── LLM — RunPod (Qwen3-VL-8B-Instruct) ───────────────────────
OPENAI_API_KEY=<your_runpod_api_key>
OPENAI_BASE_URL=https://api.runpod.ai/v2/<endpoint-id>/openai/v1
OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# ── Embedding — local Qwen ────────────────────────────────────
EMBEDDING_PROVIDER=qwen
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DIMENSIONS=2048

# ── Reranker — local Qwen ─────────────────────────────────────
RERANKER_BACKEND=qwen
RERANKER_TOP_N=5

# ── Qdrant ────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents

# ── Feature flags ─────────────────────────────────────────────
IMAGE_CAPTION_ENABLED=true

# ── API server ────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_JSON=false
```

### 3.4 Start Qdrant

```bash
docker compose up qdrant -d

# Verify it's running
curl http://localhost:6333/healthz    # → {"title":"qdrant - vector search engine"}
```

---

## 4. Unit Tests (no API keys needed — ~3–4 s)

These are fully mocked. Run them first to verify the code is intact before any live calls.

```bash
uv run pytest tests/unit/ -v
```

Expected: **127 passed, 0 failed** (with a few unrelated deprecation warnings from qdrant-client SWIG bindings).

To run a single test file:

```bash
uv run pytest tests/unit/test_embedder.py -v
uv run pytest tests/unit/test_reranker.py -v
uv run pytest tests/unit/test_image_captioner.py -v
uv run pytest tests/unit/test_generate.py -v
```

---

## 5. Integration Tests (require live API keys)

These make real network calls. Set your keys in `.env` first.

```bash
# Full integration suite
uv run pytest tests/integration/ -v -m integration

# Parse only (needs Z_AI_API_KEY)
uv run pytest tests/integration/test_pipeline_e2e.py -v

# Full ingest pipeline (needs Z_AI_API_KEY + OPENAI_API_KEY)
uv run pytest tests/integration/test_ingest_e2e.py -v
```

Tests auto-skip if the required keys are absent.

---

## 6. Manual Pipeline Testing (step by step)

Use a real PDF — ideally a scientific paper with tables, figures, and formulas.

### Step 1 — Parse only (verify GLM-OCR output)

```bash
python scripts/parse.py data/raw/paper.pdf --chunks
```

Expected output: per-chunk list showing `modality`, `text`, `page`, `is_atomic`.
Check that tables, figures, formulas appear as separate atomic chunks.

To see the raw element JSON from GLM-OCR:

```bash
python scripts/parse.py data/raw/paper.pdf --format json
```

### Step 2 — Full ingestion (parse → caption → embed → upsert)

First ingest (creates the collection):

```bash
python scripts/ingest.py data/raw/paper.pdf
```

If re-ingesting or changing dimensions:

```bash
python scripts/ingest.py data/raw/paper.pdf --overwrite
```

**What to watch for in the logs:**
- `QwenVLEmbedder loaded ... on cuda:0` (or `mps` / `cpu`) — model loaded
- `Captioning N table/formula/algorithm chunks` — RunPod LLM being called
- `Image chunk: cropped bbox, stored image_base64` — images handled locally (no LLM call)
- `Upserted N points` — vectors stored in Qdrant
- No errors in the RunPod call (if RunPod cold-starts, expect 30–90 s pause on first table chunk)

### Step 3 — Search + rerank

```bash
python scripts/search.py "your query here"

# With modality filter
python scripts/search.py "table showing accuracy results" --filter-modality table

# More candidates before reranking
python scripts/search.py "attention mechanism figure" --top-k 30 --top-n 5

# Skip reranker (raw retrieval only)
python scripts/search.py "query" --no-rerank
```

**What to check:**
- Results appear with `rerank_score` > 0 (Qwen reranker working)
- Image chunks show up for visual queries
- Table chunks show up with their markdown content in `text`/`caption`

---

## 7. API Server Testing

### Start the server

```bash
python scripts/serve.py --reload
# Server runs at http://localhost:8000
```

### Health check

```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected response:
```json
{
  "status": "ok",
  "qdrant": "ok",
  "openai": "skipped (embedding_provider=qwen)",
  "reranker_backend": "qwen"
}
```

> `openai` shows `skipped` because embedding is local Qwen — this is correct.

### Ingest via API

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/paper.pdf"
```

### Search via API

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what does table 2 show?", "top_k": 10, "top_n": 5}'
```

### RAG generation via API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise the key results from the tables"}'
```

**What to verify:**
- Response includes `answer` (text from Qwen3-VL-8B)
- Response includes `sources` array with chunks used
- Sources include `modality`, `page`, `rerank_score`
- For table/image queries, `image_base64` is present in sources (visual context was passed to VLM)

### List collections

```bash
curl http://localhost:8000/collections | python -m json.tool
```

---

## 8. GPU / Model Loading Verification

To verify the Qwen models are using GPU (if available):

```bash
python - <<'EOF'
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Embedding model device:", next(model.parameters()).device)
EOF
```

Expected: `cuda:0` (GPU) or `mps` (Apple Silicon) or `cpu` (CPU-only machine).

---

## 9. Linting

```bash
uv run ruff check src/ tests/ scripts/
```

Expected: `All checks passed.`

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Z_AI_API_KEY is required` on startup | Key missing or `.env` not loaded | Check `.env` exists in project root; verify key value |
| `QwenVLEmbedder` loads on CPU, very slow | No GPU / CUDA not set up | Normal for CPU-only — just slower. Use GPU machine for production |
| RunPod call hangs 30–90 s | Cold start (`min_workers=0`) | Set `min_workers=1` in RunPod endpoint settings |
| `Qdrant connection refused` | Qdrant container not running | `docker compose up qdrant -d` |
| `BUG: embed_chunks produced unfilled slots` | Image chunk has no base64 and no text | Check `IMAGE_CAPTION_ENABLED=true` and PDF has extractable content |
| `--overwrite` needed | Re-ingesting after changing `EMBEDDING_DIMENSIONS` | Always use `--overwrite` when dimension changes (2048 is fixed for qwen branch) |
| Health endpoint shows `status: degraded` | Qdrant not reachable | Check Docker, check `QDRANT_URL` in `.env` |
| Table captions missing / empty | RunPod endpoint offline | Verify endpoint is active in RunPod console |

---

## 11. Key Config Defaults (qwen branch)

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `qwen` | Local Qwen3-VL-Embedding-2B |
| `EMBEDDING_DIMENSIONS` | `2048` | Fixed for this branch — do not change without `--overwrite` |
| `RERANKER_BACKEND` | `qwen` | Local Qwen3-VL-Reranker-2B |
| `OPENAI_LLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Sent to RunPod endpoint |
| `IMAGE_CAPTION_ENABLED` | `true` | Images: crop only (no LLM). Tables/formulas: LLM caption via RunPod |
| `PARSER_BACKEND` | `cloud` | GLM-OCR via Z.AI MaaS |
| `QDRANT_COLLECTION_NAME` | `documents` | Change to isolate test data |
