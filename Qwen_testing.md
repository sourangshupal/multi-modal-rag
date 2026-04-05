# Qwen Branch — Testing Guide

Complete reference for testing the `qwen` branch end-to-end.

---

## Two Testing Modes

There are two ways to run and test this branch. Choose one:

| | **Mode A — FastAPI Direct** | **Mode B — Full Docker Compose** |
|---|---|---|
| **What runs in Docker** | Qdrant + Ollama + vLLM only | Everything (including the FastAPI app) |
| **FastAPI app** | Runs on your host with `python scripts/serve.py` | Runs inside the `app` container |
| **Model warm-up** | You run `python scripts/warmup.py` manually before starting the server | Automatic — the container runs `warmup.py` before `serve.py` |
| **Model loading** | Embedding + reranker load lazily on first request | Same lazy loading, but warm-up fires Ollama and polls vLLM first |
| **Best for** | Development, debugging, watching logs in real time | Production-style testing, Lightning.ai deployment |
| **`.env` addresses** | `QDRANT_URL=http://localhost:6333`, `OPENAI_BASE_URL=http://localhost:8001/v1` | `OPENAI_BASE_URL=http://vllm:8000/v1` (Docker overrides QDRANT_URL internally) |

> **Recommended for students:** Start with Mode A — you see every log line as it happens and can iterate fast. Switch to Mode B once everything works end-to-end.

---

## Stack Overview

| Component | Model / Service | Runs Where |
|---|---|---|
| **OCR / Parsing** | GLM-OCR (GLM-4V) | Ollama (local Docker) |
| **Layout detection** | PP-DocLayout-V3-L | In-process HuggingFace (auto-downloaded) |
| **Embedding** | Qwen3-VL-Embedding-2B | Local — in-process HuggingFace |
| **Reranker** | Qwen3-VL-Reranker-2B | Local — in-process HuggingFace |
| **LLM (captions + generation)** | Qwen3-VL-8B-Instruct | Local vLLM (Docker, on-GPU) |
| **Vector DB** | Qdrant | Local Docker |

> **Zero external API keys required.** Everything runs on your local GPU. The only network calls are one-time HuggingFace model downloads on first run.

---

## 1. API Keys Required

**None.** This branch runs entirely on local GPU — no external API keys of any kind are needed.

All models are downloaded automatically from HuggingFace on first run and cached locally. After the initial download everything runs fully offline.

| Model | Size | Downloaded by |
|---|---|---|
| GLM-OCR (glm-ocr:latest) | ~600 MB | `ollama pull glm-ocr:latest` |
| PP-DocLayout-V3 weights | ~400 MB | Auto on first ingest (HuggingFace) |
| Qwen3-VL-Embedding-2B | ~4–5 GB | Auto on first ingest (HuggingFace) |
| Qwen3-VL-Reranker-2B | ~4–5 GB | Auto on first search (HuggingFace) |
| Qwen3-VL-8B-Instruct (vLLM) | ~15 GB BF16 / ~5 GB AWQ | Auto on first container start |

> **Total first-run download: ~25 GB (BF16) or ~15 GB (AWQ).** All cached in Docker volumes — subsequent restarts are instant.

---

## 2. Ollama Setup — Pull the GLM-OCR Model

GLM-OCR runs locally via Ollama. You must pull the model once before starting the stack.

### If running scripts directly on the host (non-Docker mode)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the GLM-OCR model (~600 MB)
ollama pull glm-ocr:latest

# Verify it downloaded
ollama list
# Should show: glm-ocr:latest   ...   600 MB
```

### If using Docker Compose (recommended for Lightning.ai)

The `ollama` service in `docker-compose.gpu.yml` starts Ollama automatically, but you must pull the model into its volume before first use:

```bash
# Start only the Ollama container
docker compose -f docker-compose.gpu.yml up ollama -d

# Wait ~10s for Ollama to be ready, then pull the model
docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest

# Verify
docker compose -f docker-compose.gpu.yml exec ollama ollama list
# Should show: glm-ocr:latest
```

The model is stored in the `ollama_models` Docker volume — you only need to pull it once. It persists across container restarts.

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

# Qwen local models (embedding + reranker) + layout detection (PP-DocLayout-V3)
uv pip install -e ".[qwen,layout]"
```

> `[qwen]` installs `transformers`, `torch`, `accelerate`, `Pillow` (~4–5 GB download each model, first run only).
> `[layout]` installs PP-DocLayout-V3 dependencies needed for Ollama parsing mode.

### 3.3 Create `.env`

```bash
cp .env.example .env
```

**Choose the block that matches your testing mode:**

#### Mode A — FastAPI Direct (app runs on host, services in Docker)

`OPENAI_BASE_URL` uses the **host-side** port (8001) because the Python process is outside Docker.

```dotenv
# ── Parser — Ollama local ──────────────────────────────────────
PARSER_BACKEND=ollama

# ── LLM — local vLLM (host-side port) ─────────────────────────
OPENAI_API_KEY=local-token
OPENAI_BASE_URL=http://localhost:8001/v1    # ← localhost, not vllm
OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
# AWQ on L4: OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-AWQ

# ── Embedding — local Qwen ────────────────────────────────────
EMBEDDING_PROVIDER=qwen
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DIMENSIONS=2048

# ── Reranker — local Qwen ─────────────────────────────────────
RERANKER_BACKEND=qwen
RERANKER_TOP_N=5

# ── Qdrant ────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333            # ← localhost, not qdrant
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

#### Mode B — Full Docker Compose (everything in Docker)

`OPENAI_BASE_URL` uses the **internal Docker network** name. `QDRANT_URL` is overridden automatically by `docker-compose.gpu.yml` — the `.env` value is ignored.

```dotenv
# ── Parser — Ollama local ──────────────────────────────────────
PARSER_BACKEND=ollama

# ── LLM — local vLLM (internal Docker network) ────────────────
OPENAI_API_KEY=local-token
OPENAI_BASE_URL=http://vllm:8000/v1        # ← Docker internal name
OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
# AWQ on L4: OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-AWQ

# ── Embedding — local Qwen ────────────────────────────────────
EMBEDDING_PROVIDER=qwen
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DIMENSIONS=2048

# ── Reranker — local Qwen ─────────────────────────────────────
RERANKER_BACKEND=qwen
RERANKER_TOP_N=5

# ── Qdrant (value overridden inside container to http://qdrant:6333) ──
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

### 3.4 Start Infrastructure Services (required for both modes)

Both modes need Qdrant, Ollama, and vLLM running in Docker. Start them first:

```bash
# A100 / L40s — BF16
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up qdrant ollama vllm -d

# L4 — AWQ 4-bit
COMPOSE_PROFILES=local-llm-4bit docker compose -f docker-compose.gpu.yml up qdrant ollama vllm-4bit -d
```

**Pull GLM-OCR into Ollama (first time only):**

```bash
# Wait ~15s for Ollama to be ready, then:
docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest

# Verify
docker compose -f docker-compose.gpu.yml exec ollama ollama list
# → glm-ocr:latest   ...   600 MB
```

**Verify all infrastructure is healthy:**

```bash
curl http://localhost:6333/healthz    # → {"title":"qdrant - vector search engine"}
curl http://localhost:8001/health     # → HTTP 200 (vLLM ready — may take 5-15 min first run)
docker compose -f docker-compose.gpu.yml exec ollama ollama list   # → glm-ocr:latest present
```

> vLLM takes 5–15 minutes on first run (downloading ~15 GB BF16 or ~5 GB AWQ model). Watch: `docker compose -f docker-compose.gpu.yml logs -f vllm`

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

## 5. Integration Tests (require infrastructure running)

These make real calls to Qdrant, Ollama, and vLLM. Complete Section 3.4 first (all services must be healthy).

```bash
# Full integration suite
uv run pytest tests/integration/ -v -m integration

# Parse only (needs Ollama + glm-ocr model pulled)
uv run pytest tests/integration/test_pipeline_e2e.py -v

# Full ingest pipeline (needs Ollama + vLLM + Qdrant)
uv run pytest tests/integration/test_ingest_e2e.py -v
```

Tests auto-skip if the required services are not reachable.

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
- `QwenVLEmbedder loaded ... on cuda:0` (or `mps` / `cpu`) — embedding model loaded onto GPU
- `Captioning N table/formula/algorithm chunks` — local vLLM being called for captions
- `Image chunk: cropped bbox, stored image_base64` — images handled locally (no LLM call)
- `Upserted N points` — vectors successfully stored in Qdrant

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

## 7. FastAPI Server Testing

---

### Mode A — FastAPI Direct (app runs on host)

#### Step 1 — Ensure infrastructure is running

Complete Section 3.4 first. Verify before starting the server:

```bash
curl http://localhost:6333/healthz   # Qdrant — must be healthy
curl http://localhost:8001/health    # vLLM — must return HTTP 200
docker compose -f docker-compose.gpu.yml exec ollama ollama list   # glm-ocr:latest must appear
```

#### Step 2 — Warm up models (run once before starting the server)

`warmup.py` does three things before the server opens for traffic:
1. Polls vLLM `/health` until ready (handles the 5–15 min first-run download)
2. Compiles NVRTC CUDA kernels for PP-DocLayout-V3 (eliminates ~90s first-request lag)
3. Pre-loads glm-ocr into Ollama VRAM (eliminates cold-start on first parse)

```bash
python scripts/warmup.py
```

Expected output:
```
[warmup] vLLM: ready in Xs
[warmup] PP-DocLayoutV3: weights loaded on cuda:0
[warmup] PP-DocLayoutV3: ready in Xs (NVRTC kernels compiled and cached)
[warmup] GLM-OCR: model loaded into GPU VRAM in Xs
[warmup] All models hot — handing off to API server
```

> **If `warmup.py` is skipped:** the server starts fine but the first `/ingest` request will be slow — the embedding model, PP-DocLayout-V3, and GLM-OCR all cold-start simultaneously on the first call.

#### Step 3 — Start the FastAPI server

```bash
# Development mode (auto-reload on code changes)
python scripts/serve.py --reload

# Production mode
python scripts/serve.py
```

Server runs at `http://localhost:8000`. You will see startup logs:
```
Starting doc-parser API | parser=ollama | backend=qwen | collection=documents
```

#### Step 4 — Model loading sequence on first request

Because models load lazily (on first use), here is exactly when each model loads:

| Request | Model that loads | Where |
|---|---|---|
| `/ingest` (first call) | Qwen3-VL-Embedding-2B (~4–5 GB download, first run) | In-process GPU |
| `/ingest` (first call) | PP-DocLayout-V3 weights (~400 MB, first run) | In-process GPU |
| `/ingest` (first call) | GLM-OCR via Ollama | Already in VRAM (warmup loaded it) |
| `/ingest` with tables | Qwen3-VL-8B-Instruct (caption call) | vLLM container (already hot) |
| `/search` (first call) | Qwen3-VL-Reranker-2B (~4–5 GB download, first run) | In-process GPU |

> Running `warmup.py` first means vLLM and Ollama are guaranteed hot. Embedding and reranker still load on first `/ingest` and `/search` respectively — this is normal.

#### Step 5 — Health check

```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected:
```json
{
  "status": "ok",
  "qdrant": "ok",
  "openai": "skipped (embedding_provider=qwen)",
  "reranker_backend": "qwen"
}
```

> `openai: skipped` is correct — embedding uses local Qwen, not the OpenAI API.

#### Step 6 — Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/paper.pdf"
```

Watch the server logs. You should see:
```
QwenVLEmbedder loaded Qwen/Qwen3-VL-Embedding-2B on cuda:0
Captioning N table/formula/algorithm chunks      ← vLLM call
Image chunk: cropped bbox, stored image_base64   ← local crop, no LLM
Upserted N points
```

#### Step 7 — Search

```bash
# Basic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what does table 2 show?", "top_k": 10, "top_n": 5}'

# Filter by modality
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "accuracy results", "top_k": 20, "top_n": 5, "filter_modality": "table"}'
```

On first call, the reranker loads:
```
QwenVLReranker loaded Qwen/Qwen3-VL-Reranker-2B on cuda:0
```

**What to verify in the response:**
- Results have `rerank_score` > 0 (reranker is working)
- Image chunks include `image_base64` in payload
- Table chunks show markdown content in `text` or `caption`

#### Step 8 — RAG generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise the key results from the tables"}'
```

**What to verify:**
- Response includes `answer` field (text from Qwen3-VL-8B via local vLLM)
- Response includes `sources` array (chunks used as context)
- Each source has `modality`, `page`, `rerank_score`
- For table/image queries: `image_base64` is present in sources (image passed to VLM)

#### Step 9 — List collections

```bash
curl http://localhost:8000/collections | python -m json.tool
```

---

### Mode B — Full Docker Compose (everything in Docker)

In this mode the `app` container runs `warmup.py` automatically before starting the FastAPI server. **You do not run anything manually** — just start the stack and test via curl.

#### Step 1 — Start the full stack

```bash
# A100 / L40s — BF16
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d

# L4 — AWQ 4-bit
COMPOSE_PROFILES=local-llm-4bit docker compose -f docker-compose.gpu.yml up -d
```

#### Step 2 — Pull GLM-OCR model (first time only)

```bash
docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest
```

#### Step 3 — Watch the warm-up sequence

The `app` container runs `warmup.py` before accepting traffic. Follow the logs:

```bash
docker compose -f docker-compose.gpu.yml logs -f app
```

The sequence you will see:
```
[warmup] vLLM: polling http://vllm:8000/health (timeout=300s)...
[warmup] vLLM: ready in Xs
[warmup] PP-DocLayoutV3: loading weights onto GPU...
[warmup] PP-DocLayoutV3: ready in Xs (NVRTC kernels compiled and cached)
[warmup] GLM-OCR: loading glm-ocr:latest into Ollama VRAM...
[warmup] GLM-OCR: model loaded into GPU VRAM in Xs
[warmup] All models hot — handing off to API server
Starting doc-parser API | parser=ollama | backend=qwen | collection=documents
```

> On first container start, vLLM downloads the model (~15 GB BF16 / ~5 GB AWQ). The warm-up waits up to 5 minutes. Watch vLLM progress separately: `docker compose -f docker-compose.gpu.yml logs -f vllm`

#### Step 4 — Verify the stack

```bash
docker compose -f docker-compose.gpu.yml ps   # all containers should be "healthy" or "running"

curl http://localhost:8000/health | python -m json.tool
# Expected: {"status":"ok","qdrant":"ok","openai":"skipped (embedding_provider=qwen)","reranker_backend":"qwen"}
```

#### Step 5 — Test via API (same curl commands as Mode A)

```bash
# Ingest
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/paper.pdf"

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what does table 2 show?", "top_k": 10, "top_n": 5}'

# RAG generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise the key results from the tables"}'

# List collections
curl http://localhost:8000/collections | python -m json.tool
```

Watch container logs in real time:
```bash
docker compose -f docker-compose.gpu.yml logs -f app
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
| `Parsing failed: Connection refused` on port 11434 | Ollama container not running | `docker compose -f docker-compose.gpu.yml up ollama -d` |
| `model not found` during parsing | GLM-OCR not pulled into Ollama | `docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest` |
| `QwenVLEmbedder` loads on CPU, very slow | No GPU / CUDA not set up | Verify `nvidia-smi` sees the GPU; check container has GPU access |
| `Qdrant connection refused` | Qdrant container not running | `COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d` |
| vLLM call hangs / times out | vLLM still loading on first start | Watch `docker compose logs -f vllm`; wait for "Uvicorn running" |
| `BUG: embed_chunks produced unfilled slots` | Image chunk has no base64 and no text | Check `IMAGE_CAPTION_ENABLED=true` and PDF has extractable content |
| `--overwrite` needed | Re-ingesting after changing `EMBEDDING_DIMENSIONS` | Always use `--overwrite` when dimension changes (2048 is fixed for qwen branch) |
| Health endpoint shows `status: degraded` | Qdrant not reachable | Check Docker, check `QDRANT_URL` in `.env` |
| Table captions missing / empty | vLLM not yet ready | Check `curl http://localhost:8001/health` — must return 200 first |
| Parsing very slow (~30 s/page) | GLM-OCR running on CPU inside Ollama | Normal on CPU; GPU-accelerated Ollama cuts this to ~5 s/page |

---

## 11. Key Config Defaults (qwen branch)

| Setting | Default (qwen branch) | Notes |
|---|---|---|
| `PARSER_BACKEND` | `ollama` | GLM-OCR via local Ollama — no Z_AI_API_KEY needed |
| `EMBEDDING_PROVIDER` | `qwen` | Local Qwen3-VL-Embedding-2B |
| `EMBEDDING_DIMENSIONS` | `2048` | Fixed for this branch — do not change without `--overwrite` |
| `RERANKER_BACKEND` | `qwen` | Local Qwen3-VL-Reranker-2B |
| `OPENAI_LLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Served by local vLLM container |
| `OPENAI_BASE_URL` | `http://vllm:8000/v1` | Internal Docker network address for vLLM |
| `OPENAI_API_KEY` | `local-token` | Any non-empty string — vLLM does not validate tokens |
| `IMAGE_CAPTION_ENABLED` | `true` | Images: crop only (no LLM). Tables/formulas: LLM caption via local vLLM |
| `QDRANT_COLLECTION_NAME` | `documents` | Change to isolate test data |

---

## 12. Local vLLM Mode (No RunPod Required)

Run Qwen3-VL-8B-Instruct locally via vLLM instead of RunPod. Embedding and reranking are unaffected — they continue running in-process.

### 12.1 GPU Requirements

| GPU | VRAM | Variant | Compose profile |
|---|---|---|---|
| L40s (48 GB) | 48 GB | BF16 full precision | `local-llm` |
| A100 (80 GB) | 80 GB | BF16 full precision | `local-llm` |
| L4 (24 GB) | 24 GB | 4-bit AWQ (tight — ~22 GB total) | `local-llm-4bit` |

VRAM breakdown (all models, BF16): GLM-OCR ~6 GB + Embedding ~5 GB + Reranker ~5 GB + Qwen3-VL-8B ~18-20 GB ≈ 34–36 GB

### 12.2 `.env` Settings for Local vLLM

The full `.env` for local-only mode (no external API keys):

```dotenv
PARSER_BACKEND=ollama

OPENAI_API_KEY=local-token
OPENAI_BASE_URL=http://vllm:8000/v1
OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# For L4 (AWQ 4-bit):
# OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-AWQ
```

> `http://vllm:8000/v1` is the **internal Docker network** address. From your host terminal use `http://localhost:8001/...` for verification.

### 12.3 Starting the Stack

```bash
# BF16 mode (L40s / A100)
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d

# 4-bit AWQ mode (L4)
COMPOSE_PROFILES=local-llm-4bit docker compose -f docker-compose.gpu.yml up -d

# RunPod mode (default — vLLM container does NOT start)
docker compose -f docker-compose.gpu.yml up -d
```

On first run vLLM downloads the model from HuggingFace (~15 GB for BF16, ~5 GB for AWQ). The model is cached in the `huggingface_cache` Docker volume — subsequent restarts skip the download.

### 12.4 Watch vLLM Start Up

```bash
docker compose -f docker-compose.gpu.yml logs -f vllm
# Wait for: "Uvicorn running on http://0.0.0.0:8000"
```

### 12.5 Verify vLLM from the Host

```bash
# Health check (external port 8001)
curl http://localhost:8001/health
# Expected: HTTP 200

# List loaded models
curl http://localhost:8001/v1/models | python -m json.tool
# Expected: model ID matches OPENAI_LLM_MODEL

# Direct chat completion test
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer local-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "max_tokens": 30
  }'
```

### 12.7 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused` on `localhost:8001` | vLLM not started | Verify `COMPOSE_PROFILES=local-llm` was set |
| vLLM container exits immediately | OOM — not enough VRAM | Use `local-llm-4bit` profile, or upgrade to L40s/A100 |
| First startup takes 10+ min | Model downloading from HuggingFace | Normal on first run; cached after in `huggingface_cache` volume |
| App gets errors calling vLLM | Wrong port in `OPENAI_BASE_URL` | Use `http://vllm:8000/v1` (internal port 8000, not 8001) |
| `model not found` error | Model name mismatch | BF16: `Qwen/Qwen3-VL-8B-Instruct`; AWQ: `Qwen/Qwen3-VL-8B-Instruct-AWQ` |
| warmup shows "still waiting..." | Model still loading/downloading | Normal — check `docker compose logs vllm` for progress |

---

## 13. Running Everything on Lightning.ai (Full Local GPU)

This section is a complete, step-by-step guide for students running the qwen branch on a **Lightning.ai GPU instance** with no RunPod required.

> **Zero external API keys required.** Document parsing (GLM-OCR via Ollama), LLM (local vLLM), embedding, and reranker all run on the local GPU. The only network activity is one-time HuggingFace/Ollama model downloads on first run.

---

### 13.1 Choose the Right GPU Instance

| GPU | VRAM | Mode | Profile |
|---|---|---|---|
| **A100 (80 GB)** | 80 GB | BF16 full precision | `local-llm` |
| **L40s (48 GB)** | 48 GB | BF16 full precision | `local-llm` |
| **L4 (24 GB)** | 24 GB | 4-bit AWQ (tight, ~22 GB) | `local-llm-4bit` |
| T4 (16 GB) | 16 GB | **Not supported** — not enough VRAM | — |

**VRAM breakdown (BF16):** GLM-OCR ~6 GB + Embedding ~5 GB + Reranker ~5 GB + Qwen3-VL-8B ~18-20 GB = **~34-36 GB total**

---

### 13.2 Open a Terminal on the Instance

All commands below run inside the Lightning.ai terminal (SSH or the browser terminal).

---

### 13.3 Verify Prerequisites

```bash
# Confirm Docker is available
docker --version

# Confirm the GPU is visible
nvidia-smi

# If nvidia-container-toolkit is missing (rare on Lightning.ai):
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

### 13.4 Clone the Repository and Switch to qwen Branch

```bash
git clone https://github.com/sourangshupal/multi-modal-rag.git
cd multi-modal-rag
git checkout qwen
```

---

### 13.5 Create and Fill `.env`

```bash
cp .env.example .env
nano .env    # or use the Lightning.ai file editor
```

**`.env` for A100 / L40s (BF16 mode) — no external API keys needed:**

```dotenv
# ── Parser — Ollama local (no Z_AI_API_KEY) ───────────────────
PARSER_BACKEND=ollama

# ── LLM — local vLLM (BF16) ───────────────────────────────────
OPENAI_API_KEY=local-token             # any non-empty string works
OPENAI_BASE_URL=http://vllm:8000/v1   # internal Docker network address
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

**For L4 (AWQ 4-bit):** change only this one line:
```dotenv
OPENAI_LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-AWQ
```

---

### 13.6 Start the Full Stack

**A100 / L40s (BF16):**
```bash
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d
```

**L4 (AWQ 4-bit):**
```bash
COMPOSE_PROFILES=local-llm-4bit docker compose -f docker-compose.gpu.yml up -d
```

Services started:
- `qdrant` — vector database
- `ollama` — GLM-OCR inference (document parsing, local)
- `vllm` or `vllm-4bit` — Qwen3-VL-8B-Instruct LLM (local)
- `app` — FastAPI RAG server (waits for all others to be healthy before starting)

### 13.6a Pull the GLM-OCR Model into Ollama (first time only)

After the containers are up, pull the GLM-OCR model into the Ollama volume. This is a one-time step — the model persists across restarts in the `ollama_models` Docker volume.

```bash
# Wait ~15s for Ollama to be ready, then pull
docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest

# Verify it downloaded (~600 MB)
docker compose -f docker-compose.gpu.yml exec ollama ollama list
# Should show: glm-ocr:latest   ...   600 MB
```

---

### 13.7 Wait for vLLM to Load (First Run Only)

On the first run, vLLM downloads the model from HuggingFace (~15 GB for BF16, ~5 GB for AWQ). This takes **5–15 minutes** depending on network speed. The model is cached in the `huggingface_cache` Docker volume — subsequent restarts skip the download entirely.

```bash
# Watch vLLM download and startup progress:
docker compose -f docker-compose.gpu.yml logs -f vllm

# Wait for this line:
# "Uvicorn running on http://0.0.0.0:8000"
```

---

### 13.8 Verify the Full Stack is Healthy

Run these checks once vLLM has started:

```bash
# All containers should show "Up" or "healthy"
docker compose -f docker-compose.gpu.yml ps

# vLLM health (host port 8001)
curl http://localhost:8001/health
# Expected: HTTP 200

# Confirm model is loaded
curl http://localhost:8001/v1/models | python -m json.tool
# Expected: shows "Qwen/Qwen3-VL-8B-Instruct"

# Quick LLM test
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer local-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "max_tokens": 20
  }'

# Qdrant health
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine"}

# App health
curl http://localhost:8000/health | python -m json.tool
# Expected:
# {
#   "status": "ok",
#   "qdrant": "ok",
#   "openai": "skipped (embedding_provider=qwen)",
#   "reranker_backend": "qwen"
# }
```

---

### 13.9 Run the Pipeline

#### Option A — CLI scripts (recommended for learning and debugging)

```bash
# Install Python deps in a local venv (for CLI scripts)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,qwen]"

# Step 1: Ingest a PDF
python scripts/ingest.py data/raw/paper.pdf

# Step 2: Search
python scripts/search.py "what are the key results?"

# Search with modality filter
python scripts/search.py "table showing accuracy results" --filter-modality table

# Search with more candidates before reranking
python scripts/search.py "attention mechanism figure" --top-k 30 --top-n 5

# Step 3: RAG generation (via Python)
python scripts/generate.py "summarise the key results from the tables"
```

#### Option B — REST API

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/paper.pdf"

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "key results", "top_k": 10, "top_n": 5}'

# RAG generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise the key results from the tables"}'
```

---

### 13.10 What to Watch for in Logs

```bash
docker compose -f docker-compose.gpu.yml logs -f app
```

| Log message | What it means |
|---|---|
| `QwenVLEmbedder loaded ... on cuda:0` | Embedding model loaded onto GPU |
| `QwenVLReranker loaded ... on cuda:0` | Reranker loaded onto GPU |
| `Captioning N table/formula/algorithm chunks` | vLLM being called for table/formula captions |
| `Image chunk: cropped bbox, stored image_base64` | Images handled locally — no LLM call |
| `Upserted N points` | Vectors successfully stored in Qdrant |
| `[warmup] vLLM: ready in Xs` | vLLM was healthy before serving started |

---

### 13.11 Port Reference

| Service | Internal (Docker network) | External (host / Lightning.ai) |
|---|---|---|
| App (FastAPI) | `app:8000` | `http://localhost:8000` |
| vLLM | `vllm:8000` | `http://localhost:8001` |
| Qdrant HTTP | `qdrant:6333` | `http://localhost:6333` |
| Ollama (GLM-OCR) | `ollama:11434` | `http://localhost:11434` |

> **Important:** `OPENAI_BASE_URL=http://vllm:8000/v1` uses the **internal** Docker network address. When verifying from the terminal use port `8001`.

---

### 13.12 Key Reminders

| Item | Detail |
|---|---|
| No external API keys needed | `PARSER_BACKEND=ollama` + local vLLM — everything is on-GPU |
| GLM-OCR model pull | One-time step per fresh Ollama volume — see step 13.6a |
| HuggingFace models | Auto-downloaded on first run; cached in `huggingface_cache` Docker volume |
| `OPENAI_BASE_URL=http://vllm:8000/v1` | Internal Docker address — do not change for host-side verification |
| Host-side vLLM port | `http://localhost:8001/...` — for `curl` checks from the terminal |

---

### 13.13 Troubleshooting (Lightning.ai specific)

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` not found | Lightning.ai instance has no GPU | Select a GPU instance in the Lightning.ai dashboard |
| `docker: Error response from daemon: could not select device driver` | nvidia-container-toolkit missing | `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker` |
| vLLM OOM / exits immediately | Not enough VRAM for BF16 | Switch to `local-llm-4bit` profile and AWQ model, or pick a larger GPU |
| App health shows `status: degraded` | Qdrant or vLLM not yet ready | Wait 1-2 min after all containers show healthy, then retry |
| `Parsing failed: Connection refused` | Ollama not running or model not pulled | Run step 13.6a to pull `glm-ocr:latest` into the Ollama container |
| `model not found` during parsing | GLM-OCR not in Ollama volume | `docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest` |
| Table captions empty | vLLM still cold (not yet loaded) | Check `docker compose logs vllm`; wait for "Uvicorn running" message |
| `BUG: embed_chunks produced unfilled slots` | Image chunk has no base64 | Ensure `IMAGE_CAPTION_ENABLED=true` in `.env` |
| Parsing is very slow (~30 s/page) | GLM-OCR on CPU in Ollama | Normal on CPU instances; GPU-enabled Ollama reduces to ~5 s/page |
