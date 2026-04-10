# Docker Compose Testing Guide — Local GPU Stack

End-to-end runbook for the **fully containerized** Qwen multimodal RAG stack
on a single GPU host (tested on Lightning AI L40S, 46 GB VRAM).

Everything runs in Docker — the FastAPI app, embedder, reranker, layout
detector, OCR (Ollama), LLM (vLLM), and vector store (Qdrant). No external API
keys required.

> **Quick reference:** Every command in this guide assumes you are in the
> repo root. The compose project name is `multi-modal-rag` (auto-derived
> from the directory). All commands use `docker compose -f docker-compose.gpu.yml`
> — set `alias dcg='docker compose -f docker-compose.gpu.yml'` if you want
> them shorter.

---

## 0. Prerequisites

| Requirement | How to check |
|---|---|
| NVIDIA GPU + driver R510+ | `nvidia-smi` shows `Driver Version: 5xx.xx` |
| Docker + nvidia-container-toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` |
| Docker Compose v2 | `docker compose version` |
| Free disk space | ~30 GB for images + model caches |

```bash
# Driver + GPU check
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

You should see the GPU name and total VRAM. The driver must be **R510 or
newer** — earlier drivers can't run the cu128 PyTorch wheels even with the
forward-compat package baked into the Dockerfile.

---

## 1. Configure `.env`

Copy the example and set the model name vLLM will serve:

```bash
cp .env.example .env
```

Open `.env` and verify (or set) these values:

```bash
# Local LLM via vLLM (no real OpenAI key needed)
OPENAI_API_KEY=local-token
OPENAI_BASE_URL=http://vllm:8000/v1
OPENAI_LLM_MODEL=cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit

# Local embedder + reranker (Qwen multimodal)
EMBEDDING_PROVIDER=qwen
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DIMENSIONS=2048
RERANKER_BACKEND=qwen

# Local parser (Ollama-hosted GLM-OCR)
PARSER_BACKEND=ollama

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=documents
```

> **Critical:** `OPENAI_LLM_MODEL` **must exactly match** the model passed to
> vLLM at `docker-compose.gpu.yml:97`. If you change one, change the other.

---

## 2. Build and start everything

The compose file has a `local-llm` profile that includes the `vllm` service.
Set it via the `COMPOSE_PROFILES` env var:

```bash
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d --build
```

What happens (~5–10 min on first run):
1. Build the `app` image (Dockerfile.gpu): installs `cuda-compat-13-1`,
   PyTorch cu128, glmocr[layout], project deps.
2. Pull `qdrant/qdrant:v1.17.0`, `ollama/ollama:latest`,
   `vllm/vllm-openai:latest`.
3. Start all four containers.
4. `app` runs `scripts/warmup.py` which:
   - Polls vLLM `/health` until ready.
   - Loads PP-DocLayoutV3 weights to GPU (compiles NVRTC kernels).
   - Sends a tiny request to Ollama to pull `glm-ocr:latest` into VRAM.

Watch the build:

```bash
docker compose -f docker-compose.gpu.yml logs -f app
```

Wait for `Application startup complete.` from uvicorn.

---

## 3. Confirm all containers are running

```bash
docker compose -f docker-compose.gpu.yml ps
```

You want to see **four** rows, all `Up`, with `(healthy)` next to `vllm` and
`ollama`:

```
NAME                       SERVICE   STATUS                    PORTS
multi-modal-rag-app-1      app       Up X minutes              0.0.0.0:8000->8000/tcp
multi-modal-rag-ollama-1   ollama    Up X minutes (healthy)    0.0.0.0:11434->11434/tcp
multi-modal-rag-qdrant-1   qdrant    Up X minutes              0.0.0.0:6333-6334->6333-6334/tcp
multi-modal-rag-vllm-1     vllm      Up X minutes (healthy)    0.0.0.0:8001->8000/tcp
```

If any container is missing or `Restarting`, jump to **§9 Troubleshooting**.

---

## 4. Per-service health checks

Run these in order. Each one isolates a single service so you can pinpoint
where a failure starts.

### 4a. Qdrant

```bash
# Health endpoint — returns version JSON
curl -s http://localhost:6333/ | python3 -m json.tool

# List collections (will be empty until you ingest)
curl -s http://localhost:6333/collections | python3 -m json.tool
```

Expected: HTTP 200, version `1.17.0`, empty `collections` array.

### 4b. Ollama (GLM-OCR)

```bash
# Confirm the service is up
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# Verify glm-ocr:latest is loaded
docker compose -f docker-compose.gpu.yml exec ollama ollama list

# Verify it's resident on GPU (not CPU)
docker compose -f docker-compose.gpu.yml exec ollama ollama ps
```

`ollama list` must show `glm-ocr:latest`. `ollama ps` must show
`PROCESSOR: 100% GPU`. If it says `100% CPU`, GPU passthrough is broken
(see §9).

**Smoke-test the model with text-only inference:**

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "glm-ocr:latest",
  "prompt": "Reply with the single word: hello",
  "stream": false
}' | python3 -c "import json,sys; r=json.load(sys.stdin); print('response:', repr(r.get('response')))"
```

Expected: `response: 'hello'` (or close to it).

**OCR-test with the canonical prompt** (this is the prompt the SDK sends per
region — without it, glm-ocr returns empty responses):

```bash
# Use any image with text on it
IMAGE_PATH=/path/to/any_text_image.png
B64=$(base64 -w0 "$IMAGE_PATH")
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"glm-ocr:latest\",
  \"prompt\": \"Text Recognition:\",
  \"images\": [\"$B64\"],
  \"stream\": false,
  \"options\": {\"num_predict\": 256}
}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('response',''))"
```

Expected: extracted text from the image.

### 4c. vLLM (Qwen3-VL-4B-Instruct-AWQ)

```bash
# Health probe (the same one Compose uses)
curl -s http://localhost:8001/health
echo

# List served models — must show cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit
curl -s http://localhost:8001/v1/models | python3 -m json.tool

# Smoke-test chat completion
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
    "messages": [{"role": "user", "content": "Reply in one word: ready?"}],
    "max_tokens": 10
  }' | python3 -m json.tool
```

Expected: a JSON response with `choices[0].message.content`. The model name
in the request **must match** what `/v1/models` reports — copy-paste it if
unsure.

### 4d. App (FastAPI)

```bash
# OpenAPI docs (browse to confirm Swagger UI loads)
curl -sI http://localhost:8000/docs

# Hit any GET route — /docs is fine
curl -s http://localhost:8000/openapi.json | python3 -c "import json,sys; print('routes:', sorted(p for p in json.load(sys.stdin)['paths']))"
```

Expected: 200 from `/docs`, list of paths including `/ingest/file` and
`/generate`.

---

## 5. GPU verification

### 5a. Host view

```bash
nvidia-smi
```

After warmup but before any ingest/generate, you should see roughly
**~23 GB used** on the GPU:

| Process | Approximate VRAM |
|---|---|
| vLLM (Qwen3-VL-4B AWQ @ 0.3167 util) | ~14 GB |
| Ollama (GLM-OCR + KV cache) | ~6 GB |
| App: PP-DocLayoutV3 (warmed up) | ~3 GB |

The embedder (~5 GB) and reranker (~5 GB) are **lazy-loaded on first
request**, so they won't show up until you hit `/ingest/file` and
`/generate`. Total at steady state: **~33 GB / 46 GB**.

### 5b. Inside the app container

Confirm PyTorch sees the GPU:

```bash
docker compose -f docker-compose.gpu.yml exec app python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
```

Expected:
```
torch: 2.11.0+cu128
cuda available: True
device: NVIDIA L40S
```

If `cuda available: False`, the `cuda-compat-13-1` layer didn't take effect —
rebuild the app image (`up -d --build app`).

---

## 6. Inspect logs

### Per-service log commands

```bash
# All services interleaved (Ctrl+C to exit)
docker compose -f docker-compose.gpu.yml logs -f

# One service, follow live
docker compose -f docker-compose.gpu.yml logs -f app
docker compose -f docker-compose.gpu.yml logs -f vllm
docker compose -f docker-compose.gpu.yml logs -f ollama
docker compose -f docker-compose.gpu.yml logs -f qdrant

# Last N lines, no follow
docker compose -f docker-compose.gpu.yml logs --tail=200 app

# Since a relative time
docker compose -f docker-compose.gpu.yml logs --since=10m app

# Errors only
docker compose -f docker-compose.gpu.yml logs app 2>&1 | grep -iE "error|exception|traceback"
```

### What to look for during a healthy startup

**vLLM** (`logs vllm`):
- `Loading weights took ... seconds`
- `Available KV cache memory: ~7 GB`
- `Maximum concurrency for 8192 tokens per request: ...x`
- `Application startup complete.`
- `GET /health HTTP/1.1" 200 OK`

**Ollama** (`logs ollama`):
- `Listening on [::]:11434`
- `POST "/api/generate"` (during warmup)

**App** (`logs app`):
- `[warmup] vLLM: ready in 0.0s`
- `[warmup] PP-DocLayoutV3: weights loaded on cuda:0`
- `[warmup] PP-DocLayoutV3: ready in ~2s (NVRTC kernels compiled and cached)`
- `[warmup] GLM-OCR: model loaded into GPU VRAM in 0.2s`
- `[warmup] All models hot — handing off to API server`
- `Application startup complete.`

**App on first `/ingest/file`** — look for these:
- `QwenVLEmbedder loaded Qwen/Qwen3-VL-Embedding-2B on cuda:0` ← must say **cuda:0**, not `cpu`
- `Initialising reranker backend: qwen`
- `Loading Qwen/Qwen3-VL-Reranker-2B on device=cuda dtype=torch.bfloat16` ← must say **cuda**

If you see `on cpu` instead of `on cuda:0`, the GPU isn't being used. Check
§5b.

---

## 7. End-to-end smoke test

### 7a. Ingest a PDF

Place a small PDF in your working directory (e.g. `sample.pdf`).

```bash
curl -s -X POST http://localhost:8000/ingest/file \
  -F "file=@sample.pdf" \
  -F "overwrite=true" \
  -F "max_chunk_tokens=512" \
  -F "caption=true" | python3 -m json.tool
```

Expected response (after ~30–90 s for a small PDF):

```json
{
  "source_file": "sample.pdf",
  "collection": "documents",
  "chunks_upserted": 42,
  "modality_counts": {"text": 38, "image": 3, "table": 1},
  "latency_ms": 47823.5
}
```

Watch the app logs in another terminal:

```bash
docker compose -f docker-compose.gpu.yml logs -f app
```

You should see successful Ollama OCR calls (one per region), embedder
loading on `cuda:0`, and finally a Qdrant upsert.

### 7b. Verify chunks contain real text

```bash
docker compose -f docker-compose.gpu.yml exec app sh -c 'python3 -c "
import json
chunks = json.load(open(\"/app/data/chunks/sample.json\"))
print(f\"Total chunks: {len(chunks)}\")
for c in chunks[:5]:
    t = (c.get(\"text\") or \"\")[:120]
    print(f\"  page={c[\"page\"]} mod={c[\"modality\"]}: {t!r}\")
"'
```

Expected: real document text in each chunk. If you see `"##"` or `"#"`
placeholders, the GLM-OCR prompt mapping is missing — check
`ollama/config.docker.yaml` has `task_prompt_mapping` defined under
`page_loader`.

You can also query Qdrant directly:

```bash
curl -s -X POST 'http://localhost:6333/collections/documents/points/scroll' \
  -H 'Content-Type: application/json' \
  -d '{"limit": 5, "with_payload": true, "with_vector": false}' \
  | python3 -m json.tool
```

### 7c. Generate an answer

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "top_k": 20,
    "top_n": 5,
    "rerank": true,
    "max_tokens": 256
  }' | python3 -m json.tool
```

The first call triggers reranker model download (~5 GB into the HF cache
volume) so it can take 60–120 s. Subsequent calls are 5–15 s.

Expected response shape:

```json
{
  "query": "...",
  "answer": "...",
  "sources": [{"chunk_id": "...", "text": "...", "page": 1, ...}],
  "total_candidates": 20,
  "latency_ms": 9543.32
}
```

After this call, run `nvidia-smi` again — the embedder + reranker are now
loaded so VRAM usage should jump to **~33 GB**.

---

## 8. Common operations

### Restart one service (no rebuild)
```bash
docker compose -f docker-compose.gpu.yml restart app
```
> **Note:** `restart` re-runs the container's process but does NOT re-read
> `env_file`. If you edited `.env`, you need `--force-recreate` instead.

### Force-recreate a service after editing `.env` or compose file
```bash
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d --force-recreate app
```

### Rebuild after editing source code or Dockerfile
```bash
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d --build app
```

### Stop everything (containers persist, can be restarted)
```bash
docker compose -f docker-compose.gpu.yml stop
```

### Tear down everything (containers gone, volumes preserved)
```bash
docker compose -f docker-compose.gpu.yml down
```

### Tear down INCLUDING volumes (wipes Qdrant + HF cache + Ollama models)
```bash
docker compose -f docker-compose.gpu.yml down -v
```
> ⚠️ This deletes downloaded models. Next start will re-download ~15 GB.

### Open a shell inside the app container
```bash
docker compose -f docker-compose.gpu.yml exec app bash
```

### Inspect the embedder/reranker on GPU mid-run
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
docker compose -f docker-compose.gpu.yml exec app python -c "import torch; print(torch.cuda.memory_allocated()/1e9, 'GB')"
```

---

## 9. Troubleshooting

### Symptom: `nvidia-smi` shows GPU but `cuda available: False` in app
**Cause:** Driver too old for the cu128 PyTorch wheel; `cuda-compat-13-1`
not picked up.
**Fix:** Confirm `LD_LIBRARY_PATH` starts with `/usr/local/cuda-13.1/compat`
inside the container:
```bash
docker compose -f docker-compose.gpu.yml exec app printenv LD_LIBRARY_PATH
```
If missing or out of order, rebuild the app image: `up -d --build app`.

### Symptom: `/generate` returns `404 — model X does not exist`
**Cause:** `OPENAI_LLM_MODEL` in `.env` doesn't match what vLLM is serving.
**Fix:** `curl http://localhost:8001/v1/models` and copy the exact `id`
into `.env`. Then `up -d --force-recreate app` (a plain `restart` won't
re-read `.env`).

### Symptom: vLLM crashes at startup with OOM
**Cause:** `--gpu-memory-utilization` too high for the GPU shared with
ollama + embedder + reranker.
**Fix:** Lower the value at `docker-compose.gpu.yml:100`. For a 46 GB L40S
sharing with the rest of the stack, `0.3167` works (~14 GB for vLLM).

### Symptom: Chunks contain `"##"` or `"[figure]"` placeholders, no real text
**Cause:** `task_prompt_mapping` missing from `ollama/config.docker.yaml`.
GLM-OCR receives empty prompts and returns nothing; the SDK's result
formatter renders only the markdown prefix.
**Fix:** Confirm `ollama/config.docker.yaml` contains:
```yaml
page_loader:
  task_prompt_mapping:
    ocr: "Text Recognition:"
    table: "Table Recognition:"
    formula: "Formula Recognition:"
    algorithm: "Text Recognition:"
```
Then `restart app` and re-ingest with `overwrite=true`.

### Symptom: Image embedding fails with `argument of type 'NoneType' is not iterable`
**Cause:** Stale build of the app image without the embedder fix.
**Fix:** `up -d --build app`.

### Symptom: Reranker fails with `Unrecognized configuration class … for AutoModelForSequenceClassification`
**Cause:** Stale build with the old QwenVLReranker that used the wrong
AutoModel class.
**Fix:** `up -d --build app`.

### Symptom: Qdrant version skew warning in app logs
**Cause:** Old Qdrant container still running after the v1.17.0 image bump.
**Fix:** `up -d --force-recreate qdrant`.

### Symptom: Ollama `ollama ps` shows `100% CPU` instead of GPU
**Cause:** GPU passthrough not working — usually a missing
`nvidia-container-toolkit` install on the host.
**Fix:** Confirm `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
works on your host. If not, install/configure nvidia-container-toolkit.

---

## 10. Quick reference — full happy-path command sequence

```bash
# 0. Configure
cp .env.example .env
$EDITOR .env   # set OPENAI_LLM_MODEL=cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit

# 1. Start the stack
COMPOSE_PROFILES=local-llm docker compose -f docker-compose.gpu.yml up -d --build

# 2. Wait for warmup
docker compose -f docker-compose.gpu.yml logs -f app
# (Ctrl+C once you see "Application startup complete.")

# 3. Health check every service
docker compose -f docker-compose.gpu.yml ps
curl -s http://localhost:6333/ | python3 -m json.tool
curl -s http://localhost:11434/api/tags | python3 -m json.tool
curl -s http://localhost:8001/v1/models | python3 -m json.tool
curl -sI http://localhost:8000/docs

# 4. Verify GPU
nvidia-smi
docker compose -f docker-compose.gpu.yml exec app python -c "import torch; print('cuda:', torch.cuda.is_available())"

# 5. Ingest a PDF
curl -s -X POST http://localhost:8000/ingest/file \
  -F "file=@sample.pdf" -F "overwrite=true" | python3 -m json.tool

# 6. Generate
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 20, "top_n": 5, "rerank": true, "max_tokens": 256}' \
  | python3 -m json.tool

# 7. Tear down when done
docker compose -f docker-compose.gpu.yml down
```

---

## Appendix: service ↔ port reference

| Service | Container port | Host port | Purpose |
|---|---|---|---|
| `app` | 8000 | 8000 | FastAPI: `/docs`, `/ingest/file`, `/generate` |
| `qdrant` | 6333 / 6334 | 6333 / 6334 | Qdrant REST + gRPC |
| `ollama` | 11434 | 11434 | Ollama API: `/api/tags`, `/api/generate` |
| `vllm` | 8000 | 8001 | OpenAI-compatible: `/v1/models`, `/v1/chat/completions` |

Note that vLLM listens on `8000` *inside* its container but the host
port is **8001** (`8000` is taken by the app). Inside the docker network,
other services reach vLLM at `http://vllm:8000/v1`.
