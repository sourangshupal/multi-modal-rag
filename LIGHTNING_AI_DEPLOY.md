# Lightning AI — GPU Deployment Guide

Full step-by-step reference for deploying the multimodal RAG pipeline on a Lightning AI GPU instance.

---

## Recommended Instance

| Resource | Spec |
|---|---|
| GPU | T4 (16 GB VRAM) |
| RAM | 16 GB |
| CPU | 4+ cores |
| Disk | 50 GB |

---

## Step 1 — Open a Terminal in Lightning AI

From your Lightning AI Studio, open the terminal (bottom panel or `Ctrl+` `` ` ``).

---

## Step 2 — Verify GPU is Available

```bash
nvidia-smi
```

You should see the T4 listed with ~16 GB VRAM. If this fails, the instance has no GPU attached.

Verify Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## Step 3 — Clone the Repository

```bash
git clone https://github.com/sourangshupal/multi-modal-rag.git
cd multi-modal-rag
```

---

## Step 4 — Create the `.env` File

```bash
cp .env.example .env
nano .env   # or: vi .env
```

Fill in the required values:

```dotenv
# Required
OPENAI_API_KEY=sk-...
Z_AI_API_KEY=                     # leave blank — using Ollama locally

# Parser
PARSER_BACKEND=ollama
CONFIG_YAML_PATH=ollama/config.yaml

# Qdrant
QDRANT_URL=http://qdrant:6333     # set automatically by docker-compose.gpu.yml
QDRANT_COLLECTION_NAME=documents

# Embedding
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# Reranker
RERANKER_BACKEND=openai

# LLM
OPENAI_LLM_MODEL=gpt-4o
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X` for nano).

---

## Step 5 — Pull the GLM-OCR Model into Ollama

Start only the Ollama service first so the model is ready before the full stack boots:

```bash
docker compose -f docker-compose.gpu.yml up ollama -d
```

Wait ~15 seconds for Ollama to start, then pull the model:

```bash
docker compose -f docker-compose.gpu.yml exec ollama ollama pull glm-ocr:latest
```

This downloads ~600 MB. Takes 30–60 seconds on a cloud instance. You will see a progress bar.

Verify the model is available:

```bash
docker compose -f docker-compose.gpu.yml exec ollama ollama list
```

You should see `glm-ocr:latest` in the output.

---

## Step 6 — Build and Start the Full Stack

```bash
docker compose -f docker-compose.gpu.yml up --build
```

This builds the GPU app image (first build takes ~3–5 minutes due to PyTorch CUDA wheel download).

Expected startup sequence in logs:

```
qdrant-1   | ... Qdrant gRPC listening on 6334
ollama-1   | ... Listening on 127.0.0.1:11434
app-1      | ... Starting doc-parser API | parser=ollama | backend=openai | collection=documents
app-1      | ... PP-DocLayoutV3 weights loading...
app-1      | ... Application startup complete.
```

---

## Step 7 — Verify All Services are Running

```bash
docker compose -f docker-compose.gpu.yml ps
```

All four services (`app`, `qdrant`, `ollama`, `visualizer`) should show `running`.

Check the API health:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "qdrant": "ok", "openai": "ok", "reranker_backend": "openai"}
```

---

## Step 8 — Ingest a PDF

Place your PDF in `data/raw/` first:

```bash
mkdir -p data/raw
cp /path/to/your/paper.pdf data/raw/paper.pdf
```

Ingest via curl (do NOT use the Swagger UI `collection` field — leave it out):

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@data/raw/paper.pdf" \
  -F "caption=true" \
  -F "max_chunk_tokens=512"
```

Expected response:
```json
{
  "source_file": "paper.pdf",
  "collection": "documents",
  "chunks_upserted": 64,
  "modality_counts": {"text": 48, "image": 6, "table": 4, "formula": 6},
  "latency_ms": 182450.0
}
```

Enriched chunks are saved to `data/chunks/paper.json` for inspection.

---

## Step 9 — Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main contribution of this paper?",
    "top_k": 10,
    "rerank": true
  }'
```

Filter by modality (images only):

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "architecture diagram",
    "top_k": 5,
    "filter_modality": "image"
  }'
```

---

## Step 10 — Generate (Full RAG Answer)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the proposed method in detail.",
    "top_k": 20,
    "top_n": 5,
    "rerank": true,
    "max_tokens": 1024
  }'
```

---

## Service URLs

| Service | URL |
|---|---|
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Ollama visualizer | http://localhost:8501 |

---

## Useful Commands

### Follow logs for a specific service

```bash
docker compose -f docker-compose.gpu.yml logs -f app
docker compose -f docker-compose.gpu.yml logs -f ollama
docker compose -f docker-compose.gpu.yml logs -f qdrant
```

### Restart a single service (e.g. after config change)

```bash
docker compose -f docker-compose.gpu.yml restart app
```

### Stop everything

```bash
docker compose -f docker-compose.gpu.yml down
```

### Stop and wipe all volumes (full reset — deletes ingested data)

```bash
docker compose -f docker-compose.gpu.yml down -v
```

### List Qdrant collections

```bash
curl http://localhost:8000/collections
```

### Delete a collection

```bash
curl -X DELETE http://localhost:8000/collections/documents
```

### Inspect enriched chunks

```bash
# Print enriched text for all non-text chunks
python3 -c "
import json
chunks = json.load(open('data/chunks/paper.json'))
for c in chunks:
    if c['modality'] != 'text':
        print(f\"--- [{c['modality'].upper()}] page {c['page']} ---\")
        print(c['text'][:400])
        print()
"
```

### Check GPU utilization during inference

```bash
watch -n 1 nvidia-smi
```

---

## Rebuilding After Code Changes

If you pull new code from GitHub and need to rebuild the app image:

```bash
git pull origin deployment
docker compose -f docker-compose.gpu.yml up --build app
```

Qdrant and Ollama do not need rebuilding — only the `app` service changes.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `app exited with code 137` | OOM kill | Increase instance RAM or use GPU instance |
| `Read timed out (timeout=300)` | CPU too slow for OCR | Use GPU instance; timeout set to 1200s in config |
| `Collection documents doesn't exist` | Ingestion used wrong collection name | Re-ingest without the `collection` field in the request |
| `nvidia-smi` fails in container | NVIDIA Container Toolkit not installed | Use a Lightning AI GPU instance (toolkit pre-installed) |
| Ollama model not found | Model not pulled yet | Run Step 5 again |
| Port already in use | Another process on 8000/6333 | `docker compose down` then retry |
