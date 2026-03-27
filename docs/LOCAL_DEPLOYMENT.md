# Local Deployment Testing Guide

Work through this document top to bottom, one command at a time. Each section has a **verify** step so you know it worked before moving on. No automation — everything is manual and explicit.

---

## What You Are Running

```
Your machine
│
├── docker-compose up
│     ├── qdrant       (port 6333)  ← vector database
│     ├── ollama       (port 11434) ← local LLM for parsing
│     ├── app          (port 8000)  ← FastAPI backend
│     └── visualizer   (port 8501)  ← Streamlit inspector
│
└── you → curl / browser to test each service
```

---

## Prerequisites

```bash
# Check Docker is running
docker info

# Check Docker Compose version (need v2 — 'docker compose', not 'docker-compose')
docker compose version

# Check you have a .env file with keys filled in
cat .env | grep -E "Z_AI_API_KEY|OPENAI_API_KEY"
# Both lines must have actual values, not placeholders
```

If `.env` doesn't exist yet:

```bash
cp .env.example .env
# Open .env and fill in at minimum:
#   Z_AI_API_KEY=...      (skip if PARSER_BACKEND=ollama)
#   OPENAI_API_KEY=sk-...
```

---

## Step 1 — Build the Images

Build both images locally before starting anything. This catches any build errors early.

```bash
# Build the app image
docker build -t doc-parser/app:local .

# Build the visualizer image
docker build -f Dockerfile.visualizer -t doc-parser/visualizer:local .
```

**Verify:**

```bash
docker images | grep doc-parser
# Expected output:
# doc-parser/visualizer   local   <id>   <size>
# doc-parser/app          local   <id>   <size>
```

**Common build errors:**

| Error | Fix |
|-------|-----|
| `pyproject.toml not found` | Make sure you are running `docker build` from the repo root |
| `uv pip install` fails | Check Python version in `pyproject.toml` matches `3.12` |
| `libgl1` not found | Already handled in `Dockerfile` — if it still fails, run `docker build --no-cache .` |

---

## Step 2 — Start Qdrant First

Start Qdrant in isolation before bringing up the full stack. It is the only stateful service and must be healthy before the app connects to it.

```bash
docker compose up qdrant -d
```

**Verify:**

```bash
# Health check
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}

# Dashboard (open in browser)
open http://localhost:6333/dashboard
```

**Inspect the volume:**

```bash
# Confirm the named volume was created and is being used
docker volume ls | grep qdrant_data
docker volume inspect multi-modal-rag_qdrant_data
```

---

## Step 3 — Start Ollama and Pull the Model

```bash
docker compose up ollama -d
```

**Verify Ollama is running:**

```bash
curl http://localhost:11434/api/tags
# Expected: {"models":[]} (empty is fine — no models pulled yet)
```

**Pull the GLM-OCR model** (one-time, ~600 MB):

```bash
docker compose exec ollama ollama pull glm-ocr:latest
```

Watch the progress in your terminal. When done:

```bash
# Confirm model is available
docker compose exec ollama ollama list
# Expected: glm-ocr:latest   <id>   <size>   <date>
```

> The model is stored in the `ollama_models` Docker volume. It survives container restarts — you only pull once.

---

## Step 4 — Start the App

```bash
docker compose up app -d
```

**Watch the startup logs:**

```bash
docker compose logs app -f
# Wait until you see: "Application startup complete."
# Ctrl+C to stop following logs
```

**Verify the health endpoint:**

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "qdrant": "ok",
  "openai": "ok"
}
```

**If `qdrant` shows `"error"`:** the app container cannot reach Qdrant. Check:

```bash
# Is Qdrant actually running?
docker compose ps qdrant

# Can the app container resolve the hostname?
docker compose exec app curl http://qdrant:6333/healthz
```

**If `openai` shows `"error"`:** your `OPENAI_API_KEY` in `.env` is missing or invalid.

```bash
# Check the key is being passed into the container
docker compose exec app env | grep OPENAI_API_KEY
```

---

## Step 5 — Start the Visualizer

```bash
docker compose up visualizer -d
```

**Verify:**

```bash
curl http://localhost:8501/_stcore/health
# Expected: "ok"

# Open in browser
open http://localhost:8501
```

---

## Step 6 — Check All Four Services Are Up

```bash
docker compose ps
```

Expected output:

```
NAME                          STATUS          PORTS
multi-modal-rag-app-1         Up (healthy)    0.0.0.0:8000->8000/tcp
multi-modal-rag-qdrant-1      Up              0.0.0.0:6333->6333/tcp, 0.0.0.0:6334->6334/tcp
multi-modal-rag-ollama-1      Up              0.0.0.0:11434->11434/tcp
multi-modal-rag-visualizer-1  Up              0.0.0.0:8501->8501/tcp
```

All four must show `Up`. If any shows `Exit` or `Restarting`:

```bash
# Read that container's logs
docker compose logs <service-name> --tail=50
```

---

## Step 7 — Test the API Manually

Open a second terminal and test each endpoint.

### Health

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### List collections

```bash
curl -s http://localhost:8000/collections | python3 -m json.tool
# Expected: {"collections": []}  (empty until you ingest something)
```

### Ingest a document

```bash
# Use one of the sample PDFs in test_data/ if available, or any PDF you have
curl -s -X POST http://localhost:8000/ingest/file \
  -F "file=@test_data/sample.pdf" \
  | python3 -m json.tool
```

Expected response:

```json
{
  "status": "ok",
  "chunks_ingested": 42,
  "collection": "documents"
}
```

### Confirm the collection was created

```bash
curl -s http://localhost:8000/collections | python3 -m json.tool
# Expected: {"collections": ["documents"]}

# Also check in Qdrant directly
curl -s http://localhost:6333/collections | python3 -m json.tool
```

### Search

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the main contribution of this paper?", "top_k": 5}' \
  | python3 -m json.tool
```

---

## Step 8 — Test the Visualizer

Open `http://localhost:8501` in your browser.

1. Upload a PDF using the file uploader
2. Click **▶ Parse with Ollama**
3. Verify bounding boxes appear over the document pages
4. Check the element breakdown table on the right

---

## Step 9 — Inspect Volumes (confirm persistence)

```bash
# Stop the app and qdrant
docker compose stop app qdrant

# Start them again
docker compose start qdrant app

# Wait ~5 seconds, then check collections again
sleep 5
curl -s http://localhost:8000/collections | python3 -m json.tool
# The collection you ingested in Step 7 must still be there
```

This confirms the `qdrant_data` volume is working correctly. If the collection disappeared, the volume is not mounted — check `docker-compose.yml`.

---

## Step 10 — Tear Down

```bash
# Stop all containers but keep volumes (data preserved)
docker compose down

# Stop and delete all volumes (full reset — loses all ingested data)
docker compose down -v
```

After `docker compose down -v`, running `docker compose up` starts completely fresh.

---

## Useful Commands During Testing

```bash
# Follow logs for all services at once
docker compose logs -f

# Follow logs for one service
docker compose logs app -f

# Shell into the app container
docker compose exec app bash

# Shell into Qdrant container
docker compose exec qdrant sh

# Check resource usage
docker stats

# List all named volumes
docker volume ls

# Rebuild a single service without restarting others
docker compose up app --build -d

# Force rebuild from scratch (no layer cache)
docker compose build --no-cache app
docker compose up app -d
```

---

## Checkpoint: Ready for AWS

Once all the following are true, you are ready to move on to manual AWS deployment:

- [ ] `docker compose ps` shows all 4 services `Up`
- [ ] `curl http://localhost:8000/health` returns `"status": "ok"` for all components
- [ ] Ingest + Search roundtrip works end-to-end
- [ ] After `docker compose down` + `docker compose up`, ingested data is still there
- [ ] Visualizer opens in browser and parses a PDF

The next step is **manual AWS deployment** — pushing your locally-built images to ECR and running them on ECS Fargate, all from the terminal. That is covered in `DEPLOYMENT.md`.
