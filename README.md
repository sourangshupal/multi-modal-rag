<div align="center">

# 🧠 MultiModal RAG Pipeline

### From raw PDF → clean Markdown → hybrid vector search → re-ranked answers

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat-square)](https://qdrant.tech)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=flat-square)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/Ruff-linter-D7FF64?style=flat-square)](https://github.com/astral-sh/ruff)

**🏆 Ranked #1 on OmniDocBench V1.5 — 94.62 score (March 2026)**

</div>

---

## ✨ What Is This?

**MultiModal RAG** is a production-ready document intelligence pipeline. Drop in a PDF — get back clean Markdown, structured JSON, RAG-ready chunks, and natural-language answers with re-ranked context.

It understands **23 document element categories** — titles, paragraphs, tables, formulas, figures, algorithms, footnotes, and more — using the same models that power the Z.AI cloud API, running either in the cloud or **fully locally via Ollama** (no GPU required).

```
📄 PDF / Image
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1 · Parse                                    │
│  PP-DocLayout-V3  →  23 element categories          │
│  GLM-OCR 0.9B     →  text · tables · formulas       │
│  Structure-aware chunker  →  RAG-ready chunks       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2 · Ingest                                   │
│  GPT-4o image captioner  →  enriches figure chunks  │
│  Pluggable embeddings    →  OpenAI · Gemini         │
│  BM25 sparse vectors     →  feature hashing         │
│  Qdrant hybrid store     →  dense + sparse + RRF    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3 · Search & Re-rank                         │
│  Hybrid Qdrant search  →  top-k candidates          │
│  Pluggable re-ranker   →  OpenAI · Jina · BGE · Qwen│
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Phase 4 · REST API  (FastAPI)                      │
│  POST /ingest   ·   POST /search                    │
│  GET  /health   ·   GET  /collections               │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

| | Feature | Details |
|---|---|---|
| 📐 | **Layout detection** | PP-DocLayout-V3 — 23 element categories including tables, formulas, figures, algorithms |
| 🔤 | **OCR + formula recognition** | GLM-OCR 0.9B — text, HTML tables, LaTeX formulas |
| 🌐 | **Cloud or local** | Z.AI MaaS cloud API **or** fully local via Ollama (`PARSER_BACKEND=ollama`) |
| 🖼️ | **Multimodal ingestion** | GPT-4o captions every figure/image chunk — images become searchable text |
| 🔍 | **Hybrid search** | Dense + sparse (BM25) vectors fused with RRF in Qdrant |
| 🎯 | **4 re-ranker backends** | OpenAI GPT-4o-mini · Jina M0 (cloud) · BGE (local, fast) · Qwen VL (local, multimodal) |
| 🔌 | **Pluggable embeddings** | OpenAI `text-embedding-3-large/small` or Google Gemini |
| ⚡ | **Async-first** | All I/O paths use `AsyncOpenAI` + `AsyncQdrantClient` |
| 🛡️ | **Type-safe config** | Pydantic-settings — every env var validated at startup |
| 🖥️ | **Visual inspector** | Streamlit app — color-coded bboxes, polygon overlays, element breakdown |

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [CLI Reference](#-cli-reference)
- [REST API](#-rest-api)
- [Embedding Providers](#-embedding-providers)
- [Re-ranker Backends](#-re-ranker-backends)
- [Local Mode (Ollama)](#-local-mode-ollama)
- [Visual Inspector](#-visual-inspector)
- [Configuration Reference](#-configuration-reference)
- [Project Structure](#-project-structure)
- [Running Tests](#-running-tests)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)

---

## 🛠️ Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Required |
| uv | latest | Package manager — replaces pip |
| Docker | latest | For local Qdrant (optional — Qdrant Cloud works too) |
| Z.AI API key | — | [z.ai](https://z.ai) — skip if using `PARSER_BACKEND=ollama` |
| OpenAI API key | — | For captioning, embeddings, and re-ranking |

### Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## ⚡ Quick Start

```bash
# 1 · Clone / navigate to the project
cd multi-modal-rag

# 2 · Create virtual environment
uv venv --python 3.12
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3 · Install dependencies
uv pip install -e ".[dev]"

# 4 · Configure API keys
cp .env.example .env
# → Open .env and fill in Z_AI_API_KEY + OPENAI_API_KEY

# 5 · Start Qdrant
docker-compose up -d qdrant

# 6 · Parse a document
python scripts/parse.py path/to/paper.pdf --chunks

# 7 · Ingest into Qdrant
python scripts/ingest.py path/to/paper.pdf

# 8 · Search with re-ranking
python scripts/search.py "What is the transformer attention mechanism?"

# 9 · (Optional) Start the REST API
python scripts/serve.py --reload
# → Interactive docs: http://localhost:8000/docs
```

### Optional extras

```bash
uv pip install -e ".[bge]"     # 🚀 BAAI/bge-reranker — fast local re-ranking
uv pip install -e ".[qwen]"    # 👁️  Qwen VL reranker  — local multimodal re-ranking
uv pip install -e ".[gemini]"  # 💎 Google Gemini embeddings
uv pip install -e ".[layout]"  # 🦙 PP-DocLayout-V3 local detection (Ollama mode)
```

---

## 🖥️ CLI Reference

### `parse.py` — Parse PDFs into Markdown + JSON

```bash
python scripts/parse.py <input> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | *(required)* | PDF/image file **or** directory |
| `--output` | `./output/` | Where to save results |
| `--format` | `both` | `markdown`, `json`, or `both` |
| `--chunks` | off | Also write `{name}_chunks.json` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, or `WARNING` |

```bash
python scripts/parse.py paper.pdf                              # Markdown + JSON
python scripts/parse.py paper.pdf --chunks                     # + RAG chunks
python scripts/parse.py ./docs/ --format markdown --output ./parsed/
```

---

### `ingest.py` — Embed + Upsert to Qdrant

```bash
python scripts/ingest.py <input> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | *(required)* | Document file or directory |
| `--no-captions` | off | Skip GPT-4o image captioning |
| `--collection` | from `.env` | Override Qdrant collection name |
| `--overwrite` | off | Delete and recreate the collection |
| `--max-chunk-tokens` | `512` | Max tokens per chunk |

```bash
python scripts/ingest.py paper.pdf
python scripts/ingest.py ./docs/ --no-captions
python scripts/ingest.py paper.pdf --collection my_col --max-chunk-tokens 256
python scripts/ingest.py paper.pdf --overwrite          # rebuild from scratch
```

---

### `search.py` — Query + Re-rank

```bash
python scripts/search.py "<query>" [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `query` | *(required)* | Natural-language query |
| `--top-k` | `20` | Candidates before re-ranking |
| `--top-n` | from `.env` | Results to keep after re-ranking |
| `--backend` | from `.env` | `openai` · `jina` · `bge` · `qwen` |
| `--filter-modality` | all | `text` · `image` · `table` · `formula` |
| `--no-rerank` | off | Skip re-ranking, print raw retrieval |
| `--collection` | from `.env` | Override Qdrant collection name |

```bash
python scripts/search.py "attention mechanism in transformers"
python scripts/search.py "bar chart comparing accuracy" --filter-modality image --backend jina
python scripts/search.py "results table" --backend bge           # fast local
python scripts/search.py "query" --no-rerank --top-k 10          # raw retrieval
```

---

## 🌐 REST API

```bash
python scripts/serve.py [--host HOST] [--port PORT] [--workers N] [--reload]
```

Interactive docs at **`http://localhost:8000/docs`**

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Ping Qdrant + OpenAI; returns status of each |
| `GET` | `/collections` | List all Qdrant collection names |
| `POST` | `/ingest` | Ingest a document by server-side file path |
| `POST` | `/ingest/file` | Ingest a document via multipart file upload |
| `POST` | `/search` | Hybrid search + optional re-ranking |

### `POST /search`

```json
{
  "query": "transformer attention mechanism",
  "top_k": 20,
  "top_n": 5,
  "filter_modality": null,
  "rerank": true
}
```

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism"}'
```

### `POST /ingest`

```bash
# By path
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/data/paper.pdf", "overwrite": false, "caption": true}'

# File upload
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@paper.pdf" -F "overwrite=true"
```

### Error codes

| Status | Meaning |
|--------|---------|
| `400` | Bad request (unsupported file type) |
| `404` | File not found |
| `422` | Validation error |
| `500` | Parse failed |
| `502` | Upstream error (Qdrant or OpenAI) |

Every response includes an `X-Request-Id` header (8-char UUID) for log correlation.

---

## 🔌 Embedding Providers

| Provider | Model | Dimensions | Extra |
|----------|-------|-----------|-------|
| `openai` *(default)* | `text-embedding-3-large` | 3072 | — |
| `openai` | `text-embedding-3-small` | 1536 | — |
| `gemini` | `gemini-embedding-2-preview` | 3072 | `.[gemini]` |

Switch providers by editing `.env` (recreate the collection if dimensions change):

```dotenv
# Switch to text-embedding-3-small
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

```dotenv
# Switch to Gemini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-2-preview
GEMINI_API_KEY=AIzaSy...
```

> ⚠️ **Changing `EMBEDDING_DIMENSIONS`** after collection creation requires `--overwrite`.

---

## 🎯 Re-ranker Backends

| Backend | Type | Multimodal | Cost | Latency | Extra |
|---------|------|-----------|------|---------|-------|
| `openai` *(default)* | GPT-4o-mini cross-encoder | ✅ Vision | ~$0.03–0.10/q | 800ms–2s | — |
| `jina` | Jina Reranker M0 | ✅ Qwen2-VL | ~$0.01–0.02/q | 500ms–2s | `JINA_API_KEY` |
| `bge` | BAAI/bge-reranker-v2-minicpm | ✗ (uses captions) | Free | **50–100ms** | `.[bge]` |
| `qwen` | Qwen3-VL-Reranker-2B (local) | ✅ Raw images | Free | 400–800ms | `.[qwen]` |

```dotenv
RERANKER_BACKEND=openai   # or jina, bge, qwen
RERANKER_TOP_N=5
JINA_API_KEY=             # only for jina backend
```

> **How image chunks flow:** every image chunk stores both a GPT-4o caption (text) and `image_base64` (raw PNG). `openai` and `jina` pass the raw image for visual scoring; `bge` uses the caption text; `qwen` decodes and passes the raw image to the local VLM.

---

## 🦙 Local Mode (Ollama)

Run the **entire parsing pipeline locally** — no Z.AI API key, no data leaving your machine.

### Setup

```bash
# 1 · Install and start Ollama
brew install ollama        # macOS; see ollama.com for Linux
ollama serve               # leave running in a terminal

# 2 · Pull the model (~600 MB — same model as Z.AI cloud)
ollama pull glm-ocr:latest

# 3 · Install layout detection deps (~400 MB on first run)
uv pip install -e ".[layout]"
```

### Enable in `.env`

```dotenv
PARSER_BACKEND=ollama
# Z_AI_API_KEY not needed
```

All scripts and the REST API work transparently — no other changes needed.

### Parse directly from the Streamlit visualizer

```bash
uv run streamlit run ollama/visualize.py
```

Upload any PDF → click **▶ Parse with Ollama** → get color-coded results in ~30–60 s.

### Cloud vs Local

| | ☁️ Cloud (Z.AI) | 🦙 Ollama (local) |
|---|---|---|
| API key required | Yes | No |
| Cost | API credits | Free |
| Speed | Fast (cloud GPU) | 5–30s / page |
| Privacy | Data sent to Z.AI | Fully local |
| Output quality | Identical | Identical (same models) |

---

## 🎨 Visual Inspector

Two Streamlit apps — one for cloud results, one for Ollama.

### Cloud API inspector

```bash
uv run streamlit run app.py
```

Upload a PDF → parse with Z.AI → see color-coded bounding boxes per page.

### Ollama inspector

```bash
uv run streamlit run ollama/visualize.py
```

- **Parse on the fly** — upload PDF, click Parse, see results immediately
- **Load saved results** — browse previously parsed `*_elements.json` files
- **Color-coded bboxes** — 20+ element categories, each with a distinct color
- **Polygon overlays** — toggle precise polygon outlines
- **Element breakdown** — count per type per page
- **Full document Markdown** — collapsible at the bottom

> Bounding box values are normalized to 0–1000. Pixel formula: `pixel = bbox_value × image_dimension / 1000`

---

## 📦 Output Formats

<details>
<summary><strong>📝 Markdown output</strong> (<code>document.md</code>)</summary>

```markdown
# Deep Learning for Document Understanding

**Abstract:** We propose a two-stage pipeline combining layout detection with ...

## 1. Introduction

Document understanding is a fundamental challenge in NLP ...

## 2. Method

$$
\mathcal{L} = \sum_{i=1}^{N} \ell(y_i, \hat{y}_i)
$$

| Model | F1 | Notes |
|-------|----|-------|
| Ours  | 94.62 | OmniDocBench V1.5 |
```
</details>

<details>
<summary><strong>🗂️ JSON output</strong> (<code>document.json</code>)</summary>

```json
{
  "source_file": "paper.pdf",
  "total_elements": 42,
  "pages": [
    {
      "page_num": 1,
      "markdown": "# Deep Learning for Document Understanding\n\n...",
      "elements": [
        {
          "label": "document_title",
          "text": "Deep Learning for Document Understanding",
          "bbox": [103, 52, 897, 118],
          "score": 1.0,
          "reading_order": 0
        }
      ]
    }
  ]
}
```
</details>

<details>
<summary><strong>🧩 Chunks output</strong> (<code>document_chunks.json</code>)</summary>

```json
[
  {
    "chunk_id": "paper.pdf_1_0",
    "text": "# Introduction\n\nDocument understanding is a fundamental challenge ...",
    "page": 1,
    "modality": "text",
    "element_types": ["paragraph_title", "paragraph"],
    "is_atomic": false,
    "bbox": null,
    "image_base64": null,
    "caption": null,
    "source_file": "paper.pdf"
  },
  {
    "chunk_id": "paper.pdf_2_1",
    "text": "Figure 3: Accuracy vs recall curve across all models.",
    "page": 2,
    "modality": "image",
    "element_types": ["figure"],
    "is_atomic": true,
    "bbox": [100, 400, 900, 750],
    "image_base64": "<base64-encoded PNG>",
    "caption": "Figure 3: Accuracy vs recall curve across all models.",
    "source_file": "paper.pdf"
  }
]
```

> **`is_atomic: true`** — table, formula, figure, or algorithm — never split mid-element.
> **`image_base64`** — populated for image/figure chunks; stored in Qdrant payload for re-ranker access without a second lookup.

</details>

---

## ⚙️ Configuration Reference

All settings are loaded from `.env` via pydantic-settings — validated at startup.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PARSER_BACKEND` | No | `cloud` | Parser backend: `cloud` (Z.AI MaaS) \| `ollama` (local Ollama) |
| `Z_AI_API_KEY` | Yes* | — | Z.AI cloud API key (required when `PARSER_BACKEND=cloud`) |
| `OPENAI_API_KEY` | Yes† | — | OpenAI key — captioning, generation, and embeddings |
| `OPENAI_LLM_MODEL` | No | `gpt-4o` | LLM model for generation |
| `EMBEDDING_PROVIDER` | No | `openai` | `openai` \| `gemini` |
| `EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model name |
| `EMBEDDING_DIMENSIONS` | No | `3072` | Vector size (1536 for `text-embedding-3-small`) |
| `GEMINI_API_KEY` | No‡ | — | Required when `EMBEDDING_PROVIDER=gemini` |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | No | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION_NAME` | No | `documents` | Collection name |
| `RERANKER_BACKEND` | No | `openai` | `openai` \| `jina` \| `bge` \| `qwen` |
| `RERANKER_TOP_N` | No | `5` | Results to keep after re-ranking |
| `JINA_API_KEY` | No | — | Required when `RERANKER_BACKEND=jina` |
| `IMAGE_CAPTION_ENABLED` | No | `true` | GPT-4o captioning during ingestion |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` · `INFO` · `WARNING` |
| `LOG_JSON` | No | `false` | JSON-lines output for log aggregators |
| `OUTPUT_DIR` | No | `./output` | Default output directory |
| `API_HOST` | No | `0.0.0.0` | REST API bind address |
| `API_PORT` | No | `8000` | REST API bind port |
| `API_WORKERS` | No | `1` | Uvicorn worker count |

\* `Z_AI_API_KEY` not required when `PARSER_BACKEND=ollama`
† `OPENAI_API_KEY` always required for GPT-4o captioning and re-ranking
‡ `GEMINI_API_KEY` only required when `EMBEDDING_PROVIDER=gemini`; also install `.[gemini]`

---

## 🗂️ Project Structure

```
multi-modal-rag/
├── 📄 pyproject.toml              # Dependencies and tool config
├── 📄 config.yaml                 # GLM-OCR SDK cloud settings
├── 🐳 docker-compose.yml          # Qdrant local service
├── 📄 .env.example                # Template for all API keys
│
├── src/doc_parser/
│   ├── config.py                  # pydantic-settings — all env vars
│   ├── pipeline.py                # DocumentParser — wraps glmocr SDK
│   ├── post_processor.py          # Elements → Markdown
│   ├── chunker.py                 # Structure-aware chunker
│   │
│   ├── ingestion/
│   │   ├── embedder.py            # BaseEmbedder + OpenAI/Gemini + BM25
│   │   ├── image_captioner.py     # GPT-4o captions for figures
│   │   └── vector_store.py        # Qdrant hybrid search wrapper
│   │
│   ├── retrieval/
│   │   └── reranker.py            # BaseReranker + OpenAI/Jina/BGE/Qwen
│   │
│   └── api/
│       ├── app.py                 # FastAPI app factory + lifespan
│       ├── dependencies.py        # Shared singletons (client, store, reranker)
│       ├── middleware.py          # Request-Id logging middleware
│       ├── schemas.py             # Pydantic request/response models
│       └── routes/
│           ├── health.py          # GET /health, GET /collections
│           ├── ingest.py          # POST /ingest, POST /ingest/file
│           └── search.py          # POST /search
│
├── 🎨 app.py                      # Streamlit visual inspector (cloud)
│
├── 🦙 ollama/
│   ├── config.yaml                # Ollama-specific glmocr config
│   ├── test_parse.py              # CLI: parse PDF locally
│   ├── visualize.py               # Streamlit: parse + inspect locally
│   └── output/                    # Saved results: *_elements.json + *.md
│
├── 📜 scripts/
│   ├── parse.py                   # PDF → Markdown + JSON + chunks
│   ├── ingest.py                  # PDF → embed → Qdrant
│   ├── search.py                  # query → retrieve → rerank → display
│   └── serve.py                   # Launch uvicorn API server
│
└── 🧪 tests/
    ├── unit/                      # Fully mocked — no API keys needed
    └── integration/               # Require live credentials
```

---

## 🧪 Running Tests

```bash
# Fast unit tests — no API keys needed (~2 seconds)
uv run pytest tests/unit/ -v

# Specific test files
uv run pytest tests/unit/test_reranker.py -v        # 17 reranker tests
uv run pytest tests/unit/test_api_schemas.py -v     # 15 schema tests

# Integration tests — requires live credentials + running Qdrant
uv run pytest tests/integration/ -v

# All tests
uv run pytest -v
```

---

## 🔧 Development

```bash
# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Type checking
uv run mypy src/
```

**Adding a new embedding provider:** subclass `BaseEmbedder`, implement `async embed(texts) -> list[list[float]]`, add to `_PROVIDERS` in `embedder.py`, add optional dep to `pyproject.toml`.

**Adding a new re-ranker backend:** subclass `BaseReranker`, implement `async rerank(query, candidates, top_n)`, add to `_BACKENDS` in `reranker.py`.

---

## 🔍 Troubleshooting

<details>
<summary><strong>ModuleNotFoundError: No module named 'glmocr'</strong></summary>

```bash
uv pip install -e ".[dev]"
```
</details>

<details>
<summary><strong>ValidationError: Z_AI_API_KEY is required when PARSER_BACKEND=cloud</strong></summary>

Add `Z_AI_API_KEY=your-key` to `.env`, or switch to local mode:
```dotenv
PARSER_BACKEND=ollama
```
</details>

<details>
<summary><strong>ValueError: JINA_API_KEY must be set</strong></summary>

Add `JINA_API_KEY=jina_...` to `.env`, or switch re-ranker:
```dotenv
RERANKER_BACKEND=openai
```
</details>

<details>
<summary><strong>ImportError: BGE / Qwen / Gemini reranker</strong></summary>

```bash
uv pip install -e ".[bge]"     # BGE
uv pip install -e ".[qwen]"    # Qwen VL
uv pip install -e ".[gemini]"  # Gemini embeddings
```
</details>

<details>
<summary><strong>Qdrant connection refused</strong></summary>

```bash
docker-compose up -d qdrant
# or point QDRANT_URL at your Qdrant Cloud cluster
```
</details>

<details>
<summary><strong>Ollama parse fails: Connection refused</strong></summary>

```bash
ollama serve                   # start Ollama
ollama list                    # verify glm-ocr:latest is pulled
ollama pull glm-ocr:latest     # pull if missing
```
</details>

<details>
<summary><strong>Bounding boxes appear misaligned in Streamlit</strong></summary>

All `bbox_2d` values are normalized to 0–1000. Formula: `pixel_x = bbox_x × image_width / 1000`
</details>

---

## 🏗️ Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Layout detection + OCR (cloud) | `glmocr` | ≥0.1.0 |
| Layout detection + OCR (local) | `glmocr[layout]` + Ollama | ≥0.1.0 |
| PDF → image extraction | `pymupdf` | ≥1.27.2 |
| Config management | `pydantic-settings` | ≥2.8.0 |
| LLM + embeddings | `openai` | ≥2.24.0 |
| Embeddings (Gemini) | `google-genai` | ≥1.0.0 |
| Vector database | `qdrant-client` | ≥1.17.0 |
| Async HTTP | `httpx` | ≥0.28.0 |
| Token counting | `tiktoken` | ≥0.9.0 |
| Local re-ranking | `FlagEmbedding` | ≥1.3.0 |
| Local VL re-ranking | `transformers` + `torch` | ≥4.51.0 / ≥2.7.0 |
| REST API | `fastapi` + `uvicorn` | ≥0.120.0 / ≥0.34.0 |
| Visual inspector | `streamlit` | ≥1.40.0 |
| Structured logging | `loguru` | ≥0.7.0 |
| Linting | `ruff` | ≥0.11.0 |

---

<div align="center">

**Built with ❤️ · Python 3.12 · uv · OpenAI SDK v2 · Qdrant 1.17 · FastAPI · Streamlit**

</div>
