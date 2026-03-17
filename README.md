# doc-parser

A two-stage multimodal document parsing pipeline powered by **PP-DocLayout-V3** (layout detection) and **GLM-OCR 0.9B** (text/table/formula recognition), running entirely on the **Z.AI MaaS cloud API** — no GPU required.

Ranked **#1 on OmniDocBench V1.5** (94.62 score, March 2026).

---

## What This Does

Feed it a PDF or image → get back clean Markdown, structured JSON, and RAG-ready chunks.

```
PDF / Image
    ↓
Z.AI MaaS API
  ├── PP-DocLayout-V3 (detects 23 element categories: titles, paragraphs, tables, formulas, ...)
  └── GLM-OCR 0.9B    (recognizes text, renders tables as HTML, encodes formulas as LaTeX)
    ↓
Post-Processor (local)
  └── Assembles elements into structured Markdown
    ↓
Structure-Aware Chunker (local)
  └── Produces RAG-ready JSON chunks (tables/formulas never split)
```

**23 element categories detected:** document title, paragraph title, paragraph, abstract, table, formula, inline formula, figure, caption, header, footer, footnote, code block, algorithm, reference, page number, seal, and more.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Required |
| uv | latest | Package manager (replaces pip) |
| Z.AI API key | — | Get one at [z.ai](https://z.ai) |

### Install `uv` (if you don't have it)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Quick Start (5 minutes)

### Step 1 — Clone / navigate to the project

```bash
cd multi-modal-rag
```

### Step 2 — Create and activate a virtual environment

```bash
uv venv --python 3.12
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
uv pip install -e ".[dev]"
```

This installs everything: `glmocr`, `pymupdf`, `Pillow`, `pydantic`, `rich`, `tqdm`, and all dev tools.

### Step 4 — Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your Z.AI API key:

```dotenv
Z_AI_API_KEY=your-actual-api-key-here
LOG_LEVEL=INFO
OUTPUT_DIR=./output
```

> **Never commit `.env` to git.** It is already in `.gitignore`.

### Step 5 — Parse your first document

```bash
python scripts/parse.py path/to/your/document.pdf
```

Output files appear in `./output/`:
- `document.md` — full document as Markdown (with LaTeX formulas and HTML tables)
- `document.json` — structured JSON with per-element bboxes and metadata

---

## CLI Reference

```
python scripts/parse.py <input> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | *(required)* | Path to a PDF/image file **or** a directory of documents |
| `--output` | `./output/` | Where to save parsed results |
| `--format` | `both` | Output format: `markdown`, `json`, or `both` |
| `--chunks` | off | Also generate RAG-ready chunks JSON |
| `--log-level` | `INFO` | Verbosity: `DEBUG`, `INFO`, or `WARNING` |

### Examples

```bash
# Parse a single PDF — save Markdown + JSON
python scripts/parse.py paper.pdf

# Parse a PDF and also generate RAG chunks
python scripts/parse.py paper.pdf --chunks

# Parse a directory of documents, Markdown output only
python scripts/parse.py ./docs/ --format markdown --output ./parsed/

# Parse with debug logging
python scripts/parse.py paper.pdf --log-level DEBUG

# Save everything to a custom output directory
python scripts/parse.py paper.pdf --chunks --format both --output ./results/
```

### Output files produced

| Flag | Files created |
|------|--------------|
| `--format markdown` | `{name}.md` |
| `--format json` | `{name}.md` + `{name}.json` |
| `--format both` (default) | `{name}.md` + `{name}.json` |
| `--chunks` (add-on) | `{name}_chunks.json` |

---

## Output Format Examples

### Markdown output (`document.md`)

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

### JSON output (`document.json`)

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
          "bbox": [0.1, 0.05, 0.9, 0.12],
          "score": 0.97,
          "reading_order": 0
        }
      ]
    }
  ]
}
```

### Chunks output (`document_chunks.json`)

```json
[
  {
    "text": "# Introduction\n\nDocument understanding is a fundamental challenge ...",
    "chunk_id": "paper.pdf_1_0",
    "page": 1,
    "element_types": ["paragraph_title", "paragraph"],
    "bbox": null,
    "source_file": "paper.pdf",
    "is_atomic": false
  },
  {
    "text": "| Model | F1 | Notes |\n|-------|----|-------|\n| Ours | 94.62 | ... |",
    "chunk_id": "paper.pdf_1_1",
    "page": 1,
    "element_types": ["table"],
    "bbox": [0.1, 0.4, 0.9, 0.6],
    "source_file": "paper.pdf",
    "is_atomic": true
  }
]
```

> **`is_atomic: true`** means the chunk is a table, formula, or algorithm — it was never split. This is important for RAG: splitting a table mid-row produces garbage.

---

## Supported File Types

| Extension | Description |
|-----------|-------------|
| `.pdf` | PDF documents (all pages) |
| `.png` | PNG images |
| `.jpg` / `.jpeg` | JPEG images |
| `.tiff` | TIFF images |
| `.bmp` | Bitmap images |

---

## Project Structure

```
multi-modal-rag/
├── pyproject.toml              # Dependencies and tool config
├── config.yaml                 # GLM-OCR SDK cloud settings
├── .env.example                # Template for your API key
├── .env                        # Your actual keys (never commit this)
│
├── src/
│   └── doc_parser/
│       ├── config.py           # Loads Z_AI_API_KEY from .env
│       ├── pipeline.py         # DocumentParser — wraps glmocr SDK
│       ├── post_processor.py   # Converts parsed elements → Markdown
│       ├── chunker.py          # Structure-aware chunker for RAG
│       └── utils/
│           └── pdf_utils.py    # PyMuPDF helpers (page extraction, DPI)
│
├── scripts/
│   └── parse.py                # CLI entrypoint
│
├── tests/
│   ├── conftest.py
│   ├── unit/                   # 24 tests — run without API key
│   │   ├── test_post_processor.py
│   │   └── test_chunker.py
│   └── integration/            # 7 tests — require Z_AI_API_KEY
│       └── test_pipeline_e2e.py
│
├── notebooks/
│   └── 01_quickstart.ipynb    # Interactive demo notebook
│
└── data/
    ├── raw/                    # Put your source documents here
    └── processed/              # Intermediate files (gitignored)
```

---

## Using the Python API

You can also use the pipeline directly in your own Python code:

```python
import sys
sys.path.insert(0, "src")

from pathlib import Path
from doc_parser.pipeline import DocumentParser
from doc_parser.chunker import structure_aware_chunking

# Initialize parser (reads Z_AI_API_KEY from .env automatically)
parser = DocumentParser()

# Parse a document
result = parser.parse_file(Path("my_paper.pdf"))

print(f"Pages: {len(result.pages)}")
print(f"Elements: {result.total_elements}")

# Get the full Markdown
full_markdown = "\n\n".join(page.markdown for page in result.pages)

# Save to disk
result.save(Path("./output"))

# Generate RAG chunks
all_chunks = []
for page in result.pages:
    chunks = structure_aware_chunking(
        page.elements,
        source_file="my_paper.pdf",
        page=page.page_num,
        max_chunk_tokens=512,      # tune this for your use case
    )
    all_chunks.extend(chunks)

# Each chunk is ready to embed and insert into a vector store
for chunk in all_chunks:
    print(chunk.chunk_id, chunk.is_atomic, chunk.text[:80])
```

---

## Running Tests

```bash
# Unit tests only (no API key needed — runs in ~1 second)
uv run pytest tests/unit/ -v

# Integration tests (requires Z_AI_API_KEY in .env)
uv run pytest tests/integration/ -v -m integration

# All tests
uv run pytest -v
```

---

## Development

### Lint and format

```bash
# Check for lint errors
uv run ruff check src/ tests/ scripts/

# Auto-fix lint errors
uv run ruff check --fix src/ tests/ scripts/

# Format code
uv run ruff format src/ tests/ scripts/
```

### Type checking

```bash
uv run mypy src/
```

---

## Configuration Reference

### `.env` variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `Z_AI_API_KEY` | Yes | — | Your Z.AI cloud API key |
| `LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING` |
| `OUTPUT_DIR` | No | `./output` | Default output directory |

### `config.yaml` settings

| Setting | Default | Description |
|---------|---------|-------------|
| `pipeline.maas.enabled` | `true` | Use cloud MaaS API (always true) |
| `pipeline.maas.model` | `glm-ocr` | Model name on Z.AI |
| `pipeline.layout.confidence_threshold` | `0.3` | Min detection confidence (0–1) |
| `pipeline.layout.nms_threshold` | `0.5` | Non-max suppression threshold |
| `pipeline.output.include_bbox` | `true` | Include bounding boxes in output |
| `pipeline.output.max_tokens` | `8192` | Max tokens per API call |

---

## Chunking Behavior

The chunker respects document structure — it never breaks a table mid-row or splits a formula:

| Element type | Behavior |
|---|---|
| `table` | Always its own atomic chunk (`is_atomic=True`) |
| `formula` / `inline_formula` | Always its own atomic chunk |
| `algorithm` | Always its own atomic chunk |
| `document_title` / `paragraph_title` | Attaches to the next content element (no orphan headings) |
| `paragraph`, `text`, `references` | Accumulated up to `max_chunk_tokens` (default: 512) |

Token estimation: `len(text.split()) × 1.3` (accounts for subword tokenization).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'glmocr'`**
```bash
uv pip install -e ".[dev]"
```

**`ValidationError: z_ai_api_key field required`**
- Make sure `.env` exists and contains `Z_AI_API_KEY=your-key`
- Or export it: `export Z_AI_API_KEY=your-key`

**`FileNotFoundError: File not found`**
- Check the path to your document is correct
- Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

**Integration tests not running (just skipped)**
- This is correct behavior when `Z_AI_API_KEY` is not set
- Add your key to `.env` and they will run

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Layout detection + OCR (cloud) | `glmocr` | ≥0.1.0 |
| PDF → image extraction | `pymupdf` | ≥1.27.2 |
| Image processing | `Pillow` | ≥12.1.1 |
| Config management | `pydantic-settings` | ≥2.8.0 |
| Progress bars | `tqdm` | ≥4.67.0 |
| Terminal output | `rich` | ≥14.0.0 |
| Testing | `pytest` | ≥8.4.0 |
| Linting | `ruff` | ≥0.11.0 |
