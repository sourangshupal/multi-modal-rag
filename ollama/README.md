# GLM-OCR via Ollama — Local Testing

Run GLM-OCR parsing entirely on your machine (no cloud API key needed) using
[Ollama](https://ollama.com) and the `glm-ocr:latest` model.

---

## Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the GLM-OCR model

```bash
ollama pull glm-ocr:latest
```

This downloads the GLM-OCR 0.9B vision model (~600 MB). It is the same model
that powers the Z.AI cloud API — just running locally.

### 3. Install layout detection dependencies

Layout detection (PP-DocLayout-V3) requires five extra packages not bundled with
`glmocr` by default. Install them all with the `[layout]` extra:

```bash
uv pip install "glmocr[layout]"
```

This installs: `torch>=2.10`, `torchvision>=0.25`, `transformers>=5.3`,
`sentencepiece>=0.2`, `accelerate>=1.13`.

The PP-DocLayout-V3 model weights (~400 MB) are downloaded automatically from
HuggingFace Hub on first use and cached in `~/.cache/huggingface/`.

> **Apple Silicon / CPU-only:** PyTorch will run on CPU. Expect ~5–10s extra per page
> for layout detection. GPU (CUDA or MPS) is not required but speeds things up.

### 4. Verify the model is present

```bash
ollama list
# Should include a line like:
# glm-ocr:latest   abc123...   600 MB   ...
```

---

## Running the Test Script

### Start Ollama (if not already running)

```bash
ollama serve
```

Ollama runs on `http://localhost:11434` by default. Leave this terminal open.

### Basic parse (output saved automatically to `ollama/output/`)

```bash
uv run python ollama/test_parse.py data/raw/test_page1.pdf
```

Output is saved by default to `ollama/output/` — no `--output` flag needed:
- `ollama/output/test_page1.md` — extracted Markdown
- `ollama/output/test_page1_elements.json` — raw element JSON

### Parse and also print raw JSON elements to terminal

```bash
uv run python ollama/test_parse.py data/raw/test_page1.pdf --show-elements
```

### Save output to a custom directory

```bash
uv run python ollama/test_parse.py data/raw/test_page1.pdf --output ./my_results/
```

### Parse an image file

```bash
uv run python ollama/test_parse.py data/raw/figure.png
```

---

## Understanding the Output

The script prints:

```
Parser  : GLM-OCR via Ollama (glm-ocr:latest)
Config  : /path/to/ollama/config.yaml
Input   : data/raw/test_page1.pdf

Parsed in 12.3s
   Pages    : 1
   Elements : 14

------------------------------------------------------------
MARKDOWN OUTPUT
------------------------------------------------------------
# Title of the Paper

Abstract text here...
```

**Timing:** Ollama is slower than the cloud API. Expect 5–30 seconds per page
depending on your hardware (GPU vs CPU).

---

## Cloud vs Ollama: Expected Differences

| Feature | Cloud API | Ollama |
|---|---|---|
| Layout detection | PP-DocLayout-V3 (precise bboxes) | PP-DocLayout-V3 (same, `enable_layout: true`) |
| Element labels | `paragraph`, `table`, `figure`, etc. | Same label set (`doc_title`, `paragraph_title`, `table`, …) |
| Text extraction | High quality | High quality (same GLM-OCR 0.9B model) |
| Speed | Fast (cloud GPU) | 5–30s per page (local CPU/MPS/CUDA) |
| Cost | API credits | Free (local) |
| Privacy | Data sent to Z.AI | Fully local |

**Key point:** both backends run the same PP-DocLayout-V3 layout detector and
GLM-OCR 0.9B text model. The main difference is speed — the cloud API runs on
dedicated GPUs, while Ollama uses your local hardware. Output quality is
essentially equivalent for most document types.

---

## Troubleshooting

### `ERROR: Parsing failed: Connection refused`

Ollama is not running. Start it:
```bash
ollama serve
```

### `ERROR: Parsing failed: model not found`

The model name in `config.yaml` doesn't match what you pulled. Check:
```bash
ollama list
```
Update `ocr_api.model` in `ollama/config.yaml` to match exactly.

### `ERROR: glmocr not installed`

```bash
uv pip install glmocr
```

### Slow parsing

- First run is slower (model loads into memory). Subsequent calls are faster.
- If you have a GPU, Ollama will use it automatically.
- Check GPU usage: `nvidia-smi` (NVIDIA) or `sudo powermetrics --samplers gpu_power` (Apple Silicon).

### API sanity check (raw Ollama)

Test Ollama directly without the SDK:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"glm-ocr:latest","prompt":"Hello","stream":false}'
```

A successful response looks like:
```json
{"model":"glm-ocr:latest","response":"Hello! How can I help you?", ...}
```

---

## Config Reference (`ollama/config.yaml`)

```yaml
pipeline:
  maas:
    enabled: false        # Must be false to use Ollama instead of Z.AI cloud

  ocr_api:
    api_host: localhost
    api_port: 11434
    api_path: /api/generate   # Ollama native endpoint
    model: glm-ocr:latest
    api_mode: ollama_generate  # Critical — without this you get 502 errors

  enable_layout: true     # Run PP-DocLayout-V3 (requires torch + transformers)
```

The `api_mode: ollama_generate` setting is essential. It tells the SDK to format
requests for Ollama's `/api/generate` endpoint rather than the OpenAI-compatible
`/v1/chat/completions` endpoint.

### Layout detection dependencies

When `enable_layout: true`, the SDK loads PP-DocLayout-V3 via HuggingFace
Transformers. Use the official `[layout]` extra to get all required packages:

```bash
uv pip install "glmocr[layout]"
```

| Package | Version | Why needed |
|---|---|---|
| `torch` | >=2.10 | Runs the PP-DocLayout-V3 neural network |
| `torchvision` | >=0.25 | Image transforms for the detector |
| `transformers` | >=5.3 | `PPDocLayoutV3ForObjectDetection` + image processor |
| `sentencepiece` | >=0.2 | Tokenizer dependency |
| `accelerate` | >=1.13 | Model loading optimisation |
| `opencv-python` | >=4.10 | Image preprocessing for PP-DocLayout-V3 input |

The `layout.model_dir` config key (`PaddlePaddle/PP-DocLayoutV3_safetensors`) is the
HuggingFace model ID. Weights (~400 MB) are auto-downloaded on first run.

**SDK bug note:** `LayoutConfig` in glmocr 0.1.3 does not declare `id2label` as
a field, but `PPDocLayoutDetector.__init__` accesses `config.id2label` directly.
This causes `AttributeError: 'LayoutConfig' object has no attribute 'id2label'`
when `enable_layout: true` without the `layout.id2label` key in the YAML.
The fix (applied in `ollama/config.yaml`) is to explicitly set `id2label: null`
so Pydantic stores it as an extra field.

---

## Visualizer (parse directly from the UI)

The Streamlit visualizer supports direct parsing — no need to run `test_parse.py` first:

```bash
uv run streamlit run ollama/visualize.py
```

In the sidebar:
- **"Parse new PDF"** — upload any PDF and click **▶ Parse with Ollama**. A spinner shows
  while the pipeline runs (~30–60 s). Results appear immediately and are saved to
  `ollama/output/` automatically.
- **"Load saved results"** — pick any previously parsed `*_elements.json` file.

---

## Using Ollama in the Main Pipeline

Set `PARSER_BACKEND=ollama` in your `.env` to use Ollama as the parser backend
in all scripts and the REST API — no `Z_AI_API_KEY` required:

```dotenv
PARSER_BACKEND=ollama
# Z_AI_API_KEY not needed
```

Then run normally:

```bash
python scripts/parse.py data/raw/paper.pdf --chunks
python scripts/ingest.py data/raw/paper.pdf
```

---

## Next Steps

Once you have verified that local parsing works:

1. Compare the Markdown output with the cloud API output for the same document.
2. Check if element labels and bboxes are present in the JSON (`--show-elements`).
