"""Streamlit app: visualize Ollama/PP-DocLayoutV3 parsed results.

Supports two workflows:
  1. Load pre-saved results — pick any *_elements.json from ollama/output/
  2. Parse on the fly    — upload a PDF, click Parse, wait ~30s, see results
"""
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ollama Document Visualizer",
    page_icon="🦙",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
RENDER_DPI = 150
BBOX_SCALE = 1000
_CONFIG = Path(__file__).parent / "config.yaml"
OUTPUT_DIR = Path(__file__).parent / "output"

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    # Shared with cloud app
    "text":              ( 40, 160,  40),   # green
    "table":             (230, 120,   0),   # orange
    "formula":           (150,  50, 220),   # purple
    "algorithm":         (220,   0, 180),   # magenta
    "image":             (100, 100, 100),   # dark gray
    "reference":         (140,  80,  40),   # brown
    # Ollama / PP-DocLayoutV3 labels
    "doc_title":         (220,  50,  50),   # red
    "paragraph_title":   ( 30, 100, 220),   # blue
    "abstract":          ( 20, 160, 160),   # teal
    "aside_text":        (180, 100,  20),   # amber
    "figure_title":      (  0, 180, 200),   # cyan
    "footnote":          (200,  80, 120),   # pink
    "vision_footnote":   (180,  60, 140),   # rose
    "chart":             ( 80, 160,  80),   # olive
    "seal":              ( 60,  60,  60),   # near black
    "content":           ( 40, 160,  40),   # green (alias)
    "reference_content": (140,  80,  40),   # brown (alias)
    "figure":            (100, 100, 100),   # dark gray (alias)
    "inline_formula":    (180,  80, 220),   # light purple
    "caption":           (  0, 180, 200),   # cyan (alias)
    "header":            (160, 160, 160),   # light gray
    "footer":            (160, 160, 160),   # light gray
    "page_number":       (120, 120, 120),   # gray
}
DEFAULT_COLOR = (180, 180, 0)  # fallback yellow for unknown labels


def get_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_page(pdf_path: Path, page_num: int) -> Image.Image:
    """Render a PDF page as a PIL Image at RENDER_DPI."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_num)
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def draw_bboxes(img: Image.Image, elements: list[dict]) -> Image.Image:
    """Draw colored bounding boxes onto the page image."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for el in elements:
        bbox = el.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue

        label = el.get("label", "unknown")
        color = get_color(label)
        x1 = int(bbox[0] * w / BBOX_SCALE)
        y1 = int(bbox[1] * h / BBOX_SCALE)
        x2 = int(bbox[2] * w / BBOX_SCALE)
        y2 = int(bbox[3] * h / BBOX_SCALE)

        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle([x1, y1, x2, y2], fill=(*color, 35), outline=(*color, 220), width=2)

        badge_text = label.replace("_", " ")
        tx, ty = x1 + 3, max(y1 - 18, 0)
        draw.rectangle([tx - 2, ty - 1, tx + len(badge_text) * 7 + 2, ty + 14],
                       fill=(*color, 200))
        draw.text((tx, ty), badge_text, fill=(255, 255, 255))

    return img


def draw_polygons(img: Image.Image, elements: list[dict]) -> Image.Image:
    """Draw translucent polygon overlays for elements that have polygon data."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    for el in elements:
        polygon = el.get("polygon")
        if not polygon or len(polygon) < 3:
            continue

        label = el.get("label", "unknown")
        color = get_color(label)
        pts = [(int(p[0] * w / BBOX_SCALE), int(p[1] * h / BBOX_SCALE)) for p in polygon]
        draw.polygon(pts, fill=(*color, 50), outline=(*color, 240))

    return img


def build_legend(labels_present: set[str]) -> None:
    """Render a color legend for the element types found on this page."""
    st.markdown("**Legend**")
    cols = st.columns(4)
    for i, label in enumerate(sorted(labels_present)):
        r, g, b = get_color(label)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        cols[i % 4].markdown(
            f'<span style="background:{hex_color};padding:2px 8px;'
            f'border-radius:4px;color:white;font-size:0.75rem;">'
            f'{label.replace("_", " ")}</span>',
            unsafe_allow_html=True,
        )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_result(json_path: Path) -> tuple[list[list[dict]], str]:
    """Load elements JSON + paired markdown. Returns (pages, markdown_text)."""
    pages: list[list[dict]] = json.loads(json_path.read_text())
    md_path = json_path.parent / (json_path.stem.replace("_elements", "") + ".md")
    md = md_path.read_text() if md_path.exists() else ""
    return pages, md


def find_pdf(stem: str) -> Path | None:
    """Look for stem.pdf in data/raw/ relative to project root."""
    root = Path(__file__).parent.parent
    candidate = root / "data" / "raw" / f"{stem}.pdf"
    return candidate if candidate.exists() else None


def run_parser(pdf_path: Path) -> tuple[list[list[dict]], str]:
    """Parse a PDF with the local Ollama pipeline and return (pages, markdown)."""
    try:
        from glmocr import GlmOcr  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "glmocr not installed. Run: uv pip install -e '.[layout]'"
        ) from e

    parser = GlmOcr(config_path=str(_CONFIG))
    result = parser.parse(str(pdf_path), save_layout_visualization=False)

    pages: list[list[dict]] = result.json_result if isinstance(result.json_result, list) else []
    md: str = result.markdown_result or ""
    return pages, md


def save_result(stem: str, pages: list[list[dict]], md: str) -> Path:
    """Persist parsed results to ollama/output/ and return the JSON path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{stem}_elements.json"
    json_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    if md:
        (OUTPUT_DIR / f"{stem}.md").write_text(md, encoding="utf-8")
    return json_path


# ── Session state ─────────────────────────────────────────────────────────────

if "pages" not in st.session_state:
    st.session_state.pages: list[list[dict]] | None = None
if "markdown" not in st.session_state:
    st.session_state.markdown: str = ""
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path: Path | None = None
if "json_path" not in st.session_state:
    st.session_state.json_path: Path | None = None

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🦙 Ollama Document Visualizer")
st.caption("Parse any PDF locally with PP-DocLayoutV3 + GLM-OCR via Ollama, or load saved results")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Tab 1: Parse a new PDF on the fly ─────────────────────────────────────
    st.header("Parse new PDF")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF to parse",
        type=["pdf"],
        key="pdf_uploader",
        help="Requires Ollama running locally with glm-ocr:latest pulled",
    )

    if uploaded_pdf is not None:
        if st.button("▶ Parse with Ollama", type="primary", use_container_width=True):
            stem = Path(uploaded_pdf.name).stem

            # Write upload to a temp file so GlmOcr can read it by path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = Path(tmp.name)

            with st.spinner(f"Parsing {uploaded_pdf.name} — this may take 30–60 s…"):
                t0 = time.perf_counter()
                try:
                    pages, md = run_parser(tmp_path)
                    elapsed = time.perf_counter() - t0
                    json_path = save_result(stem, pages, md)
                    st.session_state.pages = pages
                    st.session_state.markdown = md
                    st.session_state.pdf_path = tmp_path
                    st.session_state.json_path = json_path
                    st.success(
                        f"Done in {elapsed:.1f}s — "
                        f"{len(pages)} page(s), "
                        f"{sum(len(p) for p in pages)} elements"
                    )
                except Exception as exc:
                    st.error(f"Parse failed: {exc}")
                    st.info(
                        "Check that Ollama is running (`ollama serve`) "
                        "and the model is pulled (`ollama pull glm-ocr:latest`)."
                    )

    st.divider()

    # ── Tab 2: Load pre-saved results ─────────────────────────────────────────
    st.header("Load saved results")

    json_files = sorted(OUTPUT_DIR.glob("*_elements.json")) if OUTPUT_DIR.exists() else []

    if not json_files:
        st.info("No saved results yet. Parse a PDF above to create some.")
    else:
        selected_name = st.selectbox(
            "JSON file",
            options=[f.name for f in json_files],
            index=0,
        )
        selected_json = OUTPUT_DIR / selected_name

        if st.session_state.json_path != selected_json:
            if st.button("Load", use_container_width=True):
                st.session_state.json_path = selected_json
                st.session_state.pages, st.session_state.markdown = load_result(selected_json)
                stem = selected_name.replace("_elements.json", "")
                st.session_state.pdf_path = find_pdf(stem)

        # PDF source for saved results (if not already set from parse)
        if (
            st.session_state.json_path == selected_json
            and st.session_state.pdf_path is None
        ):
            st.info("PDF not found in `data/raw/`. Upload it below to see page renders.")
            fallback = st.file_uploader("Upload PDF for page rendering", type=["pdf"], key="pdf_fallback")
            if fallback:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(fallback.read())
                    st.session_state.pdf_path = Path(tmp.name)

    st.divider()

    # ── Display options ────────────────────────────────────────────────────────
    st.header("Display options")
    show_content = st.checkbox("Show element content", value=False)
    show_markdown = st.checkbox("Show page Markdown", value=False)
    show_polygons = st.checkbox("Show polygons (precise outlines)", value=False)


# ── Main area ─────────────────────────────────────────────────────────────────

pages: list[list[dict]] | None = st.session_state.pages
pdf_path: Path | None = st.session_state.pdf_path

if pages is None:
    st.info("Upload and parse a PDF, or load a saved result from the sidebar.")
    st.stop()

total_pages = len(pages)
page_idx = st.slider("Page", min_value=1, max_value=total_pages, value=1) - 1
elements = pages[page_idx]

col_img, col_detail = st.columns([3, 2])

with col_img:
    st.subheader(f"Page {page_idx + 1} — {len(elements)} elements detected")

    if pdf_path:
        img = render_page(pdf_path, page_idx)
        img_with_boxes = draw_bboxes(img, elements)
        if show_polygons:
            img_with_boxes = draw_polygons(img_with_boxes, elements)
        st.image(img_with_boxes, use_container_width=True)
    else:
        st.warning("No PDF loaded — upload one in the sidebar to see the rendered page.")

    if elements:
        labels_present = {el.get("label", "unknown") for el in elements}
        build_legend(labels_present)

with col_detail:
    st.subheader("Element breakdown")
    counts = Counter(el.get("label", "unknown") for el in elements)
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        r, g, b = get_color(label)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(
            f'<span style="background:{hex_color};padding:1px 6px;border-radius:3px;'
            f'color:white;font-size:0.8rem;">{label.replace("_", " ")}</span> '
            f"× **{count}**",
            unsafe_allow_html=True,
        )

    if show_content and elements:
        st.divider()
        st.subheader("Elements (index order)")
        for el in sorted(elements, key=lambda e: e.get("index", 0)):
            label = el.get("label", "unknown")
            content = el.get("content", "")
            r, g, b = get_color(label)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            preview = content[:60] + ("…" if len(content) > 60 else "")
            with st.expander(f"[{el.get('index', '?')}] {label.replace('_', ' ')} — {preview}"):
                st.markdown(
                    f'<span style="background:{hex_color};padding:1px 6px;'
                    f'border-radius:3px;color:white;">{label}</span>',
                    unsafe_allow_html=True,
                )
                st.text(content)
                st.caption(f"bbox_2d: {el.get('bbox_2d', [])}")
                polygon = el.get("polygon")
                if polygon:
                    st.caption(f"polygon: {len(polygon)} points")

    if show_markdown:
        st.divider()
        st.subheader("Page Markdown")
        st.info("Per-page markdown is not available in the Ollama output format.")

# ── Full document markdown ─────────────────────────────────────────────────────
with st.expander("Full document Markdown", expanded=False):
    md = st.session_state.markdown
    if md:
        st.markdown(md)
    else:
        st.info("_No markdown available._")
