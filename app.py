"""Streamlit app: upload a PDF and visualize GLM-OCR detected elements with bounding boxes."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / "src"))

from doc_parser.config import get_settings
from doc_parser.pipeline import DocumentParser, ParseResult

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GLM-OCR Document Visualizer",
    page_icon="📄",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

# SDK processes PDFs at 96 DPI (confirmed: 612pt × 96/72 = 816px matches bbox values)
SDK_DPI = 96

# Display at higher resolution for clarity, then scale bboxes up to match
DISPLAY_DPI = 150
SCALE = DISPLAY_DPI / SDK_DPI  # bbox coords × SCALE = display pixel coords

# Color map: element label → (R, G, B)
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "document_title":  (220,  50,  50),   # red
    "paragraph_title": ( 30, 100, 220),   # blue
    "abstract":        ( 20, 160, 160),   # teal
    "paragraph":       ( 40, 160,  40),   # green
    "text":            ( 40, 160,  40),   # green
    "table":           (230, 120,   0),   # orange
    "formula":         (150,  50, 220),   # purple
    "inline_formula":  (180,  80, 220),   # light purple
    "figure_caption":  (  0, 180, 200),   # cyan
    "caption":         (  0, 180, 200),   # cyan
    "code_block":      (200, 180,   0),   # yellow
    "algorithm":       (220,   0, 180),   # magenta
    "footnotes":       (200,  80, 120),   # pink
    "reference":       (140,  80,  40),   # brown
    "header":          (160, 160, 160),   # light gray
    "footer":          (160, 160, 160),   # light gray
    "page_number":     (120, 120, 120),   # gray
    "image":           (100, 100, 100),   # dark gray
    "seal":            ( 60,  60,  60),   # near black
}
DEFAULT_COLOR = (180, 180, 0)  # fallback yellow for unknown labels


def get_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_page(pdf_path: Path, page_num: int) -> Image.Image:
    """Render a PDF page as a PIL Image at DISPLAY_DPI."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_num)
        mat = fitz.Matrix(DISPLAY_DPI / 72, DISPLAY_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def draw_bboxes(img: Image.Image, elements: list) -> Image.Image:
    """Draw colored bounding boxes and labels onto the image."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    for el in elements:
        if not el.bbox or len(el.bbox) != 4:
            continue

        label = el.label
        color = get_color(label)
        x1, y1, x2, y2 = [int(v * SCALE) for v in el.bbox]

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Semi-transparent fill
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(*color, 35),
            outline=(*color, 220),
            width=2,
        )

        # Label badge at top-left of box
        badge_text = label.replace("_", " ")
        tx, ty = x1 + 3, max(y1 - 18, 0)
        draw.rectangle([tx - 2, ty - 1, tx + len(badge_text) * 7 + 2, ty + 14],
                       fill=(*color, 200))
        draw.text((tx, ty), badge_text, fill=(255, 255, 255))

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


# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result: ParseResult | None = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path: Path | None = None

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📄 GLM-OCR Document Visualizer")
st.caption("Upload a PDF to detect and visualize document elements (PP-DocLayout-V3 + GLM-OCR 0.9B)")

# Sidebar: upload + controls
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        # Save upload to temp file (needed for PyMuPDF + glmocr)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = Path(tmp.name)

        if (
            st.session_state.pdf_path is None
            or st.session_state.pdf_path.name != tmp_path.name
        ):
            st.session_state.pdf_path = tmp_path
            st.session_state.result = None  # reset on new upload

        if st.button("Parse Document", type="primary", use_container_width=True):
            with st.spinner("Sending to GLM-OCR MaaS API…"):
                try:
                    parser = DocumentParser()
                    st.session_state.result = parser.parse_file(tmp_path)
                    st.success(
                        f"Done — {len(st.session_state.result.pages)} pages, "
                        f"{st.session_state.result.total_elements} elements"
                    )
                except Exception as e:
                    st.error(f"Parse failed: {e}")

    st.divider()
    st.header("Display options")
    show_text = st.checkbox("Show element text", value=False)
    show_markdown = st.checkbox("Show page Markdown", value=False)

# ── Main area ─────────────────────────────────────────────────────────────────

result: ParseResult | None = st.session_state.result
pdf_path: Path | None = st.session_state.pdf_path

if result is None:
    st.info("Upload a PDF and click **Parse Document** to get started.")
    st.stop()

# Page selector
total_pages = len(result.pages)
page_idx = st.slider("Page", min_value=1, max_value=total_pages, value=1) - 1
page_result = result.pages[page_idx]
elements = page_result.elements

# Columns: image (left) | details (right)
col_img, col_detail = st.columns([3, 2])

with col_img:
    st.subheader(f"Page {page_idx + 1} — {len(elements)} elements detected")

    img = render_page(pdf_path, page_idx)
    img_with_boxes = draw_bboxes(img, elements)
    st.image(img_with_boxes, use_container_width=True)

    # Legend for labels on this page
    if elements:
        labels_present = {el.label for el in elements}
        build_legend(labels_present)

with col_detail:
    # Summary counts
    st.subheader("Element breakdown")
    from collections import Counter
    counts = Counter(el.label for el in elements)
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        r, g, b = get_color(label)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(
            f'<span style="background:{hex_color};padding:1px 6px;border-radius:3px;'
            f'color:white;font-size:0.8rem;">{label.replace("_", " ")}</span> '
            f"× **{count}**",
            unsafe_allow_html=True,
        )

    # Element list
    if show_text and elements:
        st.subheader("Elements (reading order)")
        for el in sorted(elements, key=lambda e: e.reading_order):
            r, g, b = get_color(el.label)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            with st.expander(
                f"[{el.reading_order}] {el.label.replace('_', ' ')} — "
                f"{el.text[:60]}{'…' if len(el.text) > 60 else ''}"
            ):
                st.markdown(
                    f'<span style="background:{hex_color};padding:1px 6px;'
                    f'border-radius:3px;color:white;">{el.label}</span>',
                    unsafe_allow_html=True,
                )
                st.text(el.text)
                st.caption(f"bbox: {[int(v) for v in el.bbox]}")

    # Page markdown
    if show_markdown:
        st.subheader("Page Markdown")
        st.markdown(page_result.markdown or "_No markdown for this page._")

# ── Full document markdown tab ────────────────────────────────────────────────
with st.expander("Full document Markdown", expanded=False):
    st.markdown(result.full_markdown or "_No markdown available._")
