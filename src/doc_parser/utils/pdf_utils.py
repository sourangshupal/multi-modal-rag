"""PyMuPDF helpers for PDF → image extraction and validation."""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"})


def pdf_page_to_image(pdf_path: Path, page_num: int, dpi: int = 300) -> Image.Image:
    """Extract a single PDF page as a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        page_num: Zero-based page index.
        dpi: Resolution in dots per inch (default 300 for high quality).

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        IndexError: If page_num is out of range.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        if page_num >= len(doc):
            raise IndexError(f"Page {page_num} out of range (document has {len(doc)} pages)")
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # type: ignore[arg-type]
    finally:
        doc.close()


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Page count as an integer.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def validate_input_file(file_path: Path) -> None:
    """Validate that a file exists and has a supported extension.

    Args:
        file_path: Path to the input document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
