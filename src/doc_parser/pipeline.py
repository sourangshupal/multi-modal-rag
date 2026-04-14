"""Main document parsing pipeline wrapping the GLM-OCR SDK."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from doc_parser.config import get_settings
from doc_parser.post_processor import assemble_markdown, save_to_json
from doc_parser.utils.pdf_utils import count_pdf_pages

logger = logging.getLogger(__name__)

try:
    from glmocr import GlmOcr  # type: ignore[import]
    _GLMOCR_AVAILABLE = True
except ImportError:
    _GLMOCR_AVAILABLE = False
    logger.warning(
        "glmocr package not installed. Install with: uv pip install glmocr"
    )


@dataclass
class ParsedElement:
    """A single detected and recognized document element.

    Attributes:
        label: Element category (e.g. 'document_title', 'paragraph', 'table').
        text: Recognized text content.
        bbox: Bounding box [x1, y1, x2, y2] in normalized coordinates.
        score: Detection confidence score (0.0–1.0).
        reading_order: Sequence index for correct reading order assembly.
    """

    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int


@dataclass
class PageResult:
    """Parsing result for a single document page.

    Attributes:
        page_num: One-based page number.
        elements: All detected elements on this page.
        markdown: Assembled Markdown string for this page.
    """

    page_num: int
    elements: list[ParsedElement] = field(default_factory=list)
    markdown: str = ""


@dataclass
class ParseResult:
    """Complete parsing result for a document.

    Attributes:
        source_file: Path to the original document.
        pages: Per-page parsing results.
        total_elements: Sum of all elements across all pages.
    """

    source_file: str
    pages: list[PageResult] = field(default_factory=list)
    total_elements: int = 0
    full_markdown: str = ""  # Full document markdown from SDK (preferred over per-page assembly)

    @classmethod
    def from_sdk_result(cls, raw: Any, source_file: str) -> ParseResult:
        """Build a ParseResult from a raw GLM-OCR SDK PipelineResult.

        The SDK returns a PipelineResult with:
        - json_result: list[list[dict]] — one inner list per page, each dict has
          'index', 'label', 'content', 'bbox_2d'
        - markdown_result: str — full document markdown for the entire document

        Args:
            raw: PipelineResult from GlmOcr.parse().
            source_file: Path to the source document.

        Returns:
            Populated ParseResult instance.
        """
        pages: list[PageResult] = []

        # json_result is a list-of-lists: [page][element]
        raw_pages: list[list[dict[str, Any]]] = getattr(raw, "json_result", [])

        # markdown_result is a single string for the whole document (not per-page)
        full_markdown: str = getattr(raw, "markdown_result", "") or ""

        for page_idx, raw_elements in enumerate(raw_pages):
            page_num = page_idx + 1
            elements: list[ParsedElement] = []

            for raw_el in raw_elements:
                bbox_2d = raw_el.get("bbox_2d", [0, 0, 1, 1])
                el = ParsedElement(
                    label=raw_el.get("label", "paragraph"),
                    text=raw_el.get("content", ""),
                    bbox=[float(v) for v in bbox_2d],
                    score=1.0,  # SDK does not provide a confidence score
                    reading_order=raw_el.get("index", len(elements)),
                )
                elements.append(el)

            # Per-page markdown assembled from elements (used for chunking metadata)
            markdown = assemble_markdown(elements)  # type: ignore[arg-type]
            pages.append(PageResult(page_num=page_num, elements=elements, markdown=markdown))

        total_elements = sum(len(p.elements) for p in pages)
        return cls(
            source_file=source_file,
            pages=pages,
            total_elements=total_elements,
            full_markdown=full_markdown,
        )

    def save(self, output_dir: Path) -> None:
        """Save this result as Markdown and JSON files.

        Args:
            output_dir: Directory to write output files.
        """
        save_to_json(self, output_dir)


class DocumentParser:
    """Main document parsing pipeline using GLM-OCR MaaS API.

    Wraps the glmocr SDK to provide structured document parsing
    via the Z.AI cloud API (PP-DocLayout-V3 + GLM-OCR 0.9B).

    Example:
        parser = DocumentParser()
        result = parser.parse_file(Path("document.pdf"))
        result.save(Path("./output"))
    """

    def __init__(self) -> None:
        """Initialize the DocumentParser with settings from environment."""
        if not _GLMOCR_AVAILABLE:
            raise ImportError(
                "glmocr package is required. Install with: uv pip install glmocr"
            )
        settings = get_settings()
        # Only pass api_key in cloud mode. In Ollama/self-hosted mode, passing
        # any key (even a valid one) causes GlmOcr to override the YAML config
        # and force MaaS mode regardless of maas.enabled=false.
        api_key = (
            settings.z_ai_api_key.get_secret_value()
            if settings.parser_backend == "cloud" and settings.z_ai_api_key
            else None
        )
        self._parser = GlmOcr(
            config_path=settings.config_yaml_path,
            api_key=api_key,
        )
        logger.info("DocumentParser initialized with config: %s", settings.config_yaml_path)

    def parse_file(self, file_path: str | Path) -> ParseResult:
        """Parse a single PDF or image file.

        Args:
            file_path: Path to the document to parse (PDF or image).

        Returns:
            ParseResult with structured elements and assembled Markdown.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If glmocr is not installed.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Parsing file: %s", file_path)

        settings = get_settings()
        parse_kwargs: dict[str, Any] = {}
        if file_path.suffix.lower() == ".pdf":
            total_pages = count_pdf_pages(file_path)
            if settings.parser_backend == "cloud":
                # Cloud/MaaS mode: must pass explicit page range or the SDK
                # defaults to page 1 only.
                parse_kwargs["start_page_id"] = 0
                parse_kwargs["end_page_id"] = total_pages - 1
                logger.info("PDF has %d pages (PyMuPDF) — parsing all via MaaS", total_pages)
            else:
                # Ollama/self-hosted mode: the glmocr SDK silently drops
                # start_page_id/end_page_id — the internal pypdfium2 page
                # loader handles page range itself.  PyMuPDF and pypdfium2 may
                # report different page counts for certain PDFs; the count below
                # is informational only.
                logger.info(
                    "PDF has %d pages (PyMuPDF) — Ollama mode uses pypdfium2 "
                    "internally; page count may differ slightly",
                    total_pages,
                )

        if settings.parser_backend == "ollama":
            parse_kwargs["save_layout_visualization"] = False

        raw = self._parser.parse(str(file_path), **parse_kwargs)
        result = ParseResult.from_sdk_result(raw, source_file=str(file_path))
        if file_path.suffix.lower() == ".pdf" and len(result.pages) != total_pages:
            logger.warning(
                "Parsed %s: glmocr returned %d pages but PyMuPDF counted %d. "
                "The PDF may have a blank/unreadable trailing page or pypdfium2 "
                "and PyMuPDF disagree on its structure.",
                file_path.name,
                len(result.pages),
                total_pages,
            )
        else:
            logger.info(
                "Parsed %s: %d pages, %d elements",
                file_path.name,
                len(result.pages),
                result.total_elements,
            )
        return result

    def parse_batch(
        self,
        file_paths: list[Path],
        output_dir: Path,
    ) -> list[ParseResult]:
        """Parse multiple files with progress tracking.

        Args:
            file_paths: List of paths to documents to parse.
            output_dir: Directory to save output files.

        Returns:
            List of ParseResult objects, one per input file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ParseResult] = []

        for fp in tqdm(file_paths, desc="Parsing documents", unit="file"):
            try:
                result = self.parse_file(fp)
                result.save(output_dir)
                results.append(result)
            except Exception as e:
                logger.error("Failed to parse %s: %s", fp, e, exc_info=True)
                raise

        return results
