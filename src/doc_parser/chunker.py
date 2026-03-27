"""Structure-aware and document-aware chunkers for RAG-ready document chunks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from doc_parser.post_processor import ElementLike

logger = logging.getLogger(__name__)

# Token-count heuristic multiplier: word count * 1.3 approximates subword tokenization
# for typical English text (e.g. BPE tokenizers such as tiktoken cl100k_base).
_TOKEN_WORD_RATIO: float = 1.3

# Labels that must never be split across chunks
ATOMIC_LABELS: frozenset[str] = frozenset(
    {"table", "formula", "inline_formula", "algorithm", "image", "figure"}
)

# Labels that are headings/titles — attach to following content element.
# figure_title is the caption label emitted by PP-DocLayout-V3 for figure captions;
# it is treated as a title so it binds forward to the next image/figure atomic chunk.
TITLE_LABELS: frozenset[str] = frozenset(
    {"document_title", "paragraph_title", "figure_title"}
)

# Modality classification sets
_IMAGE_TYPES: frozenset[str] = frozenset({"image", "figure"})
_TABLE_TYPES: frozenset[str] = frozenset({"table"})
_FORMULA_TYPES: frozenset[str] = frozenset({"formula", "inline_formula"})
_ALGORITHM_TYPES: frozenset[str] = frozenset({"algorithm"})


def _infer_modality(element_types: list[str]) -> str:
    """Derive chunk modality from element label(s).

    Args:
        element_types: List of element labels in the chunk.

    Returns:
        One of: "image", "table", "formula", "algorithm", "text".
    """
    types = frozenset(element_types)
    if types & _IMAGE_TYPES:
        return "image"
    if types & _TABLE_TYPES:
        return "table"
    if types & _FORMULA_TYPES:
        return "formula"
    if types & _ALGORITHM_TYPES:
        return "algorithm"
    return "text"


@dataclass
class Chunk:
    """A RAG-ready document chunk.

    Attributes:
        text: The chunk text content (or AI caption for image chunks).
        chunk_id: Unique identifier in format "{source_file}_{page}_{idx}".
        page: Page number the chunk came from (first page for cross-page chunks).
        element_types: List of element labels included in this chunk.
        bbox: Bounding box [x1, y1, x2, y2] or None if multi-element chunk.
        source_file: Source document filename.
        is_atomic: True for atomic elements (tables, formulas, images) that must not be split.
        modality: Content type — "text" | "image" | "table" | "formula" | "algorithm".
        image_base64: Base64-encoded PNG of the cropped region (set by image_captioner).
        caption: AI-generated caption text (set by image_captioner).
    """

    text: str
    chunk_id: str
    page: int
    element_types: list[str]
    bbox: list[float] | None
    source_file: str
    is_atomic: bool
    modality: str = field(default="text")
    image_base64: str | None = field(default=None)
    caption: str | None = field(default=None)


def _estimate_tokens(text: str) -> int:
    """Estimate token count using word count heuristic.

    Args:
        text: Input text string.

    Returns:
        Estimated token count (word count * 1.3, rounded down).
        The 1.3 multiplier accounts for subword tokenization in typical English text.
    """
    return int(len(text.split()) * _TOKEN_WORD_RATIO)


def _split_text_into_sub_chunks(text: str, max_tokens: int) -> list[str]:
    """Split a single large text block into sub-chunks that fit within max_tokens.

    Splits on whitespace boundaries to avoid cutting mid-word.

    Args:
        text: The text to split.
        max_tokens: Maximum tokens per sub-chunk.

    Returns:
        List of text sub-chunks each within the token limit.
    """
    words = text.split()
    words_per_chunk = max(1, int(max_tokens / _TOKEN_WORD_RATIO))
    sub_chunks = []
    for i in range(0, len(words), words_per_chunk):
        sub_chunks.append(" ".join(words[i : i + words_per_chunk]))
    return sub_chunks


def document_aware_chunking(
    pages: list[tuple[int, list[ElementLike]]],
    source_file: str,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Chunk a whole document across ALL pages in a single pass.

    Unlike ``structure_aware_chunking`` (single-page), this processes every page
    as one continuous element stream so that:

    - A section heading on the last line of page N attaches to content on page N+1
      instead of becoming an orphan chunk.
    - A ``figure_title`` (figure caption label) is linked directly to the
      following ``image``/``figure`` atomic chunk rather than floating into a
      surrounding text chunk.

    Rules (same as structure_aware_chunking, extended for cross-page):
    - Atomic elements always get their own chunk; never split or merged.
    - ``figure_title`` is intercepted before the atomic flush and prepended to the
      figure/image chunk text so caption and visual are co-located.
    - Section/document/paragraph titles attach forward to the next content element
      even across a page boundary.
    - Text elements accumulate up to ``max_chunk_tokens``; overflow starts a new chunk.
    - Chunk ``page`` is set to the page of the first element that entered the chunk.

    Args:
        pages: Ordered list of ``(page_num, elements)`` pairs, one per document page.
        source_file: Source document filename used in chunk_id generation.
        max_chunk_tokens: Maximum estimated tokens per text chunk (default 512).

    Returns:
        List of Chunk objects in document reading order.
    """
    # Flatten: (page_num, element) sorted by (page_num, reading_order)
    all_pairs: list[tuple[int, ElementLike]] = [
        (page_num, el)
        for page_num, elements in pages
        for el in elements
    ]
    if not all_pairs:
        return []

    all_pairs.sort(key=lambda x: (x[0], x[1].reading_order))

    chunks: list[Chunk] = []
    chunk_idx = 0

    # Accumulator state
    current_texts: list[str] = []
    current_labels: list[str] = []
    current_tokens: int = 0
    current_page: int = all_pairs[0][0]   # page of the first accumulated element

    # Pending title (section heading or figure caption waiting to attach forward)
    pending_title: str | None = None
    pending_title_label: str | None = None
    pending_title_page: int = current_page

    def flush_current() -> None:
        nonlocal current_texts, current_labels, current_tokens, chunk_idx
        nonlocal pending_title, pending_title_label, pending_title_page, current_page

        if not current_texts and pending_title is None:
            return

        texts_to_flush: list[str] = []
        labels_to_flush: list[str] = []
        # Orphan title (never got attached to content) — emit on its own page
        page_to_use = pending_title_page if (pending_title and not current_texts) else current_page

        if pending_title is not None:
            texts_to_flush.append(pending_title)
            labels_to_flush.append(pending_title_label or "paragraph_title")
            pending_title = None
            pending_title_label = None

        texts_to_flush.extend(current_texts)
        labels_to_flush.extend(current_labels)

        if not texts_to_flush:
            return

        chunk = Chunk(
            text="\n\n".join(texts_to_flush),
            chunk_id=f"{source_file}_{page_to_use}_{chunk_idx}",
            page=page_to_use,
            element_types=labels_to_flush,
            bbox=None,
            source_file=source_file,
            is_atomic=False,
            modality=_infer_modality(labels_to_flush),
        )
        chunks.append(chunk)
        chunk_idx += 1
        current_texts = []
        current_labels = []
        current_tokens = 0

    for page_num, element in all_pairs:
        label = element.label
        text = element.text.strip()

        if label in ATOMIC_LABELS:
            # Figure-title linkage: a pending figure_title is the caption of THIS
            # figure/image element — intercept it before flush_current() would emit
            # it as a standalone orphan chunk.
            figure_caption: str | None = None
            if pending_title is not None and pending_title_label == "figure_title":
                figure_caption = pending_title
                pending_title = None
                pending_title_label = None

            flush_current()

            # Build atomic chunk, prepending the figure caption when present
            if figure_caption:
                atomic_text = f"{figure_caption}\n\n{text}" if text else figure_caption
                atomic_labels = ["figure_title", label]
            else:
                atomic_text = text
                atomic_labels = [label]

            atomic_chunk = Chunk(
                text=atomic_text,
                chunk_id=f"{source_file}_{page_num}_{chunk_idx}",
                page=page_num,
                element_types=atomic_labels,
                bbox=element.bbox,
                source_file=source_file,
                is_atomic=True,
                modality=_infer_modality(atomic_labels),
            )
            chunks.append(atomic_chunk)
            chunk_idx += 1
            continue

        if not text:
            continue

        if label in TITLE_LABELS:
            # Flush accumulated content before starting a new heading context,
            # but NOT if this is a figure_title following accumulated text —
            # that case is handled above at the next atomic element.
            if current_texts:
                flush_current()
            elif pending_title is not None:
                # Two consecutive titles with no content between them → flush orphan
                flush_current()
            pending_title = text
            pending_title_label = label
            pending_title_page = page_num
            continue

        # Regular content element
        token_estimate = _estimate_tokens(text)
        pending_tokens = _estimate_tokens(pending_title) if pending_title else 0

        if token_estimate > max_chunk_tokens:
            flush_current()
            sub_chunks = _split_text_into_sub_chunks(text, max_chunk_tokens)
            for sub_text in sub_chunks:
                chunk = Chunk(
                    text=sub_text,
                    chunk_id=f"{source_file}_{page_num}_{chunk_idx}",
                    page=page_num,
                    element_types=[label],
                    bbox=None,
                    source_file=source_file,
                    is_atomic=False,
                    modality=_infer_modality([label]),
                )
                chunks.append(chunk)
                chunk_idx += 1
            continue

        if current_texts and (current_tokens + token_estimate + pending_tokens > max_chunk_tokens):
            flush_current()

        # Absorb pending title into accumulator as the heading for this content
        if pending_title is not None:
            if not current_texts:
                current_page = pending_title_page  # chunk's page = where heading started
            current_texts.append(pending_title)
            current_labels.append(pending_title_label or "paragraph_title")
            current_tokens += _estimate_tokens(pending_title)
            pending_title = None
            pending_title_label = None

        if not current_texts:
            current_page = page_num  # first element sets the chunk's page

        current_texts.append(text)
        current_labels.append(label)
        current_tokens += token_estimate

        if current_tokens >= max_chunk_tokens:
            flush_current()

    flush_current()
    return chunks


def structure_aware_chunking(
    elements: list[ElementLike],
    source_file: str,
    page: int,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Chunk a single page's elements respecting structure boundaries.

    For multi-page documents prefer ``document_aware_chunking`` which preserves
    section context across page boundaries and links figure captions correctly.

    Rules:
    - Atomic elements (table, formula, algorithm, image, figure) → always their own chunk.
    - ``figure_title`` is linked to the following image/figure atomic chunk.
    - Section/document/paragraph titles attach forward to the next content element.
    - Text elements accumulate until max_chunk_tokens is reached.

    Args:
        elements: List of parsed elements from a single page.
        source_file: Source document filename for chunk_id generation.
        page: Page number for chunk_id generation.
        max_chunk_tokens: Maximum tokens per chunk (default 512).

    Returns:
        List of Chunk objects ready for vector store ingestion.
    """
    # Delegate to document_aware_chunking with a single page
    return document_aware_chunking([(page, elements)], source_file, max_chunk_tokens)
