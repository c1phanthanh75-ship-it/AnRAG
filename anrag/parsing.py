from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import fitz

from anrag.llm import OllamaLLM
from anrag.models import ParsedBlock
from anrag.ocr import PaddleOCRProcessor
from anrag.text import normalize_text


_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVX]+)\.?\s+\S+", re.IGNORECASE)
_TABLE_HINT_RE = re.compile(r"^\s*(table|bảng)\s+\d+", re.IGNORECASE)
_FIGURE_HINT_RE = re.compile(r"^\s*(figure|fig\.|hình)\s+\d+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Author-zone filtering
# Academic papers (arXiv, IEEE, ACM, ...) have a header zone on page 1 that
# contains author names, affiliations, emails, and ORCID IDs.  These lines
# carry zero retrieval value and inflate chunk count.
#
# Strategy: state-machine over the sorted line list.
# - Zone opens  after the TITLE block (heading_level=1, page 1).
# - Zone closes when the parser sees an "Abstract" heading or a numbered
#   section heading (e.g. "1. Introduction").
# - Inside the zone, lines matching _AUTHOR_LINE_PATTERNS are discarded.
#   Lines that do NOT match any pattern are kept (could be subtitle, etc.)
# ---------------------------------------------------------------------------

_ABSTRACT_HEADING_RE = re.compile(
    r"^\s*(abstract|tóm\s*tắt|résumé|zusammenfassung|摘要)\s*$",
    re.IGNORECASE,
)
_SECTION_START_RE = re.compile(
    r"^\s*(\d+\.|\bI\b\.?)\s+\S",
    re.IGNORECASE,
)

# Each pattern matches lines that should be dropped when inside the author zone.
_AUTHOR_LINE_PATTERNS: list[re.Pattern[str]] = [
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # ORCID link or bare ORCID id
    re.compile(r"\borcid\.org\b|\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b"),
    # Unicode superscript at line start (affiliation marker: e.g. ¹University of ...)
    re.compile(r"^[¹²³⁴⁵⁶⁷⁸⁹⁰†‡*]+\s*\S"),
    # Institution keywords (EN + FR + DE + IT + ES)
    re.compile(r"\b(university|université|universität|università|universidad)\b", re.IGNORECASE),
    re.compile(r"\b(institute|institution|laboratory|laboratories)\b", re.IGNORECASE),
    re.compile(r"\b(department|dept\.?|faculty|college|school\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(center|centre)\s+(for|of)\b", re.IGNORECASE),
    # Vietnamese institution keywords
    re.compile(r"\b(trường|viện|khoa|bộ\s*môn|đại\s*học|học\s*viện)\b", re.IGNORECASE),
    # Author-credit annotations
    re.compile(r"\b(equal\s+contribution|corresponding\s+author|work\s+done\s+at)\b", re.IGNORECASE),
    # Comma-separated capitalised name list: "A. Smith, B. Jones, C. Lee"
    re.compile(
        r"^([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝ][^\n,]{1,30})"
        r"(,\s*[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝ][^\n,]{1,30}){2,}$"
    ),
    # Names with trailing Unicode superscript footnote markers: "Author Name¹²"
    re.compile(r"[\w\s.,\-]+[¹²³⁴⁵⁶⁷⁸⁹]+\s*$"),
]


def _is_author_zone_line(text: str) -> bool:
    """Return True if *text* looks like an author/affiliation line."""
    return any(pattern.search(text) for pattern in _AUTHOR_LINE_PATTERNS)


def _ends_author_zone(text: str) -> bool:
    """Return True if *text* is a heading that closes the author zone."""
    return bool(_ABSTRACT_HEADING_RE.match(text) or _SECTION_START_RE.match(text))


@dataclass
class LayoutLine:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool
    block_index: int
    line_index: int
    column: int
    in_table: bool = False


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def parse_pdf(
    path: str | Path,
    *,
    enable_ocr: bool = True,
    ocr_lang: str | None = "en",
    ocr_min_confidence: float = 0.35,
    ocr_render_dpi: int = 180,
    ocr_text_detection_model_dir: str | None = None,
    ocr_text_recognition_model_dir: str | None = None,
    visual_dir: str | Path | None = None,
    enable_vision_caption: bool = False,
    llm: OllamaLLM | None = None,
) -> list[ParsedBlock]:
    """Parse PDF into layout-preserving blocks before chunking.

    The parser keeps headings, paragraphs, table regions, figures, page/column
    order, and scanned-page blocks. OCR is applied to scanned pages and visual
    crops when enabled.
    """

    path = Path(path)
    doc = fitz.open(str(path))
    visual_root = Path(visual_dir) if visual_dir else path.parent / ".anrag_visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    ocr = _build_ocr(
        enable=enable_ocr,
        lang=ocr_lang,
        min_confidence=ocr_min_confidence,
        text_detection_model_dir=ocr_text_detection_model_dir,
        text_recognition_model_dir=ocr_text_recognition_model_dir,
    )
    raw_blocks: list[ParsedBlock] = []
    lines: list[LayoutLine] = []
    line_sizes: list[float] = []

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        page_blocks: list[ParsedBlock] = []
        page_tables = _extract_tables(page, page_no)
        page_blocks.extend(page_tables)

        data = page.get_text("dict")
        table_bboxes = [block.bbox for block in page_tables if block.bbox]
        text_line_count = 0
        for block_index, block in enumerate(data.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_index, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                text = normalize_text(" ".join(span.get("text", "") for span in spans))
                if not text:
                    continue
                bbox = tuple(float(value) for value in line.get("bbox", block.get("bbox", (0, 0, 0, 0))))
                size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                bold = any("bold" in span.get("font", "").lower() for span in spans)
                center_x = (bbox[0] + bbox[2]) / 2
                column = 0 if center_x < page.rect.width / 2 else 1
                in_table = any(_intersects(bbox, table_bbox) for table_bbox in table_bboxes)
                lines.append(
                    LayoutLine(
                        text=text,
                        page=page_no,
                        bbox=bbox,
                        font_size=size,
                        bold=bold,
                        block_index=block_index,
                        line_index=line_index,
                        column=column,
                        in_table=in_table,
                    )
                )
                if not in_table:
                    line_sizes.append(size)
                    text_line_count += 1

        page_figures = _extract_figures(
            page=page,
            page_no=page_no,
            source_path=path,
            visual_dir=visual_root,
            ocr=ocr,
            ocr_render_dpi=ocr_render_dpi,
            enable_vision_caption=enable_vision_caption,
            llm=llm,
            skip_full_page_figures=text_line_count == 0,
        )

        if text_line_count == 0:
            scan_image = _render_region(
                page=page,
                page_no=page_no,
                source_path=path,
                visual_dir=visual_root,
                bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                prefix="scan",
                dpi=ocr_render_dpi,
            )
            ocr_result = ocr.extract_text(scan_image) if ocr else None
            scan_text = ocr_result.text if ocr_result and ocr_result.text else f"[Scanned page {page_no}: OCR text not available]"
            page_blocks.append(
                ParsedBlock(
                    id=_stable_id("scan", path, page_no),
                    text=scan_text,
                    page=page_no,
                    kind="scanned_page",
                    page_end=page_no,
                    bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                    metadata={
                        "requires_ocr": not bool(ocr_result and ocr_result.text),
                        "ocr_applied": bool(ocr),
                        "ocr_confidence": ocr_result.confidence if ocr_result else None,
                        "visual_path": str(scan_image),
                        "layout_role": "scanned_page",
                    },
                )
            )
        else:
            page_blocks.extend(page_figures)

        raw_blocks.extend(page_blocks)

    if not lines and not raw_blocks:
        doc.close()
        return []

    body_size = median(line_sizes) if line_sizes else 10.0
    heading_sizes = _heading_sizes(line_sizes, body_size)
    text_blocks = _lines_to_blocks(lines, heading_sizes, body_size)
    merged = _attach_hierarchy(sorted(raw_blocks + text_blocks, key=_layout_sort_key))
    doc.close()
    return merged


def _extract_tables(page: fitz.Page, page_no: int) -> list[ParsedBlock]:
    if not hasattr(page, "find_tables"):
        return []
    blocks: list[ParsedBlock] = []
    try:
        tables = page.find_tables()
    except Exception:
        return []

    for table_index, table in enumerate(getattr(tables, "tables", [])):
        rows = table.extract()
        row_texts = [" | ".join(normalize_text(str(cell or "")) for cell in row) for row in rows]
        text = "\n".join(row for row in row_texts if row.strip(" |"))
        if not text:
            continue
        bbox = tuple(float(value) for value in table.bbox)
        blocks.append(
            ParsedBlock(
                id=_stable_id("tbl", page_no, table_index, text[:80]),
                text=text,
                page=page_no,
                kind="table",
                page_end=page_no,
                bbox=bbox,
                metadata={"table_index": table_index, "layout_role": "table"},
            )
        )
    return blocks


def _extract_figures(
    page: fitz.Page,
    page_no: int,
    source_path: Path,
    visual_dir: Path,
    ocr: PaddleOCRProcessor | None,
    ocr_render_dpi: int,
    enable_vision_caption: bool,
    llm: OllamaLLM | None,
    skip_full_page_figures: bool = False,
) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    for image_index, image in enumerate(page.get_images(full=True)):
        xref = image[0]
        rects = page.get_image_rects(xref)
        for rect_index, rect in enumerate(rects):
            bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            if skip_full_page_figures and _covers_most_of_page(bbox, page.rect):
                continue
            image_path = _render_region(
                page=page,
                page_no=page_no,
                source_path=source_path,
                visual_dir=visual_dir,
                bbox=bbox,
                prefix="figure",
                dpi=ocr_render_dpi,
                index=image_index,
            )
            ocr_result = ocr.extract_text(image_path) if ocr else None
            vision_summary = ""
            if enable_vision_caption and llm:
                try:
                    vision_summary = llm.describe_image(str(image_path))
                except Exception:
                    vision_summary = ""
            text = _visual_text(page_no, ocr_result.text if ocr_result else "", vision_summary)
            blocks.append(
                ParsedBlock(
                    id=_stable_id("fig", page_no, image_index, rect_index, bbox),
                    text=text,
                    page=page_no,
                    kind="figure",
                    page_end=page_no,
                    bbox=bbox,
                    metadata={
                        "image_index": image_index,
                        "xref": xref,
                        "layout_role": "figure",
                        "visual_path": str(image_path),
                        "ocr_applied": bool(ocr),
                        "ocr_confidence": ocr_result.confidence if ocr_result else None,
                        "has_vision_caption": bool(vision_summary),
                    },
                )
            )
    return blocks


def _build_ocr(
    enable: bool,
    lang: str | None,
    min_confidence: float,
    text_detection_model_dir: str | None,
    text_recognition_model_dir: str | None,
) -> PaddleOCRProcessor | None:
    if not enable:
        return None
    try:
        return PaddleOCRProcessor(
            lang=lang,
            min_confidence=min_confidence,
            text_detection_model_dir=text_detection_model_dir,
            text_recognition_model_dir=text_recognition_model_dir,
        )
    except Exception:
        return None


def _render_region(
    page: fitz.Page,
    page_no: int,
    source_path: Path,
    visual_dir: Path,
    bbox: tuple[float, float, float, float],
    prefix: str,
    dpi: int,
    index: int = 0,
) -> Path:
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    clip = fitz.Rect(*bbox)
    pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    doc_dir = visual_dir / source_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    image_path = doc_dir / f"{prefix}_p{page_no}_{index}.png"
    pixmap.save(str(image_path))
    return image_path


def _visual_text(page_no: int, ocr_text: str, vision_summary: str) -> str:
    parts = [f"[Figure on page {page_no}]"]
    if ocr_text:
        parts.append(f"OCR text:\n{ocr_text}")
    if vision_summary:
        parts.append(f"Visual summary:\n{vision_summary}")
    return "\n\n".join(parts)


def _covers_most_of_page(bbox: tuple[float, float, float, float], page_rect: fitz.Rect) -> bool:
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    page_area = float(page_rect.width * page_rect.height)
    return page_area > 0 and area / page_area > 0.82


def _lines_to_blocks(lines: list[LayoutLine], heading_sizes: list[float], body_size: float) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    paragraph_buffer: list[LayoutLine] = []

    # Author-zone state machine
    # in_author_zone=True  → between TITLE heading and Abstract/section-1 heading on page 1
    in_author_zone: bool = False
    title_seen: bool = False

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        text = normalize_text(" ".join(line.text for line in paragraph_buffer))
        bbox = _union_bbox([line.bbox for line in paragraph_buffer])
        page_start = min(line.page for line in paragraph_buffer)
        page_end = max(line.page for line in paragraph_buffer)
        blocks.append(
            ParsedBlock(
                id=_stable_id("para", len(blocks), page_start, text[:100]),
                text=text,
                page=page_start,
                kind="paragraph",
                page_end=page_end,
                bbox=bbox,
                metadata={
                    "columns": sorted({line.column for line in paragraph_buffer}),
                    "block_indices": sorted({line.block_index for line in paragraph_buffer}),
                    "layout_role": "paragraph",
                },
            )
        )
        paragraph_buffer = []

    for line in sorted((item for item in lines if not item.in_table), key=_line_sort_key):
        heading_level = _heading_level(line, heading_sizes, body_size)
        text = line.text.strip()

        if heading_level is not None:
            # Check whether this heading closes the author zone
            if in_author_zone and _ends_author_zone(text):
                in_author_zone = False

            flush_paragraph()
            blocks.append(
                ParsedBlock(
                    id=_stable_id("hdr", len(blocks), line.page, text),
                    text=text,
                    page=line.page,
                    kind="heading",
                    level=heading_level,
                    page_end=line.page,
                    bbox=line.bbox,
                    font_size=line.font_size,
                    metadata={"column": line.column, "layout_role": "heading"},
                )
            )
            # Open author zone after the title heading on page 1
            if not title_seen and heading_level == 1 and line.page == 1:
                title_seen = True
                in_author_zone = True
            continue

        # Non-heading line: close author zone if it looks like a real section start
        if in_author_zone and _ends_author_zone(text):
            in_author_zone = False

        # Drop lines inside the author zone that match author/affiliation patterns
        if in_author_zone and _is_author_zone_line(text):
            continue

        if paragraph_buffer and _starts_new_paragraph(paragraph_buffer[-1], line):
            flush_paragraph()
        paragraph_buffer.append(line)

    flush_paragraph()
    return blocks


def _attach_hierarchy(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    heading_stack: list[ParsedBlock] = []
    result: list[ParsedBlock] = []
    for block in blocks:
        if block.kind == "heading":
            while heading_stack and (heading_stack[-1].level or 0) >= (block.level or 1):
                heading_stack.pop()
            parent = heading_stack[-1] if heading_stack else None
            block.parent_id = parent.id if parent else None
            block.hierarchy_path = [item.text for item in heading_stack] + [block.text]
            heading_stack.append(block)
        else:
            parent = heading_stack[-1] if heading_stack else None
            block.parent_id = parent.id if parent else None
            block.hierarchy_path = [item.text for item in heading_stack]
        result.append(block)
    return result


def _heading_sizes(sizes: list[float], body_size: float) -> list[float]:
    rounded = sorted({round(size, 1) for size in sizes}, reverse=True)
    return [size for size in rounded if size >= body_size + 1.0][:5]


def _heading_level(line: LayoutLine, heading_sizes: list[float], body_size: float) -> int | None:
    text = line.text.strip()
    
    # Filter out pure math formulas / numbers by requiring at least one word of 2+ letters
    if not re.search(r'[a-zA-Z\u00C0-\u1EF9]{2,}', text):
        return None
        
    # Filter out 1-2 token chunks unless they contain a solid word (like "Abstract" or "Method")
    if len(text.split()) < 3 and not re.search(r'[a-zA-Z\u00C0-\u1EF9]{4,}', text):
        return None

    rounded = round(line.font_size, 1)
    if _TABLE_HINT_RE.match(text) or _FIGURE_HINT_RE.match(text):
        return None
    if rounded in heading_sizes and len(text) <= 180:
        return heading_sizes.index(rounded) + 1
    if line.bold and len(text) <= 140 and _NUMBERED_HEADING_RE.match(text):
        return min(4, text.split()[0].count(".") + 1)
    if _NUMBERED_HEADING_RE.match(text) and len(text) <= 120 and line.font_size >= body_size:
        return min(5, text.split()[0].count(".") + 1)
    return None


def _starts_new_paragraph(prev: LayoutLine, current: LayoutLine) -> bool:
    if current.page != prev.page:
        return True
    if current.column != prev.column:
        return True
    vertical_gap = current.bbox[1] - prev.bbox[3]
    if vertical_gap > max(prev.font_size, current.font_size) * 1.3:
        return True
    if prev.text.endswith((".", "?", "!", "。", "！", "？")) and vertical_gap > max(prev.font_size, current.font_size) * 0.35:
        return True
    return False


def _line_sort_key(line: LayoutLine) -> tuple[int, int, float, float]:
    return (line.page, line.column, line.bbox[1], line.bbox[0])


def _layout_sort_key(block: ParsedBlock) -> tuple[int, int, float, float]:
    bbox = block.bbox or (0.0, 0.0, 0.0, 0.0)
    column = int(block.metadata.get("column", 0))
    if "columns" in block.metadata:
        columns = block.metadata.get("columns") or [0]
        column = min(columns)
    return (block.page, column, bbox[1], bbox[0])


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _union_bbox(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
