from __future__ import annotations

from pathlib import Path
from typing import Any

import hashlib

from anrag.config import Settings, benchmark_eval_settings, get_settings
from anrag.documents import document_id_for_file
from anrag.llm import OllamaLLM
from anrag.models import ParsedBlock
from anrag.parsing import parse_pdf
from anrag.text_parsing import parse_plain_text


class BenchmarkParser:
    """Parse benchmark corpora into ``ParsedBlock`` lists for the anRAG pipeline.

    Flow: BenchmarkParser → ParsedBlock → ingest_blocks() → anRAG retrieval.
    """

    def parse_benchmark(
        self,
        path: str | Path,
        *,
        settings: Settings | None = None,
        llm: OllamaLLM | None = None,
        eval_mode: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[ParsedBlock]]:
        """Parse a benchmark corpus. PDF files are fully supported today.

        Accepts a single ``.pdf`` file or a directory tree of PDFs.
        Returns ``{doc_id: blocks}`` ready for ``ingest_blocks()``.

        When ``eval_mode=True``, OCR/VLM/LLM are disabled for retrieval-only evaluation.
        """
        path = Path(path)
        settings = benchmark_eval_settings(settings) if eval_mode else (settings or get_settings())
        pdf_kwargs = self._pdf_kwargs(settings, None if eval_mode else llm, kwargs)

        if path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return {document_id_for_file(path): self._parse_pdf(path, **pdf_kwargs)}
            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8-sig")
                doc_id = document_id_for_file(path)
                return {doc_id: self.parse_text(text, doc_id=doc_id)}
            if suffix in {".json", ".jsonl"}:
                from anrag.benchmark_gt import load_official_benchmark

                documents, _ = load_official_benchmark(path)
                return documents

        if path.is_dir():
            from anrag.benchmark_gt import detect_benchmark_format, load_official_benchmark

            if detect_benchmark_format(path) == "beir":
                documents, _ = load_official_benchmark(path, fmt="beir")
                return documents

            documents: dict[str, list[ParsedBlock]] = {}
            for pdf_path in sorted(path.rglob("*.pdf")):
                documents[document_id_for_file(pdf_path)] = self._parse_pdf(pdf_path, **pdf_kwargs)
            for text_path in sorted(path.rglob("*.txt")):
                text = text_path.read_text(encoding="utf-8-sig")
                documents[document_id_for_file(text_path)] = self.parse_text(
                    text,
                    doc_id=document_id_for_file(text_path),
                )
            if not documents:
                raise FileNotFoundError(f"No PDF/text files found under benchmark path: {path}")
            return documents

        raise FileNotFoundError(f"Benchmark path does not exist: {path}")

    def parse_text(self, text: str, *, doc_id: str = "doc_text") -> list[ParsedBlock]:
        """Parse plain text into heading/paragraph blocks with hierarchy."""
        if not text.strip():
            return []
        return parse_plain_text(text, doc_id=doc_id or _document_id_for_text(text))

    def parse_jsonl(self, path: str | Path) -> list[ParsedBlock]:
        """Parse a JSONL benchmark file into layout blocks. Reserved for future use."""
        raise NotImplementedError(
            "parse_jsonl() is reserved for a future JSONL benchmark loader. "
            f"path={Path(path)!r}"
        )

    def _parse_pdf(self, path: Path, **kwargs: Any) -> list[ParsedBlock]:
        return parse_pdf(path, **kwargs)

    @staticmethod
    def _pdf_kwargs(settings: Settings, llm: OllamaLLM | None, overrides: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "enable_ocr": settings.enable_ocr,
            "ocr_lang": settings.ocr_lang,
            "ocr_min_confidence": settings.ocr_min_confidence,
            "ocr_render_dpi": settings.ocr_render_dpi,
            "ocr_text_detection_model_dir": settings.ocr_text_detection_model_dir,
            "ocr_text_recognition_model_dir": settings.ocr_text_recognition_model_dir,
            "visual_dir": settings.visual_dir,
            "enable_vision_caption": settings.enable_vision_caption,
            "llm": llm,
        }
        defaults.update(overrides)
        return defaults


def _document_id_for_text(text: str) -> str:
    digest = hashlib.sha1(f"text|{len(text)}|{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"
