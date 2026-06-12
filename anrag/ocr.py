from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OCRText:
    text: str
    confidence: float | None = None


class PaddleOCRProcessor:
    def __init__(
        self,
        lang: str | None = "en",
        min_confidence: float = 0.35,
        text_detection_model_dir: str | None = None,
        text_recognition_model_dir: str | None = None,
    ):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        kwargs: dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        }
        if text_detection_model_dir:
            kwargs["text_detection_model_dir"] = text_detection_model_dir
        if text_recognition_model_dir:
            kwargs["text_recognition_model_dir"] = text_recognition_model_dir
        self.engine = PaddleOCR(**kwargs)
        self.min_confidence = min_confidence

    def extract_text(self, image_path: str | Path) -> OCRText:
        try:
            if hasattr(self.engine, "predict"):
                result = self.engine.predict(str(image_path))
            else:
                result = self.engine.ocr(str(image_path), cls=True)
        except Exception:
            return OCRText(text="", confidence=None)
        return _parse_paddle_result(result, self.min_confidence)


def _parse_paddle_result(result: Any, min_confidence: float) -> OCRText:
    texts: list[str] = []
    scores: list[float] = []

    def add_text(text: Any, score: Any = None) -> None:
        if text is None:
            return
        clean = str(text).strip()
        if not clean:
            return
        confidence = _to_float(score)
        if confidence is not None and confidence < min_confidence:
            return
        texts.append(clean)
        if confidence is not None:
            scores.append(confidence)

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or item.get("texts") or item.get("text")
            rec_scores = item.get("rec_scores") or item.get("scores") or item.get("score")
            if isinstance(rec_texts, list):
                if isinstance(rec_scores, list):
                    for text, score in zip(rec_texts, rec_scores, strict=False):
                        add_text(text, score)
                else:
                    for text in rec_texts:
                        add_text(text, rec_scores)
            else:
                add_text(rec_texts, rec_scores)
            return

        if hasattr(item, "json"):
            try:
                visit(item.json)
                return
            except Exception:
                pass

        if isinstance(item, (list, tuple)):
            if len(item) == 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                add_text(item[1][0], item[1][1])
                return
            for value in item:
                visit(value)

    visit(result)
    confidence = sum(scores) / len(scores) if scores else None
    return OCRText(text="\n".join(texts), confidence=confidence)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
