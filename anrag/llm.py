from __future__ import annotations

import re
from typing import Iterable

import ollama

from anrag.models import Chunk


def _detect_lang(text: str) -> str:
    """Return 'vi' if text looks Vietnamese, else 'en'."""
    vi_chars = re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text.lower())
    if len(vi_chars) >= 1:
        return 'vi'
    return 'en'


_LANG_INSTRUCTION = {
    'vi': 'BẮT BUỘC trả lời hoàn toàn bằng tiếng Việt.',
    'en': 'Answer in English.',
}


class OllamaLLM:
    def __init__(self, model: str = "qwen3:8b", host: str | None = None, vision_model: str | None = None):
        self.model = model
        self.vision_model = vision_model
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt, stream=False)
        return str(response.get("response", "")).strip()

    def generate_stream(self, prompt: str) -> Iterable[str]:
        for chunk in self.client.generate(model=self.model, prompt=prompt, stream=True):
            if "response" in chunk:
                yield chunk["response"]

    def rewrite_query(self, query: str) -> str:
        lang = _detect_lang(query)
        lang_instr = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION['en'])
        prompt = (
            "Rewrite the user query for document retrieval. "
            "Return only the rewritten query, nothing else.\n\n"
            f"Query: {query}\n\n"
            f"{lang_instr} Keep the same language as the query."
        )
        if 'qwen3' in self.model.lower():
            prompt += " /no_think"
        rewritten = self.generate(prompt)
        return rewritten or query

    def answer(self, query: str, contexts: Iterable[Chunk]) -> str:
        context_text = "\n\n".join(
            f"[{index + 1}] page {chunk.page_start}-{chunk.page_end} "
            f"path={' > '.join(chunk.hierarchy_path)}\n{chunk.text}"
            for index, chunk in enumerate(contexts)
        )
        lang = _detect_lang(query)
        lang_instr = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION['en'])
        prompt = (
            "You are an AnchorRAG assistant. Answer using only the provided context. "
            "If the context is insufficient, say that it is insufficient.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{query}\n\n"
            f"CRITICAL INSTRUCTION: {lang_instr}\n"
            "Answer:"
        )
        return self.generate(prompt)

    def answer_stream(self, query: str, contexts: Iterable[Chunk]) -> Iterable[str]:
        context_text = "\n\n".join(
            f"[{index + 1}] page {chunk.page_start}-{chunk.page_end} "
            f"path={' > '.join(chunk.hierarchy_path)}\n{chunk.text}"
            for index, chunk in enumerate(contexts)
        )
        lang = _detect_lang(query)
        lang_instr = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION['en'])
        prompt = (
            "You are an AnchorRAG assistant. Answer using only the provided context. "
            "If the context is insufficient, say that it is insufficient.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{query}\n\n"
            f"CRITICAL INSTRUCTION: {lang_instr}\n"
            "Answer:"
        )
        return self.generate_stream(prompt)

    def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        if not self.vision_model:
            return ""
        image_prompt = prompt or (
            "Describe this PDF visual element for retrieval. Focus on chart/table structure, "
            "labels, numbers, visual relationships, and any visible text. Return concise text."
        )
        with open(image_path, "rb") as handle:
            image_bytes = handle.read()
        response = self.client.generate(
            model=self.vision_model,
            prompt=image_prompt,
            images=[image_bytes],
            stream=False,
        )
        return str(response.get("response", "")).strip()
