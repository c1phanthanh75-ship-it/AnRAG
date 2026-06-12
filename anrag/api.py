from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json

from anrag.ablation import run_ablations
from anrag.config import benchmark_eval_settings, get_settings
from anrag.embedding import EmbeddingBackend
from anrag.llm import OllamaLLM
from anrag.pipeline import ingest_pdf
from anrag.rerank import CrossEncoderReranker
from anrag.retrieval import AnchorRetriever
from anrag.store import SQLiteTreeStore


settings = get_settings()
store = SQLiteTreeStore(settings.sqlite_path)
embedder = EmbeddingBackend(settings.embedding_mode, settings.embedding_model, settings.embedding_dim)
llm = OllamaLLM(settings.ollama_model, settings.ollama_host, settings.vision_model)
reranker = None
if settings.enable_cross_encoder:
    try:
        reranker = CrossEncoderReranker(
            settings.cross_encoder_model,
            lambda_mmr=settings.reranker_lambda_mmr,
            mmr_top_n=settings.reranker_mmr_top_n,
        )
    except Exception:
        reranker = None
retriever = AnchorRetriever(
    store=store,
    index_dir=str(settings.index_dir),
    embedder=embedder,
    llm=llm,
    reranker=reranker,
    confidence_threshold=settings.anchor_confidence_threshold,
    dense_weight=settings.hybrid_dense_weight,
    sparse_weight=settings.hybrid_sparse_weight,
    rrf_k=settings.hybrid_rrf_k,
)

app = FastAPI(title="AnchorRAG Prototype")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static")


class QueryRequest(BaseModel):
    query: str
    mode: str = "anchor"
    budget_tokens: int | None = None
    generate_answer: bool = True
    doc_ids: list[str] | None = None
    rewrite_query: bool = True


class BenchmarkRequest(BaseModel):
    benchmark_path: str
    qa_path: str | None = None
    top_k: int = 8
    budget_tokens: int | None = None
    benchmark_format: str | None = None


@app.on_event("startup")
def startup() -> None:
    if not embedder.is_semantic:
        import warnings
        warnings.warn(
            "AnchorRAG is running with HashingVectorizer embeddings. "
            "Install sentence-transformers for production-quality retrieval.",
            RuntimeWarning,
            stacklevel=1,
        )
    if store.all_chunks():
        retriever.rebuild_indexes()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/documents")
def documents() -> dict:
    return {
        "documents": store.list_documents(),
        "embedding_quality": embedder.quality_tier,
    }


@app.get("/api/visual/{doc_name}/{file_name}", response_model=None)
def visual(doc_name: str, file_name: str):
    path = (settings.visual_dir / doc_name / file_name).resolve()
    root = settings.visual_dir.resolve()
    if root not in path.parents or not path.exists():
        return Response(status_code=404)
    return FileResponse(path)


@app.post("/api/ingest")
def ingest(file: UploadFile = File(...)) -> dict:
    destination = settings.upload_dir / file.filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    doc_id = ingest_pdf(destination, settings, store, llm=llm)
    retriever.rebuild_indexes()
    return {"doc_id": doc_id, "documents": store.list_documents()}


def _run_query(request: QueryRequest, *, stream_answer: bool = False):
    budget = request.budget_tokens or settings.context_budget_tokens
    common = {
        "budget_tokens": budget,
        "generate_answer": request.generate_answer,
        "stream_answer": stream_answer,
        "doc_ids": request.doc_ids,
        "rewrite_query": request.rewrite_query,
    }
    if request.mode in {"baseline", "plain_rag"}:
        return retriever.plain_rag(request.query, **common)
    return retriever.retrieve(request.query, **common)


@app.post("/api/benchmark")
def benchmark_eval(request: BenchmarkRequest) -> dict:
    """Retrieval-only ablation benchmark without LLM, VLM, or OCR."""
    eval_settings = benchmark_eval_settings(settings)
    report = run_ablations(
        request.benchmark_path,
        qa_path=request.qa_path,
        settings=eval_settings,
        llm=None,
        budget_tokens=request.budget_tokens,
        top_k=request.top_k,
        eval_mode=True,
        benchmark_format=request.benchmark_format,
    )
    payload = report.to_dict()
    payload["eval_mode"] = {
        "llm": False,
        "vlm": False,
        "ocr": False,
        "rewrite_query": False,
    }
    return payload


@app.post("/api/query")
def query(request: QueryRequest) -> dict:
    result = _run_query(request)
    return {
        "answer": result.answer,
        "trace": result.trace,
        "anchors": [hit.__dict__ for hit in result.anchors],
        "contexts": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "anchor_type": chunk.anchor_type,
                "path": chunk.hierarchy_path,
                "tokens": chunk.token_count,
                "chunk_role": chunk.metadata.get("chunk_role"),
                "block_kind": chunk.metadata.get("block_kind"),
                "ocr_applied": chunk.metadata.get("ocr_applied"),
                "ocr_confidence": chunk.metadata.get("ocr_confidence"),
                "visual_url": _visual_url(chunk.metadata.get("visual_path")),
            }
            for chunk in result.contexts
        ],
    }


@app.post("/api/query_stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    result = _run_query(request, stream_answer=True)

    def event_generator():
        # First send the contexts and trace
        meta = {
            "trace": result.trace,
            "anchors": [hit.__dict__ for hit in result.anchors],
            "contexts": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "anchor_type": chunk.anchor_type,
                    "path": chunk.hierarchy_path,
                    "tokens": chunk.token_count,
                    "chunk_role": chunk.metadata.get("chunk_role"),
                    "block_kind": chunk.metadata.get("block_kind"),
                    "ocr_applied": chunk.metadata.get("ocr_applied"),
                    "ocr_confidence": chunk.metadata.get("ocr_confidence"),
                    "visual_url": _visual_url(chunk.metadata.get("visual_path")),
                }
                for chunk in result.contexts
            ],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        if request.generate_answer and isinstance(result.answer, type((_ for _ in []))): # Check if generator
            for chunk in result.answer:
                # Send text chunks
                yield f"event: chunk\ndata: {json.dumps(chunk)}\n\n"
        elif request.generate_answer and isinstance(result.answer, str):
            yield f"event: chunk\ndata: {json.dumps(result.answer)}\n\n"
            
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/reindex")
def reindex() -> dict:
    retriever.rebuild_indexes()
    return {"ok": True, "documents": store.list_documents()}


def _visual_url(visual_path: object) -> str | None:
    if not visual_path:
        return None
    path = Path(str(visual_path))
    try:
        resolved = path.resolve()
        root = settings.visual_dir.resolve()
        if root not in resolved.parents:
            return None
        rel = resolved.relative_to(root)
    except Exception:
        return None
    if len(rel.parts) != 2:
        return None
    return f"/api/visual/{quote(rel.parts[0])}/{quote(rel.parts[1])}"
