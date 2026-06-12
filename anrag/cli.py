from __future__ import annotations

import argparse
import json

from anrag.config import get_settings
from anrag.embedding import EmbeddingBackend
from anrag.llm import OllamaLLM
from anrag.ablation import run_ablations
from anrag.pipeline import ingest_pdf
from anrag.rerank import CrossEncoderReranker
from anrag.retrieval import AnchorRetriever
from anrag.store import SQLiteTreeStore


def build_retriever() -> AnchorRetriever:
    settings = get_settings()
    store = SQLiteTreeStore(settings.sqlite_path)
    embedder = EmbeddingBackend(settings.embedding_mode, settings.embedding_model, settings.embedding_dim)
    if not embedder.is_semantic:
        import sys
        print(
            "[WARNING] EmbeddingBackend is using HashingVectorizer — retrieval quality is reduced. "
            "Install sentence-transformers for semantic search.",
            file=sys.stderr,
        )
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
        store,
        str(settings.index_dir),
        embedder,
        llm,
        reranker,
        settings.anchor_confidence_threshold,
        dense_weight=settings.hybrid_dense_weight,
        sparse_weight=settings.hybrid_sparse_weight,
        rrf_k=settings.hybrid_rrf_k,
    )
    if store.all_chunks():
        retriever.rebuild_indexes()
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="anrag")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser("ingest")
    ingest_cmd.add_argument("pdf")

    query_cmd = sub.add_parser("query")
    query_cmd.add_argument("query")
    query_cmd.add_argument("--mode", choices=["anchor", "baseline", "plain_rag"], default="anchor")
    query_cmd.add_argument("--no-answer", action="store_true")
    query_cmd.add_argument("--budget", type=int, default=None)

    ablation_cmd = sub.add_parser("ablation")
    ablation_cmd.add_argument("benchmark", help="PDF file or directory of PDFs")
    ablation_cmd.add_argument("--qa", default=None, help="Optional QA file (anrag JSONL, HotpotQA, BeIR dir, KILT)")
    ablation_cmd.add_argument("--format", dest="benchmark_format", default=None, choices=["anrag", "hotpotqa", "beir", "kilt"])
    ablation_cmd.add_argument("--top-k", type=int, default=8)
    ablation_cmd.add_argument("--budget", type=int, default=None)

    args = parser.parse_args()
    settings = get_settings()
    store = SQLiteTreeStore(settings.sqlite_path)

    if args.command == "ablation":
        budget = args.budget or settings.context_budget_tokens
        report = run_ablations(
            args.benchmark,
            qa_path=args.qa,
            budget_tokens=budget,
            top_k=args.top_k,
            benchmark_format=args.benchmark_format,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "ingest":
        doc_id = ingest_pdf(args.pdf, settings, store, llm=OllamaLLM(settings.ollama_model, settings.ollama_host, settings.vision_model))
        retriever = build_retriever()
        retriever.rebuild_indexes()
        print(json.dumps({"doc_id": doc_id, "documents": store.list_documents()}, ensure_ascii=False, indent=2))
        return

    retriever = build_retriever()
    budget = args.budget or settings.context_budget_tokens
    if args.mode in {"baseline", "plain_rag"}:
        result = retriever.plain_rag(args.query, budget_tokens=budget, generate_answer=not args.no_answer)
    else:
        result = retriever.retrieve(args.query, budget_tokens=budget, generate_answer=not args.no_answer)
    print(
        json.dumps(
            {
                "answer": result.answer,
                "trace": result.trace,
                "anchors": [hit.__dict__ for hit in result.anchors],
                "contexts": [
                    {
                        "id": chunk.id,
                        "page": [chunk.page_start, chunk.page_end],
                        "anchor_type": chunk.anchor_type,
                        "chunk_role": chunk.metadata.get("chunk_role"),
                        "block_kind": chunk.metadata.get("block_kind"),
                        "ocr_applied": chunk.metadata.get("ocr_applied"),
                        "ocr_confidence": chunk.metadata.get("ocr_confidence"),
                        "visual_path": chunk.metadata.get("visual_path"),
                        "path": chunk.hierarchy_path,
                        "text": chunk.text[:500],
                    }
                    for chunk in result.contexts
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
