from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from anrag.benchmark import BenchmarkParser
from anrag.benchmark_gt import (
    align_questions_with_documents,
    build_relevance_map,
    detect_benchmark_format,
    load_anrag_questions,
    load_official_benchmark,
    resolve_gold_official,
    resolve_run_format,
    split_doc_ids,
)
from anrag.benchmark_types import BenchmarkFormat, BenchmarkQuestion
from anrag.chunking import ChunkingMode
from anrag.config import Settings, benchmark_eval_settings, get_settings
from anrag.embedding import EmbeddingBackend
from anrag.llm import OllamaLLM
from anrag.metrics import RetrievalMetrics, average_metrics, evaluate_ranking
from anrag.models import ParsedBlock
from anrag.pipeline import RetrievalMode, ingest_blocks
from anrag.retrieval import AnchorRetriever
from anrag.store import SQLiteTreeStore


class AblationName(str, Enum):
    BASELINE = "baseline"
    HIERARCHY = "hierarchy"
    FULL = "full"


@dataclass(frozen=True)
class AblationSpec:
    name: AblationName
    chunking_mode: ChunkingMode
    retrieval_mode: RetrievalMode
    mark_anchors: bool
    label: str


@dataclass
class AblationTiming:
    ingest_seconds: float
    index_seconds: float
    retrieval_total_seconds: float
    retrieval_avg_seconds: float
    retrieval_p95_seconds: float
    total_seconds: float


@dataclass
class GoldSanityReport:
    zero_gold_count: int
    avg_gold_size: float
    question_count: int


@dataclass
class AblationScore:
    name: str
    key: str
    metrics: RetrievalMetrics
    query_count: int
    timing: AblationTiming
    gold_sanity: GoldSanityReport

    @property
    def recall_at_k(self) -> float:
        return self.metrics.recall_at_k


@dataclass
class ComponentContribution:
    hierarchy_pct: float
    anchor_pct: float
    expansion_pct: float
    baseline_score: float
    hierarchy_score: float
    anchor_only_score: float
    full_score: float
    total_gain: float
    hierarchy_gain: float
    anchor_gain: float
    expansion_gain: float


@dataclass
class TimeCostReport:
    parse_seconds: float
    total_seconds: float
    by_mode: dict[str, AblationTiming]
    hierarchy_overhead_pct: float
    anchor_overhead_pct: float
    expansion_overhead_pct: float
    full_anrag_avg_query_ms: float
    plain_rag_avg_query_ms: float


@dataclass
class AblationReport:
    scores: list[AblationScore]
    contributions: ComponentContribution
    time_cost: TimeCostReport
    questions: list[BenchmarkQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        full_score = next((score for score in self.scores if score.key == "full"), None)
        return {
            "summary": {
                "primary_metric": "recall_at_k",
                "hierarchy_contribution_pct": self.contributions.hierarchy_pct,
                "anchor_contribution_pct": self.contributions.anchor_pct,
                "expansion_contribution_pct": self.contributions.expansion_pct,
                "total_quality_gain": self.contributions.total_gain,
                "full_anrag_metrics": full_score.metrics.to_dict() if full_score else {},
                "gold_sanity": asdict(full_score.gold_sanity) if full_score else {},
            },
            "contributions": asdict(self.contributions),
            "time_cost": {
                "parse_seconds": self.time_cost.parse_seconds,
                "total_seconds": self.time_cost.total_seconds,
                "hierarchy_overhead_pct": self.time_cost.hierarchy_overhead_pct,
                "anchor_overhead_pct": self.time_cost.anchor_overhead_pct,
                "expansion_overhead_pct": self.time_cost.expansion_overhead_pct,
                "full_anrag_avg_query_ms": self.time_cost.full_anrag_avg_query_ms,
                "plain_rag_avg_query_ms": self.time_cost.plain_rag_avg_query_ms,
                "by_mode": {key: asdict(timing) for key, timing in self.time_cost.by_mode.items()},
            },
            "scores": [
                {
                    "name": score.name,
                    "key": score.key,
                    "query_count": score.query_count,
                    "timing": asdict(score.timing),
                    "metrics": score.metrics.to_dict(),
                    "gold_sanity": asdict(score.gold_sanity),
                }
                for score in self.scores
            ],
            "questions": [
                {
                    "question": item.question,
                    "doc_id": item.doc_id,
                    "gold_chunk_ids": sorted(item.gold_chunk_ids),
                    "gold_doc_ids": sorted(item.gold_doc_ids),
                }
                for item in self.questions
            ],
        }


ABLATION_SPECS: dict[AblationName, AblationSpec] = {
    AblationName.BASELINE: AblationSpec(
        name=AblationName.BASELINE,
        chunking_mode="fixed",
        retrieval_mode="baseline",
        mark_anchors=False,
        label="Plain RAG baseline: fixed chunk + hybrid retrieval",
    ),
    AblationName.HIERARCHY: AblationSpec(
        name=AblationName.HIERARCHY,
        chunking_mode="hierarchy",
        retrieval_mode="baseline",
        mark_anchors=False,
        label="Fixed chunk + parent-child hierarchy",
    ),
    AblationName.FULL: AblationSpec(
        name=AblationName.FULL,
        chunking_mode="semantic",
        retrieval_mode="full",
        mark_anchors=True,
        label="Full anRAG: hierarchy + anchor + expansion",
    ),
}

INTERNAL_SPECS: dict[str, AblationSpec] = {
    "anchor_only": AblationSpec(
        name=AblationName.FULL,
        chunking_mode="semantic",
        retrieval_mode="anchor_only",
        mark_anchors=True,
        label="Semantic hierarchy + anchor (no expansion)",
    ),
}


def load_questions(
    path: str | Path | None,
    documents: dict[str, list[ParsedBlock]],
    *,
    benchmark_format: BenchmarkFormat | None = None,
) -> list[BenchmarkQuestion]:
    if path is None:
        return _default_questions(documents)

    path = Path(path)
    fmt = benchmark_format or detect_benchmark_format(path)
    if fmt in {"hotpotqa", "beir", "kilt", "ragbench"}:
        _, questions = load_official_benchmark(path, fmt=fmt)
        return questions
    return load_anrag_questions(path)


def _default_questions(documents: dict[str, list[ParsedBlock]]) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    for doc_id, blocks in documents.items():
        for block in blocks:
            if block.kind != "paragraph" or len(block.text.split()) < 6:
                continue
            words = block.text.split()[:4]
            questions.append(BenchmarkQuestion(question=" ".join(words), doc_id=doc_id))
            if len(questions) >= 5:
                return questions
    return questions


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[index]


def compute_contributions(
    baseline_score: float,
    hierarchy_score: float,
    anchor_only_score: float,
    full_score: float,
) -> ComponentContribution:
    total_gain = full_score - baseline_score
    hierarchy_gain = max(0.0, hierarchy_score - baseline_score)
    anchor_gain = max(0.0, anchor_only_score - hierarchy_score)
    expansion_gain = max(0.0, full_score - anchor_only_score)

    if total_gain <= 0:
        return ComponentContribution(
            hierarchy_pct=0.0,
            anchor_pct=0.0,
            expansion_pct=0.0,
            baseline_score=baseline_score,
            hierarchy_score=hierarchy_score,
            anchor_only_score=anchor_only_score,
            full_score=full_score,
            total_gain=0.0,
            hierarchy_gain=hierarchy_gain,
            anchor_gain=anchor_gain,
            expansion_gain=expansion_gain,
        )

    return ComponentContribution(
        hierarchy_pct=round(100.0 * hierarchy_gain / total_gain, 2),
        anchor_pct=round(100.0 * anchor_gain / total_gain, 2),
        expansion_pct=round(100.0 * expansion_gain / total_gain, 2),
        baseline_score=baseline_score,
        hierarchy_score=hierarchy_score,
        anchor_only_score=anchor_only_score,
        full_score=full_score,
        total_gain=round(total_gain, 4),
        hierarchy_gain=round(hierarchy_gain, 4),
        anchor_gain=round(anchor_gain, 4),
        expansion_gain=round(expansion_gain, 4),
    )


def compute_time_overhead(
    baseline_seconds: float,
    hierarchy_seconds: float,
    anchor_only_seconds: float,
    full_seconds: float,
) -> tuple[float, float, float]:
    total_extra = full_seconds - baseline_seconds
    if total_extra <= 0:
        return 0.0, 0.0, 0.0

    hierarchy_extra = max(0.0, hierarchy_seconds - baseline_seconds)
    anchor_extra = max(0.0, anchor_only_seconds - hierarchy_seconds)
    expansion_extra = max(0.0, full_seconds - anchor_only_seconds)
    return (
        round(100.0 * hierarchy_extra / total_extra, 2),
        round(100.0 * anchor_extra / total_extra, 2),
        round(100.0 * expansion_extra / total_extra, 2),
    )


@dataclass
class AblationEvalResult:
    metrics: RetrievalMetrics
    timing: AblationTiming
    gold_sanity: GoldSanityReport

    @property
    def recall_at_k(self) -> float:
        return self.metrics.recall_at_k


def _retrieve(
    retriever: AnchorRetriever,
    spec: AblationSpec,
    question: BenchmarkQuestion,
    budget_tokens: int,
    top_k: int,
    *,
    rewrite_query: bool = False,
) -> tuple[list[str], float]:
    # FIX: resolve doc_ids per-question so retrieval is scoped correctly
    # (resolves merge conflict — fa13e49 version is correct)
    doc_ids = split_doc_ids(question.doc_id) or None
    common = {
        "budget_tokens": budget_tokens,
        "top_k": top_k,
        "generate_answer": False,
        "doc_ids": doc_ids,
        "rewrite_query": rewrite_query,
    }
    if spec.retrieval_mode == "baseline":
        result = retriever.plain_rag(question.question, **common)
    elif spec.retrieval_mode == "anchor_only":
        result = retriever.anchor_only(question.question, **common)
    else:
        result = retriever.retrieve(question.question, **common)
    latency = float(result.trace.get("latency_seconds", 0.0))
    return [chunk.id for chunk in result.contexts], latency


def _evaluate_spec(
    spec: AblationSpec,
    documents: dict[str, list[ParsedBlock]],
    questions: list[BenchmarkQuestion],
    settings: Settings,
    store: SQLiteTreeStore,
    retriever: AnchorRetriever,
    *,
    budget_tokens: int,
    top_k: int,
    resolve_gold_ids: Callable[[BenchmarkQuestion, SQLiteTreeStore], set[str]],
    rewrite_query: bool = False,
    retrieval_depth: int = 20,
) -> AblationEvalResult:
    run_start = time.perf_counter()

    store.clear_all()
    ingest_start = time.perf_counter()
    for doc_id, blocks in documents.items():
        ingest_blocks(
            doc_id,
            f"{doc_id}.benchmark",
            f"benchmark://{doc_id}",
            blocks,
            settings,
            store,
            chunking_mode=spec.chunking_mode,
            mark_anchor_tags=spec.mark_anchors,
        )
    ingest_seconds = time.perf_counter() - ingest_start

    index_start = time.perf_counter()
    retriever.rebuild_indexes()
    index_seconds = time.perf_counter() - index_start

    metric_rows: list[RetrievalMetrics] = []
    query_latencies: list[float] = []
    gold_sizes: list[int] = []
    fetch_k = max(top_k, retrieval_depth, 20)
    for question in questions:
        gold_ids = resolve_gold_ids(question, store)
        gold_sizes.append(len(gold_ids))
        relevance = build_relevance_map(question, store, gold_ids)
        retrieved_ids, latency = _retrieve(
            retriever,
            spec,
            question,
            budget_tokens,
            fetch_k,
            rewrite_query=rewrite_query,
        )
        metric_rows.append(
            evaluate_ranking(
                retrieved_ids,
                gold_ids,
                k=top_k,
                relevance_grades=relevance,
            )
        )
        query_latencies.append(latency)

    retrieval_total = sum(query_latencies)
    timing = AblationTiming(
        ingest_seconds=round(ingest_seconds, 3),
        index_seconds=round(index_seconds, 3),
        retrieval_total_seconds=round(retrieval_total, 3),
        retrieval_avg_seconds=round(retrieval_total / len(query_latencies), 3) if query_latencies else 0.0,
        retrieval_p95_seconds=round(_percentile(query_latencies, 0.95), 3),
        total_seconds=round(time.perf_counter() - run_start, 3),
    )
    gold_sanity = GoldSanityReport(
        zero_gold_count=sum(1 for size in gold_sizes if size == 0),
        avg_gold_size=round(sum(gold_sizes) / len(gold_sizes), 4) if gold_sizes else 0.0,
        question_count=len(gold_sizes),
    )
    return AblationEvalResult(
        metrics=average_metrics(metric_rows),
        timing=timing,
        gold_sanity=gold_sanity,
    )


def run_ablations_from_documents(
    documents: dict[str, list[ParsedBlock]],
    *,
    qa_path: str | Path | None = None,
    questions: list[BenchmarkQuestion] | None = None,
    settings: Settings | None = None,
    llm: OllamaLLM | None = None,
    budget_tokens: int | None = None,
    top_k: int = 8,
    resolve_gold_ids: Callable[[BenchmarkQuestion, SQLiteTreeStore], set[str]] | None = None,
    sqlite_path: str | Path | None = None,
    benchmark_format: BenchmarkFormat | None = None,
) -> AblationReport:
    settings = benchmark_eval_settings(settings)
    budget_tokens = budget_tokens or settings.context_budget_tokens
    gold_resolver = resolve_gold_ids or resolve_gold_official
    if questions is None:
        questions = load_questions(qa_path, documents, benchmark_format=benchmark_format)
    questions = align_questions_with_documents(questions, documents)

    store_path = sqlite_path or settings.sqlite_path
    store = SQLiteTreeStore(store_path)
    embedder = EmbeddingBackend(settings.embedding_mode, settings.embedding_model, settings.embedding_dim)
    retriever = AnchorRetriever(
        store,
        str(settings.index_dir),
        embedder,
        llm=None,
        confidence_threshold=settings.anchor_confidence_threshold,
        dense_weight=settings.hybrid_dense_weight,
        sparse_weight=settings.hybrid_sparse_weight,
        rrf_k=settings.hybrid_rrf_k,
    )

    eval_runs = {
        "baseline": _evaluate_spec(
            ABLATION_SPECS[AblationName.BASELINE],
            documents,
            questions,
            settings,
            store,
            retriever,
            budget_tokens=budget_tokens,
            top_k=top_k,
            resolve_gold_ids=gold_resolver,
            rewrite_query=False,
        ),
        "hierarchy": _evaluate_spec(
            ABLATION_SPECS[AblationName.HIERARCHY],
            documents,
            questions,
            settings,
            store,
            retriever,
            budget_tokens=budget_tokens,
            top_k=top_k,
            resolve_gold_ids=gold_resolver,
            rewrite_query=False,
        ),
        "anchor_only": _evaluate_spec(
            INTERNAL_SPECS["anchor_only"],
            documents,
            questions,
            settings,
            store,
            retriever,
            budget_tokens=budget_tokens,
            top_k=top_k,
            resolve_gold_ids=gold_resolver,
            rewrite_query=False,
        ),
        "full": _evaluate_spec(
            ABLATION_SPECS[AblationName.FULL],
            documents,
            questions,
            settings,
            store,
            retriever,
            budget_tokens=budget_tokens,
            top_k=top_k,
            resolve_gold_ids=gold_resolver,
            rewrite_query=False,
        ),
    }

    contributions = compute_contributions(
        eval_runs["baseline"].recall_at_k,
        eval_runs["hierarchy"].recall_at_k,
        eval_runs["anchor_only"].recall_at_k,
        eval_runs["full"].recall_at_k,
    )
    hierarchy_overhead_pct, anchor_overhead_pct, expansion_overhead_pct = compute_time_overhead(
        eval_runs["baseline"].timing.total_seconds,
        eval_runs["hierarchy"].timing.total_seconds,
        eval_runs["anchor_only"].timing.total_seconds,
        eval_runs["full"].timing.total_seconds,
    )
    time_cost = TimeCostReport(
        parse_seconds=0.0,
        total_seconds=round(sum(run.timing.total_seconds for run in eval_runs.values()), 3),
        by_mode={key: run.timing for key, run in eval_runs.items()},
        hierarchy_overhead_pct=hierarchy_overhead_pct,
        anchor_overhead_pct=anchor_overhead_pct,
        expansion_overhead_pct=expansion_overhead_pct,
        full_anrag_avg_query_ms=round(eval_runs["full"].timing.retrieval_avg_seconds * 1000, 1),
        plain_rag_avg_query_ms=round(eval_runs["baseline"].timing.retrieval_avg_seconds * 1000, 1),
    )
    scores = [
        AblationScore(
            name=ABLATION_SPECS[AblationName.BASELINE].label,
            key="baseline",
            metrics=eval_runs["baseline"].metrics,
            query_count=len(questions),
            timing=eval_runs["baseline"].timing,
            gold_sanity=eval_runs["baseline"].gold_sanity,
        ),
        AblationScore(
            name=ABLATION_SPECS[AblationName.HIERARCHY].label,
            key="hierarchy",
            metrics=eval_runs["hierarchy"].metrics,
            query_count=len(questions),
            timing=eval_runs["hierarchy"].timing,
            gold_sanity=eval_runs["hierarchy"].gold_sanity,
        ),
        AblationScore(
            name=INTERNAL_SPECS["anchor_only"].label,
            key="anchor_only",
            metrics=eval_runs["anchor_only"].metrics,
            query_count=len(questions),
            timing=eval_runs["anchor_only"].timing,
            gold_sanity=eval_runs["anchor_only"].gold_sanity,
        ),
        AblationScore(
            name=ABLATION_SPECS[AblationName.FULL].label,
            key="full",
            metrics=eval_runs["full"].metrics,
            query_count=len(questions),
            timing=eval_runs["full"].timing,
            gold_sanity=eval_runs["full"].gold_sanity,
        ),
    ]
    return AblationReport(
        scores=scores,
        contributions=contributions,
        time_cost=time_cost,
        questions=questions,
    )


def run_ablations(
    benchmark_path: str | Path,
    *,
    qa_path: str | Path | None = None,
    settings: Settings | None = None,
    llm: OllamaLLM | None = None,
    budget_tokens: int | None = None,
    top_k: int = 8,
    resolve_gold_ids: Callable[[BenchmarkQuestion, SQLiteTreeStore], set[str]] | None = None,
    eval_mode: bool = True,
    benchmark_format: BenchmarkFormat | None = None,
) -> AblationReport:
    settings = benchmark_eval_settings(settings)
    parser = BenchmarkParser()
    fmt = resolve_run_format(benchmark_path, qa_path, benchmark_format)

    parse_start = time.perf_counter()
    if fmt in {"hotpotqa", "beir", "kilt", "ragbench"}:
        source_path = Path(qa_path or benchmark_path)
        documents, questions = load_official_benchmark(source_path, fmt=fmt)
        parse_seconds = round(time.perf_counter() - parse_start, 3)
        report = run_ablations_from_documents(
            documents,
            questions=questions,
            settings=settings,
            llm=llm,
            budget_tokens=budget_tokens,
            top_k=top_k,
            resolve_gold_ids=resolve_gold_ids,
            benchmark_format=fmt,
        )
    else:
        documents = parser.parse_benchmark(
            benchmark_path,
            settings=settings,
            llm=None if eval_mode else llm,
            eval_mode=eval_mode,
        )
        parse_seconds = round(time.perf_counter() - parse_start, 3)
        report = run_ablations_from_documents(
            documents,
            qa_path=qa_path,
            settings=settings,
            llm=llm,
            budget_tokens=budget_tokens,
            top_k=top_k,
            resolve_gold_ids=resolve_gold_ids,
            benchmark_format="anrag",
        )
    report.time_cost.parse_seconds = parse_seconds
    report.time_cost.total_seconds = round(report.time_cost.total_seconds + parse_seconds, 3)
    return report
