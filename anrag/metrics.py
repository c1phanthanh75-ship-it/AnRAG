from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics. Primary focus: ``recall_at_k`` (configured k)."""

    recall_at_k: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, float],
    k: int = 10,
) -> float:
    if not relevance:
        return 0.0

    def dcg(ranks: list[str]) -> float:
        score = 0.0
        for index, item_id in enumerate(ranks[:k], start=1):
            rel = relevance.get(item_id, 0.0)
            if rel > 0:
                score += rel / math.log2(index + 1)
        return score

    ideal = sorted(relevance.items(), key=lambda item: item[1], reverse=True)
    ideal_ids = [item_id for item_id, _ in ideal]
    ideal_dcg = dcg(ideal_ids)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(retrieved_ids) / ideal_dcg


def evaluate_ranking(
    retrieved_ids: list[str],
    gold_ids: set[str],
    *,
    k: int = 8,
    relevance_grades: dict[str, float] | None = None,
) -> RetrievalMetrics:
    binary_relevance = relevance_grades or {item_id: 1.0 for item_id in gold_ids}
    return RetrievalMetrics(
        recall_at_k=round(recall_at_k(retrieved_ids, gold_ids, k), 4),
        recall_at_5=round(recall_at_k(retrieved_ids, gold_ids, 5), 4),
        recall_at_10=round(recall_at_k(retrieved_ids, gold_ids, 10), 4),
        recall_at_20=round(recall_at_k(retrieved_ids, gold_ids, 20), 4),
        mrr=round(mrr(retrieved_ids, gold_ids), 4),
        ndcg_at_10=round(ndcg_at_k(retrieved_ids, binary_relevance, k=10), 4),
    )


def average_metrics(rows: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not rows:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = len(rows)
    return RetrievalMetrics(
        recall_at_k=round(sum(row.recall_at_k for row in rows) / count, 4),
        recall_at_5=round(sum(row.recall_at_5 for row in rows) / count, 4),
        recall_at_10=round(sum(row.recall_at_10 for row in rows) / count, 4),
        recall_at_20=round(sum(row.recall_at_20 for row in rows) / count, 4),
        mrr=round(sum(row.mrr for row in rows) / count, 4),
        ndcg_at_10=round(sum(row.ndcg_at_10 for row in rows) / count, 4),
    )
