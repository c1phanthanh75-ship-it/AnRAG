from anrag.metrics import evaluate_ranking, mrr, ndcg_at_k, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=2) == 0.5
    assert recall_at_k(["a"], set(), k=2) == 0.0


def test_mrr():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_10():
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], relevance, k=10) == 1.0
    assert ndcg_at_k(["c", "b", "a"], relevance, k=10) < 1.0


def test_evaluate_ranking_bundle():
    metrics = evaluate_ranking(["a", "b", "c", "d"], {"a", "c"}, k=8)
    assert metrics.recall_at_k == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr == 1.0
