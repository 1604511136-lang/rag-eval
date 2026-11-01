import pytest

from src.eval_retrieval import evaluate_retrieval, mrr, ndcg_at_k, recall_at_k


def test_recall():
    ranked = ["A", "X", "B", "Y"]
    gold = {"A", "B"}
    assert recall_at_k(ranked, gold, 1) == 0.5
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert recall_at_k(ranked, {"Z"}, 10) == 0.0


def test_mrr():
    assert mrr(["A", "X", "B"], {"A", "B"}) == 1.0
    assert mrr(["X", "A"], {"A"}) == 0.5
    assert mrr(["X", "Y"], {"A"}) == 0.0


def test_ndcg():
    ranked = ["A", "X", "B"]
    gold = {"A", "B"}
    # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1.5
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.63093...
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(0.9197, abs=1e-3)


def test_evaluate_retrieval_shape():
    qa = [{"question": "q", "gold_key": "A"}]
    result = evaluate_retrieval(qa, lambda q: ["A", "X", "B"], k_values=(1, 5))
    assert "mrr" in result
    assert "recall@1" in result
    assert "recall@5" in result
    assert result["recall@5"] == 1.0
    assert result["mrr"] == 1.0
