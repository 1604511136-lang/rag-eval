"""检索质量评测：Recall@k / nDCG@k / MRR。

纯函数实现，便于单元测试。gold 用 chunk_id 集合表示命中的「标准答案 chunk」。
"""

from __future__ import annotations

import math
from collections.abc import Callable


def _dcg(rel: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    ideal = [1.0] * min(len(gold), k)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    rel = [1.0 if cid in gold else 0.0 for cid in ranked[:k]]
    return _dcg(rel) / ideal_dcg


def mrr(ranked: list[str], gold: set[str]) -> float:
    """第一个相关结果的倒数排名（1-indexed），整列表内无命中则为 0。"""
    for i, cid in enumerate(ranked):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    qa_items: list[dict],
    search_fn: Callable[[str], list[str]],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """对 QA 集计算聚合指标。

    qa_items 每项含 question 与 gold_key（section 级 relevance key）；
    search_fn 返回按相关度排序的 key 列表。
    """
    agg = {k: {"recall": [], "ndcg": []} for k in k_values}
    mrr_list: list[float] = []

    for item in qa_items:
        ranked = search_fn(item["question"])
        gold = {item["gold_key"]}
        mrr_list.append(mrr(ranked, gold))
        for k in k_values:
            agg[k]["recall"].append(recall_at_k(ranked, gold, k))
            agg[k]["ndcg"].append(ndcg_at_k(ranked, gold, k))

    result = {"mrr": sum(mrr_list) / len(mrr_list)}
    for k in k_values:
        result[f"recall@{k}"] = sum(agg[k]["recall"]) / len(agg[k]["recall"])
        result[f"ndcg@{k}"] = sum(agg[k]["ndcg"]) / len(agg[k]["ndcg"])
    return result
