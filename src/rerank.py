"""重排序（rerank）：用 cross-encoder 对检索候选做语义精排。

思路：先由 BM25/dense/hybrid 召回 top-N 候选（快、召回率高但排序粗），
再用 cross-encoder（如硅基流动免费的 BAAI/bge-reranker-v2-m3）对候选逐个
计算 query-doc 相关度，精排取 top-k。cross-encoder 对 query 与文档做联合编码，
比双塔 embedding 的余弦相似度排序更准，能显著提升 MRR / recall@1。
"""

from __future__ import annotations

import requests

from src import config


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[int]:
    """返回按相关度降序的原始下标列表（长度 ≤ top_n）。

    ``documents`` 顺序与调用方传入的候选列表一致，返回的是这些候选在
    原列表中的下标，便于调用方映射回 chunk_id。
    """
    if not documents:
        return []
    n = min(top_n or len(documents), len(documents))
    resp = requests.post(
        f"{config.BASE_URL.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {config.API_KEY}"},
        json={
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": n,
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    # 按相关度降序排序（不依赖服务端是否已排序），返回原下标
    ordered = sorted(results, key=lambda r: -float(r.get("relevance_score", 0.0)))
    return [r["index"] for r in ordered]
