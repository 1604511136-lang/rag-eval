"""检索：BM25 / dense（向量）/ hybrid（RRF 融合）。

对同一组 chunk 构建 Retriever，支持三种策略返回按相关度排序的 chunk_id 列表，
供评测统一调用。
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src import config, llm


def tokenize(text: str) -> list[str]:
    """简单分词：英文单词/数字 + 中文字符与二元组（bigram），作为 BM25 的 token。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    zh = re.findall(r"[一-鿿]", text)
    for i, ch in enumerate(zh):
        tokens.append(ch)
        if i + 1 < len(zh):
            tokens.append(ch + zh[i + 1])
    return tokens


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Retriever:
    """基于一组 chunk 的检索器。dense 检索依赖 llm.embed 计算 query 向量。"""

    def __init__(self, chunks: list[dict], embeddings: list[list[float]]):
        self.chunks = list(chunks)
        self.embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

    def _rank_bm25(self, query: str, top_k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        idxs = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i]["chunk_id"] for i in idxs]

    def _rank_dense(self, query: str, top_k: int) -> list[str]:
        q = _normalize(np.asarray([llm.embed([query])[0]], dtype=np.float32))
        sims = (self.embeddings @ q.T).ravel()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i]["chunk_id"] for i in idxs]

    @staticmethod
    def _rrf_fuse(rankings: list[list[str]], k: int, top_k: int) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top_k]

    def search(self, query: str, strategy: str = "hybrid", top_k: int = config.TOP_K) -> list[str]:
        if strategy == "bm25":
            return self._rank_bm25(query, top_k)
        if strategy == "dense":
            return self._rank_dense(query, top_k)
        if strategy == "hybrid":
            bm25 = self._rank_bm25(query, top_k)
            dense = self._rank_dense(query, top_k)
            return self._rrf_fuse([bm25, dense], k=config.RRF_K, top_k=top_k)
        raise ValueError(f"未知检索策略: {strategy}")
