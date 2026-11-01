import numpy as np

from src.retrieve import Retriever, tokenize


CHUNKS = [
    {"chunk_id": "c0", "text": "FastAPI 是一个 Python Web 框架"},
    {"chunk_id": "c1", "text": "香蕉是一种常见的水果"},
    {"chunk_id": "c2", "text": "路径参数使用 path parameter 声明"},
]


def make_retriever(monkeypatch, query_vec):
    # 三个 chunk 的向量分别对齐三个坐标轴，便于断言 dense 排序
    emb = np.eye(3, dtype=np.float32)
    monkeypatch.setattr("src.llm.embed", lambda texts: [query_vec] * len(texts))
    return Retriever(CHUNKS, emb)


def test_tokenize_english_and_chinese():
    toks = tokenize("FastAPI 香蕉")
    assert "fastapi" in toks
    assert "香蕉" in toks  # 中文 bigram
    assert "香" in toks and "蕉" in toks


def test_bm25_ranks_chinese_keyword(monkeypatch):
    r = make_retriever(monkeypatch, [0, 0, 0])
    ranked = r.search("水果", strategy="bm25", top_k=3)
    assert ranked[0] == "c1"


def test_dense_ranks_by_similarity(monkeypatch):
    r = make_retriever(monkeypatch, [1.0, 0.0, 0.0])
    ranked = r.search("任意问题", strategy="dense", top_k=3)
    assert ranked[0] == "c0"


def test_hybrid_returns_list(monkeypatch):
    r = make_retriever(monkeypatch, [1.0, 0.0, 0.0])
    ranked = r.search("水果", strategy="hybrid", top_k=3)
    assert len(ranked) == 3
    assert len(set(ranked)) == 3  # 无重复
