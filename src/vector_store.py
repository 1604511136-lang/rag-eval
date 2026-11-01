"""向量库（Chroma）封装：把 chunk 的向量持久化到本地 Chroma 集合，用于 dense 检索。

相比 numpy 的余弦相似度矩阵，Chroma 提供：
- 持久化（重启不丢，向量随语料一起落盘）
- 元数据过滤、增量 upsert、近似最近邻（HNSW）
- 生产级向量库接口（可平滑迁移到服务端部署）

retrieve.py 的 Retriever 会优先走这里做 dense 检索；集合不存在时回退到
numpy 余弦相似度（保证离线单测与旧数据仍可用）。
"""

from __future__ import annotations

import chromadb

from src import config

_client = None
_client_path: str | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client, _client_path
    path = str(config.chroma_dir())
    if _client is None or _client_path != path:
        _client = chromadb.PersistentClient(path=path)
        _client_path = path
    return _client


def collection_name(size: int) -> str:
    return f"{config.CORPUS_NAME}-chunks-{size}"


def get_collection(size: int):
    return _get_client().get_or_create_collection(
        name=collection_name(size), metadata={"hnsw:space": "cosine"}
    )


def upsert(size: int, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """把一组 chunk 及其向量写入集合（按 chunk_id 增量 upsert）。"""
    col = get_collection(size)
    col.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"], "url": c["url"]}
                   for c in chunks],
        embeddings=embeddings,
    )


def exists(size: int) -> bool:
    """集合是否存在且非空（不创建）。"""
    try:
        return _get_client().get_collection(collection_name(size)).count() > 0
    except Exception:
        return False


def search(size: int, query_embedding: list[float], top_k: int) -> list[str]:
    """按余弦相似度返回 top_k 个 chunk_id（降序）。"""
    res = get_collection(size).query(
        query_embeddings=[query_embedding], n_results=top_k
    )
    return list(res["ids"][0])
