"""交互式问答：输入问题，返回带引用的答案与来源。

用法：
    python scripts/ask.py "FastAPI 怎么声明查询参数？"   # 单次问答
    python scripts/ask.py                                # 进入交互模式（输入 q 退出）

默认使用最优配置：chunk_size=512 + hybrid 检索，top-k=5。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import generate, ingest, report
from src.retrieve import Retriever

CHUNK_SIZE = 512
STRATEGY = "hybrid"
TOP_K = 5


def build():
    chunks = ingest.load_chunks(CHUNK_SIZE)
    retriever = Retriever(chunks, report.load_embeddings(CHUNK_SIZE))
    by_id = {c["chunk_id"]: c for c in chunks}
    return chunks, retriever, by_id


def answer(question: str, retriever, by_id) -> tuple[str, list[dict]]:
    ranked = retriever.search(question, STRATEGY, TOP_K)
    ctx = [by_id[cid] for cid in ranked if cid in by_id]
    ans = generate.generate(question, ctx)
    return ans, ctx


def print_sources(ctx: list[dict]) -> None:
    for c in ctx:
        title = c["section"] or c["source"]
        print(f"  - [{title}] {c['url']}")


def main() -> None:
    _, retriever, by_id = build()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ans, ctx = answer(question, retriever, by_id)
        print(f"\n问题：{question}\n")
        print(f"回答：{ans}\n")
        print("来源：")
        print_sources(ctx)
        return

    print("RAG 问答（FastAPI 中文文档 · chunk=512 + hybrid）")
    print("输入问题后回车；输入 q / quit / exit 退出。\n")
    while True:
        q = input("> ").strip()
        if q.lower() in ("q", "quit", "exit", ""):
            break
        ans, ctx = answer(q, retriever, by_id)
        print(f"\n回答：{ans}\n")
        print("来源：")
        print_sources(ctx)
        print()


if __name__ == "__main__":
    main()
