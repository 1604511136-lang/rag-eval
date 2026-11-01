"""一键跑通：下载语料 → 切分 → embedding → 检索评测 → 答案评测 → 报告。

用法：
    python scripts/run_all.py                      # 首次：含下载语料
    python scripts/run_all.py --skip-download      # 已下载过语料
    python scripts/run_all.py --answer-limit 10    # 答案评测只跑前 10 道（控 API 成本）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config, ingest, llm, report, vector_store


def embed_and_save(chunks: list[dict], size: int) -> None:
    texts = [c["text"] for c in chunks]
    print(f"[embed] 计算 {len(texts)} 个 chunk 的向量 (chunk_size={size}) ...")
    vecs = llm.embed(texts)
    np.save(config.CHUNK_DIR / f"embeddings_{size}.npy", np.asarray(vecs, dtype=np.float32))
    vector_store.upsert(size, chunks, vecs)
    print(f"[embed] 已保存 embeddings_{size}.npy 并写入 Chroma 集合 "
          f"{vector_store.collection_name(size)}")


def load_qa(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 检索问答系统端到端评测")
    ap.add_argument("--corpus", default=config.CORPUS_NAME,
                    help="语料名称（见 src/config.py 的 CORPORA）")
    ap.add_argument("--skip-download", action="store_true", help="跳过语料下载（使用已下载的语料）")
    ap.add_argument("--qa", default=None, help="评测 QA 集路径（默认 <benchmark>/qa.jsonl）")
    ap.add_argument("--answer-limit", type=int, default=20, help="答案质量评测的题目数")
    ap.add_argument("--no-rerank", action="store_true", help="跳过 cross-encoder 重排序评测（省 API 调用）")
    ap.add_argument("--ingest-only", action="store_true", help="只做下载+切分+embedding，跳过评测")
    args = ap.parse_args()

    # 切换语料（重定位数据目录，须在加载 chunk / embedding 之前）
    config.use_corpus(args.corpus)
    meta = config.corpus_meta()
    subpath = f"{meta['tar_top']}/{meta['doc_subpath']}"

    # 1. 语料
    if args.skip_download:
        md_files = sorted((config.RAW_DIR / subpath).glob("**/*.md"))
        if not md_files:
            raise SystemExit("未找到已下载语料，请先去掉 --skip-download")
    else:
        md_files = ingest.download_corpus()

    # 2. 切分
    for size in config.CHUNK_SIZES:
        ingest.save_chunks(ingest.ingest(md_files, size), size)

    # 3. embedding
    for size in config.CHUNK_SIZES:
        embed_and_save(ingest.load_chunks(size), size)

    if args.ingest_only:
        print("[run] ingest-only 完成，跳过评测")
        return

    # 4. 检索评测
    qa_path = Path(args.qa) if args.qa else config.BENCHMARK_DIR / "qa.jsonl"
    qa = load_qa(qa_path)
    retrieval_results = report.run_retrieval_grid(qa)
    if not args.no_rerank:
        retrieval_results.update(report.run_rerank_grid(qa))

    # 5. 答案质量（只评最优配置，控成本）
    best_size, best_strategy = report.best_config(retrieval_results)
    print(f"[eval] 最优配置: chunk_size={best_size} + {best_strategy}")
    answer_results = report.run_answer_eval(qa, best_size, best_strategy, limit=args.answer_limit)

    # 6. 报告
    report.write_report(retrieval_results, answer_results)


if __name__ == "__main__":
    main()
