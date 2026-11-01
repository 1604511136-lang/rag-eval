"""评测编排与报告生成。

1. 遍历 chunk_size × 检索策略，计算检索指标；
2. 对最优配置跑答案质量评测（LLM-as-judge）；
3. 输出 report.md 对比表 + 图表 + 结论。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能出图
import matplotlib.pyplot as plt
import numpy as np

# 让图表能正常显示中文标签（Windows 用微软雅黑，其他平台回退）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from src import config, generate, ingest
from src.eval_answer import judge_correctness, judge_faithfulness
from src.eval_retrieval import evaluate_retrieval
from src.retrieve import Retriever

STRATEGIES = ("bm25", "dense", "hybrid")


def load_embeddings(chunk_size: int) -> np.ndarray:
    path = config.CHUNK_DIR / f"embeddings_{chunk_size}.npy"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行 embedding 步骤")
    return np.load(path)


def _chunk_key(c: dict) -> str:
    """section 级 relevance key，与 chunk 大小无关，用于跨切分公平对比。"""
    return f"{c['source']}||{c['section']}"


def run_retrieval_grid(qa_items: list[dict]) -> dict:
    """返回 {(chunk_size, strategy): 指标dict}。"""
    results: dict = {}
    for size in config.CHUNK_SIZES:
        chunks = ingest.load_chunks(size)
        key_by_id = {c["chunk_id"]: _chunk_key(c) for c in chunks}
        retriever = Retriever(chunks, load_embeddings(size))
        for strategy in STRATEGIES:
            def search(q, r=retriever, kb=key_by_id, s=strategy):
                ranked = r.search(q, s)
                # chunk_id -> section key，并去重（保留首个出现顺序）
                return list(dict.fromkeys(kb[cid] for cid in ranked if cid in kb))

            metrics = evaluate_retrieval(qa_items, search, k_values=config.K_VALUES)
            results[(size, strategy)] = metrics
            print(f"[eval] chunk={size} strategy={strategy} "
                  f"recall@10={metrics['recall@10']:.3f} mrr={metrics['mrr']:.3f}")
    return results


def _search_ranked(retriever: Retriever, text_by_id: dict, question: str,
                   strategy: str, candidate_k: int, top_k: int) -> list[str]:
    """返回按相关度排序的 chunk_id；strategy 以 '+rerank' 结尾时先召回再 cross-encoder 重排。"""
    if strategy.endswith("+rerank"):
        from src import rerank
        base = strategy[: -len("+rerank")]
        ranked = retriever.search(question, base, candidate_k)
        docs = [text_by_id[cid] for cid in ranked if cid in text_by_id]
        order = rerank.rerank(question, docs, top_n=top_k)
        return [ranked[i] for i in order if i < len(ranked)]
    return retriever.search(question, strategy, top_k)


def run_rerank_grid(qa_items: list[dict], strategy: str = "hybrid") -> dict:
    """在 base 策略召回 top-N 后加 cross-encoder 重排，返回 {(size, "<strategy>+rerank"): 指标}。"""
    results: dict = {}
    for size in config.CHUNK_SIZES:
        chunks = ingest.load_chunks(size)
        key_by_id = {c["chunk_id"]: _chunk_key(c) for c in chunks}
        text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
        retriever = Retriever(chunks, load_embeddings(size))

        def search(q, r=retriever, kb=key_by_id, tb=text_by_id):
            ranked = _search_ranked(
                r, tb, q, f"{strategy}+rerank", config.RERANK_CANDIDATE_K, config.TOP_K
            )
            return list(dict.fromkeys(kb[cid] for cid in ranked if cid in kb))

        metrics = evaluate_retrieval(qa_items, search, k_values=config.K_VALUES)
        results[(size, f"{strategy}+rerank")] = metrics
        print(f"[eval] chunk={size} {strategy}+rerank "
              f"recall@10={metrics['recall@10']:.3f} mrr={metrics['mrr']:.3f}")
    return results


def _resolve(chunks: list[dict], ranked_ids: list[str]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in chunks}
    return [by_id[cid] for cid in ranked_ids if cid in by_id]


def run_answer_eval(qa_items: list[dict], size: int, strategy: str,
                    top_k: int = 5, limit: int | None = None) -> dict:
    """对指定配置跑答案质量评测，返回 faithfulness/correctness 均值。"""
    chunks = ingest.load_chunks(size)
    retriever = Retriever(chunks, load_embeddings(size))
    text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    items = qa_items[:limit] if limit else qa_items

    faith_scores, corr_scores = [], []
    for item in items:
        ranked = _search_ranked(
            retriever, text_by_id, item["question"], strategy,
            config.RERANK_CANDIDATE_K, top_k,
        )
        context_chunks = _resolve(chunks, ranked)
        answer = generate.generate(item["question"], context_chunks)
        context_text = "\n".join(c["text"] for c in context_chunks)

        f = judge_faithfulness(item["question"], context_text, answer)
        c = judge_correctness(item["question"], item["gold_answer"], answer)
        if f is not None:
            faith_scores.append(f)
        if c is not None:
            corr_scores.append(c)

    return {
        "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else None,
        "correctness": sum(corr_scores) / len(corr_scores) if corr_scores else None,
        "n": len(items),
    }


def best_config(retrieval_results: dict) -> tuple[int, str]:
    return max(retrieval_results, key=lambda k: (
        retrieval_results[k]["recall@10"], retrieval_results[k]["mrr"]
    ))


def _pct_change(new: float, old: float) -> float:
    """百分比变化；旧值为 0 时返回 0（无基线可比，避免除零）。"""
    if old == 0:
        return 0.0
    return (new - old) / old * 100


def write_report(retrieval_results: dict, answer_results: dict | None,
                 out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 对比表 ----
    rows = []
    for (size, strategy), m in sorted(retrieval_results.items()):
        rows.append(
            f"| {size} | {strategy} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
            f"{m['recall@10']:.3f} | {m['ndcg@10']:.3f} | {m['mrr']:.3f} |"
        )
    table = "\n".join([
        "| chunk_size | 策略 | recall@1 | recall@5 | recall@10 | ndcg@10 | MRR |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ])

    # ---- 结论 ----
    best_size, best_strategy = best_config(retrieval_results)
    best = retrieval_results[(best_size, best_strategy)]
    base = retrieval_results[(512, "bm25")]
    recall_improv = _pct_change(best["recall@10"], base["recall@10"])
    mrr_improv = _pct_change(best["mrr"], base["mrr"])
    r1_improv = _pct_change(best["recall@1"], base["recall@1"])
    conclusion = (
        f"最优配置为 chunk_size={best_size} + {best_strategy} 检索"
        f"（recall@10={best['recall@10']:.3f}，MRR={best['mrr']:.3f}，recall@1={best['recall@1']:.3f}）。\n\n"
        f"相比 BM25 基线（chunk=512），MRR 提升 {mrr_improv:.1f}%、recall@1 提升 {r1_improv:.1f}%、"
        f"recall@10 提升 {recall_improv:.1f}%。由于 recall@10 已接近饱和，"
        "MRR / recall@1 更能体现排序质量——hybrid 检索把正确段落排得更靠前。"
    )

    # 若包含重排序结果，额外给出「hybrid vs hybrid+rerank」的排序质量对比
    if (512, "hybrid+rerank") in retrieval_results and (512, "hybrid") in retrieval_results:
        hr = retrieval_results[(512, "hybrid+rerank")]
        h = retrieval_results[(512, "hybrid")]
        mrr_d = _pct_change(hr["mrr"], h["mrr"])
        r1_d = _pct_change(hr["recall@1"], h["recall@1"])
        conclusion += (
            f"\n\n在 hybrid 召回基础上叠加 cross-encoder 重排序（{config.RERANK_MODEL}），"
            f"chunk=512 时 MRR 由 {h['mrr']:.3f} 提升到 {hr['mrr']:.3f}（+{mrr_d:.1f}%），"
            f"recall@1 由 {h['recall@1']:.3f} 提升到 {hr['recall@1']:.3f}（+{r1_d:.1f}%），"
            "说明重排序把正确段落排得更靠前。"
        )

    # ---- 图表 ----
    _plot_retrieval(retrieval_results, out_dir / "retrieval.png")

    answer_section = ""
    if answer_results and answer_results.get("faithfulness") is not None:
        answer_section = (
            "\n## 答案质量（LLM-as-judge，1-5 分）\n\n"
            f"- faithfulness: {answer_results['faithfulness']:.2f}\n"
            f"- correctness: {answer_results['correctness']:.2f}\n"
        )

    md = (
        "# RAG 检索评测报告\n\n"
        f"## 检索指标\n\n{table}\n\n"
        f"## 结论\n\n{conclusion}\n"
        f"{answer_section}\n"
        "## 图表\n\n![检索指标对比](retrieval.png)\n"
    )

    report_path = out_dir / "report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"[report] 已生成 {report_path}")
    return report_path


def _plot_retrieval(results: dict, out_path: Path) -> None:
    configs = [f"{s}/{st}" for (s, st) in sorted(results)]
    recalls = [results[k]["recall@10"] for k in sorted(results)]
    mrrs = [results[k]["mrr"] for k in sorted(results)]

    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, recalls, width, label="recall@10", color="#4C72B0")
    ax.bar(x + width / 2, mrrs, width, label="MRR", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("分数")
    ax.set_title("不同 chunk_size 与检索策略的检索质量")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report] 已保存图表 {out_path}")
