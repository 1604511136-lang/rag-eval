"""端到端冒烟测试：用 fake 的 embed/chat 跑通全流程，不依赖网络与 API。"""

import numpy as np
import pytest

from src import config, ingest, llm, report


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    # 把数据/输出目录重定向到临时目录
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "CHUNK_DIR", tmp_path / "chunks")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")

    rng = np.random.RandomState(0)

    def fake_embed(texts):
        return [rng.randn(16).tolist() for _ in texts]

    def fake_chat(messages, **kwargs):
        return "5"  # 评测打分恒为 5

    monkeypatch.setattr("src.llm.embed", fake_embed)
    monkeypatch.setattr("src.llm.chat", fake_chat)

    # 造 3 篇样例 markdown
    md_files = []
    for i in range(3):
        p = config.RAW_DIR / f"fastapi-master/docs/zh/docs/tutorial/doc{i}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"# 章节 {i}\n\nFastAPI 的路径参数与查询参数说明。\n\n## 小节\n\n内容 {i}。",
            encoding="utf-8",
        )
        md_files.append(p)
    return md_files


def test_pipeline_end_to_end(fake_env):
    md_files = fake_env

    for size in config.CHUNK_SIZES:
        chunks = ingest.ingest(md_files, size)
        ingest.save_chunks(chunks, size)
        emb = np.asarray(llm.embed([c["text"] for c in chunks]), dtype=np.float32)
        np.save(config.CHUNK_DIR / f"embeddings_{size}.npy", emb)

    qa = [{
        "id": "q0",
        "question": "路径参数怎么用？",
        "gold_answer": "...",
        "gold_chunks": ["c00000"],
        "gold_key": "fastapi-master/docs/zh/docs/tutorial/doc0.md||章节 0",
    }]

    results = report.run_retrieval_grid(qa)
    assert len(results) == len(config.CHUNK_SIZES) * 3  # 3 种策略

    best = report.best_config(results)
    answer = report.run_answer_eval(qa, *best, limit=1)
    assert answer["faithfulness"] == 5.0
    assert answer["correctness"] == 5.0

    path = report.write_report(results, answer)
    assert path.exists()
    assert (config.OUTPUT_DIR / "retrieval.png").exists()
