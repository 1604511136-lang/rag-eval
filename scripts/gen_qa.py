"""半自动生成评测 QA 集：从 chunk 抽取若干条，用 LLM 生成「问题 + 标准答案」，
并自动标注 gold_chunk（即该 chunk）。生成后请人工抽查约 20% 再用于评测。

用法：
    python scripts/gen_qa.py 50 512   # 生成 50 道，基于 chunk_size=512 的语料
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingest, llm

GEN_PROMPT = """请根据下面的文档片段，生成一道具体、可回答的问题，并给出简短的标准答案。
要求：问题必须能仅凭该片段回答；不要编造片段之外的信息。
只输出一个 JSON 对象，格式：{{"question": "...", "answer": "..."}}
文档片段：
{text}"""


def _extract_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    argv = sys.argv[1:]
    corpus = config.CORPUS_NAME
    if "--corpus" in argv:
        i = argv.index("--corpus")
        corpus = argv[i + 1]
        del argv[i:i + 2]
    n = int(argv[0]) if argv else 50
    size = int(argv[1]) if len(argv) > 1 else 512

    config.use_corpus(corpus)
    chunks = ingest.load_chunks(size)
    sample = random.sample(chunks, min(n, len(chunks)))

    config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.BENCHMARK_DIR / "qa.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(sample):
            raw = llm.chat([{"role": "user", "content": GEN_PROMPT.format(text=c["text"])}])
            parsed = _extract_json(raw)
            if not parsed or not parsed.get("question") or not parsed.get("answer"):
                print(f"[gen_qa] 跳过 chunk {c['chunk_id']}（解析失败）")
                continue
            item = {
                "id": f"q{written:03d}",
                "question": parsed["question"].strip(),
                "gold_answer": parsed["answer"].strip(),
                "gold_chunks": [c["chunk_id"]],
                "gold_key": f"{c['source']}||{c['section']}",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"[gen_qa] 已生成 {written} 道 QA 到 {out_path}（请人工抽查约 20%）")


if __name__ == "__main__":
    main()
