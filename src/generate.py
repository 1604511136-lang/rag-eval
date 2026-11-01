"""RAG 生成：把检索到的 chunk 组装成 prompt，让 LLM 生成带引用的答案。"""

from __future__ import annotations

from src import llm

SYSTEM = (
    "你是文档问答助手。请只依据提供的文档片段回答问题；"
    "若片段中找不到答案，请明确回答「文档中未提及」。"
    "回答时用 [n] 标注引用来源编号。"
)


def build_messages(question: str, chunks: list[dict],
                   history: list[dict] | None = None) -> list[dict]:
    """组装 RAG 生成消息；传入 ``history``（本轮之前的问答）则附带多轮上下文。"""
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    messages = [{"role": "system", "content": SYSTEM}]
    for t in (history or [])[-4:]:
        messages.append({"role": "user", "content": t["question"]})
        messages.append({"role": "assistant", "content": t["answer"]})
    messages.append({
        "role": "user",
        "content": f"文档片段：\n{context}\n\n问题：{question}",
    })
    return messages


def generate(question: str, chunks: list[dict], history: list[dict] | None = None,
             model: str | None = None) -> str:
    return llm.chat(build_messages(question, chunks, history), model=model)
