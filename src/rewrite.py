"""查询改写（query rewriting / HyDE-lite）。

口语化问题常带指代词、省略和低区分度词，直接拿去检索会漏召回。这里在检索前
用 LLM 把问题改写成「关键词密集、术语规范化」的检索查询，再喂给 BM25 / dense /
hybrid；而重排与生成仍用原始问题，保证忠实于用户意图。

完整 HyDE 会先让 LLM 生成一段「假设答案文档」，再用这段文档的向量去检索；
此处采用更轻量、每问一次小模型调用的查询改写，成本低且可肉眼验证，作为
HyDE 的低成本替代。
"""

from __future__ import annotations

from src import config, llm

_SYSTEM = (
    "你是检索查询改写助手。把用户的问题改写成更利于从技术文档中"
    "检索出相关段落的查询：补充关键词、同义词与英文术语，去掉口语化表达和指代词。"
    "只输出改写后的查询本身，不要任何解释，不要句末标点。"
)


def rewrite_query(question: str, history: list[dict] | None = None,
                  model: str | None = None) -> str:
    """把问题改写成检索查询；关闭开关或失败时原样返回，保证检索不中断。

    传入 ``history``（本轮之前的 ``[{question, answer}]``）时，会结合对话历史
    把指代、省略（如「它呢？」「那 query 参数呢？」）解析成独立、关键词密集的查询。
    """
    if not config.QUERY_REWRITE:
        return question
    try:
        messages = [{"role": "system", "content": _SYSTEM}]
        if history:
            turns = "\n".join(
                f"问：{t['question']}\n答：{t['answer']}" for t in history[-4:]
            )
            messages.append({
                "role": "user",
                "content": f"对话历史：\n{turns}\n\n请把当前问题改写成独立的检索查询：{question}",
            })
        else:
            messages.append({"role": "user", "content": f"原问题：{question}"})
        out = llm.chat(messages, temperature=0.0, max_tokens=128, model=model)
        out = out.strip().strip('"').strip("'").strip()
        return out or question
    except Exception:
        return question
