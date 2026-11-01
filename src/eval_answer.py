"""答案质量评测：LLM-as-judge。

- faithfulness：回答是否由检索到的上下文支撑（是否编造）
- correctness：回答与标准答案的一致性
均为 1-5 分，由 LLM 输出单个整数，解析后返回。
"""

from __future__ import annotations

import re

from src import llm

FAITHFUL_PROMPT = """你是一个严格的事实一致性评测器。请判断「回答」是否完全由「给定文档片段」支撑、没有编造任何片段之外的内容。
问题：{question}
给定文档片段：
{context}
回答：{answer}
只输出一个 1-5 的整数：5=完全由文档支撑；3=部分支撑、有少量不一致；1=大量编造。"""

CORRECT_PROMPT = """你是一个答案评测器。请判断「回答」与「标准答案」的一致性（不要求逐字相同，语义一致即可）。
问题：{question}
标准答案：{gold_answer}
回答：{answer}
只输出一个 1-5 的整数：5=完全正确；3=部分正确；1=完全错误。"""


def parse_score(text: str | None) -> float | None:
    if not text:
        return None
    for m in re.findall(r"\d+", text):
        v = int(m)
        if 1 <= v <= 5:
            return float(v)
    return None


def judge_faithfulness(question: str, context: str, answer: str) -> float | None:
    prompt = FAITHFUL_PROMPT.format(question=question, context=context, answer=answer)
    return parse_score(llm.chat([{"role": "user", "content": prompt}]))


def judge_correctness(question: str, gold_answer: str, answer: str) -> float | None:
    prompt = CORRECT_PROMPT.format(question=question, gold_answer=gold_answer, answer=answer)
    return parse_score(llm.chat([{"role": "user", "content": prompt}]))
