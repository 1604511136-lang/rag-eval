"""OpenAI-compatible LLM / Embedding 客户端封装。

统一走 `openai` SDK，通过 base_url/api_key 可切换不同服务商
（OpenAI / DeepSeek / 硅基流动 / 智谱等），便于评测时解耦模型。
"""

from __future__ import annotations

from openai import OpenAI

from src import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    return _client


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024,
         model: str | None = None) -> str:
    """调用对话模型，返回文本内容。"""
    model = model or config.CHAT_MODEL
    resp = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def chat_stream(messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024,
                model: str | None = None):
    """流式调用对话模型，逐段 yield 文本增量。"""
    model = model or config.CHAT_MODEL
    stream = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def embed(texts: list[str], model: str | None = None, batch_size: int = 32) -> list[list[float]]:
    """批量获取文本 embedding，返回按输入顺序排列的向量列表。"""
    model = model or config.EMBED_MODEL
    client = _get_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # 按 index 排序，避免个别服务商返回顺序不一致
        for item in sorted(resp.data, key=lambda d: d.index):
            out.append(item.embedding)
    return out
