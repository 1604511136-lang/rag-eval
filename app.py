"""RAG 问答 Web 界面后端。

提供 POST /api/ask（检索 + 生成）并托管前端静态页面。
启动：uvicorn app:app --reload
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config, generate, ingest, llm, report, rerank, rewrite, video
from src.retrieve import Retriever

HISTORY_PATH = config.DATA_DIR / "history.jsonl"
HISTORY_MAX = 50


def _append_history(item: dict) -> None:
    """追加一条历史，只保留最近 HISTORY_MAX 条。"""
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        items = []
        if HISTORY_PATH.exists():
            items = [json.loads(l) for l in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        items.append(item)
        items = items[-HISTORY_MAX:]
        HISTORY_PATH.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass  # 历史记录失败不影响主流程

app = FastAPI(title="FastAPI 文档问答", version="1.0.0")

CHUNK_SIZE = 512
STRATEGY = "hybrid"
TOP_K = 5

_retriever: Retriever | None = None
_by_id: dict = {}


def _get_retriever() -> tuple[Retriever, dict]:
    global _retriever, _by_id
    if _retriever is None:
        chunks = ingest.load_chunks(CHUNK_SIZE)
        _retriever = Retriever(chunks, report.load_embeddings(CHUNK_SIZE))
        _by_id = {c["chunk_id"]: c for c in chunks}
    return _retriever, _by_id


# ---- 多轮对话：按 session 保留最近若干轮上下文（内存态，重启即清）----
_SESSIONS: dict[str, list[dict]] = {}
SESSION_MAX = 6          # 每会话保留的最近轮数
SESSION_TOTAL_MAX = 200  # 会话数上限，超出淘汰最旧


def _session_history(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    return _SESSIONS.get(session_id, [])


def _remember(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return
    if len(_SESSIONS) >= SESSION_TOTAL_MAX and session_id not in _SESSIONS:
        _SESSIONS.pop(next(iter(_SESSIONS)))
    hist = _SESSIONS.setdefault(session_id, [])
    hist.append({"question": question, "answer": answer})
    del hist[:-SESSION_MAX]


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        retriever, by_id = _get_retriever()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"未找到数据，请先运行 scripts/run_all.py（{e}）")

    # 结合对话历史改写查询（解析指代/省略），用于检索与重排；生成仍用原始问题 + 历史
    history = _session_history(req.session_id)
    retrieval_query = rewrite.rewrite_query(question, history)
    candidates = retriever.search(retrieval_query, STRATEGY, config.RERANK_CANDIDATE_K)
    docs = [by_id[cid]["text"] for cid in candidates if cid in by_id]
    order = rerank.rerank(retrieval_query, docs, top_n=TOP_K)
    ranked = [candidates[i] for i in order if i < len(candidates)]
    ctx = [by_id[cid] for cid in ranked if cid in by_id]
    answer = generate.generate(question, ctx, history)
    sources = [
        {"title": c["section"] or c["source"], "url": c["url"], "text": c["text"][:240]}
        for c in ctx
    ]
    _append_history({
        "question": question, "answer": answer,
        "sources": [{"title": s["title"], "url": s["url"], "text": s["text"]} for s in sources],
        "ts": datetime.now().strftime("%m-%d %H:%M"),
    })
    _remember(req.session_id, question, answer)
    return {"question": question, "answer": answer, "sources": sources,
            "rewritten_query": retrieval_query}


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """流式问答（SSE）：先返回来源，再逐段返回答案增量。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        retriever, by_id = _get_retriever()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"未找到数据，请先运行 scripts/run_all.py（{e}）")

    history = _session_history(req.session_id)
    retrieval_query = rewrite.rewrite_query(question, history)
    candidates = retriever.search(retrieval_query, STRATEGY, config.RERANK_CANDIDATE_K)
    docs = [by_id[cid]["text"] for cid in candidates if cid in by_id]
    order = rerank.rerank(retrieval_query, docs, top_n=TOP_K)
    ranked = [candidates[i] for i in order if i < len(candidates)]
    ctx = [by_id[cid] for cid in ranked if cid in by_id]
    sources = [
        {"title": c["section"] or c["source"], "url": c["url"], "text": c["text"][:240]}
        for c in ctx
    ]

    def gen():
        def emit(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        yield emit({"type": "sources", "question": question,
                    "rewritten_query": retrieval_query, "sources": sources})
        full = ""
        try:
            for chunk in llm.chat_stream(generate.build_messages(question, ctx, history)):
                full += chunk
                yield emit({"type": "token", "text": chunk})
        except Exception as e:  # 生成失败时告知前端
            yield emit({"type": "error", "detail": str(e)})
            return
        _append_history({
            "question": question, "answer": full,
            "sources": [{"title": s["title"], "url": s["url"], "text": s["text"]} for s in sources],
            "ts": datetime.now().strftime("%m-%d %H:%M"),
        })
        _remember(req.session_id, question, full)
        yield emit({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/api/search")
def search(req: SearchRequest):
    """语义检索：只返回相关段落，不生成答案。"""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空")

    try:
        retriever, by_id = _get_retriever()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"未找到数据，请先运行 scripts/run_all.py（{e}）")

    retrieval_query = rewrite.rewrite_query(query)
    candidates = retriever.search(retrieval_query, STRATEGY, config.RERANK_CANDIDATE_K)
    docs = [by_id[cid]["text"] for cid in candidates if cid in by_id]
    order = rerank.rerank(query, docs, top_n=req.top_k)
    ranked = [candidates[i] for i in order if i < len(candidates)]
    results = [
        {"title": by_id[cid]["section"] or by_id[cid]["source"],
         "url": by_id[cid]["url"],
         "text": by_id[cid]["text"][:240]}
        for cid in ranked if cid in by_id
    ]
    return {"query": query, "rewritten_query": retrieval_query, "results": results}


@app.get("/api/videos")
def videos(q: str, top_n: int = 3):
    """为问题检索 B 站相关讲解视频（失败则降级为站内搜索链接）。"""
    keyword = q.strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="缺少搜索词")
    return video.search(keyword, top_n)


@app.get("/api/history")
def history():
    if not HISTORY_PATH.exists():
        return {"items": []}
    items = [json.loads(l) for l in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"items": list(reversed(items[-HISTORY_MAX:]))}


@app.delete("/api/history")
def clear_history():
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    return {"ok": True}


# ---- 评测工具 ----
_JOBS: dict[str, subprocess.Popen] = {}


def _output_dir(corpus: str) -> Path:
    try:
        config.corpus_meta(corpus)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return config.output_dir(corpus)


@app.get("/api/eval/corpora")
def eval_corpora():
    return {"corpora": [
        {"name": n, "label": m["label"],
         "has_report": (config.output_dir(n) / "report.md").exists()}
        for n, m in config.CORPORA.items()
    ]}


@app.get("/api/eval/report")
def eval_report(corpus: str = "fastapi-zh"):
    p = _output_dir(corpus) / "report.md"
    if not p.exists():
        return {"corpus": corpus, "available": False}
    return {"corpus": corpus, "available": True,
            "markdown": p.read_text(encoding="utf-8")}


@app.get("/api/eval/chart")
def eval_chart(corpus: str = "fastapi-zh"):
    p = _output_dir(corpus) / "retrieval.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="图表不存在，请先运行评测")
    return FileResponse(p, media_type="image/png")


@app.post("/api/eval/run")
def eval_run(corpus: str = "fastapi-zh"):
    _output_dir(corpus)  # 校验语料名
    if corpus in _JOBS and _JOBS[corpus].poll() is None:
        return {"running": True, "corpus": corpus}
    proc = subprocess.Popen(
        [sys.executable, "scripts/run_all.py", "--corpus", corpus,
         "--skip-download", "--answer-limit", "5"],
        cwd=Path(__file__).parent,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _JOBS[corpus] = proc
    return {"running": True, "corpus": corpus}


@app.get("/api/eval/status")
def eval_status(corpus: str = "fastapi-zh"):
    if corpus in _JOBS:
        return {"running": _JOBS[corpus].poll() is None}
    return {"running": False}


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
