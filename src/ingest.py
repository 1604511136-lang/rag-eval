"""语料下载、markdown 解析、切分与入库。

流程：下载 FastAPI 中文文档 → 按标题切分 → 按 chunk_size 切成窗口 → 保存 chunks.jsonl。
切分大小以「字符数」近似 token 数（对中文约为 1 字符 ≈ 1 token）。
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path

import requests

from src import config

def download_corpus(dest: Path | None = None) -> list[Path]:
    """下载并解压当前语料的 markdown，返回 .md 文件路径列表。"""
    dest = dest or config.RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)
    meta = config.corpus_meta()
    subpath = f"{meta['tar_top']}/{meta['doc_subpath']}"

    print(f"[ingest] 下载语料: {meta['label']} ({meta['tarball_url']})")
    resp = requests.get(meta["tarball_url"], timeout=120)
    resp.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.isfile() and m.name.startswith(subpath) and m.name.endswith(".md")
        ]
        print(f"[ingest] 解压 {len(members)} 个 markdown 文件到 {dest}")
        tar.extractall(dest, members=members)

    return sorted((dest / subpath).glob("**/*.md"))


def _split_sections(md_text: str) -> list[tuple[str, str]]:
    """按标题把 markdown 切成 (标题路径, 正文) 列表，忽略代码块内的 #。"""
    sections: list[tuple[str, str]] = []
    headings: list[str] = []
    body: list[str] = []
    in_code = False

    def flush() -> None:
        title = " > ".join(headings)
        text = "\n".join(body).strip()
        if text:
            sections.append((title, text))
        body.clear()

    for line in md_text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            body.append(line)
            continue
        if not in_code:
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                flush()  # 先归属旧标题，再更新标题路径
                headings = headings[: len(m.group(1)) - 1] + [m.group(2).strip()]
                continue
        body.append(line)
    flush()
    return sections


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    """把长文本按 chunk_size 切成带 overlap 的窗口，最后一块延伸到文本末尾。"""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    step = max(chunk_size - overlap, 1)
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return pieces


def chunk_markdown(md_text: str, source: str, url: str,
                   chunk_size: int, overlap: int) -> list[dict]:
    """把一篇 markdown 切成 chunk 列表（不含 chunk_id，由调用方统一编号）。"""
    chunks: list[dict] = []
    for title, body in _split_sections(md_text):
        full = f"{title}\n{body}".strip() if title else body
        for piece in _split_long(full, chunk_size, overlap):
            chunks.append({
                "text": piece,
                "source": source,
                "section": title,
                "url": url,
                "chunk_size": chunk_size,
            })
    return chunks


def _source_to_url(source: str) -> str:
    """把仓库内相对路径映射为文档站 URL（语料由 config 决定）。

    例：fastapi-master/docs/zh/docs/tutorial/first-steps.md
        -> https://fastapi.tiangolo.com/zh/tutorial/first-steps/
    """
    meta = config.corpus_meta()
    rel = source.split(meta["doc_subpath"], 1)[-1]
    if rel.endswith("index.md"):
        rel = rel[: -len("index.md")]
    elif rel.endswith(".md"):
        rel = rel[: -len(".md")] + "/"
    return meta["url_prefix"].rstrip("/") + "/" + rel


def ingest(md_files: list[Path], chunk_size: int, overlap: int | None = None) -> list[dict]:
    """对所有 md 文件做切分，返回带全局唯一 chunk_id 的 chunk 列表。"""
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    all_chunks: list[dict] = []
    for path in md_files:
        md_text = path.read_text(encoding="utf-8", errors="ignore")
        # 用相对于 raw 目录的路径作为 source（统一正斜杠，保证跨平台一致）
        source = path.relative_to(config.RAW_DIR).as_posix()
        url = _source_to_url(source)
        all_chunks.extend(chunk_markdown(md_text, source, url, chunk_size, overlap))

    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"c{i:05d}"
    return all_chunks


def save_chunks(chunks: list[dict], chunk_size: int) -> Path:
    """保存 chunks 到 data/chunks/chunks_{size}.jsonl。"""
    config.CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CHUNK_DIR / f"chunks_{chunk_size}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[ingest] 已保存 {len(chunks)} 个 chunk 到 {out}")
    return out


def load_chunks(chunk_size: int) -> list[dict]:
    out = config.CHUNK_DIR / f"chunks_{chunk_size}.jsonl"
    if not out.exists():
        raise FileNotFoundError(f"未找到 {out}，请先运行 ingest")
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
