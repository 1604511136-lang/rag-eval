"""全局配置：从 .env 读取 API 与模型信息，以及评测相关常量。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 固定加载项目根目录下的 .env，避免受运行时 cwd 影响
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---- API / 模型 ----
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-ai/DeepSeek-V3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "1") == "1"  # 检索前是否用 LLM 改写查询

# ---- 评测超参数 ----
CHUNK_SIZES = (256, 512, 1024)   # 三种切分大小（按字符数近似 token）
CHUNK_OVERLAP = 50
TOP_K = 10                       # 检索返回的 top-k 上限
RRF_K = 60                       # hybrid 融合 RRF 的平滑常数
K_VALUES = (1, 3, 5, 10)         # 评测用的 k
RERANK_CANDIDATE_K = 20          # 重排序前召回候选数（先召回 20，再精排取 top-10）

# ---- 语料注册表 ----
# 每个语料定义「下载地址、tarball 顶层目录、文档子路径、文档站 URL 前缀」。
# 新增语料只需在这里加一条，切分 / 检索 / 评测 / 报告全链路即可复用。
CORPORA = {
    "fastapi-zh": {
        "label": "FastAPI 中文文档",
        "tarball_url": "https://codeload.github.com/fastapi/fastapi/tar.gz/refs/heads/master",
        "tar_top": "fastapi-master",
        "doc_subpath": "docs/zh/docs/",
        "url_prefix": "https://fastapi.tiangolo.com/zh/",
    },
    "fastapi-en": {
        "label": "FastAPI 英文文档",
        "tarball_url": "https://codeload.github.com/fastapi/fastapi/tar.gz/refs/heads/master",
        "tar_top": "fastapi-master",
        "doc_subpath": "docs/en/docs/",
        "url_prefix": "https://fastapi.tiangolo.com/",
    },
}

CORPUS_NAME = os.getenv("CORPUS", "fastapi-zh")


# ---- 路径 ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _dirs_for(name: str) -> tuple[Path, Path, Path, Path]:
    """默认语料沿用历史路径（data/{raw,chunks,benchmark,output}），
    其余语料放入 data/<name>/ 子目录，互不干扰。"""
    if name == "fastapi-zh":
        base = DATA_DIR
        return base / "raw", base / "chunks", base / "benchmark", base / "output"
    base = DATA_DIR / name
    return base / "raw", base / "chunks", base / "benchmark", base / "output"


RAW_DIR, CHUNK_DIR, BENCHMARK_DIR, OUTPUT_DIR = _dirs_for(CORPUS_NAME)


def corpus_meta(name: str | None = None) -> dict:
    n = name or CORPUS_NAME
    if n not in CORPORA:
        raise ValueError(f"未知语料: {n}，可选 {sorted(CORPORA)}")
    return CORPORA[n]


def use_corpus(name: str) -> None:
    """切换到指定语料（重定位数据目录）。须在任何加载 chunk / embedding 之前调用。"""
    global CORPUS_NAME, RAW_DIR, CHUNK_DIR, BENCHMARK_DIR, OUTPUT_DIR
    corpus_meta(name)  # 校验名称
    CORPUS_NAME = name
    RAW_DIR, CHUNK_DIR, BENCHMARK_DIR, OUTPUT_DIR = _dirs_for(name)


def output_dir(name: str | None = None) -> Path:
    """某语料的报告产物目录（data/<name>/output）。"""
    return _dirs_for(name or CORPUS_NAME)[3]


def chroma_dir() -> Path:
    """向量库（Chroma）持久化目录，按语料隔离（data/<corpus>/chunks/chroma）。"""
    return CHUNK_DIR / "chroma"
