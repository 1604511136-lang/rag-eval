# RAG 检索问答系统 + 评测基准

[![eval](https://github.com/1604511136-lang/rag-eval/actions/workflows/eval.yml/badge.svg)](https://github.com/1604511136-lang/rag-eval/actions/workflows/eval.yml)

一个基于 **FastAPI 官方文档（中/英两套语料）** 的检索增强生成（RAG）问答系统。重点不在「能回答问题」，而在于一套**可量化、可复用**的评测基准：对比不同切分大小与检索策略下的 `Recall@k / nDCG@k / MRR`，用 LLM-as-judge 评估答案忠实度与正确性，并把这套评测做成了网页看板。

> 一句话能写进简历：*「构建 FastAPI 文档检索问答系统，搭建含 Recall@k/MRR/nDCG 与 LLM-judge 的可复用评测基准，在中文/英文两套语料上量化对比 hybrid+rerank 相比 BM25 基线 MRR 提升 15.8%、recall@1 提升 16.2%」。*

## ✨ 特性

### 检索与生成
- **检索**：BM25（词法基线）/ Dense（向量余弦相似度）/ Hybrid（RRF 融合）三种策略
- **重排序**：cross-encoder（bge-reranker-v2-m3）对 hybrid 召回的 top-N 候选精排，提升 MRR / recall@1
- **查询改写**：检索前用 LLM 把口语化问题改写成关键词密集的检索查询（HyDE 的低成本替代），重排与生成仍用原始问题
- **多轮对话**：按 session 保留最近若干轮上下文，结合历史把「它呢？」这类指代追问解析成独立查询再检索，生成时附带对话历史
- **切分**：按 markdown 标题切分，支持 `256 / 512 / 1024` 三种 chunk 大小与重叠
- **生成**：带 `[n]` 引用编号的答案，引用可跳转官方文档

### 评测基准（核心）
- 检索指标：`Recall@k`、`nDCG@k`、`MRR`（客观、免 LLM）
- 答案质量：LLM-as-judge 的 faithfulness / correctness（1-5 分）
- 自动生成 `report.md` 对比表 + 结论 + matplotlib 图表

### 多语料可复用
- 语料注册表（`CORPORA`）一键切换，已支持 FastAPI 中/英文文档；新增语料（Vue / Django）只需加一条配置

### Web 应用（PWA，可装到手机）
- **问答**：检索 + 生成 + 引用来源 + 自动联动 **B 站相关讲解视频**
- **语义检索**：只返回相关段落、不生成答案
- **评测看板**（`/eval.html`）：网页切换语料、看指标表/图表、一键运行评测
- **历史记录**：服务端持久化最近 50 条，点击还原历史答案
- **PWA**：manifest + Service Worker，支持「添加到主屏幕」全屏运行
- 模型全走 **OpenAI-compatible API**，可切换 OpenAI / DeepSeek / 硅基流动 / 智谱

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（默认硅基流动，可换成 OpenAI / DeepSeek 等）
```

### 3. 生成评测 QA 集（半自动，需人工抽查约 20%）

```bash
python scripts/gen_qa.py 50 512   # 生成 50 道题，基于 512-token 语料
```

### 4. 一键跑通全流程

```bash
python scripts/run_all.py            # 首次：下载语料 → 切分 → embedding → 评测 → 报告
python scripts/run_all.py --skip-download   # 之后可跳过下载
python scripts/run_all.py --skip-download --no-rerank  # 跳过重排序评测（省 API 调用）
```

产物在 `data/output/`：`report.md`（对比表 + 结论）与 `retrieval.png`（图表）。

### 5. 换一套语料（证明评测方法可复用）

语料在 `src/config.py` 的 `CORPORA` 注册表里定义，切分 / 检索 / 评测 / 报告全链路自动复用：

```bash
# FastAPI 英文文档（第二套语料，产物落在 data/fastapi-en/）
python scripts/run_all.py --corpus fastapi-en --ingest-only   # 下载 + 切分 + embedding
python scripts/gen_qa.py --corpus fastapi-en 30 512           # 半自动生成 30 道 QA
python scripts/run_all.py --corpus fastapi-en --skip-download --answer-limit 5
```

新增语料（如 Vue / Django 文档）只需在 `CORPORA` 加一条配置，无需改动检索与评测代码。

## 🌐 部署 / 手机访问

本地起 Web 问答界面：`python -m uvicorn app:app`，浏览器打开 http://127.0.0.1:8000。

- **局域网内手机访问**：服务绑定 `0.0.0.0` 后，手机连同一 WiFi 打开 `http://电脑IP:8000`（Windows 双击 [start.bat](start.bat) 即可）。
- **公网部署（手机随时随地、不依赖电脑）**：见 [DEPLOY.md](DEPLOY.md)。

## 🧪 测试

```bash
pip install -r requirements-dev.txt
pytest
```

测试用构造样例验证切分逻辑、检索指标的数学正确性与三种检索策略，**不依赖网络或 API**。

## 🤖 CI 自动评测（GitHub Actions）

[.github/workflows/eval.yml](.github/workflows/eval.yml) 定义了两个任务：

- **test**：每次 push / PR 跑 `pytest`（离线、无需密钥）
- **eval**：跑完整 `run_all.py`，把 `report.md` + `retrieval.png` 作为 artifact 上传；未配置密钥时自动跳过

只需在仓库 **Settings → Secrets and variables → Actions** 添加一个 `OPENAI_API_KEY`（硅基流动等），可选 `OPENAI_BASE_URL`。

## 🧠 原理

1. **检索**：query 与 chunk 分别用 embedding 模型向量化，dense 检索取余弦相似度 top-k；BM25 作为词法基线；hybrid 用 Reciprocal Rank Fusion（RRF）融合两者排序。
2. **重排序**：hybrid 先召回 top-20 候选，再用 cross-encoder（对 query 与文档联合编码）逐个计算相关度精排取 top-10，比双塔 embedding 的余弦相似度排序更准。
3. **评测**：`gold_chunks` 标注每道题对应的标准文档片段，检索结果与之比对计算 Recall@k / nDCG@k / MRR。
4. **答案质量**：检索到的片段 + 问题交给 LLM 生成带引用的答案，再用 LLM 对忠实度、正确性打分。

## 📁 项目结构

```
rag-eval/
├── app.py                # Web 接口：问答 / 检索 / 视频 / 历史 / 评测看板
├── src/
│   ├── config.py         # 全局配置（.env + 语料注册表 CORPORA）
│   ├── llm.py            # OpenAI-compatible 客户端封装
│   ├── ingest.py         # 语料下载 / markdown 切分
│   ├── retrieve.py       # BM25 / dense / hybrid 检索
│   ├── rerank.py         # cross-encoder 重排序
│   ├── rewrite.py        # 检索查询改写（HyDE-lite）
│   ├── generate.py       # RAG 生成（带引用）
│   ├── video.py          # B 站视频搜索（防盗链降级）
│   ├── eval_retrieval.py # Recall@k / nDCG@k / MRR
│   ├── eval_answer.py    # LLM-as-judge
│   └── report.py         # 评测编排 + 报告/图表
├── scripts/
│   ├── run_all.py        # 端到端一键脚本（--corpus 切换语料）
│   └── gen_qa.py         # 半自动生成 QA 集
├── static/               # 前端：问答页 + 评测看板 + PWA
├── tests/                # 单元测试
├── data/                 # 语料产物（默认语料 + data/<corpus>/ 子目录，均 gitignore）
└── requirements.txt
```

## 🛠 技术栈

| 层 | 技术 |
| --- | --- |
| 检索 | BM25（rank-bm25）+ 向量余弦相似度（numpy）+ RRF 融合 |
| 重排序 | cross-encoder（bge-reranker-v2-m3，走 `/rerank` 接口） |
| Embedding / LLM | OpenAI-compatible API（BGE-M3 / DeepSeek / Qwen 等） |
| 评测 | Recall@k / nDCG@k / MRR + LLM-as-judge |
| 报告 | matplotlib |
| Web | FastAPI + 原生 JS（无前端框架） |
| 视频 / 历史 | B 站搜索 API（buvid3 + 防盗链降级）+ JSONL 持久化 |
| 前端 | PWA（manifest + Service Worker） |

## 📄 License

[MIT](LICENSE)
