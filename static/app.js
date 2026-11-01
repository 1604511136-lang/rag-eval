"use strict";

const form = document.getElementById("form");
const input = document.getElementById("input");
const submitBtn = document.getElementById("submit");
const result = document.getElementById("result");
const chips = document.getElementById("chips");
const historyEl = document.getElementById("history");
const historyList = document.getElementById("historyList");
const clearBtn = document.getElementById("clearHistory");

// 每次页面加载生成一个会话 ID，后端据此保留多轮上下文
const sessionId = window.crypto && crypto.randomUUID
  ? crypto.randomUUID()
  : Math.random().toString(36).slice(2);

// 轻量 markdown 渲染：处理代码块 / 行内代码 / 加粗 / 引用编号，其余按纯文本
function mdToHtml(text) {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // 先抽出代码（块 + 行内），避免后续规则误改其中的 [n]、** 等
  const stash = [];
  html = html.replace(/```(\w*)\n?([\s\S]*?)```|`([^`]+)`/g, (m, lang, block, inline) => {
    const code = block !== undefined ? block : inline;
    stash.push(
      block !== undefined
        ? `<pre><code>${code}</code></pre>`
        : `<code>${code}</code>`
    );
    return `\u0000${stash.length - 1}\u0000`;
  });

  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[(\d+)\]/g, '<sup class="cite">$1</sup>');
  html = html.replace(/\n/g, "<br>");

  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[+i]);
}

function sourceList(sources) {
  return (sources || [])
    .map((s, i) => {
      const title = escapeHtml(s.title || s.url);
      const snippet = escapeHtml((s.text || "").replace(/<[^>]*>/g, ""));
      return (
        `<li class="source">` +
        `<span class="num">${i + 1}</span>` +
        `<span class="source-body">` +
        `<a href="${s.url}" target="_blank" rel="noopener noreferrer">${title}</a>` +
        `<span class="snippet">${snippet}</span>` +
        `</span>` +
        `</li>`
      );
    })
    .join("");
}

function render(data) {
  result.innerHTML =
    `<div class="card q-card">` +
    `<p class="q">${escapeHtml(data.question)}</p>` +
    `<div class="answer">${mdToHtml(data.answer)}</div>` +
    `</div>` +
    `<div class="card">` +
    `<p class="sources-title">参考来源</p>` +
    `<ul class="sources">${sourceList(data.sources)}</ul>` +
    `</div>` +
    `<div class="card">` +
    `<p class="sources-title">相关视频讲解</p>` +
    `<div id="videoList"><div class="spinner">正在B站找视频…</div></div>` +
    `</div>`;
  result.hidden = false;
  loadVideos(data.question);
}

async function loadVideos(question) {
  const box = document.getElementById("videoList");
  if (!box) return;
  try {
    const resp = await fetch(`/api/videos?q=${encodeURIComponent(question)}`);
    const data = await resp.json();
    if (data.fallback || !(data.videos || []).length) {
      box.innerHTML =
        `<a class="video-fallback" href="${data.search_url}" target="_blank" rel="noopener noreferrer">在B站搜索相关讲解 →</a>`;
      return;
    }
    box.innerHTML =
      `<ul class="video-list">` +
      data.videos
        .map(
          (v) =>
            `<li class="video-item">` +
            (v.cover ? `<img src="${v.cover}" alt="" loading="lazy" referrerpolicy="no-referrer" />` : "") +
            `<span class="v-meta">` +
            `<a href="${v.url}" target="_blank" rel="noopener noreferrer">${escapeHtml(v.title)}</a>` +
            `<div class="v-sub">${escapeHtml(v.author)} · ${v.play ? v.play + " 播放" : ""}</div>` +
            `</span>` +
            `</li>`
        )
        .join("") +
      `</ul>`;
  } catch (err) {
    box.innerHTML = "";
  }
}

function renderSearch(data) {
  const n = (data.results || []).length;
  result.innerHTML =
    `<div class="card q-card">` +
    `<p class="q">${escapeHtml(data.query)}</p>` +
    `<p class="search-meta">检索到 ${n} 条相关段落` +
    (data.rewritten_query && data.rewritten_query !== data.query
      ? " · 已优化检索词：" + escapeHtml(data.rewritten_query)
      : "") +
    `</p>` +
    `</div>` +
    `<div class="card">` +
    `<p class="sources-title">相关段落</p>` +
    `<ul class="sources">${sourceList(data.results)}</ul>` +
    `</div>`;
  result.hidden = false;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function ask(question) {
  question = question.trim();
  if (!question) return;

  input.value = question;
  submitBtn.disabled = true;
  submitBtn.textContent = "提问中…";
  result.hidden = false;
  result.innerHTML =
    `<div class="card q-card">` +
    `<p class="q">${escapeHtml(question)}</p>` +
    `<div class="answer" id="answerBox"><span class="stream-cursor">▍</span></div>` +
    `</div>` +
    `<div class="card">` +
    `<p class="sources-title">参考来源</p>` +
    `<p class="rewrite-hint" id="rewriteHint" hidden></p>` +
    `<ul class="sources" id="sourceBox"></ul>` +
    `</div>` +
    `<div class="card">` +
    `<p class="sources-title">相关视频讲解</p>` +
    `<div id="videoList"><div class="spinner">正在B站找视频…</div></div>` +
    `</div>`;

  const answerBox = document.getElementById("answerBox");
  const sourceBox = document.getElementById("sourceBox");

  try {
    const resp = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `请求失败（${resp.status}）`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let sources = [];
    let answerText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.type === "sources") {
          sources = payload.sources;
          sourceBox.innerHTML = sourceList(sources);
          const hint = document.getElementById("rewriteHint");
          if (payload.rewritten_query && payload.rewritten_query !== question && hint) {
            hint.textContent = "已优化检索词：" + payload.rewritten_query;
            hint.hidden = false;
          }
        } else if (payload.type === "token") {
          answerText += payload.text;
          answerBox.innerHTML = mdToHtml(answerText) + `<span class="stream-cursor">▍</span>`;
        } else if (payload.type === "error") {
          throw new Error(payload.detail);
        }
      }
    }
    answerBox.innerHTML = mdToHtml(answerText);
    loadHistory();
    loadVideos(question);
  } catch (err) {
    result.innerHTML = `<div class="error">出错了：${escapeHtml(err.message)}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "提问";
  }
}

// 语义检索：只返回相关段落，不生成答案
async function search(query) {
  query = query.trim();
  if (!query) return;
  input.value = query;
  submitBtn.disabled = true;
  submitBtn.textContent = "检索中…";
  result.hidden = false;
  result.innerHTML = `<div class="spinner">正在检索相关段落…</div>`;
  try {
    const resp = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `请求失败（${resp.status}）`);
    renderSearch(data);
  } catch (err) {
    result.innerHTML = `<div class="error">出错了：${escapeHtml(err.message)}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "检索";
  }
}

let mode = "ask";

function setMode(m) {
  mode = m;
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === m)
  );
  submitBtn.textContent = m === "search" ? "检索" : "提问";
  input.placeholder =
    m === "search" ? "输入关键词，检索相关文档段落…" : "输入你的问题…";
}

document.getElementById("mode").addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-btn");
  if (btn) setMode(btn.dataset.mode);
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (mode === "search") search(input.value);
  else ask(input.value);
});

chips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) ask(chip.textContent.trim());
});

input.focus();

// ---- 历史记录 ----
let historyCache = [];

async function loadHistory() {
  try {
    const resp = await fetch("/api/history");
    const { items } = await resp.json();
    historyCache = items || [];
    if (!historyCache.length) {
      historyEl.hidden = true;
      return;
    }
    historyEl.hidden = false;
    historyList.innerHTML = historyCache
      .map(
        (it, i) =>
          `<li class="history-item">` +
          `<button class="history-q" type="button">${escapeHtml(it.question)}</button>` +
          `<span class="history-ts">${escapeHtml(it.ts || "")}</span>` +
          `<span class="history-idx" hidden>${i}</span>` +
          `</li>`
      )
      .join("");
  } catch (err) {
    /* 忽略 */
  }
}

function restoreHistory(i) {
  const it = historyCache[i];
  if (!it) return;
  input.value = it.question;
  result.innerHTML =
    `<div class="card q-card">` +
    `<p class="q">${escapeHtml(it.question)}</p>` +
    `<div class="answer">${mdToHtml(it.answer)}</div>` +
    `</div>` +
    `<div class="card">` +
    `<p class="sources-title">参考来源</p>` +
    `<ul class="sources">${sourceList(it.sources)}</ul>` +
    `</div>`;
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth" });
}

historyList.addEventListener("click", (e) => {
  const item = e.target.closest(".history-item");
  if (!item) return;
  restoreHistory(+item.querySelector(".history-idx").textContent);
});

clearBtn.addEventListener("click", async () => {
  try {
    await fetch("/api/history", { method: "DELETE" });
  } catch (err) {
    /* 忽略 */
  }
  historyCache = [];
  historyEl.hidden = true;
});

loadHistory();

// 注册 Service Worker（仅 HTTPS / localhost 生效；HTTP 下静默跳过，用于 PWA 离线缓存）
if ("serviceWorker" in navigator &&
    (location.protocol === "https:" || location.hostname === "localhost")) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
