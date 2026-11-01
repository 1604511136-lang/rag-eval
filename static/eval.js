"use strict";

const corpusTabs = document.getElementById("corpusTabs");
const reportEl = document.getElementById("report");
const runBtn = document.getElementById("runBtn");
const runStatus = document.getElementById("runStatus");

let currentCorpus = null;
let pollTimer = null;

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(s) {
  return escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderTable(rows) {
  const cells = rows.map((r) =>
    r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim())
  );
  const header = cells[0];
  const body = cells.slice(2); // 跳过 |---|---| 分隔行
  return (
    "<table><thead><tr>" +
    header.map((h) => `<th>${inline(h)}</th>`).join("") +
    "</tr></thead><tbody>" +
    body
      .map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>")
      .join("") +
    "</tbody></table>"
  );
}

// 针对本报告格式的轻量 markdown 渲染（标题 / 表格 / 列表 / 图片 / 段落）
function mdToHtml(md) {
  const lines = md.split("\n");
  let html = "";
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i]);
        i++;
      }
      html += renderTable(rows);
      continue;
    }
    if (line.startsWith("### ")) { html += `<h3>${inline(line.slice(4))}</h3>`; i++; continue; }
    if (line.startsWith("## ")) { html += `<h2>${inline(line.slice(3))}</h2>`; i++; continue; }
    if (line.startsWith("# ")) { html += `<h1>${inline(line.slice(2))}</h1>`; i++; continue; }
    if (/^\s*-\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*-\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*-\s+/, ""));
        i++;
      }
      html += `<ul>${items.map((x) => `<li>${inline(x)}</li>`).join("")}</ul>`;
      continue;
    }
    if (/^!\[/.test(line)) {
      html += `<img class="chart" src="/api/eval/chart?corpus=${encodeURIComponent(currentCorpus)}" alt="检索指标对比" />`;
      i++;
      continue;
    }
    if (line.trim() === "") { i++; continue; }
    html += `<p>${inline(line)}</p>`;
    i++;
  }
  return html;
}

async function loadReport() {
  if (!currentCorpus) return;
  reportEl.innerHTML = `<div class="spinner">加载评测报告…</div>`;
  try {
    const resp = await fetch(`/api/eval/report?corpus=${encodeURIComponent(currentCorpus)}`);
    const data = await resp.json();
    if (!data.available) {
      reportEl.innerHTML =
        `<div class="card"><p class="empty">暂无评测报告，点击上方「运行评测」生成。</p></div>`;
      return;
    }
    reportEl.innerHTML = `<div class="card report-card">${mdToHtml(data.markdown)}</div>`;
  } catch (err) {
    reportEl.innerHTML = `<div class="error">加载失败：${escapeHtml(err.message)}</div>`;
  }
}

async function loadCorpora() {
  const resp = await fetch("/api/eval/corpora");
  const { corpora } = await resp.json();
  corpusTabs.innerHTML = corpora
    .map(
      (c) =>
        `<button class="corpus-tab${c.name === currentCorpus ? " active" : ""}" data-corpus="${c.name}">` +
        `${c.label}${c.has_report ? "" : "（未评测）"}</button>`
    )
    .join("");
  if (!currentCorpus && corpora.length) {
    currentCorpus = corpora[0].name;
    loadReport();
  }
}

corpusTabs.addEventListener("click", (e) => {
  const tab = e.target.closest(".corpus-tab");
  if (!tab) return;
  currentCorpus = tab.dataset.corpus;
  loadCorpora().then(loadReport);
});

runBtn.addEventListener("click", async () => {
  if (!currentCorpus) return;
  runBtn.disabled = true;
  runStatus.hidden = false;
  runStatus.textContent = "评测运行中，请稍候…";
  try {
    await fetch(`/api/eval/run?corpus=${encodeURIComponent(currentCorpus)}`, { method: "POST" });
  } catch (err) {
    runStatus.textContent = "启动失败：" + err.message;
    runBtn.disabled = false;
    return;
  }
  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/eval/status?corpus=${encodeURIComponent(currentCorpus)}`);
      const { running } = await resp.json();
      if (!running) {
        clearInterval(pollTimer);
        runStatus.hidden = true;
        runBtn.disabled = false;
        loadCorpora().then(loadReport);
      }
    } catch (err) {
      clearInterval(pollTimer);
      runStatus.textContent = "查询状态失败：" + err.message;
      runBtn.disabled = false;
    }
  }, 3000);
});

loadCorpora();
