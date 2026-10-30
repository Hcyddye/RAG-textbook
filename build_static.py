# -*- coding: utf-8 -*-
"""
生成「可双击打开的离线演示页」。

做三件事：
1) 用已建好的索引，对 6 个示例问题真实跑一遍检索，记录命中块与相似度、耗时；
2) 读取 qa_results.json 里已经生成好的答案；
3) 把数据内嵌进单文件 HTML，双击即可打开，不需要 Python 服务、不联网。

运行：.venv\\Scripts\\python.exe build_static.py
"""
import html
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "rag_index"
MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
QA_FILE = ROOT / "qa_results.json"
# 同一份内容写两个名字：中文名给本地双击，index.html 给 GitHub Pages 当首页
OUT_FILES = [ROOT / "RAG界面.html", ROOT / "index.html"]

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5


def md_to_html(text: str) -> str:
    """把答案的轻量 markdown（加粗/行内码/有序无序列表）转成 HTML 片段。"""

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    out, buf, mode = [], [], None

    def flush():
        nonlocal buf, mode
        if not buf:
            return
        if mode == "ul":
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ul>")
        elif mode == "ol":
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ol>")
        else:
            out.append("<p>" + "<br>".join(inline(x) for x in buf) + "</p>")
        buf, mode = [], None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # 列表项之间常有空行，此时不能断开列表，否则每项都会重新从 1 编号
            if mode is None:
                flush()
            continue
        m_ul = re.match(r"^[-*]\s+(.*)$", line)
        m_ol = re.match(r"^\d+[.)]\s*(.*)$", line)
        if m_ul:
            if mode not in (None, "ul"):
                flush()
            mode = "ul"
            buf.append(m_ul.group(1))
        elif m_ol:
            if mode not in (None, "ol"):
                flush()
            mode = "ol"
            buf.append(m_ol.group(1))
        else:
            if mode is not None:
                flush()
            buf.append(line)
    flush()
    return "\n".join(out)


def main() -> None:
    print("[1/3] 读取答案与索引 …")
    qa = json.loads(QA_FILE.read_text(encoding="utf-8"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_DIR / "nlp_book_flat.faiss"))
    print(f"      {len(chunks):,} 块 · 索引 {index.ntotal:,} × {index.d} 维 · 问题 {len(qa)} 个")

    print("[2/3] 加载嵌入模型并检索 …")
    embedder = SentenceTransformer(
        str(MODEL_DIR) if MODEL_DIR.exists() else "BAAI/bge-small-zh-v1.5", device="cpu"
    )

    records = []
    for item in qa:
        q = item["question"]
        t0 = time.perf_counter()
        vec = np.asarray(
            embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True), dtype="float32"
        )
        scores, ids = index.search(vec, TOP_K)
        ms = (time.perf_counter() - t0) * 1000

        hits = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = chunks[int(i)]
            body = c["text"].replace("\n", " ").strip()
            # 切分点可能落在句子中间，去掉开头的孤立标点，读起来更干净
            body = re.sub(r"^[，。、；：,.;:！？!?%·）)\]】」』\s]+", "", body)
            hits.append(
                {
                    "s": round(float(s), 4),
                    "ch": c["chapter"],
                    "sec": html.escape(str(c["section"])),
                    "ps": c["page_start"],
                    "pe": c["page_end"],
                    "t": html.escape(body[:240]) + (" …" if len(body) > 240 else ""),
                }
            )
        records.append(
            {"q": q, "a": md_to_html(item["answer"]), "ms": round(ms), "hits": hits}
        )
        print(f"      {q[:26]:<28} {ms:6.1f} ms  Top1={scores[0][0]:.3f}")

    print("[3/3] 套模板生成 HTML …")
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    page = TEMPLATE.replace("__DATA__", payload)
    for out in OUT_FILES:
        out.write_text(page, encoding="utf-8")
        print(f"      已写出 {out.name}  {out.stat().st_size / 1024:.0f} KB")
    print("      双击 RAG界面.html 即可打开；index.html 用于 GitHub Pages 首页")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>《自然语言处理导论》知识问答 · 演示</title>
<style>
:root{
  --bg:#f5f6f8; --card:#ffffff; --line:#e5e7eb; --line-soft:#eef0f2;
  --ink:#111827; --ink-2:#374151; --ink-3:#6b7280; --ink-4:#9ca3af;
  --accent:#2563eb; --accent-soft:#eff6ff; --accent-line:#dbeafe;
  --radius:12px; --shadow:0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0f1115; --card:#171a21; --line:#272c36; --line-soft:#20242c;
    --ink:#f3f4f6; --ink-2:#d4d8de; --ink-3:#9aa3af; --ink-4:#6f7783;
    --accent:#7aa7f7; --accent-soft:#18202e; --accent-line:#2b3a52;
    --shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",
              "Hiragino Sans GB","Source Han Sans SC",sans-serif;
  -webkit-font-smoothing:antialiased; line-height:1.6;
}
.wrap{max-width:1240px; margin:0 auto; padding:24px 24px 40px;}

/* 页头 */
.hdr{padding:2px 2px 18px; border-bottom:1px solid var(--line); margin-bottom:20px;}
.titlerow{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;}
.hdr h1{margin:0; font-size:21px; font-weight:600; letter-spacing:-.2px;}
.tag{font-size:11.5px; color:var(--accent); background:var(--accent-soft);
     border:1px solid var(--accent-line); border-radius:999px; padding:2px 9px; white-space:nowrap;}
.sub{margin:8px 0 0; font-size:13px; color:var(--ink-3); line-height:1.75; max-width:800px;}
.sub code{font-size:12px; color:var(--accent); background:var(--accent-soft);
          border:1px solid var(--accent-line); border-radius:4px; padding:0 4px;}
.chips{margin-top:13px; display:flex; flex-wrap:wrap; gap:6px;}
.chip{font-size:11.5px; color:var(--ink-3); background:var(--card);
      border:1px solid var(--line); border-radius:6px; padding:3px 9px; white-space:nowrap;}
.notice{margin-top:14px; padding:10px 13px; font-size:12.5px; line-height:1.7;
        color:var(--ink-2); background:var(--accent-soft);
        border:1px solid var(--accent-line); border-radius:9px;}
.notice b{color:var(--accent); font-weight:600;}

/* 布局 */
.grid{display:grid; grid-template-columns:360px minmax(0,1fr); gap:18px; align-items:start;}
.card{background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
      box-shadow:var(--shadow);}
.left{padding:16px 16px 14px; position:sticky; top:16px;}
.lb{display:block; font-size:12.5px; font-weight:500; color:var(--ink-2); margin-bottom:7px;}
textarea{width:100%; resize:vertical; font:inherit; font-size:14px; line-height:1.75;
         padding:11px 12px; border-radius:9px; color:var(--ink); background:var(--card);
         border:1px solid var(--line); outline:none; transition:border-color .15s, box-shadow .15s;}
textarea::placeholder{color:var(--ink-4);}
textarea:focus{border-color:var(--accent); box-shadow:0 0 0 3px rgba(37,99,235,.09);}
.btnrow{display:flex; gap:8px; margin-top:10px;}
button{font:inherit; font-size:14px; font-weight:500; min-height:40px; border-radius:9px;
       cursor:pointer; transition:all .15s;}
.primary{flex:3; border:1px solid var(--accent); background:var(--accent); color:#fff;}
.primary:hover{filter:brightness(1.07);}
.ghost{flex:1; border:1px solid var(--line); background:transparent; color:var(--ink-2); font-size:13px;}
.ghost:hover{border-color:var(--accent); color:var(--accent);}
.exlabel{margin:18px 0 8px; font-size:12.5px; font-weight:500; color:var(--ink-2);}
.exlist{display:flex; flex-direction:column; gap:6px;}
.ex{text-align:left; font-size:13px; line-height:1.6; color:var(--ink-2); min-height:0;
    padding:9px 11px; background:var(--bg); border:1px solid var(--line);
    border-radius:8px; cursor:pointer; transition:all .15s;}
.ex:hover{border-color:var(--accent-line); background:var(--accent-soft); color:var(--accent);}
.ex.on{border-color:var(--accent); background:var(--accent-soft); color:var(--accent); font-weight:500;}
.tip{margin-top:12px; font-size:11.5px; color:var(--ink-4); line-height:1.7;}

/* 右栏 */
.right{display:flex; flex-direction:column; gap:14px; min-width:0;}
.ans{padding:16px 20px 16px;}
.status{font-size:11.5px; color:var(--ink-4); font-variant-numeric:tabular-nums;
        margin:0 0 12px; padding-bottom:9px; border-bottom:1px solid var(--line-soft);}
.answer{font-size:14.5px; line-height:1.9; color:var(--ink-2); min-height:190px;}
.answer h5{font-size:14px; font-weight:600; color:var(--ink-4); margin:2px 0 10px;}
.answer p{margin:0 0 10px;}
.answer ul,.answer ol{padding-left:22px; margin:6px 0 12px;}
.answer li{margin-bottom:6px;}
.answer strong{color:var(--ink); font-weight:600;}
.answer code{font-size:12.5px; background:var(--accent-soft); color:var(--accent);
             padding:1px 5px; border-radius:4px;}
.evbox{padding:14px 18px 6px;}
.evtitle{font-size:13px; font-weight:500; color:var(--ink-2); margin-bottom:10px;}
.evscroll{max-height:460px; overflow-y:auto; padding-right:6px;}
.evscroll::-webkit-scrollbar{width:8px;}
.evscroll::-webkit-scrollbar-thumb{background:var(--line); border-radius:4px;}
.ev-lead{font-size:11.5px; color:var(--ink-4); margin:2px 0 10px;}
.ev{border:1px solid var(--line); border-radius:10px; padding:11px 13px;
    margin-bottom:9px; transition:border-color .15s;}
.ev:hover{border-color:var(--accent-line);}
.ev-weak{opacity:.88;}
.ev-head{display:flex; align-items:center; gap:9px; margin-bottom:7px; flex-wrap:wrap;}
.ev-rank{flex:none; width:20px; height:20px; border-radius:6px; background:var(--accent-soft);
         color:var(--accent); border:1px solid var(--accent-line); font-size:11.5px;
         font-weight:600; display:inline-flex; align-items:center; justify-content:center;}
.ev-meta{font-size:12px; color:var(--ink-3); flex:1; min-width:0;
         overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.ev-score{font-size:11.5px; color:var(--ink-4); font-variant-numeric:tabular-nums;}
.ev-bar{height:3px; background:var(--line-soft); border-radius:2px; overflow:hidden; margin-bottom:8px;}
.ev-bar i{display:block; height:100%; background:var(--accent); border-radius:2px; opacity:.55;}
.ev-strong .ev-bar i{opacity:.85;}
.ev-weak .ev-bar i{opacity:.3;}
.ev-text{font-size:13px; color:var(--ink-2); line-height:1.78;}
.ev-empty{font-size:12.5px; color:var(--ink-4); padding:14px 2px 18px;}

footer{margin-top:18px; padding-top:13px; border-top:1px solid var(--line);
       font-size:11.5px; color:var(--ink-4); line-height:1.85;}

@media (max-width: 900px){
  .wrap{padding:16px 14px 30px;}
  .grid{grid-template-columns:1fr;}
  .left{position:static;}
}
</style>
</head>
<body>
<div class="wrap">

  <header class="hdr">
    <div class="titlerow">
      <h1>《自然语言处理导论》知识问答</h1>
      <span class="tag">RAG · 答案可溯源到页码</span>
    </div>
    <p class="sub">从 604 页教材中检索相关段落，由大模型依据检索到的原文作答；
      每条结论标注 <code>[编号]</code>，资料不足时明确拒答，不编造。</p>
    <div class="chips">
      <span class="chip">604 页 PDF · 正文 506 页</span>
      <span class="chip">1,700 个文本块</span>
      <span class="chip">bge-small-zh · 512 维</span>
      <span class="chip">FAISS IndexFlatIP</span>
      <span class="chip">Hit@5 100% · MRR 0.825</span>
      <span class="chip">端到端 1.9–3.1 s</span>
    </div>
    <div class="notice">
      本页是<b>预生成的静态演示</b>：6 组问答都是真实跑完检索与生成后存档的结果，
      检索耗时、相似度、页码均为实测值；不调用 API、不需要后端，所以打开即看。
      想要自由提问，按 README 在本地启动服务即可。
    </div>
  </header>

  <main class="grid">
    <aside class="card left">
      <label class="lb" for="q">你的问题</label>
      <textarea id="q" rows="4" placeholder="点下方示例载入，或输入一个已收录的问题…"></textarea>
      <div class="btnrow">
        <button id="ask" class="primary">提问</button>
        <button id="reset" class="ghost">清空</button>
      </div>
      <div class="exlabel">示例问题（点击直接看结果）</div>
      <div id="exlist" class="exlist"></div>
      <div class="tip">离线版只内置这 6 组已存档结果；输入其他问题会提示未收录。</div>
    </aside>

    <section class="right">
      <div class="card ans">
        <div id="status" class="status">等待提问</div>
        <div id="answer" class="answer">
          <h5>使用说明</h5>
          <p>点左侧任意一个问题，右栏会显示<b>模型生成的答案</b>与<b>它的出处</b>。</p>
          <p>答案里的 [1] [2] 对应下方证据卡的编号，可以逐条核查到具体章节与页码；
             如果教材里确实没写，模型会明确说「资料中没有相关内容」，而不是编一个。</p>
        </div>
      </div>
      <div class="card evbox">
        <div class="evtitle">检索证据 · 答案的出处</div>
        <div id="evidence" class="evscroll">
          <div class="ev-empty">提问后，这里显示命中的文本块、相似度与页码</div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    链路：问题 → bge 向量化 → FAISS 检索 Top-K → 拼装带编号与页码的 Prompt → DeepSeek 生成。
    证据卡右侧数字为余弦相似度，条形长度按相似度缩放；相似度偏低（&lt; 0.50）的块可能只是字面相近，
    模型若判断资料不相关会明确拒答。
  </footer>
</div>

<script>
const DATA = __DATA__;

const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").replace(/[\s，。？！、；：,.?!;:「」“”"'（）()]/g, "").toLowerCase();

let current = -1;

function evHTML(hits){
  let s = `<div class="ev-lead">共 ${hits.length} 条，按相似度降序 · 编号与答案中的 [n] 一一对应</div>`;
  hits.forEach((h, i) => {
    const tone = h.s >= 0.60 ? "strong" : h.s >= 0.50 ? "mid" : "weak";
    const pct = Math.max(6, Math.min(100, (h.s - 0.35) / 0.45 * 100));
    s += `<div class="ev ev-${tone}">
      <div class="ev-head">
        <span class="ev-rank">${i + 1}</span>
        <span class="ev-meta">第 ${h.ch} 章 · ${h.sec} · p.${h.ps}–${h.pe}</span>
        <span class="ev-score">${h.s.toFixed(3)}</span>
      </div>
      <div class="ev-bar"><i style="width:${pct.toFixed(0)}%"></i></div>
      <div class="ev-text">${h.t}</div>
    </div>`;
  });
  return s;
}

function render(i){
  const d = DATA[i];
  current = i;
  $("q").value = d.q;
  $("status").textContent = `检索 Top-${d.hits.length} · ${d.ms} ms · 生成完成 · 答案 ${(d.a.replace(/<[^>]+>/g, "")).length} 字`;
  $("answer").innerHTML = d.a;
  $("evidence").innerHTML = evHTML(d.hits);
  document.querySelectorAll(".ex").forEach((el, k) => el.classList.toggle("on", k === i));
  $("evidence").scrollTop = 0;
}

function lookup(text){
  const n = norm(text);
  if (!n) return -1;
  for (let i = 0; i < DATA.length; i++) if (norm(DATA[i].q) === n) return i;
  for (let i = 0; i < DATA.length; i++) {
    const k = norm(DATA[i].q);
    if (n.length >= 6 && (k.includes(n) || n.includes(k))) return i;
  }
  return -1;
}

function ask(){
  const i = lookup($("q").value);
  if (i >= 0) { render(i); return; }
  current = -1;
  $("status").textContent = "未收录该问题（离线版）";
  $("answer").innerHTML =
    `<h5>这个页面是离线存档版</h5>
     <p>它只保存了 6 组真实问答结果，不能对任意新问题做检索与生成——那需要有后端服务在跑。</p>
     <p>想看完整交互：双击项目目录里的 <code>启动演示.bat</code>，等浏览器自动打开
        <code>127.0.0.1:7860</code>，左侧输入框就能问任意问题（每次提问会调用一次 DeepSeek API）。</p>
     <p>或者直接点左边任一个示例问题，看这条链路跑出来的答案和证据。</p>`;
  $("evidence").innerHTML = `<div class="ev-empty">未检索：离线页面没有向量索引，这里只展示已存档的证据。</div>`;
  document.querySelectorAll(".ex").forEach((el) => el.classList.remove("on"));
}

$("exlist").innerHTML = DATA.map((d, i) =>
  `<button class="ex" data-i="${i}">${d.q}</button>`).join("");
document.querySelectorAll(".ex").forEach((el) => {
  el.addEventListener("click", () => render(Number(el.dataset.i)));
});
$("ask").addEventListener("click", ask);
$("reset").addEventListener("click", () => {
  current = -1;
  $("q").value = "";
  $("status").textContent = "等待提问";
  $("answer").innerHTML = `<h5>使用说明</h5>
    <p>点左侧任意一个问题，右栏会显示<b>模型生成的答案</b>与<b>它的出处</b>。</p>
    <p>答案里的 [1] [2] 对应下方证据卡的编号，可以逐条核查到具体章节与页码；
       如果教材里确实没写，模型会明确说「资料中没有相关内容」，而不是编一个。</p>`;
  $("evidence").innerHTML = `<div class="ev-empty">提问后，这里显示命中的文本块、相似度与页码</div>`;
  document.querySelectorAll(".ex").forEach((el) => el.classList.remove("on"));
});
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
});

// 打开即有内容：默认展示第一组问答，避免首屏空白；支持 #q3 定位到第 3 题
const m = (location.hash || "").match(/^#q(\d+)$/);
const start = m ? Math.min(DATA.length - 1, Math.max(0, Number(m[1]) - 1)) : 0;
if (DATA.length) render(start);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
