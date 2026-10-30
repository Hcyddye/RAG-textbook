# -*- coding: utf-8 -*-
"""
RAG 教材知识问答 · 交互式演示（答辩用）

直接复用 rag_index/ 里已建好的索引，不重新解析 PDF。
启动：  .venv\\Scripts\\python.exe demo_app.py
"""
import html
import os
import time
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "rag_index"
MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
KEY_FILE = ROOT / "deepseek_api_key.txt"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.1
MAX_CHARS = 1800

SYSTEM_PROMPT = (
    "你是一个严谨的教材知识问答助手，作答依据只有用户提供的【参考资料】。请遵守以下规则：\n"
    "1) 只使用参考资料中的信息作答，不得引入资料之外的知识，不得编造；\n"
    "2) 若参考资料不足以回答问题，请直接说明“提供的资料中没有相关内容”，并指出还缺少什么；\n"
    "3) 回答中的关键结论都用 [编号] 标注出处，编号必须与参考资料一一对应；\n"
    "4) 用中文分点作答：先给结论，再给必要的解释；术语、公式保持与资料一致。"
)

EXAMPLES = [
    "n 元语言模型遇到数据稀疏问题，可以用哪些平滑方法？",
    "CYK 算法是怎么做句法分析的？",
    "BLEU 指标如何计算机器翻译质量？它有什么缺点？",
    "开放领域问答中的「检索-阅读理解」架构是怎么工作的？",
    "知识图谱向量表示中 TransE 的基本思想是什么？",
    "这本书里有没有讲 GPT-4 的参数量是多少？",
]

# ---------------------------------------------------------------- 资源加载
print("[1/3] 加载文本块与向量索引 …")
CHUNKS = __import__("json").loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
INDEX = faiss.read_index(str(INDEX_DIR / "nlp_book_flat.faiss"))
print(f"      {len(CHUNKS):,} 个文本块 · 索引 {INDEX.ntotal:,} 条向量 × {INDEX.d} 维")

print("[2/3] 加载嵌入模型 …")
EMBEDDER = SentenceTransformer(
    str(MODEL_DIR) if MODEL_DIR.exists() else "BAAI/bge-small-zh-v1.5", device="cpu"
)

print("[3/3] 检查大模型 Key …")
# 优先读环境变量（部署场景），其次读本地文件（课程演示场景）
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
if not API_KEY and KEY_FILE.exists():
    API_KEY = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
CLIENT = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL) if API_KEY else None
print("      已配置，问答可正常生成" if CLIENT else "      未配置，将运行在「仅检索」模式")


# ---------------------------------------------------------------- 核心链路
def embed_query(text):
    v = EMBEDDER.encode([QUERY_PREFIX + text], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")


def search(question, top_k=5):
    scores, ids = INDEX.search(embed_query(question), int(top_k))
    return [{**CHUNKS[int(i)], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]


def build_messages(question, hits):
    blocks, used = [], 0
    for i, h in enumerate(hits, 1):
        block = (
            f"[{i}] 《自然语言处理导论》第{h['chapter']}章 {h['section']}，"
            f"p.{h['page_start']}-{h['page_end']}\n{h['text']}"
        )
        if used + len(block) > MAX_CHARS:
            break
        blocks.append(block)
        used += len(block)
    user_prompt = (
        f"【参考资料】\n{chr(10).join(chr(10) * 2 + b for b in blocks)}\n\n"
        f"【问题】\n{question}\n\n"
        f"请严格依据上述参考资料回答，并在关键结论后标注 [编号]。"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}], used


def render_evidence(hits):
    if not hits:
        return "<div class='ev-empty'>没有检索到相关内容</div>"
    out = [
        f"<div class='ev-lead'>共 {len(hits)} 条，按相似度降序 · "
        f"编号与答案中的 [n] 一一对应</div>"
    ]
    for i, h in enumerate(hits, 1):
        s = h["score"]
        tone = "strong" if s >= 0.60 else "mid" if s >= 0.50 else "weak"
        pct = max(6.0, min(100.0, (s - 0.35) / 0.45 * 100))
        body = html.escape(h["text"][:240].replace("\n", " "))
        if len(h["text"]) > 240:
            body += " …"
        out.append(
            f"<div class='ev ev-{tone}'>"
            f"<div class='ev-head'>"
            f"<span class='ev-rank'>{i}</span>"
            f"<span class='ev-meta'>第 {h['chapter']} 章 · {html.escape(str(h['section']))} · "
            f"p.{h['page_start']}–{h['page_end']}</span>"
            f"<span class='ev-score'>{s:.3f}</span></div>"
            f"<div class='ev-bar'><i style='width:{pct:.0f}%'></i></div>"
            f"<div class='ev-text'>{body}</div></div>"
        )
    return "".join(out)


# ---------------------------------------------------------------- 交互
def ask(question, top_k):
    question = (question or "").strip()
    if not question:
        yield ANSWER_PLACEHOLDER, "<div class='ev-empty'>请先输入问题</div>", "等待提问"
        return

    t0 = time.time()
    hits = search(question, top_k)
    evidence = render_evidence(hits)
    t_search = (time.time() - t0) * 1000

    head = f"检索 Top-{len(hits)} · {t_search:.0f} ms"
    if CLIENT is None:
        yield "", evidence, head + " · 未配置 Key，仅展示检索证据"
        return
    yield "", evidence, head + " · 正在生成…"

    messages, used = build_messages(question, hits)
    acc = ""
    try:
        stream = CLIENT.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            temperature=TEMPERATURE, max_tokens=900, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                acc += delta
                yield acc, evidence, head + f" · 生成中 {len(acc)} 字"
    except Exception as exc:  # noqa: BLE001
        yield f"**调用失败**：`{type(exc).__name__}: {exc}`", evidence, head + " · 失败"
        return

    total = (time.time() - t0) * 1000
    yield acc, evidence, (
        f"检索 {t_search:.0f} ms · 生成 {total - t_search:.0f} ms · "
        f"参考 {used} 字符 ≈ {used / 1.6:.0f} token · 答案 {len(acc)} 字"
    )


ANSWER_PLACEHOLDER = (
    "在左侧输入问题，或展开「示例问题」点一个。\n\n"
    "答案中的 `[编号]` 对应下方证据卡的编号，可以逐条核查到具体页码；"
    "资料不足时模型会明确说不知道，而不是编造。"
)
EVIDENCE_PLACEHOLDER = "<div class='ev-empty'>提问后，这里显示命中的文本块、相似度与页码</div>"

# ---------------------------------------------------------------- 样式
CSS = """
:root {
  --bg:#f5f6f8; --card:#ffffff; --line:#e5e7eb; --line-soft:#eef0f2;
  --ink:#111827; --ink-2:#374151; --ink-3:#6b7280; --ink-4:#9ca3af;
  --accent:#2563eb; --accent-soft:#eff6ff; --accent-line:#dbeafe;
  --radius:12px; --shadow:0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
}
body.dark, .dark {
  --bg:#0f1115; --card:#171a21; --line:#272c36; --line-soft:#20242c;
  --ink:#f3f4f6; --ink-2:#d4d8de; --ink-3:#9aa3af; --ink-4:#6f7783;
  --accent:#7aa7f7; --accent-soft:#18202e; --accent-line:#2b3a52;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}
body, .gradio-container {background: var(--bg) !important;}
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;
  padding: 22px 24px 34px !important;}
footer {display:none !important;}

/* ---------- 页头 ---------- */
#hdr {padding: 2px 2px 16px 2px; margin-bottom: 18px;
      border-bottom: 1px solid var(--line);}
#hdr .titlerow {display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;}
#hdr h1 {font-size: 21px !important; font-weight: 600 !important; letter-spacing:-.2px;
         margin: 0 !important; color: var(--ink);}
#hdr .tag {font-size: 11.5px; color: var(--accent); background: var(--accent-soft);
           border: 1px solid var(--accent-line); border-radius: 999px; padding: 2px 9px;
           white-space: nowrap;}
#hdr .sub {font-size: 13px; color: var(--ink-3); line-height: 1.75; margin-top: 7px;
           max-width: 780px;}
#hdr .sub code {font-size: 12px; color: var(--accent); background: var(--accent-soft);
           border: 1px solid var(--accent-line); border-radius: 4px; padding: 0 4px;}
#hdr .chips {margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px;}
#hdr .chip {font-size: 11.5px; color: var(--ink-3); background: var(--card);
            border: 1px solid var(--line); border-radius: 6px; padding: 3px 9px;
            white-space: nowrap;}

/* ---------- 左栏控制区 ---------- */
#left {background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
       padding: 14px 14px 10px 14px; box-shadow: var(--shadow); position: sticky; top: 12px;}
/* 清掉 Gradio 组件自带的灰底，否则标签、滑块背后会出现一块块灰蓝色方块 */
#left .block, #left .form, #left .label-wrap, #left .head {
       background: transparent !important; box-shadow: none !important; border: none !important;}
#left label, #left label span, #left .label-wrap span, #left .head span {
       background: transparent !important; border-radius: 0 !important;
       box-shadow: none !important; padding-left: 0 !important;}
#left label span, #left .label-wrap span {font-size: 12.5px !important; font-weight: 500 !important;
       color: var(--ink-2) !important;}
#qbox textarea {font-size: 14px !important; line-height: 1.75 !important;
       border-radius: 9px !important; padding: 11px 12px !important;
       background: var(--card) !important; border: 1px solid var(--line) !important;
       color: var(--ink) !important; box-shadow: none !important;}
#qbox textarea::placeholder {color: var(--ink-4) !important;}
#qbox textarea:focus {border-color: var(--accent) !important;
       box-shadow: 0 0 0 3px rgba(37,99,235,.09) !important;}
#qbox {border: none !important; background: transparent !important; box-shadow: none !important;
       padding: 0 !important; margin-bottom: -6px !important;}
#ask-btn {font-size: 14px !important; font-weight: 500 !important; border-radius: 9px !important;
       min-height: 40px !important;}
#clear-btn {border-radius: 9px !important; font-size: 13px !important; min-height: 40px !important;
       background: transparent !important; border: 1px solid var(--line) !important;
       color: var(--ink-2) !important; box-shadow: none !important;}
#clear-btn:hover {border-color: var(--accent) !important; color: var(--accent) !important;}
#kbar {background: transparent !important; border: none !important; box-shadow: none !important;
       padding: 4px 0 0 0 !important; overflow: visible !important;}
#kbar .head, #kbar label {font-size: 12.5px !important;}
#kbar .info {font-size: 11.5px !important; color: var(--ink-4) !important;}
#kbar .reset-button {display: none !important;}
#kbar .tab-like-container {min-width: 0 !important;}
#kbar input[type="number"] {width: 52px !important; min-width: 52px !important;
       font-size: 12.5px !important; padding: 3px 4px !important;}
#kbar .min_value, #kbar .max_value {font-size: 11px !important; color: var(--ink-4) !important;}
#exbox {margin-top: 4px;}
#exbox .gr-examples, #exbox .examples {background: transparent !important;}

/* ---------- 右栏答案区 ---------- */
#right {min-width: 0;}
#status {font-size: 11.5px; color: var(--ink-4); font-variant-numeric: tabular-nums;
         min-height: 16px; margin: 0 0 12px 0; padding-bottom: 9px;
         border-bottom: 1px solid var(--line-soft);}
#status .prose {font-size: 11.5px !important; color: var(--ink-4) !important;}
#status .prose p {margin: 0 !important;}
#answer-card {background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
              box-shadow: var(--shadow); padding: 18px 20px 14px 20px; min-height: 236px;}
#answer-card .block, #answer-card .form {background: transparent !important;
              border: none !important; box-shadow: none !important;}
#answer {min-height: 190px;}
#answer .prose {font-size: 14.5px !important; line-height: 1.9 !important;
                max-width: none !important; color: var(--ink-2) !important;}
#answer .prose h5 {font-size: 14px !important; font-weight: 600 !important;
                   color: var(--ink-4) !important; margin: 2px 0 10px 0 !important;}
#answer .prose p {margin: 0 0 10px 0 !important;}
#answer .prose ol, #answer .prose ul {padding-left: 22px !important; margin: 6px 0 12px 0 !important;}
#answer .prose li {margin-bottom: 6px !important;}
#answer .prose strong {color: var(--ink) !important; font-weight: 600 !important;}
#answer .prose code {font-size: 12.5px !important; background: var(--accent-soft) !important;
                     color: var(--accent) !important; padding: 1px 5px !important;
                     border-radius: 4px !important;}

/* ---------- 证据卡 ---------- */
#evbox {margin-top: 12px;}
#evbox .label-wrap, #evbox summary {font-size: 13px !important; font-weight: 500 !important;
                                    color: var(--ink-2) !important;}
#evscroll {max-height: 460px; overflow-y: auto; padding-right: 4px;}
#evscroll::-webkit-scrollbar {width: 8px;}
#evscroll::-webkit-scrollbar-thumb {background: var(--line); border-radius: 4px;}
.ev-lead {font-size: 11.5px; color: var(--ink-4); margin: 2px 0 10px 2px;}
.ev {border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px;
     margin-bottom: 9px; background: var(--card); transition: border-color .15s;}
.ev:hover {border-color: var(--accent-line);}
.ev-weak {opacity: .88;}
.ev-head {display: flex; align-items: center; gap: 9px; margin-bottom: 7px; flex-wrap: wrap;}
.ev-rank {flex: none; width: 20px; height: 20px; border-radius: 6px; background: var(--accent-soft);
          color: var(--accent); border: 1px solid var(--accent-line); font-size: 11.5px;
          font-weight: 600; display: inline-flex; align-items: center; justify-content: center;}
.ev-meta {font-size: 12px; color: var(--ink-3); flex: 1; min-width: 0;
          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.ev-score {font-size: 11.5px; color: var(--ink-4); font-variant-numeric: tabular-nums;}
.ev-bar {height: 3px; background: var(--line-soft); border-radius: 2px; overflow: hidden;
         margin-bottom: 8px;}
.ev-bar i {display: block; height: 100%; background: var(--accent); border-radius: 2px; opacity: .55;}
.ev-strong .ev-bar i {opacity: .85;}
.ev-weak .ev-bar i {opacity: .3;}
.ev-text {font-size: 13px; color: var(--ink-2); line-height: 1.78;}
.ev-empty {font-size: 12.5px; color: var(--ink-4); padding: 14px 2px;}

/* ---------- 页脚 ---------- */
#foot {font-size: 11.5px; color: var(--ink-4); line-height: 1.8; margin-top: 14px;
       padding-top: 12px; border-top: 1px solid var(--line);}

@media (max-width: 860px) {
  .gradio-container {padding: 14px 14px 24px !important;}
  #left {position: static;}
}
"""

with gr.Blocks(title="RAG 教材知识问答") as demo:
    gr.HTML(
        """
        <div id='hdr'>
          <div class='titlerow'>
            <h1>《自然语言处理导论》知识问答</h1>
            <span class='tag'>RAG · 答案可溯源到页码</span>
          </div>
          <div class='sub'>
            从 604 页教材中检索相关段落，由大模型依据检索到的原文作答；
            每条结论标注 <code>[编号]</code>，资料不足时明确拒答，不编造。
          </div>
          <div class='chips'>
            <span class='chip'>604 页 PDF · 正文 506 页</span>
            <span class='chip'>1,700 个文本块</span>
            <span class='chip'>bge-small-zh · 512 维</span>
            <span class='chip'>FAISS IndexFlatIP</span>
            <span class='chip'>Hit@5 100% · MRR 0.825</span>
            <span class='chip'>端到端 1.9–3.1 s</span>
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=False, elem_id="main-row"):
        with gr.Column(scale=4, min_width=340, elem_id="left"):
            question = gr.Textbox(
                label="你的问题", lines=4, max_lines=10, elem_id="qbox",
                placeholder="例如：n 元语言模型遇到数据稀疏问题，可以用哪些平滑方法？",
            )
            with gr.Row():
                submit = gr.Button("提问", variant="primary", scale=3, elem_id="ask-btn")
                clear = gr.Button("清空", scale=1, elem_id="clear-btn")
            top_k = gr.Slider(1, 10, value=5, step=1, elem_id="kbar",
                              label="检索块数 Top-K",
                              info="块数越多证据越全，但上下文更长、成本更高")
            with gr.Accordion("示例问题（点击直接提问）", open=False, elem_id="exbox"):
                gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question],
                            label=None, examples_per_page=6)

        with gr.Column(scale=7, min_width=0, elem_id="right"):
            with gr.Column(elem_id="answer-card"):
                status = gr.Markdown("等待提问", elem_id="status")
                answer = gr.Markdown(ANSWER_PLACEHOLDER, elem_id="answer")

            with gr.Accordion("检索证据 · 答案的出处", open=True, elem_id="evbox"):
                with gr.Column(elem_id="evscroll"):
                    evidence = gr.HTML(EVIDENCE_PLACEHOLDER)

    gr.HTML(
        "<div id='foot'>"
        "链路：问题 → bge 向量化 → FAISS 检索 Top-K → 拼装带编号与页码的 Prompt → DeepSeek 生成。"
        "证据卡右侧的数字为余弦相似度，条形长度按相似度缩放；相似度偏低（&lt; 0.50）的块可能只是字面相近，"
        "模型若判断资料不相关会明确拒答。"
        "</div>"
    )

    outputs = [answer, evidence, status]
    submit.click(ask, inputs=[question, top_k], outputs=outputs,
                 show_progress="hidden", concurrency_limit=1)
    question.submit(ask, inputs=[question, top_k], outputs=outputs,
                    show_progress="hidden", concurrency_limit=1)
    clear.click(lambda: ("", ANSWER_PLACEHOLDER, EVIDENCE_PLACEHOLDER, "等待提问"),
                outputs=[question, answer, evidence, status], show_progress="hidden")


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
