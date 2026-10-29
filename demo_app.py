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
    out = []
    for i, h in enumerate(hits, 1):
        s = h["score"]
        tone = "strong" if s >= 0.60 else "mid" if s >= 0.50 else "weak"
        body = html.escape(h["text"][:320].replace("\n", " "))
        if len(h["text"]) > 320:
            body += " …"
        out.append(
            f"<div class='ev ev-{tone}'>"
            f"<div class='ev-head'>"
            f"<span class='ev-rank'>[{i}]</span>"
            f"<span class='ev-score'>{s:.3f}</span>"
            f"<span class='ev-meta'>第 {h['chapter']} 章 · {html.escape(str(h['section']))} · "
            f"p.{h['page_start']}–{h['page_end']}</span></div>"
            f"<div class='ev-text'>{body}</div></div>"
        )
    return "".join(out)


# ---------------------------------------------------------------- 交互
def ask(question, top_k):
    question = (question or "").strip()
    if not question:
        yield "", "<div class='ev-empty'>请先输入问题</div>", "等待提问"
        return

    t0 = time.time()
    hits = search(question, top_k)
    evidence = render_evidence(hits)
    t_search = (time.time() - t0) * 1000

    head = f"检索 **Top-{len(hits)}** · {t_search:.0f} ms"
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
        f"检索 **Top-{len(hits)}** · {t_search:.0f} ms ｜ "
        f"生成 {total - t_search:.0f} ms ｜ 参考 {used} 字符 ≈ {used / 1.6:.0f} token ｜ "
        f"答案 {len(acc)} 字"
    )


ANSWER_PLACEHOLDER = (
    "**答案会显示在这里。**\n\n"
    "关键结论后带 `[编号]`，对应下方证据卡的编号，可逐条核查到具体页码。"
)
EVIDENCE_PLACEHOLDER = "<div class='ev-empty'>提问后，这里显示命中的文本块、相似度与页码</div>"

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
#hdr {border-bottom: 1px solid #e5e7eb; padding-bottom: 14px; margin-bottom: 6px;}
#hdr h1 {font-size: 22px !important; font-weight: 500 !important; margin: 0 0 6px 0 !important; color: #111827;}
#hdr .sub {font-size: 13px; color: #6b7280; line-height: 1.7;}
.chips {margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px;}
.chip {font-size: 12px; color: #1d4ed8; background: #eff6ff; border: 1px solid #dbeafe;
       border-radius: 999px; padding: 3px 10px; white-space: nowrap;}
#status {font-size: 12px; color: #6b7280; min-height: 18px;}
.ev {border: 1px solid #e5e7eb; border-left: 3px solid #d1d5db; border-radius: 6px;
     padding: 10px 12px; margin-bottom: 8px; background: #fcfcfd;}
.ev-strong {border-left-color: #2563eb;}
.ev-mid {border-left-color: #60a5fa;}
.ev-weak {border-left-color: #d1d5db; opacity: .85;}
.ev-head {display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px; flex-wrap: wrap;}
.ev-rank {font-size: 12px; font-weight: 500; color: #1d4ed8;}
.ev-score {font-size: 12px; color: #6b7280; font-variant-numeric: tabular-nums;}
.ev-meta {font-size: 12px; color: #9ca3af;}
.ev-text {font-size: 13px; color: #374151; line-height: 1.75;}
.ev-empty {font-size: 13px; color: #9ca3af; padding: 10px 0;}
#answer {min-height: 190px;}
#answer-box {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
             padding: 14px 16px 6px 16px; min-height: 300px;}
#answer-box .prose {font-size: 14px !important; line-height: 1.85 !important; max-width: none !important;}
"""

with gr.Blocks(title="RAG 教材知识问答") as demo:
    gr.HTML(
        """
        <div id='hdr'>
          <h1>《自然语言处理导论》知识问答系统</h1>
          <div class='sub'>
            输入问题，系统从 604 页教材中检索相关段落，由大模型生成<strong>带页码引用</strong>的答案；
            参考资料不足时明确拒答，不编造。
          </div>
          <div class='chips'>
            <span class='chip'>604 页 PDF</span>
            <span class='chip'>1700 个文本块</span>
            <span class='chip'>bge-small-zh 512 维</span>
            <span class='chip'>FAISS IndexFlatIP</span>
            <span class='chip'>Hit@5 100% / MRR 0.825</span>
            <span class='chip'>端到端 1.9–3.1 s</span>
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=380):
            question = gr.Textbox(
                label="你的问题", lines=3, max_lines=8,
                placeholder="例如：n 元语言模型遇到数据稀疏问题，可以用哪些平滑方法？",
            )
            with gr.Row():
                submit = gr.Button("提问", variant="primary", scale=2)
                clear = gr.Button("清空", scale=1)
            top_k = gr.Slider(1, 10, value=5, step=1,
                              label="检索块数 Top-K", info="块越多证据越全，但上下文越长、成本越高")
            gr.Examples(examples=[[q] for q in EXAMPLES[:4]], inputs=[question], label="示例问题（点击填入）")

        with gr.Column(scale=7, elem_id="answer-box"):
            status = gr.Markdown("等待提问", elem_id="status")
            answer = gr.Markdown(ANSWER_PLACEHOLDER, elem_id="answer")

    with gr.Accordion("检索证据 —— 答案的出处（可与答案中的 [编号] 逐一对照）", open=True):
        evidence = gr.HTML(EVIDENCE_PLACEHOLDER)

    gr.HTML(
        "<div style='font-size:12px;color:#9ca3af;line-height:1.7;margin-top:6px;'>"
        "链路：问题 → bge 向量化 → FAISS 检索 Top-K → 拼装带编号与页码的 Prompt → DeepSeek 生成。"
        "答案中的 [编号] 对应下方证据编号，可逐条核查；资料不足时模型会明确拒答而不是编造。"
        "</div>"
    )

    submit.click(ask, inputs=[question, top_k], outputs=[answer, evidence, status])
    question.submit(ask, inputs=[question, top_k], outputs=[answer, evidence, status])
    clear.click(lambda: ("", ANSWER_PLACEHOLDER, EVIDENCE_PLACEHOLDER, "等待提问"),
                outputs=[question, answer, evidence, status])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
