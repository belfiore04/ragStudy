import html
import time
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document
from io_readers import convert_to_pdf_with_libreoffice, pdf_page_to_image
from streamlit_markmap import markmap
import streamlit.components.v1 as components

def _render_block_container(kind: str, title: str | None = None):
    """
    统一的“卡片”容器：
    - kind: mcq | card | mindmap | evidence
    - title: 可选标题
    用法:
        with _render_block_container("mcq", "练习题"):
            ... 这里放具体内容 ...
    """
    labels = {
        "mcq": "📘 练习题",
        "card": "📌 知识卡片",
        "mindmap": "🧠 思维导图",
        "evidence": "📎 依据",
    }
    label = labels.get(kind, "")
    header_text = ""
    if label and title:
        header_text = f"**{label} · {title}**"
    elif label:
        header_text = f"**{label}**"
    elif title:
        header_text = f"**{title}**"

    container = st.container(border=True)
    with container:
        if header_text:
            st.markdown(header_text)
    # 再返回这个 container，方便 with 继续往里写
    return container


def render_evidence_cards(proj, hits: List[Document]):
    """
    在一个小 expander 里展示依据列表。
    不再占整块大卡片，由上层决定放在哪个位置。
    """
    if not hits:
        return

    with st.expander("📎 依据", expanded=False):
        for d in hits:
            meta = d.metadata or {}
            src = meta.get("source", "?")
            tag = Path(src).name
            page = meta.get("page")
            slide = meta.get("slide")
            label = f"{tag} · " + (f"P{page}" if page else (f"S{slide}" if slide else ""))

            with st.expander(label):
                src_path = proj.files_dir / tag
                shown = False
                if src_path.exists():
                    preview_pdf = (
                        src_path
                        if src_path.suffix.lower() == ".pdf"
                        else convert_to_pdf_with_libreoffice(src_path, proj.preview_dir / "pdf")
                    )
                    page_num = page or slide
                    if preview_pdf and page_num:
                        img = pdf_page_to_image(preview_pdf, page_num)
                        if img is not None:
                            st.image(img, use_column_width=True)
                            shown = True
                if not shown:
                    txt = d.page_content or ""
                    st.write(txt[:1000] + ("..." if len(txt) > 1000 else ""))


def render_mcq_block(proj, data: Dict[str, Any], qid: str):
    question = data.get("question", "") or "(无题干)"
    opts = data.get("options", []) or []

    choice_key = f"mcq_choice_{qid}"
    submit_key = f"mcq_submit_{qid}"
    feedback_key = f"mcq_feedback_{qid}"

    # 整个题目放在统一卡片容器里
    with _render_block_container("mcq", None):
        st.markdown(f"**题目：** {question}")

        if not opts:
            st.info("暂无选项。")
            return

        sel = st.radio("选择一个选项：", opts, key=choice_key)
        c1, c2 = st.columns(2)
        if c1.button("提交", key=submit_key):
            your_letter = (sel or "").strip()[:1].upper()
            correct = (data.get("answer", "").strip()[:1].upper())
            st.session_state[feedback_key] = {
                "your": your_letter,
                "correct": correct,
                "logged": False,
            }

        fb = st.session_state.get(feedback_key)
        if fb:
            if fb["your"] == fb["correct"]:
                st.success("✅ 回答正确")
            else:
                st.error(f"❌ 回答错误，正确答案：{fb['correct']}")
                if not fb.get("logged"):
                    try:
                        now_ts = int(time.time())
                        proj.log_wrong({
                            "t": now_ts,
                            "q": data.get("question"),
                            "opts": opts,
                            "ans": data.get("answer", "").strip()[:1].upper(),
                            "ua": fb["your"],
                            "rationale": data.get("rationale", ""),
                            "box": 1,
                            "last": now_ts,
                        })
                        fb["logged"] = True
                        st.session_state[feedback_key] = fb
                    except Exception as e:
                        st.warning(f"写入错题本失败：{e}")

        with st.expander("查看答案与解析"):
            st.write(data.get("rationale", "暂无解析"))


def render_card_block(text: str):
    """
    知识卡片：用统一卡片容器包一整段 Markdown，让 Markdown 自己解析成大标题/小标题/列表。
    """
    with _render_block_container("card", None):
        # 这里不要做任何正则清洗，直接让 markdown 渲染
        st.markdown(text or "", unsafe_allow_html=False)


def render_mindmap_block(text: str):
    """
    思维导图：同样用卡片容器包裹，内部仍用 Markdown 解析层级列表。
    """
    with _render_block_container("mindmap", None):
        """
    自己用 markmap-autoloader 渲染思维导图，
    这样 iframe 里的 CSS 完全由我们控制，可以改字体颜色 / 分支颜色。
    """
        if not text:
            return

        escaped_md = html.escape(text)

        html_code = f"""
        <!DOCTYPE html>
        <html class="markmap-dark">
        <head>
          <meta charset="utf-8" />
          <style>
            html, body {{
              margin: 0;
              padding: 0;
              width: 100%;
              height: 100%;
              background: transparent;
            }}

            .markmap {{
              position: relative;
              width: 100%;
              height: 100%;
              /* 这里可以继续覆盖变量 */
              --markmap-text-color: #eeeeee;
              --markmap-link-color: #88c0d0;
              --markmap-code-bg: #2e3440;
              --markmap-code-color: #d8dee9;
              font: 300 16px/20px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              color: var(--markmap-text-color);
            }}

            .markmap > svg {{
              width: 100%;
              height: 100%;
            }}
          </style>
        </head>
        <body>
          <div class="markmap">
            <script type="text/template">
        {escaped_md}
            </script>
          </div>

          <script>
            window.markmap = {{
              autoLoader: {{
                toolbar: true
              }},
            }};
          </script>
          <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader@0.18.12"></script>
        </body>
        </html>
        """


        # 这里决定 iframe 本身有多高，相当于“可视高度”
        components.html(html_code, height=500, scrolling=True)

def render_answer_with_evidence(
    proj,
    answer_text: str,
    hits: List[Document] | None,
):
    """
    左边显示回答，右边一个小“📎 依据”按钮（expander）。
    """
    col_ans, col_ev = st.columns([5, 1])

    with col_ans:
        st.markdown(answer_text or "")

    with col_ev:
        if hits:
            render_evidence_cards(proj, hits)
