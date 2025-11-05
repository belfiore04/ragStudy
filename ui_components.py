import time
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document
from io_readers import convert_to_pdf_with_libreoffice, pdf_page_to_image




def render_evidence_cards(proj, hits: List[Document]):
    st.markdown("**依据**")
    for d in hits:
        meta = d.metadata or {}
        src = meta.get("source", "?")
        tag = Path(src).name
        page = meta.get("page")
        slide = meta.get("slide")
        with st.expander(f"{tag} · " + (f"P{page}" if page else (f"S{slide}" if slide else ""))):
            src_path = proj.files_dir / tag
            shown = False
            if src_path.exists():
                preview_pdf = src_path if src_path.suffix.lower() == ".pdf" else convert_to_pdf_with_libreoffice(src_path, proj.preview_dir / "pdf")
                page_num = page or slide
                if preview_pdf and page_num:
                    img = pdf_page_to_image(preview_pdf, page_num)
                    if img is not None:
                        st.image(img, use_column_width=True)
                        shown = True
            if not shown:
                st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))

def render_mcq_block(proj, data: Dict[str, Any], qid: str):
    st.markdown(f"**Question:** {data.get('question','')}")
    opts = data.get("options", [])
    choice_key = f"mcq_choice_{qid}"
    submit_key = f"mcq_submit_{qid}"
    feedback_key = f"mcq_feedback_{qid}"


    sel = st.radio("选项", opts, key=choice_key)
    c1, c2 = st.columns(2)
    if c1.button("提交", key=submit_key):
        your_letter = (sel or "").strip()[:1].upper()
        correct = (data.get("answer", "").strip()[:1].upper())
        st.session_state[feedback_key] = {"your": your_letter, "correct": correct, "logged": False}


    fb = st.session_state.get(feedback_key)
    if fb:
        if fb["your"] == fb["correct"]:
            st.success("正确")
        else:
            st.error(f"错误。答案：{fb['correct']}")
            if not fb.get("logged"):
                try:
                    proj.log_wrong({
                    "t": int(time.time()),
                    "q": data.get("question"),
                    "opts": opts,
                    "ans": data.get("answer", "").strip()[:1].upper(),
                    "ua": fb["your"],
                    "rationale": data.get("rationale", ""),
                    "box": 1,
                    "last": int(time.time()),
                    })
                    fb["logged"] = True
                    st.session_state[feedback_key] = fb
                except Exception as e:
                    st.warning(f"写入错题本失败：{e}")
    with st.expander("答案与解析"):
        st.write(data.get("rationale", "N/A"))    