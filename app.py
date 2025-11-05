import os
import re
import json
import time
import shutil
from pathlib import Path
from typing import List
import streamlit as st


from config import DEFAULT_INDEX_ROOT, K_RETRIEVE_DEFAULT
from project import Project
from utils import now_ts, due_wrong
from io_readers import read_pdf, read_pptx, read_docx, read_txt
from rag_core import get_embeddings, split_docs, build_index, save_index, try_load_index, retrieve
from llm import get_llm, rag_answer, gen_mcq, gen_card_or_map
from ui_components import render_evidence_cards, render_mcq_block


st.set_page_config(page_title="📘 学习助手（模块化）", page_icon="📘", layout="wide")


# --- 全局状态 ---
if "index_root" not in st.session_state:
    st.session_state["index_root"] = str(DEFAULT_INDEX_ROOT)
if "dev_mode" not in st.session_state:
    st.session_state["dev_mode"] = False
if "project_id" not in st.session_state:
    st.session_state["project_id"] = None

# --- 侧栏 ---
st.sidebar.checkbox("开发者模式", key="dev_mode")
INDEX_ROOT = Path(st.session_state["index_root"]).resolve()
INDEX_ROOT.mkdir(parents=True, exist_ok=True)


# 列出项目
projects: List[Project] = []
for p in sorted(INDEX_ROOT.glob("*/project.json")):
    proj = Project(p.parent); proj.load_meta(); projects.append(proj)


# --- 项目选择/创建页 ---
st.title("📘 学习助手 · 项目选择")
cols = st.columns([2, 1])
with cols[0]:
    st.subheader("已存在的项目")
    if not projects:
        st.info("暂无项目。右侧创建一个。")
    else:
        for proj in projects:
            name = proj.meta.get("name", proj.root.name)
            files = proj.meta.get("files", [])
            tstr = time.strftime("%Y-%m-%d %H:%M", time.localtime(proj.meta.get("created_at", now_ts())))
            with st.container(border=True):
                st.markdown(f"**{name}** · {tstr}")
                st.caption("文件：" + ", ".join([Path(f).name for f in files]))
                c1, c2 = st.columns(2)
                if c1.button("打开", key=f"open_{proj.root.name}"):
                    st.session_state["project_id"] = proj.root.name; st.rerun()
                if c2.button("删除", key=f"del_{proj.root.name}"):
                    shutil.rmtree(proj.root, ignore_errors=True); st.rerun()

with cols[1]:
    st.subheader("创建新项目")
    new_name = st.text_input("项目名称", value=f"proj_{now_ts()}")
    up_files = st.file_uploader("上传 PDF / PPTX / DOCX / TXT", type=["pdf","pptx","docx","txt"], accept_multiple_files=True)
    if st.button("创建并构建索引", type="primary"):
        if not up_files:
            st.warning("请先上传至少一个文件。")
        else:
            proj_dir = INDEX_ROOT / new_name
            proj = Project(proj_dir)
            if proj.exists():
                st.error("同名项目已存在。更换名称。")
            else:
                proj.root.mkdir(parents=True, exist_ok=True)
                proj.files_dir.mkdir(parents=True, exist_ok=True)


                docs_all = []
                files_meta = []
                progress = st.progress(0, text="保存文件…")


                # 1) 保存 + 解析
                for idx, f in enumerate(up_files, start=1):
                    b = f.read()
                    (proj.files_dir / f.name).write_bytes(b)
                    files_meta.append(str(proj.files_dir / f.name))
                    progress.progress(min(5 + int(idx/ max(1,len(up_files))*10), 15), text=f"读取 {f.name}")
                    ext = f.name.lower().split(".")[-1]
                    if ext == "pdf":
                        docs_all += read_pdf(b, f.name)
                    elif ext == "pptx":
                        docs_all += read_pptx(b, f.name)
                    elif ext == "docx":
                        docs_all += read_docx(b, f.name)
                    elif ext == "txt":
                        docs_all += read_txt(b, f.name)


                # 2) 切分
                progress.progress(30, text="分块中…")
                chunks = split_docs(docs_all)


                # 3) 嵌入与索引
                progress.progress(45, text="计算向量…")
                _ = get_embeddings()
                progress.progress(60, text="建立索引…")
                from langchain_community.vectorstores import FAISS
                vs = FAISS.from_documents(chunks, _)


                # 4) 保存
                save_index(vs, proj.index_dir)
                progress.progress(85, text="写入元数据…")
                proj.meta = {"name": new_name, "created_at": now_ts(), "files": files_meta}
                proj.save_meta()
                progress.progress(100, text="完成")
                st.success("项目已创建。点击左侧打开。")
                st.rerun()


# --- 聊天页 ---
if st.session_state.get("project_id"):
    proj = Project(INDEX_ROOT / st.session_state["project_id"])
    if not proj.exists():
        st.error("项目不存在。"); st.stop()
    proj.load_meta()


    vs = try_load_index(proj.index_dir)
    if not vs:
        st.error("索引未找到。"); st.stop()


    st.title(f"💬 {proj.meta.get('name', proj.root.name)}")
    st.caption("像 ChatGPT 一样提问；也支持 /quiz、/card、/map 指令")


    # 侧栏：切换项目 + 历史
    st.sidebar.divider(); st.sidebar.subheader("项目")
    for p in projects:
        name = p.meta.get("name", p.root.name)
        if st.sidebar.button(name, key=f"switch_{p.root.name}"):
            st.session_state["project_id"] = p.root.name; st.rerun()


    st.sidebar.divider(); st.sidebar.subheader("历史对话")
    chats = proj.load_chats(limit=200)
    for i, rec in enumerate(chats[-30:]):
        role = rec.get("role","user"); kind = rec.get("kind","msg"); short = rec.get("text"," ").strip().replace("\n"," ")[:40]
        st.sidebar.caption(f"{i+1}. {role[:1]}/{kind}: {short}")


    # 渲染历史
    for i, rec in enumerate(chats):
        role = rec.get("role","user"); kind = rec.get("kind","msg")
        with st.chat_message("assistant" if role == "assistant" else "user"):
            if kind == "msg":
                st.markdown(rec.get("text",""))
            elif kind == "answer":
                st.markdown(rec.get("text",""))
                if rec.get("hits"):
                    from langchain.schema import Document
                    render_evidence_cards(proj, [Document(page_content=h["content"], metadata=h["meta"]) for h in rec["hits"]])
            elif kind == "mcq":
                render_mcq_block(proj, rec.get("data", {}), qid=str(rec.get("qid") or rec.get("t") or f"mcq_{i}"))
            elif kind in ("card","mindmap"):
                st.markdown(rec.get("text",""))
    # 输入区
    user_msg = st.chat_input("输入问题、或 /quiz 关键词，/card 主题，/map 主题")
    if user_msg:
        # 立即回显
        with st.chat_message("user"): st.markdown(user_msg)
        proj.append_chat({"t": now_ts(), "role": "user", "kind": "msg", "text": user_msg})


        llm = get_llm(); devlog = {}
        lower = user_msg.lower()
        try_quiz = ("/quiz" in lower) or ("生成题目" in user_msg) or ("测验" in user_msg)
        try_card = ("/card" in lower) or ("知识卡片" in user_msg)
        try_map = ("/map" in lower) or ("思维导图" in user_msg)


        with st.chat_message("assistant"):
            if try_quiz:
                topic = re.sub(r"^.*?/quiz", "", lower).strip() or user_msg
                hits = retrieve(vs, topic, k=8)
                ctx = "\n\n".join(d.page_content[:600] for d in hits)
                try:
                    with st.spinner("生成题目中…"):
                        data = gen_mcq(llm, ctx, devlog)
                except Exception as e:
                    devlog["error_mcq"] = str(e); st.error(f"生成题目失败：{e}")
                    data = {"question":"生成失败","options":[],"answer":"","rationale":""}
                qid = str(int(time.time()*1000))
                render_mcq_block(proj, data, qid)
                proj.append_chat({"t": now_ts(), "role": "assistant", "kind": "mcq", "qid": qid, "data": data})


            elif try_card or try_map:
                topic = re.sub(r"^.*?/(card|map)", "", lower).strip() or user_msg
                hits = retrieve(vs, topic, k=10)
                ctx = "\n\n".join(d.page_content[:800] for d in hits)
                mode = "card" if try_card else "mindmap"
                try:
                    with st.spinner("生成内容中…"):
                        out = gen_card_or_map(llm, ctx, mode, devlog)
                    st.markdown(out)
                    proj.append_chat({"t": now_ts(), "role": "assistant", "kind": mode, "text": out})
                except Exception as e:
                    devlog["error_cardmap"] = str(e); st.error(f"生成内容失败：{e}")


            else:
                try:
                    with st.spinner("生成回答中…"):
                        ans, hits = rag_answer(llm, vs, user_msg, k=K_RETRIEVE_DEFAULT, devlog=devlog)
                    st.markdown(ans)
                    render_evidence_cards(proj, hits)
                    proj.append_chat({
                    "t": now_ts(), "role": "assistant", "kind": "answer", "text": ans,
                    "hits": [{"content": h.page_content, "meta": h.metadata} for h in hits],
                    })
                except Exception as e:
                    devlog["error_answer"] = str(e); st.error(f"生成回答失败：{e}")


            if st.session_state.get("dev_mode"):
                with st.expander("🔧 开发者模式：Prompt & 原始返回"):
                    for k, v in devlog.items():
                        st.markdown(f"**{k}**"); st.code(v)
# --- 底部：错题本与导出 ---
st.divider()
tab1, tab2 = st.tabs(["错题本", "导出与备份"])
with tab1:
    if not st.session_state.get("project_id"):
        st.info("请选择或创建一个项目后使用错题本。")
    else:
        proj = Project(INDEX_ROOT / st.session_state["project_id"])
        items = proj.load_wrong()
        st.caption(f"总错题：{len(items)}")
        due = due_wrong(items)
        if due:
            st.warning(f"需要复习：{len(due)}")
            keep = items[:]
            for i, it in enumerate(due[:10], 1):
                st.markdown(f"**{i}. {it.get('q','(no question)')}**")
                st.write("\n".join(it.get("opts", [])))
                c1, c2, c3 = st.columns(3)
                if c1.button("掌握", key=f"up_{i}"):
                    it["box"] = min(it.get("box",1)+1, 3); it["last"] = now_ts()
                if c2.button("仍错", key=f"down_{i}"):
                    it["box"] = 1; it["last"] = now_ts()
                if c3.button("删除", key=f"del_{i}"):
                    it["del"] = True
                keep = [it for it in keep if not it.get("del")]
                with open(proj.wrong_path, "w", encoding="utf-8") as f:
                    for it in keep:
                        f.write(json.dumps(it, ensure_ascii=False)+"\n")
        else:
            st.info("没有到期的复习项。")


with tab2:
    if not st.session_state.get("project_id"):
        st.info("选择/创建项目后可导出。")
    else:
        proj = Project(INDEX_ROOT / st.session_state["project_id"])
        colA, colB = st.columns(2)
        if proj.chat_path.exists():
            st.download_button("导出对话 JSONL", data=proj.chat_path.read_bytes(), file_name=f"{proj.root.name}_chats.jsonl")
        if proj.meta_path.exists():
            st.download_button("导出项目元数据", data=proj.meta_path.read_bytes(), file_name=f"{proj.root.name}_meta.json")
        if proj.wrong_path.exists():
            st.download_button("导出错题本 JSONL", data=proj.wrong_path.read_bytes(), file_name=f"{proj.root.name}_wrong.jsonl")