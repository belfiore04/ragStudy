# views.py
import os
import time
import json
import shutil
from pathlib import Path
from typing import List
import streamlit as st
from config import DEFAULT_INDEX_ROOT, K_RETRIEVE_DEFAULT
from project import Project
from utils import now_ts, due_wrong
from io_readers import read_pdf, read_pptx, read_docx, read_txt
from rag_core import get_embeddings, split_docs, save_index, try_load_index, retrieve
from llm import get_llm
from ui_components import render_evidence_cards, render_mcq_block
from utils import slugify_name
from llm import get_llm
from tools import execute_plan, llm_make_plan, run_tool, llm_route_tool

def render_new_project_view(projects: List[Project], INDEX_ROOT: Path):
    st.title("RAG学习助手")

    cols = st.columns([2, 1])

    # 左列：项目列表
    with cols[0]:
        st.subheader("已存在的项目")
        if not projects:
            st.info("暂无项目。右侧创建一个。")
        else:
            for proj in projects:
                name = proj.meta.get("name", proj.root.name)
                files = proj.meta.get("files", [])
                tstr = time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(proj.meta.get("created_at", now_ts()))
                )
                with st.container(border=True):
                    st.markdown(f"**{name}** · {tstr}")
                    st.caption("文件：" + ", ".join([Path(f).name for f in files]))
                    c1, c2 = st.columns(2)
                    if c1.button("打开", key=f"open_{proj.root.name}"):
                        st.session_state["project_id"] = proj.root.name
                        st.session_state["view"] = "对话"
                        st.rerun()
                    if c2.button("删除", key=f"del_{proj.root.name}"):
                        shutil.rmtree(proj.root, ignore_errors=True)
                        st.rerun()

    # 右列：新建项目
    with cols[1]:
        st.subheader("创建新项目")
        new_name = st.text_input("项目名称", placeholder="请输入项目名称")

        up_files = st.file_uploader(
            "上传 PDF / PPTX / DOCX / TXT",
            type=["pdf", "pptx", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("创建并构建索引", type="primary"):
            display_name = new_name.strip()
            if not display_name:
                st.warning("请填写项目名称。")
            elif not up_files:
                st.warning("请先上传至少一个文件。")
            else:
                # 目录名：安全的 ascii slug
                dir_name = slugify_name(display_name)
                proj_dir = INDEX_ROOT / dir_name
                proj = Project(proj_dir)
                if proj.exists():
                    st.error(f"目录名 {dir_name} 已存在。请换一个项目名称。")
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
                        progress.progress(
                            min(5 + int(idx / max(1, len(up_files)) * 10), 15),
                            text=f"读取 {f.name}"
                        )
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
                    proj.meta = {
                        "name": display_name,          # 显示中文名
                        "dir_name": dir_name,          # 目录名（可选）
                        "created_at": now_ts(),
                        "files": files_meta
                    }
                    proj.save_meta()
                    progress.progress(100, text="完成")
                    st.success("项目已创建。可以在左侧“历史项目”里打开。")
                    st.rerun()


def render_chat_view(INDEX_ROOT: Path):
    if not st.session_state.get("project_id"):
        st.title("💬 对话")
        st.info("请先在左侧选择一个项目，或切换到“新建项目”页创建。")
    else:
        proj = Project(INDEX_ROOT / st.session_state["project_id"])
        if not proj.exists():
            st.error("项目不存在。")
            st.stop()
        proj.load_meta()

        vs = try_load_index(proj.index_dir)
        if not vs:
            st.error("索引未找到。")
            st.stop()

        st.title(f"💬 {proj.meta.get('name', proj.root.name)}")
        st.caption("像 ChatGPT 一样提问；也支持 /quiz、/card、/map 指令")

        chats = proj.load_chats(limit=200)

        # 历史对话（只在主区显示）
        for i, rec in enumerate(chats):
            role = rec.get("role", "user")
            kind = rec.get("kind", "msg")
            with st.chat_message("assistant" if role == "assistant" else "user"):
                if kind == "msg":
                    st.markdown(rec.get("text", ""))
                elif kind == "answer":
                    st.markdown(rec.get("text", ""))
                    if rec.get("hits"):
                        from langchain.schema import Document
                        render_evidence_cards(
                            proj,
                            [Document(page_content=h["content"], metadata=h["meta"]) for h in rec["hits"]]
                        )
                elif kind == "mcq":
                    render_mcq_block(
                        proj,
                        rec.get("data", {}),
                        qid=str(rec.get("qid") or rec.get("t") or f"mcq_{i}")
                    )
                elif kind in ("card", "mindmap"):
                    st.markdown(rec.get("text", ""))

        # 输入区
        user_msg = st.chat_input("输入问题、或 /quiz 关键词，/card 主题，/map 主题")
        if user_msg:
            # 立即回显
            with st.chat_message("user"):
                st.markdown(user_msg)
            proj.append_chat({
                "t": now_ts(),
                "role": "user",
                "kind": "msg",
                "text": user_msg
            })

            llm = get_llm()
            devlog = {}
            text = user_msg.strip()
            use_plan = any(k in text for k in [
                "/plan",
                "综合训练",
                "系统复习",
                "综合复习",
                "一套练习",
                "出一套题",
                "完整复习",
            ])
            with st.chat_message("assistant"):
                if use_plan:
                    # 1) 先让 LLM 生成学习 plan
                    plan = llm_make_plan(llm, user_msg, devlog)
                    # 2) 再按 plan 执行多个工具
                    records = execute_plan(
                        plan=plan,
                        proj=proj,
                        vs=vs,
                        llm=llm,
                        user_msg=user_msg,
                        devlog=devlog,
                    )
                else:
                    mode, topic = llm_route_tool(llm, user_msg)

                    records = run_tool(
                        mode=mode,
                        proj=proj,
                        vs=vs,
                        llm=llm,
                        user_msg=user_msg,
                        topic=topic,
                        devlog=devlog,
                    )
    
                    # 写入 assistant 侧聊天记录
                    for rec in records:
                        proj.append_chat(rec)
    
                    if st.session_state.get("dev_mode"):
                        with st.expander("🔧 开发者模式：Prompt & 原始返回"):
                            for k, v in devlog.items():
                                st.markdown(f"**{k}**")
                                st.code(v)


def render_wrongbook_view(INDEX_ROOT: Path):
    st.title("🧠 错题本")
    if not st.session_state.get("project_id"):
        st.info("请先在左侧选择一个项目。")
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
                    it["box"] = min(it.get("box", 1) + 1, 3)
                    it["last"] = now_ts()
                if c2.button("仍错", key=f"down_{i}"):
                    it["box"] = 1
                    it["last"] = now_ts()
                if c3.button("删除", key=f"del_{i}"):
                    it["del"] = True
            keep = [it for it in keep if not it.get("del")]
            with open(proj.wrong_path, "w", encoding="utf-8") as f:
                for it in keep:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        else:
            st.info("没有到期的复习项。")


def render_export_view(INDEX_ROOT: Path):
    st.title("💾 导出与备份")
    if not st.session_state.get("project_id"):
        st.info("请先在左侧选择一个项目。")
    else:
        proj = Project(INDEX_ROOT / st.session_state["project_id"])
        colA, colB = st.columns(2)
        with colA:
            if proj.chat_path.exists():
                st.download_button(
                    "导出对话 JSONL",
                    data=proj.chat_path.read_bytes(),
                    file_name=f"{proj.root.name}_chats.jsonl"
                )
            if proj.meta_path.exists():
                st.download_button(
                    "导出项目元数据",
                    data=proj.meta_path.read_bytes(),
                    file_name=f"{proj.root.name}_meta.json"
                )
            if proj.wrong_path.exists():
                st.download_button(
                    "导出错题本 JSONL",
                    data=proj.wrong_path.read_bytes(),
                    file_name=f"{proj.root.name}_wrong.jsonl"
                )
        with colB:
            if proj.index_dir.exists():
                import io, zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(proj.index_dir):
                        for f in files:
                            full = Path(root) / f
                            zf.write(full, full.relative_to(proj.index_dir))
                st.download_button(
                    "导出索引 ZIP",
                    data=buf.getvalue(),
                    file_name=f"{proj.root.name}_index.zip",
                    mime="application/zip",
                )
