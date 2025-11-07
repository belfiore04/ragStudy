from pathlib import Path
from typing import List
import streamlit as st
from config import DEFAULT_INDEX_ROOT, K_RETRIEVE_DEFAULT
from project import Project
from views import (
    render_new_project_view,
    render_chat_view,
    render_wrongbook_view,
    render_export_view,
)

st.set_page_config(page_title="RAG学习助手", page_icon="📘", layout="wide")

# ============= 全局状态 =============
if "index_root" not in st.session_state:
    st.session_state["index_root"] = str(DEFAULT_INDEX_ROOT)
if "dev_mode" not in st.session_state:
    st.session_state["dev_mode"] = False
if "project_id" not in st.session_state:
    st.session_state["project_id"] = None
if "view" not in st.session_state:
        st.session_state["view"] = "新建项目"   # 默认页

INDEX_ROOT = Path(st.session_state["index_root"]).resolve()
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# 列出所有项目
projects: List[Project] = []
for p in sorted(INDEX_ROOT.glob("*/project.json")):
    proj = Project(p.parent)
    proj.load_meta()
    projects.append(proj)


# ============= 侧边栏导航 =============
st.sidebar.markdown("### 页面")
if st.sidebar.button("新建项目"):
    st.session_state["view"] = "新建项目"
    st.rerun()

st.sidebar.markdown("### 历史项目")
if not projects:
    st.sidebar.caption("暂无项目")
else:
    for p in projects:
        name = p.meta.get("name", p.root.name)
        if st.sidebar.button(name, key=f"switch_{p.root.name}"):
            st.session_state["project_id"] = p.root.name
            st.session_state["view"] = "对话"
            st.rerun()

st.sidebar.markdown("### 工具")
if st.sidebar.button("错题本"):
    st.session_state["view"] = "错题本"
    st.rerun()

if st.sidebar.button("导出与备份"):
    st.session_state["view"] = "导出与备份"
    st.rerun()

st.sidebar.markdown("### 设置")
st.sidebar.checkbox("开发者模式", key="dev_mode")

# 当前视图
view = st.session_state["view"]

# ============ 路由到各视图 ============
if view == "新建项目":
    render_new_project_view(projects, INDEX_ROOT)
elif view == "对话":
    render_chat_view(INDEX_ROOT)
elif view == "错题本":
    render_wrongbook_view(INDEX_ROOT)
elif view == "导出与备份":
    render_export_view(INDEX_ROOT)
else:
    st.error(f"未知视图：{view}")