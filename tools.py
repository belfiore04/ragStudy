# tools.py
import re
import time
from typing import Dict, Any, List, Tuple

import streamlit as st
from langchain.schema import Document

from rag_core import retrieve
from llm import rag_answer, gen_mcq, gen_card_or_map
from ui_components import render_evidence_cards, render_mcq_block
from utils import now_ts


def detect_tool(user_msg: str) -> Tuple[str, str]:
    """
    根据用户输入判断使用哪种工具：
    mode:
      - "quiz": 出题
      - "card": 知识卡片
      - "map": 思维导图
      - "answer": 普通 RAG 问答
    返回 (mode, topic)
    """
    lower = user_msg.lower()
    mode = "answer"
    topic = user_msg

    # 出题
    if ("/quiz" in lower) or ("生成题目" in user_msg) or ("测验" in user_msg):
        mode = "quiz"
        after = re.sub(r"^.*?/quiz", "", lower).strip()
        topic = after or user_msg
        return mode, topic

    # 知识卡片
    if ("/card" in lower) or ("知识卡片" in user_msg):
        mode = "card"
        after = re.sub(r"^.*?/card", "", lower).strip()
        topic = after or user_msg
        return mode, topic

    # 思维导图
    if ("/map" in lower) or ("思维导图" in user_msg):
        mode = "map"
        after = re.sub(r"^.*?/map", "", lower).strip()
        topic = after or user_msg
        return mode, topic

    # 默认普通问答
    return mode, topic


def run_tool(
    mode: str,
    proj,
    vs,
    llm,
    user_msg: str,
    topic: str,
    devlog: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    执行对应“工具”，负责：
    - 检索 / 调 LLM
    - 在当前的 st.chat_message("assistant") 容器内渲染 UI
    - 返回需要写入 chat.jsonl 的记录列表
    """
    records: List[Dict[str, Any]] = []

    if mode == "quiz":
        hits_r = retrieve(vs, topic, k=8)
        ctx = "\n\n".join(d.page_content[:600] for d in hits_r)
        try:
            with st.spinner("生成题目中…"):
                data = gen_mcq(llm, ctx, devlog)
        except Exception as e:
            devlog["error_mcq"] = str(e)
            st.error(f"生成题目失败：{e}")
            data = {
                "question": "生成失败",
                "options": [],
                "answer": "",
                "rationale": "",
            }
        qid = str(int(time.time() * 1000))
        render_mcq_block(proj, data, qid)
        records.append({
            "t": now_ts(),
            "role": "assistant",
            "kind": "mcq",
            "qid": qid,
            "data": data,
        })
        return records

    if mode in ("card", "map"):
        hits_r = retrieve(vs, topic, k=10)
        ctx = "\n\n".join(d.page_content[:800] for d in hits_r)
        mode_cardmap = "card" if mode == "card" else "mindmap"
        try:
            with st.spinner("生成内容中…"):
                out = gen_card_or_map(llm, ctx, mode_cardmap, devlog)
            st.markdown(out)
            records.append({
                "t": now_ts(),
                "role": "assistant",
                "kind": mode_cardmap,
                "text": out,
            })
        except Exception as e:
            devlog["error_cardmap"] = str(e)
            st.error(f"生成内容失败：{e}")
        return records

    # 默认：普通 RAG 问答
    try:
        with st.spinner("生成回答中…"):
            ans, hits_r = rag_answer(
                llm, vs, user_msg,
                k=4,   # 这里你也可以用 K_RETRIEVE_DEFAULT，在调用方传进来
                devlog=devlog,
            )
        st.markdown(ans)
        render_evidence_cards(proj, hits_r)
        records.append({
            "t": now_ts(),
            "role": "assistant",
            "kind": "answer",
            "text": ans,
            "hits": [
                {"content": h.page_content, "meta": h.metadata}
                for h in hits_r
            ],
        })
    except Exception as e:
        devlog["error_answer"] = str(e)
        st.error(f"生成回答失败：{e}")

    return records
# tools.py 中新增
def llm_route_tool(llm, user_msg: str) -> Tuple[str, str]:
    """
    让 LLM 决定使用哪种工具。
    返回 (mode, topic)，mode 同 detect_tool:
      - "answer" / "quiz" / "card" / "map"
    """
    system_prompt = (
        "你是一个学习助手的路由器。"
        "用户会向你提问，你需要选择一个最合适的工具来处理：\n"
        "- answer: 普通问答，适合一般解释、理解、推导。\n"
        "- quiz: 生成单选题，适合用户说“测试一下”“出几道题”等。\n"
        "- card: 生成知识卡片，适合用户说“整理成知识点”“做卡片”等。\n"
        "- map: 生成思维导图，适合用户说“梳理结构”“画个思维导图”等。\n\n"
        "你只做决策，不直接回答内容。\n"
        "请严格返回 JSON，格式为：\n"
        '{"tool": "answer|quiz|card|map", "topic": "<用来检索的主题字符串>"}\n'
        "不要输出任何多余文字。"
    )

    user_prompt = f"用户输入是：{user_msg}\n请决定工具并给出合适的检索主题。"

    # 这里直接用 llm.invoke 简单调用
    out = llm.invoke(system_prompt + "\n\n" + user_prompt)
    text = getattr(out, "content", str(out)).strip()

    import json
    try:
        data = json.loads(text)
        tool = str(data.get("tool", "answer")).strip().lower()
        topic = str(data.get("topic", "")).strip() or user_msg
        if tool not in ("answer", "quiz", "card", "map"):
            tool = "answer"
        return tool, topic
    except Exception:
        # 解析失败就回退到默认
        return "answer", user_msg
