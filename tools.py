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
import json

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
    统一的工具路由入口：
    1) 先用规则处理显式指令（/quiz, /card, /map、中文关键词），直接返回；
    2) 再用 LLM 选择 tool: answer | quiz | card | map。

    返回: (mode, topic)
    """
    text = user_msg.strip()
    lower = text.lower()

    # === 1. 规则优先：兼容旧的显式指令 & 中文关键词 ===
    # 出题
    if ("/quiz" in lower) or ("生成题目" in text) or ("测验" in text) or ("出几道题" in text):
        # 提取 /quiz 后面的部分作为 topic，兼容以前写法
        after = re.sub(r"^.*?/quiz", "", lower).strip()
        topic = after or text
        return "quiz", topic

    # 知识卡片
    if ("/card" in lower) or ("知识卡片" in text) or ("整理成知识点" in text) or ("做成卡片" in text):
        after = re.sub(r"^.*?/card", "", lower).strip()
        topic = after or text
        return "card", topic

    # 思维导图
    if ("/map" in lower) or ("思维导图" in text) or ("梳理结构" in text) or ("画个导图" in text):
        after = re.sub(r"^.*?/map", "", lower).strip()
        topic = after or text
        return "map", topic

    # === 2. 没有显式工具指令，交给 LLM 决策 ===
    system_prompt = (
        "你是一个学习助手的路由器，只负责选择最合适的工具，不直接回答问题。\n"
        "可选工具：\n"
        "- answer: 普通问答，解释概念、推导、总结等。\n"
        "- quiz: 生成 1 道单选题，适合用户说“出题”“测一测”“练习题”等。\n"
        "- card: 生成知识卡片，适合用户说“整理成知识点”“做卡片”等。\n"
        "- map: 生成思维导图，适合用户说“帮我梳理结构”“列个框架”等。\n\n"
        "你必须输出 JSON，格式严格为：\n"
        '{"tool": "answer|quiz|card|map", "topic": "<用于检索的主题>"}\n'
        "不要输出任何多余文字。"
    )

    user_prompt = f"用户输入是：{text}\n请根据用户意图选择一个工具，并给出合适的检索主题。"

    out = llm.invoke(system_prompt + "\n\n" + user_prompt)
    raw = getattr(out, "content", str(out)).strip()

    # 开发者模式下方便调试
    if st.session_state.get("dev_mode"):
        st.session_state["dev_router_raw"] = raw

    try:
        data = json.loads(raw)
        tool = str(data.get("tool", "answer")).strip().lower()
        topic = str(data.get("topic", "")).strip() or text
        if tool not in ("answer", "quiz", "card", "map"):
            tool = "answer"
        return tool, topic
    except Exception:
        # 解析失败：兜底为普通回答
        return "answer", text

def llm_make_plan(llm, user_msg: str, devlog: Dict[str, Any]) -> Dict[str, Any]:
    """
    让 LLM 规划一个多步学习 plan。

    期望返回结构示例:
    {
      "steps": [
        {"tool": "quiz", "topic": "导数基础", "n_questions": 5},
        {"tool": "card", "topic": "导数基础"},
        {"tool": "map",  "topic": "导数基础"}
      ]
    }

    tool 只能是: answer | quiz | card | map
    n_questions 仅在 quiz 时生效，默认 1
    """
    text = user_msg.strip()

    system_prompt = (
        "你是一个学习计划规划器，负责为用户设计一小段学习 session。\n"
        "可用的工具有：\n"
        "- answer: 普通 RAG 问答，解释概念、推导、总结等。\n"
        "- quiz: 生成 1 道单选题，用于练习或自测。\n"
        "- card: 生成“知识卡片”，提炼核心概念与要点。\n"
        "- map: 生成“思维导图”，梳理知识结构。\n\n"
        "你要根据用户需求，生成 1～3 个步骤的学习计划，步骤按顺序执行。\n"
        "常见模式例如：\n"
        "- 先 quiz 几道题再用 answer 讲解\n"
        "- 或者 quiz 多道题 + card 总结 + map 梳理\n\n"
        "请严格输出 JSON，格式为：\n"
        '{\"steps\": [\n'
        '  {\"tool\": \"quiz|answer|card|map\", \"topic\": \"主题\", \"n_questions\": 可选整数},\n'
        '  ... 最多 3 步\n'
        ']}\n'
        "注意：不要输出任何多余文字，不要加注释。"
    )

    user_prompt = f"用户输入是：{text}\n请给出一个合适的学习 plan。"

    out = llm.invoke(system_prompt + "\n\n" + user_prompt)
    raw = getattr(out, "content", str(out)).strip()

    devlog["plan_raw"] = raw

    try:
        plan = json.loads(raw)
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("no valid steps")
        # 规范化每个 step
        norm_steps = []
        for s in steps[:3]:  # 最多 3 步
            tool = str(s.get("tool", "answer")).strip().lower()
            if tool not in ("answer", "quiz", "card", "map"):
                tool = "answer"
            topic = str(s.get("topic", "")).strip() or text
            n_q = s.get("n_questions", 1)
            try:
                n_q = int(n_q)
            except Exception:
                n_q = 1
            if n_q < 1:
                n_q = 1
            if n_q > 10:
                n_q = 10
            norm_steps.append({
                "tool": tool,
                "topic": topic,
                "n_questions": n_q,
            })
        plan = {"steps": norm_steps}
        devlog["plan_json"] = json.dumps(plan, ensure_ascii=False, indent=2)
        return plan
    except Exception as e:
        devlog["plan_error"] = f"{type(e).__name__}: {e}"
        # 兜底：退化成一个 answer 步骤
        return {"steps": [{"tool": "answer", "topic": text, "n_questions": 1}]}

def execute_plan(
    plan: Dict[str, Any],
    proj,
    vs,
    llm,
    user_msg: str,
    devlog: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    按 plan 依次执行多个工具步骤。

    返回：所有步骤产生的聊天记录列表（用于写入 chat.jsonl）
    """
    records_all: List[Dict[str, Any]] = []

    steps = plan.get("steps") or []
    if not isinstance(steps, list) or not steps:
        # 兜底：退回单工具路由
        mode, topic = llm_route_tool(llm, user_msg)
        return run_tool(
            mode=mode,
            proj=proj,
            vs=vs,
            llm=llm,
            user_msg=user_msg,
            topic=topic,
            devlog=devlog,
        )

    # 依次执行每一步
    for idx, step in enumerate(steps, start=1):
        tool = step.get("tool", "answer")
        topic = step.get("topic") or user_msg
        n_questions = int(step.get("n_questions", 1) or 1)

        # 方便在 dev_mode 下看到每一步的信息
        devlog[f"step_{idx}_tool"] = tool
        devlog[f"step_{idx}_topic"] = topic
        devlog[f"step_{idx}_n_questions"] = n_questions

        if tool == "quiz" and n_questions > 1:
            # 多道题：循环调用已有 quiz 工具
            for j in range(n_questions):
                step_records = run_tool(
                    mode="quiz",
                    proj=proj,
                    vs=vs,
                    llm=llm,
                    user_msg=user_msg,
                    topic=topic,
                    devlog=devlog,
                )
                records_all.extend(step_records)
        else:
            step_records = run_tool(
                mode=tool,
                proj=proj,
                vs=vs,
                llm=llm,
                user_msg=user_msg,
                topic=topic,
                devlog=devlog,
            )
            records_all.extend(step_records)

    return records_all
