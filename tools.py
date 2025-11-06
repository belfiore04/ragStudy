# tools.py
import re
import time
from typing import Dict, Any, List, Tuple, Optional
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
    history: Optional[List[Dict[str, Any]]] = None,  # 新增
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
            q = topic or user_msg
            ans, hits_r = rag_answer(
                llm, vs, q,
                k=4,   # 这里你也可以用 K_RETRIEVE_DEFAULT，在调用方传进来
                devlog=devlog,
                history=history
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
     "你是一个学习计划规划器，负责为用户设计一小段“像老师上课一样”的学习 session。\n"
     "你不会直接讲解知识本身，只负责规划后续要调用的工具。\n"
     "可用的工具有：\n"
     "- answer: 普通 RAG 问答，用来讲解概念、例题讲解、总结要点，相当于老师“讲一段”。\n"
     "- quiz: 生成 1 道单选题，用来练习或自测，相当于老师“出一道题让学生做一下”。\n"
     "- card: 生成“知识卡片”，提炼核心概念与要点，相当于老师“最后帮学生整理一个小抄”。\n"
     "- map: 生成“思维导图”，梳理知识结构，相当于老师“帮学生画一个知识结构图”。\n\n"
     "你的任务是：根据用户的需求，规划 1～6 个步骤的学习流程，每一步调用一个工具。\n"
     "流程要尽量像老师带着学生走一小节课，比如：\n"
     "- 先用 answer 简要讲解，然后用 quiz 出 1～2 道题检查理解；\n"
     "- 或者先 quiz 出题让学生暴露问题，再用 answer 讲解，再用 card 帮学生整理要点；\n"
     "- 或者 answer 讲解 + map 梳理结构；\n"
     "- 如果用户只问了一个很小的问题，也可以只给 1 步（例如只用 answer）。\n\n"
     "关于字段含义：\n"
     "- tool: 只能是 \"answer\" | \"quiz\" | \"card\" | \"map\"。\n"
     "- topic: 用于检索的主题，一般来自用户的问题，也可以稍作抽象，例如“导数基础”“自底向上优先分析的优点”等。\n"
     "- n_questions: 仅在 tool=quiz 时生效，表示这一阶段要出多少道题，默认 1，通常建议 1～3 道，不要超过 10。\n\n"
     "请根据用户需求，设计最多 6 步，步骤按顺序执行，体现一定的教学节奏（讲解→练习→总结 或 练习→讲解→整理 等）。\n"
     "严格输出 JSON，格式为：\n"
     "{\\\"steps\\\": [\n"
     "  {\\\"tool\\\": \\\"quiz|answer|card|map\\\", \\\"topic\\\": \\\"主题\\\", \\\"n_questions\\\": 可选整数},\n"
     "  ... 最多 3 步\n"
     "]}\n"
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
        for s in steps[:6]:  # 最多 3 步
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

def llm_should_use_plan(llm, user_msg: str, devlog: Dict[str, Any]) -> bool:
    """
    让 LLM 决定当前这条消息是否需要使用多工具 plan。

    典型需要 plan 的情况：
    - 用户明确说“系统复习”“综合训练”“出一套题”“完整复习”之类；
    - 问题本身比较大、涉及多个知识点，需要“先练题再总结/梳理结构”。

    返回 True 用 plan，False 就用单工具路由。
    """
    text = user_msg.strip()

    system_prompt = (
        "你是学习助手的调度器，要判断是否需要一个多步骤的学习计划(plan)。\n"
        "可选策略：\n"
        "- False: 单一步骤，用一个工具(answer/quiz/card/map)就可以解决。\n"
        "- True: 使用多步骤 plan，比如“先出几道题，再总结/画思维导图”等。\n\n"
        "当用户有这些倾向时倾向于 True：\n"
        "- 说“系统复习、综合训练、出一套题、完整复习、来一套练习”等。\n"
        "- 问题明显很大，像“帮我全面掌握第 X 章”“从头梳理这个专题”等。\n"
        "当只是单个问题、单个概念、一个小练习时，选择 False。\n\n"
        "你必须只输出 JSON：{\"use_plan\": true 或 false}\n"
        "不要输出任何其他文字。"
    )

    user_prompt = f"用户输入是：{text}\n请判断是否需要 plan。"

    out = llm.invoke(system_prompt + "\n\n" + user_prompt)
    raw = getattr(out, "content", str(out)).strip()
    devlog["plan_decide_raw"] = raw

    import json
    try:
        data = json.loads(raw)
        val = data.get("use_plan", False)
        # 兼容 'true'/'false' 字符串
        if isinstance(val, str):
            val = val.strip().lower() == "true"
        return bool(val)
    except Exception as e:
        devlog["plan_decide_error"] = f"{type(e).__name__}: {e}"
        # 兜底：默认不用 plan
        return False
