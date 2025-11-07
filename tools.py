# tools.py
import re
import time
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
from langchain.schema import Document
from rag_core import retrieve
from llm import rag_answer, gen_mcq, gen_card_or_map
from utils import now_ts
import json
from ui_components import (
    render_evidence_cards,
    render_mcq_block,
    render_card_block,
    render_mindmap_block,
    render_answer_with_evidence,
)
def run_tool(
    mode: str,
    proj,
    vs,
    llm,
    user_msg: str,
    topic: str,
    devlog: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
    role: Optional[str] = None,
    strictness: str = "strict",
    extra_context: str = "",
    instruction: str = "",  
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
                data = gen_mcq(
                    llm,
                    ctx,
                    devlog,
                    role=role,
                    strictness=strictness,
                    extra_context=extra_context,
                    instruction = instruction
                )
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
                out = gen_card_or_map(
                    llm,
                    ctx,
                    mode_cardmap if mode_cardmap in ("card", "mindmap") else "card",
                    devlog,
                    role=role,
                    strictness=strictness,
                    extra_context=extra_context,
                    instruction = instruction
                )
            if mode_cardmap == "card":
                render_card_block(out)
            else:
                render_mindmap_block(out)

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

    # 默认：answer（用 topic 作为问题，避免把用户的流程指令传进回答）
    try:
        with st.spinner("生成回答中…"):
            q = topic or user_msg
            ans, hits_r = rag_answer(
                llm, vs, q,
                k=4,
                devlog=devlog,
                history=history,
                role=role,
                strictness=strictness,
                extra_context=extra_context,
                instruction = instruction
            )
        docs = [Document(page_content=h.page_content, metadata=h.metadata) for h in hits_r]
        render_answer_with_evidence(proj, ans, docs)
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
    让 LLM 规划一个多步学习 plan，并保留教案字段。
    - 保留并规范化: id/tool/topic/instruction/role/strictness/n_questions/read_keys/write_key/output_format
    - 其余未知字段透传
    - 校验/默认值:
        tool ∈ {answer,quiz,card,map}
        role ∈ {intro_quiz,explain,check_understanding,summary}
        strictness ∈ {strict,soft,free}
        output_format ∈ {text,mcq_json,markdown}，默认映射: answer→text, quiz→mcq_json, card/map→markdown
        n_questions: 仅 quiz 使用，范围 1..10
        read_keys: 仅允许引用已出现的 write_key（前向依赖会被丢弃）
    """
    text = user_msg.strip()

    system_prompt = (
        "你是一个教学“教案员”。你不直接讲解知识，也不生成最终内容。"
        "你的任务是按“老师上课”的节奏，规划 1–6 个步骤的教学流程，供下游工具执行。\n"
        "可用工具：answer(讲解/总结, 输出text)、quiz(单选题, 输出mcq_json)、"
        "card(知识卡片, 输出markdown)、map(思维导图, 输出markdown)。\n"
        "每步需要字段：id/tool/topic/instruction/role/strictness/n_questions/read_keys/write_key/output_format。\n"
        "role ∈ {intro_quiz,explain,check_understanding,summary}；strictness ∈ {strict,soft,free}。\n"
        "read_keys 引用之前步骤的 write_key；若无依赖则为空数组。\n"
        "严格输出 JSON：{\"steps\":[{...}]}\n"
        "不要输出任何多余文字。"
    )
    user_prompt = f"用户输入：{text}\n请给出一个合适的教学 plan。"

    out = llm.invoke(system_prompt + "\n\n" + user_prompt)
    raw = getattr(out, "content", str(out)).strip()
    devlog["plan_raw"] = raw

    # 尝试解析 JSON（容错：从文本中提取第一个 {...}）
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}

    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        # 兜底：退化为单步 answer
        plan = {"steps": [{"id": "s1", "tool": "answer", "topic": text, "role": "explain",
                           "strictness": "strict", "read_keys": [], "write_key": None,
                           "output_format": "text"}]}
        devlog["plan_json"] = json.dumps(plan, ensure_ascii=False, indent=2)
        return plan

    allowed_tools = {"answer", "quiz", "card", "map"}
    allowed_roles = {"intro_quiz", "explain", "check_understanding", "summary"}
    allowed_strict = {"strict", "soft", "free"}
    allowed_fmt = {"text", "mcq_json", "markdown"}
    default_fmt = {"answer": "text", "quiz": "mcq_json", "card": "markdown", "map": "markdown"}

    norm_steps: List[Dict[str, Any]] = []
    seen_write_keys: set[str] = set()

    for i, s in enumerate(steps[:6], start=1):
        if not isinstance(s, dict):
            continue
        sn = dict(s)  # 透传未知字段

        # id
        sid = str(sn.get("id") or f"s{i}")
        sn["id"] = sid

        # tool
        tool = str(sn.get("tool", "answer")).lower()
        if tool not in allowed_tools:
            tool = "answer"
        sn["tool"] = tool

        # topic
        topic = str(sn.get("topic", "") or text).strip()
        sn["topic"] = topic

        # instruction
        if "instruction" in sn:
            sn["instruction"] = str(sn["instruction"])
        else:
            # 给个简短默认说明
            sn["instruction"] = {
                "answer": "老师讲解，先直观解释后分点说明，要点清晰。",
                "quiz": "生成一题单选题，难度与角色匹配。",
                "card": "生成知识卡片，提炼核心要点与易错点。",
                "map": "生成思维导图，最多四级节点。"
            }[tool]

        # role
        role = str(sn.get("role") or ("explain" if tool == "answer"
                                      else "intro_quiz" if tool == "quiz"
                                      else "summary")).lower()
        if role not in allowed_roles:
            role = "explain" if tool == "answer" else ("intro_quiz" if tool == "quiz" else "summary")
        sn["role"] = role

        # strictness
        strict = str(sn.get("strictness", "strict")).lower()
        if strict not in allowed_strict:
            strict = "strict"
        sn["strictness"] = strict

        # n_questions
        if tool == "quiz":
            try:
                nq = int(sn.get("n_questions", 1))
            except Exception:
                nq = 1
            sn["n_questions"] = max(1, min(10, nq))
        else:
            sn.pop("n_questions", None)

        # read_keys
        rk = sn.get("read_keys", [])
        if not isinstance(rk, list):
            rk = [rk]
        rk = [str(x) for x in rk if isinstance(x, (str, int))]
        # 只保留已出现的 write_key（去掉前向依赖）
        rk = [k for k in rk if k in seen_write_keys]
        sn["read_keys"] = rk

        # write_key
        wk = sn.get("write_key")
        wk = str(wk) if wk not in (None, "") else None
        sn["write_key"] = wk
        if wk:
            seen_write_keys.add(wk)

        # output_format
        ofmt = sn.get("output_format")
        ofmt = str(ofmt).lower() if ofmt else None
        ofmt = ofmt if ofmt in allowed_fmt else default_fmt[tool]
        sn["output_format"] = ofmt

        norm_steps.append(sn)

    plan = {"steps": norm_steps}
    devlog["plan_json"] = json.dumps(plan, ensure_ascii=False, indent=2)
    return plan


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
    - 支持教案字段：role/strictness/instruction/read_keys/write_key/output_format/n_questions
    - 用简易“黑板” artifacts 在步骤间传递产物；
    - 每步执行时仅使用 topic 驱动工具，避免读取用户的流程指令；
    - 将 role/strictness/extra_context 传入 run_tool 以控制生成风格与依赖上下文。
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

    # 黑板：用于跨步骤传递产物
    artifacts: Dict[str, str] = {}

    def _build_extra_context(read_keys: List[str]) -> str:
        if not read_keys:
            return ""
        parts = []
        for k in read_keys:
            v = artifacts.get(k)
            if v:
                parts.append(f"[{k}]\n{str(v)}")
        # 统一截断，防止 prompt 过长
        extra = "\n\n".join(parts)
        return extra[:1200] if len(extra) > 1200 else extra

    def _artifact_from_records(recs: List[Dict[str, Any]]) -> str:
        """
        将一步产生的记录整合为可写入黑板的字符串：
        - answer: 取 text
        - mcq: 将 data JSON 串化；多题用 JSON Lines 拼接
        - card/mindmap: 取 text
        其他类型返回空串
        """
        buf: List[str] = []
        for r in recs:
            kind = r.get("kind")
            if kind == "answer":
                t = r.get("text") or ""
                if t:
                    buf.append(str(t))
            elif kind == "mcq":
                data = r.get("data")
                try:
                    buf.append(json.dumps(data, ensure_ascii=False))
                except Exception:
                    # 兜底：直接转字符串
                    buf.append(str(data))
            elif kind in ("card", "mindmap"):
                t = r.get("text") or ""
                if t:
                    buf.append(str(t))
        return "\n".join(buf).strip()

    # 依次执行每一步
    for idx, step in enumerate(steps, start=1):
        tool = step.get("tool", "answer")
        topic = (step.get("topic") or user_msg or "").strip()
        n_questions = int(step.get("n_questions", 1) or 1)
        role = step.get("role")
        strictness = step.get("strictness", "strict")
        instruction = step.get("instruction", "")  # 仅用于 devlog 记录
        read_keys: List[str] = step.get("read_keys", []) or []
        write_key = step.get("write_key")
        output_format = step.get("output_format")  # 仅用于 devlog 记录

        # devlog 标注本步信息
        devlog[f"step_{idx}_tool"] = tool
        devlog[f"step_{idx}_topic"] = topic
        devlog[f"step_{idx}_n_questions"] = n_questions
        devlog[f"step_{idx}_role"] = role or ""
        devlog[f"step_{idx}_strictness"] = strictness
        devlog[f"step_{idx}_instruction"] = instruction
        devlog[f"step_{idx}_read_keys"] = ",".join(read_keys)
        devlog[f"step_{idx}_write_key"] = write_key or ""
        devlog[f"step_{idx}_output_format"] = output_format or ""

        # 拼装跨步依赖上下文
        extra_context = _build_extra_context(read_keys)
        devlog[f"step_{idx}_extra_context_len"] = len(extra_context)

        # 执行
        step_records: List[Dict[str, Any]] = []
        if tool == "quiz" and n_questions > 1:
            # 多题：循环调用 quiz
            for j in range(n_questions):
                sub = run_tool(
                    mode="quiz",
                    proj=proj,
                    vs=vs,
                    llm=llm,
                    user_msg=f"(auto) quiz {j+1}/{n_questions} for {topic}",
                    topic=topic,
                    devlog=devlog,
                    role=role,
                    strictness=strictness,
                    extra_context=extra_context,
                    instruction = instruction
                )
                step_records.extend(sub)
        else:
            sub = run_tool(
                mode=tool,
                proj=proj,
                vs=vs,
                llm=llm,
                user_msg=f"(auto) {tool} for {topic}",
                topic=topic,
                devlog=devlog,
                role=role,
                strictness=strictness,
                extra_context=extra_context,
                instruction = instruction
            )
            step_records.extend(sub)

        # 累计到总记录
        records_all.extend(step_records)

        # 写入黑板
        if write_key:
            art = _artifact_from_records(step_records)
            if art:
                artifacts[write_key] = art
                devlog[f"step_{idx}_artifact_written"] = f"{write_key}:{len(art)}chars"
            else:
                devlog[f"step_{idx}_artifact_written"] = f"{write_key}:<empty>"

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
