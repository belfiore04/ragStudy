# tools.py
import re
import time
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
from langchain.schema import Document
from rag_core import retrieve
from llm import _rewrite_query_if_needed, rag_answer, gen_mcq, gen_card_or_map
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
    q = topic or user_msg
    if mode == "quiz":
        hits_r = retrieve(vs, topic, k=8)
        ctx = "\n\n".join(d.page_content[:600] for d in hits_r)
        try:
            data = gen_mcq(
                llm,
                ctx,
                devlog,
                strictness=strictness,
                extra_context=extra_context,
                instruction = instruction,
                topic = topic
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
            out = gen_card_or_map(
                llm,
                ctx,
                mode_cardmap if mode_cardmap in ("card", "mindmap") else "card",
                devlog,
                strictness=strictness,
                extra_context=extra_context,
                instruction = instruction,
                topic = topic
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
        q = topic or user_msg
        ans, hits_r = rag_answer(
            llm, vs, q,
            k=4,
            devlog=devlog,
            strictness=strictness,
            extra_context=extra_context,
            instruction = instruction
        )
        docs = [Document(page_content=h.page_content, metadata=h.metadata) for h in hits_r]
        render_evidence_cards(proj, docs)
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
def llm_route_tool(
        llm,
        user_msg: str,
        devlog: Optional[Dict[str, Any]] = None, 
        history: Optional[List[Dict[str, Any]]] = None,
        ) -> Tuple[str, str]:
    """
    统一的工具路由入口：
    1) 先用规则处理显式指令（/quiz, /card, /map、中文关键词），直接返回；
    2) 再用 LLM 选择 tool: answer | quiz | card | map。

    返回: (mode, topic)
    """
    text = user_msg.strip()
    #重写query
    if devlog is None:
        devlog = {}

    # 1) 改写一次（之后全局都用 rewritten_q）
    with st.spinner("正在分析并重建问题"):
        text = _rewrite_query_if_needed(llm, user_msg, history, devlog)
    devlog["route_original_q"] = user_msg
    devlog["route_rewritten_q"] = text

    # # === 1. 规则优先：兼容旧的显式指令 & 中文关键词 ===
    # # 出题
    # if ("/quiz" in lower) or ("生成题目" in text) or ("测验" in text) or ("出几道题" in text):
    #     # 提取 /quiz 后面的部分作为 topic，兼容以前写法
    #     after = re.sub(r"^.*?/quiz", "", lower).strip()
    #     topic = after or text
    #     return "quiz", topic

    # # 知识卡片
    # if ("/card" in lower) or ("知识卡片" in text) or ("整理成知识点" in text) or ("做成卡片" in text):
    #     after = re.sub(r"^.*?/card", "", lower).strip()
    #     topic = after or text
    #     return "card", topic

    # # 思维导图
    # if ("/map" in lower) or ("思维导图" in text) or ("梳理结构" in text) or ("画个导图" in text):
    #     after = re.sub(r"^.*?/map", "", lower).strip()
    #     topic = after or text
    #     return "map", topic

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


def llm_make_plan(llm, user_msg: str, devlog: Dict[str, Any],  history: Optional[List[Dict[str, Any]]] = None,) -> Dict[str, Any]:
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
    example = ('''
        ```json
{
  "steps": [
    {
      "id": 1,
      "tool": "quiz",
      "topic": "自底向上概念的直观解释和初步引入",
      "instruction": "你在教学计划的开头位置，保持题目简单，带有引入性质",
      "strictness": "soft",
      "read_keys": [],
      "write_key": "pre_knowledge"
    },
    {
      "id": 2,
      "tool": "answer",
      "topic": "自底向上的基本定义和核心思想",
      "instruction": "让你的语言保持生动",
      "strictness": "strict",
      "read_keys": ["pre_knowledge"],
      "write_key": "basic_concept"
    },
    {
      "id": 3,
      "tool": "card",
      "topic": "自底向上的步骤",
      "instruction": "详细解释抽象的步骤，最好举出例子",
      "strictness": "strict",
      "read_keys": ["basic_concept"],
      "write_key": "key_features"
    },
    {
      "id": 4,
      "tool": "quiz",
      "topic": "自底向上定义理解检测",
      "instruction": "请不要脱离上面给出的定义范围出题",
      "strictness": "strict",
      "read_keys": ["basic_concept", "key_features"],
      "write_key": "understanding_level"
    },
    {
      "id": 5,
      "tool": "map",
      "topic": "自底向上知识体系",
      "instruction": "总体总结一遍上面的知识，只注意宏观完整即可",
      "strictness": "soft",
      "read_keys": ["basic_concept", "key_features", "understanding_level"],
      "write_key": "knowledge_map"
    },
    {
      "id": 6,
      "tool": "answer",
      "topic": "自底向上的应用",
      "instruction": "你需要保持内容丰富充实，你在回答的结尾部分，所以需要让你的回答带有总结收束性质",
      "strictness": "soft"
      "read_keys": ["knowledge_map"],
      "write_key": "final_summary",
    }
  ]
}
```
''')
    system_prompt = (
    '''你是一个教学“教案员”。你不直接讲解知识，也不生成最终内容。\n
        你的任务是按“老师上课”的节奏，规划 1–6 个步骤的教学流程，供下游工具执行。\n
        你的工具分别是：answer(讲解/总结, 输出text)、quiz(单选题, 输出mcq_json)、
        card(知识卡片, 输出markdown)、map(思维导图, 输出markdown)。\n
        你要注意对整体教学节奏和流程的把控，根据用户输入自行给出一个“上下衔接连贯”的教学安排。\n
        以下是几种常见的教学节奏供你参考（但不必拘泥）：
        文字引入-出题-文字讲解-总结/出题-文字讲解-出题-总结/先总结-分别出题-最后讲解。\n
        你在回答时“必须”遵守这些规则：\n
        1.你的回答应该是JSON形式：{\"steps\":[{...}]}；\n
        2.每一个step需要包含这些字段：id（1、2……）/tool/topic/instruction/strictness/read_keys/write_key；\n
        3.tool字段描述这一步需要使用的工具，范围：answer、quiz、card、map。
        其中，answer和quiz应该是你教学计划的主要内容。
        map只在你认为有必要梳理所有知识的宏观结构时或用户明确指名时使用，且应该放在结尾或接近结尾。
        card只在你认为有必要提取出讲的内容里较难、复杂、不易理解的知识时或用户明确指名时使用，且应该放在结尾。
        如果你认为知识非常简单、基础或者没有什么好提取的，你可以不规划 card。
        card和map都起到总结梳理作用，因此“禁止”在其中出现前面没有讲过的内容，且它们之间互相不应重复。
        请你在规划topic和read_keys时想好。\n
        4.topic字段应该填入在你的教学节奏中这一步的教学内容，
        请使你的教学内容精准、带有逻辑且互不重复，下游工具将会严格按照你的教学内容生成；\n
        5.insruction字段应该填入你觉得为了让下游工具更好地生成内容，它需要知道的额外信息。
        这些可能包括：在整个计划中的位置、回答的详细程度、难易度等；\n
        6.strictness字段描述这一步回答的严谨程度，范围：strict、soft。如果涉及到开放知识的生成
        （比如例子、引入、补充知识等），可以使用soft，否则使用strict；\n
        7.为了实现后一步在生成时能看到之前步骤的结果，你有一块黑板。\n
        你可以通过read_keys和write_key字段来控制这一步要不要向黑板写或者从黑板读。\n
        比如你在step2的write_key写了"aaa"，在step3的read_keys写了"aaa"，
        那step3就会在生成时在黑板上找到名为"aaa"的项，内容就是step2的生成内容。\n
        你“禁止”输出任何多余文字。\n
        以下是你回答的一个示例：\n
        '''
        + example
    )
    rewritten_q = _rewrite_query_if_needed(llm, text, history, devlog)
    user_prompt = f"用户输入：{rewritten_q}\n请开始给出你的plan。"
    prompt = system_prompt + "\n\n" + user_prompt
    devlog["plan_prompt"] = prompt
    out = llm.invoke(prompt)
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
        plan = {"steps": [{"id": "1", "tool": "answer", "topic": text, "instruction": "请你详细解释",
                           "strictness": "strict", "read_keys": [], "write_key": []}]}
        devlog["plan_json"] = json.dumps(plan, ensure_ascii=False, indent=2)
        return plan

    allowed_tools = {"answer", "quiz", "card", "map"}
    allowed_strict = {"strict", "soft"}

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

        # strictness
        strict = str(sn.get("strictness", "strict")).lower()
        if strict not in allowed_strict:
            strict = "strict"
        sn["strictness"] = strict

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
    label_map = {
        "answer": "讲解回答",
        "quiz": "练习题",
        "card": "知识卡片",
        "map": "思维导图",
    }
    # 依次执行每一步
    for idx, step in enumerate(steps, start=1):
        tool = step.get("tool", "answer")
        topic = (step.get("topic") or user_msg or "").strip()
        strictness = step.get("strictness", "strict")
        instruction = step.get("instruction", "")  # 仅用于 devlog 记录
        read_keys: List[str] = step.get("read_keys", []) or []
        write_key = step.get("write_key")

        # devlog 标注本步信息
        devlog[f"step_{idx}_tool"] = tool
        devlog[f"step_{idx}_topic"] = topic
        devlog[f"step_{idx}_strictness"] = strictness
        devlog[f"step_{idx}_instruction"] = instruction
        devlog[f"step_{idx}_read_keys"] = ",".join(read_keys)
        devlog[f"step_{idx}_write_key"] = write_key or ""

        # 拼装跨步依赖上下文
        extra_context = _build_extra_context(read_keys)
        devlog[f"step_{idx}_extra_context_len"] = len(extra_context)
        label = label_map.get(tool, "内容")
        base_msg = f"第 {idx} 步  正在生成{label}：{topic}"
        # 执行
        step_records: List[Dict[str, Any]] = []
        with st.spinner(base_msg):
            sub = run_tool(
                mode=tool,
                proj=proj,
                vs=vs,
                llm=llm,
                user_msg=f"(auto) {tool} for {topic}",
                topic=topic,
                devlog=devlog,
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
