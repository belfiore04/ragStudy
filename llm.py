import re
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from typing import Tuple, List, Dict, Any
from langchain.schema import Document
from config import MODEL_NAME, MODEL_BASE_URL, API_ENV_KEY
from rag_core import retrieve, format_hits


@st.cache_resource(show_spinner=False)
def get_llm():
    key = __import__("os").getenv(API_ENV_KEY, "").strip()
    if not key:
        st.error("未检测到 API Key。请设置环境变量 DEEPSEEK_API_KEY。")
        st.stop()
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=key,
        openai_api_base=MODEL_BASE_URL,
        temperature=0,
        timeout=60,
        max_retries=2,
    )
def rag_answer(
        llm: ChatOpenAI,
        vs,
        q: str,
        k: int,
        devlog: Dict[str, Any],
        history: List[Dict[str, Any]] | None = None,
        role: str | None = None,
        strictness: str = "strict",
        extra_context: str = "",
        instruction: str = ""
) -> Tuple[str, List[Document]]:
    # 1) 按需改写检索 query（可能等于原 q）
    rewritten_q = _rewrite_query_if_needed(llm, q, history, devlog)

    # 2) 检索
    hits = retrieve(vs, rewritten_q, k)
    ctx = format_hits(hits)

    # 3) 角色提示
    if role == "summary":
        role_hint = (
            "你现在处于“总结”环节：\n"
            "- 用简洁条目回顾核心要点；\n"
            "- 不引入新的具体事实。\n"
        )
    else:  # 默认为讲解
        role_hint = (
            "你现在处于“讲解”环节：\n"
            "- 先给出直观解释；\n"
            "- 再分点写出关键要点、步骤、注意点；\n"
            "- 如合适，可给一个很小的例子帮助理解。\n"
        )

    # 4) 严格度策略
    if str(strictness).lower() == "soft":
        strict_hint = (
            "严格度=soft：以 [CONTEXT] 为主，允许少量通用教学衔接语，"
            "但不要引入材料未出现的具体事实。\n"
        )
    else:
        strictness = "strict"
        strict_hint = (
            "严格度=strict：仅基于 [CONTEXT] 回答。若依据不足，直接回答“依据不足”。\n"
        )

    # 5) 组装可选的“前序产物”
    prev_part = ""
    extra_context = (extra_context or "").strip()
    if extra_context:
        prev_part = f"\n[PREVIOUS_ARTIFACTS]\n{extra_context[:1200]}"

    inst_part = ""
    instruction = (instruction or "").strip()
    if instruction:
        inst_part = f"\n[TEACHER_INSTRUCTION]\n{instruction[:400]}"
    
    # 6) Prompt
    prompt = (
        "You are a teacher-like study assistant.\n"
        "Answer strictly based on [CONTEXT]"
        + (" and [PREVIOUS_ARTIFACTS]" if prev_part else "")
        + ".\n"
        + strict_hint
        + role_hint
        + "回答使用中文。\n\n"
        f"{inst_part}\n"
        f"Original question: {q}\n"
        f"Search query (maybe rewritten): {rewritten_q}\n\n"
        f"[CONTEXT]\n{ctx}"
        + prev_part
    )

    devlog["prompt"] = prompt
    devlog["role"] = role or ""
    devlog["strictness"] = strictness
    devlog["extra_context_len"] = len(extra_context)
    devlog["instruction"] = instruction


    out = llm.invoke(prompt)
    devlog["raw"] = getattr(out, "content", str(out))
    return out.content, hits
def gen_mcq(
    llm: ChatOpenAI,
    context: str,
    devlog: Dict[str, Any],
    role: str | None = None,
    strictness: str = "strict",
    extra_context: str = "",
    instruction: str = ""
) -> Dict[str, Any]:
    # 角色与严格度提示
    if role == "intro_quiz":
        role_hint = "本题为引入题，难度偏简单，侧重基础概念或直观理解。"
    elif role == "check_understanding":
        role_hint = "本题用于检验刚讲过的要点，可稍微综合，但不应过难。"
    else:
        role_hint = "生成一题合适的单选题。"

    if str(strictness).lower() == "soft":
        strict_hint = (
            "严格度=soft：以 [MATERIAL] 为主，允许少量通用教学措辞；"
            "不得捏造材料中未出现的具体事实。"
        )
    else:
        strictness = "strict"
        strict_hint = (
            "严格度=strict：仅基于 [MATERIAL] 出题。若信息不足，"
            "请在 rationale 中说明“依据不足”，题干与选项仍需尽量基于已有材料。"
        )

    prev_part = ""
    extra_context = (extra_context or "").strip()
    if extra_context:
        prev_part = f"\n[PREVIOUS_ARTIFACTS]\n{extra_context[:800]}"

    inst_part = ""
    instruction = (instruction or "").strip()
    if instruction:
        inst_part = f"\n[TEACHER_INSTRUCTION]\n{instruction[:300]}"
    prompt = (
        "From the [MATERIAL]"
        + (" and [PREVIOUS_ARTIFACTS]" if prev_part else "")
        + ", create ONE single-choice question.\n"
        f"{role_hint}\n"
        f"{strict_hint}\n"
        f"{inst_part}\n"
        'Return JSON exactly: {"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","rationale":"..."}\n'
        "回答的非格式部分请用中文。\n"
        f"[MATERIAL]\n{context[:2500]}"
        + prev_part
    )

    devlog["prompt_mcq"] = prompt
    devlog["mcq_role"] = role or ""
    devlog["mcq_strictness"] = strictness
    devlog["instruction"] = instruction

    text = llm.invoke(prompt).content
    devlog["raw_mcq"] = text
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {
            "question": "Parse failed",
            "options": [],
            "answer": "",
            "rationale": text
        }
    return data
def gen_card_or_map(
    llm: ChatOpenAI,
    context: str,
    mode: str,
    devlog: Dict[str, Any],
    role: str | None = None,
    strictness: str = "strict",
    extra_context: str = "",
    instruction :str= ""
) -> str:
    # 产物形态
    if mode == "card":
        instr = (
            "生成‘知识卡片’的 Markdown，包含：\n"
            "- 核心定义\n- 关键公式/定理\n- 解题步骤\n- 易错点\n- 例题要点\n- 记忆提示\n"
        )
    else:
        instr = "生成层级化‘思维导图’（Markdown 缩进列表，最多 4 层）。"

    # 角色与严格度提示
    if role == "summary":
        role_hint = "这是总结性产物：提炼要点，避免引入新事实。"
    else:
        role_hint = "生成用于学习复习的结构化摘要。"

    if str(strictness).lower() == "soft":
        strict_hint = (
            "严格度=soft：以 [MATERIAL] 为主，允许少量通用教学措辞；"
            "不得捏造材料中未出现的具体事实。"
        )
    else:
        strictness = "strict"
        strict_hint = "严格度=strict：只根据 [MATERIAL] 生成；若信息不足则保持简洁。"

    prev_part = ""
    extra_context = (extra_context or "").strip()
    if extra_context:
        prev_part = f"\n[PREVIOUS_ARTIFACTS]\n{extra_context[:1200]}"

    inst_part = ""
    instruction = (instruction or "").strip()
    if instruction:
        inst_part = f"\n[TEACHER_INSTRUCTION]\n{instruction[:400]}"

    no_codeblock_hint = (
        "重要要求：\n"
        "- 不要在生成的中间使用代码块语法 ``` ，也不要输出任何 ```。\n"
        "- 如果要展示文法产生式、公式等，请用普通行或列表的形式书写，例如：\n"
        "  - S→bAb\n"
        "  - A→(B | a\n"
        "而不是放在 ``` 包裹的代码块中。\n"
    )
    prompt = (
        f"{instr}\n"
        f"{role_hint}\n"
        f"{strict_hint}\n"
        f"{no_codeblock_hint}\n"
        "输出用中文。\n\n"
        f"{inst_part}\n\n"
        f"[MATERIAL]\n{context[:4000]}"
        + prev_part
    )

    devlog["prompt_cardmap"] = prompt
    devlog["cardmap_role"] = role or ""
    devlog["cardmap_strictness"] = strictness
    devlog["instruction"] = instruction
    
    out = llm.invoke(prompt).content
    devlog["raw_cardmap"] = out
    return out


def _build_last_turn(history: List[Dict[str, Any]] | None) -> str:
    """
    只取最近一轮 user + assistant，对话摘要给改写用。
    """
    if not history:
        return ""

    # 从后往前找最近的 user / assistant 文本
    user_text = ""
    asst_text = ""
    for rec in reversed(history):
        role = rec.get("role")
        text = rec.get("text", "")
        if not text:
            continue
        if role == "assistant" and not asst_text:
            asst_text = text
        elif role == "user" and not user_text:
            user_text = text
        if user_text and asst_text:
            break

    if not user_text and not asst_text:
        return ""

    lines = []
    if user_text:
        lines.append(f"User: {user_text}")
    if asst_text:
        lines.append(f"Assistant: {asst_text}")
    return "\n".join(lines)

def _rewrite_query_if_needed(
    llm,
    q: str,
    history: List[Dict[str, Any]] | None,
    devlog: Dict[str, Any],
) -> str:
    """
    只有在存在指代 / 不清晰时才改写；否则原样返回 q。
    """
    last_turn = _build_last_turn(history)
    if not last_turn:
        return q

    prompt = (
        "你是一个“查询改写”助手。\n"
        "用户当前的问题里如果出现“这个东西”“它”“这类方法”等指代，"
        "请结合最近一轮对话，把问题改写成一个自包含、完整、具体的中文问题。\n"
        "如果当前问题本身已经足够清晰，不需要依赖上下文就能理解，"
        "那就原样输出当前问题，不要改写。\n"
        "只输出最终的问题文本，不要添加任何解释或前后缀。\n\n"
        f"[最近一轮对话]\n{last_turn}\n\n"
        f"[当前问题]\n{q}"
    )

    out = llm.invoke(prompt).content.strip()
    devlog["q_rewritten"] = out
    # 防止 LLM 弄丢信息：返回空就退回 q
    return out or q
