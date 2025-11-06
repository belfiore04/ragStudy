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
) -> Tuple[str, List[Document]]:
    # 1) 先按需改写检索 query（可能等于原 q）
    rewritten_q = _rewrite_query_if_needed(llm, q, history, devlog)
    hits = retrieve(vs, rewritten_q, k)
    ctx = format_hits(hits)
    prompt = (
        "你是一个仿老师的学习助手。你需要严格依照 [CONTEXT]来回答问题。 "
        "如果依据不足，请你诚实回答'依据不足'。\n"
        "回答要求：\n"
        "1) 用中文回答；\n"
        "2) 先用简短的自然语言解释核心概念或结论；\n"
        "3) 然后用要点式（bullet points）分条展开说明：关键理由、步骤、注意点；\n"
        "4) 如果适合，可以顺带给出一个简单的小例子帮助理解。\n\n"
        f"Question: {q}\n\n[CONTEXT]\n{ctx}"
    )

    devlog["prompt"] = prompt
    out = llm.invoke(prompt)
    devlog["raw"] = getattr(out, "content", str(out))
    return out.content, hits

def gen_mcq(llm: ChatOpenAI, context: str, devlog: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
    "From the [MATERIAL], create ONE single-choice question. "
    "Provide options A-D, the correct letter, and one-sentence rationale. "
    'Return JSON: {"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","rationale":"..."}'
    "\n回答的非格式部分请用中文。\n"
    f"[MATERIAL]\n{context[:2500]}"
    )
    devlog["prompt_mcq"] = prompt
    text = llm.invoke(prompt).content
    devlog["raw_mcq"] = text
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {"question": "Parse failed", "options": [], "answer": "", "rationale": text}
    return data

def gen_card_or_map(llm: ChatOpenAI, context: str, mode: str, devlog: Dict[str, Any]) -> str:
    if mode == "card":
        instr = (
        "将学习材料整理成‘知识卡片’的 Markdown。结构包含：\n"
        "- 核心定义\n- 关键公式/定理\n- 解题步骤\n- 易错点\n- 例题要点\n- 记忆提示\n"
        )
    else:
        instr = "将学习材料提炼为层级化‘思维导图’（Markdown 缩进列表，最多 4 层）。"
    prompt = instr + "\n只根据[MATERIAL]内容，不要外推。输出用中文。\n\n[MATERIAL]\n" + context[:4000]
    devlog["prompt_cardmap"] = prompt
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
