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

def rag_answer(llm: ChatOpenAI, vs, q: str, k: int, devlog: Dict[str, Any]) -> Tuple[str, List[Document]]:
    hits = retrieve(vs, q, k)
    ctx = format_hits(hits)
    prompt = (
    "You are a study assistant. Answer strictly based on [CONTEXT]. "
    "If evidence is insufficient, say 'Insufficient evidence'. "
    "Use bullet points. End with '参考: <file:P/S,...>'.\n"
    "回答的非格式部分请用中文。"
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