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
        role: str | None = None,
        strictness: str = "strict",
        extra_context: str = "",
        instruction: str = ""
) -> Tuple[str, List[Document]]:
    # q是话题，已不是问题
    # 1) 按需改写检索 query（可能等于原 q）


    # 2) 检索
    hits = retrieve(vs, q, k)
    ctx = format_hits(hits)

    # 4) 严格度策略
    if str(strictness).lower() == "soft":
        strict_hint = (
            "你可以以教案为参考，自行发挥来执行好你的指示。"
        )
    else:
        strictness = "strict"
        strict_hint = (
            '''你需要以教案作为主要参考去输出教学内容。
            如果依据不足，你可以适当补充一些内容，但总体上不能和教案冲突。\n'''
        )

    # 5) 组装可选的“前序产物”
    prev_part = ""
    extra_context = (extra_context or "").strip()
    if extra_context:
        prev_part = f"{extra_context[:1200]}"

    inst_part = ""
    instruction = (instruction or "").strip()
    if instruction:
        inst_part = f"{instruction[:400]}"
    
    #6) Prompt
    prompt = (
        "你是一位辅助学习的老师，你和其他多位老师合作一起帮助一位同学学习。这次你需要给这位同学教的知识主题是："
        f"{q}。\n"
        "你必须严格遵守知识主题进行讲解，不可以多回答或者少回答。"
        "在讲解时，你需要遵循你的教案员给你的指示："
        f"{inst_part}。\n"
        + ("你在回答时可能需要参考前面老师已经讲解的内容：\n" if prev_part else "")
        + f"{prev_part}\n"
        +"你有一本教案，"
        +f"{strict_hint}\n"
        +"教案的内容如下：\n"
        f"{ctx}\n"
        +"你可以开始生成了。"
    )


    devlog["prompt"] = prompt
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
    strictness: str = "strict",
    extra_context: str = "",
    instruction: str = "",
    topic: str = ""
) -> Dict[str, Any]:
    # 角色与严格度提示

    if str(strictness).lower() == "soft":
        strict_hint = (
            "你可以以教案为参考，自行发挥来执行好你的指示。"
        )
    else:
        strictness = "strict"
        strict_hint = (
            '''你需要以教案作为主要参考去输出教学内容。
            如果依据不足，你可以适当补充一些内容，但总体上不能和教案冲突。\n'''
        )

    prev_part = ""
    extra_context = (extra_context or "").strip()
    if extra_context:
        prev_part = f"{extra_context[:800]}"

    inst_part = ""
    instruction = (instruction or "").strip()
    if instruction:
        inst_part = f"{instruction[:300]}"
    
    prompt = (
        "你是一位出单选题的老师，你和其他多位老师合作一起帮助一位同学学习。这次你需要给这位同学出题的主题是："
        f"{topic}。\n"
        "你必须严格遵守主题出一道单选题。"
        "在出题时，你需要遵循你的教案员给你的指示："
        f"{inst_part}。\n"
        +'你的回答格式必须为JSON样式：{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","rationale":"..."}\n'
        +"你在rationale部分用一句话来解释这道题，并且带出这道题的答案字母。\n"
        + ("你在出题时可能需要参考前面老师已经讲解的内容：\n" if prev_part else "")
        + f"{prev_part}\n"
        +"你有一本教案，"
        +f"{strict_hint}\n"
        +"教案的内容如下：\n"
        f"{context}\n"
        +"你可以开始出题了。"
    )
    devlog["prompt_mcq"] = prompt
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
    strictness: str = "strict",
    extra_context: str = "",
    instruction :str= "",
    topic:str = ""
) -> str:
    # 产物形态
    if mode == "card":
        instr = (
            '''你是一位给学生做知识卡片的老师，你和其他多位老师合作一起帮助一位同学学习。
            你必须保证你的知识卡片足够简短、凝练。
            你的输出需要使用Markdown格式。
            这次你需要给这位同学做的知识卡片主题是：'''
        )
    else:
        instr = (
            '''你是一位给学生做思维导图的老师，你和其他多位老师合作一起帮助一位同学学习。
            你的输出需要使用Markdown格式。
            这次你需要给这位同学做的思维导图主题是：'''
        )


    if str(strictness).lower() == "soft":
        strict_hint = (
            "你可以以教案为参考，自行发挥来执行好你的指示。"
        )
    else:
        strictness = "strict"
        strict_hint = (
            '''你需要以教案作为主要参考去输出教学内容。
            如果依据不足，你可以适当补充一些内容，但总体上不能和教案冲突。\n'''
        )

    prev_part = ""
    extra_context = (extra_context or "").strip()
    if extra_context:
        prev_part = f"{extra_context[:1200]}"

    inst_part = ""
    instruction = (instruction or "").strip()
    if instruction:
        inst_part = f"{instruction[:400]}"

    no_codeblock_hint = (
        "重要要求：\n"
        "- 不要在生成的中间使用代码块语法 ``` ，也不要输出任何 ```。\n"
        "- 如果要展示文法产生式、公式等，请用普通行或列表的形式书写，例如：\n"
        "  - S→bAb\n"
        "  - A→(B | a\n"
        "而不是放在 ``` 包裹的代码块中。\n"
    )
    prompt = (
        instr
        +f"{topic}。\n"            
        "你必须严格遵守知识主题进行制作，不可以多写或者少写。"
        "在制作时，你需要遵循你的教案员给你的指示："
        f"{inst_part}。\n"
        + ("你在回答时可能需要参考前面老师已经讲解的内容：\n" if prev_part else "")
        + f"{prev_part}\n"
        +"你有一本教案，"
        +f"{strict_hint}\n"
        +"教案的内容如下：\n"
        f"{context}\n"
        +"你可以开始生成了。"
        +f"{no_codeblock_hint}\n"
    )

    devlog["prompt_cardmap"] = prompt

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
