import json
from pathlib import Path
from typing import List, Optional
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import EMB_MODEL


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True})




def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)




def build_index(docs: List[Document]) -> FAISS:
    chunks = split_docs(docs)
    emb = get_embeddings()
    return FAISS.from_documents(chunks, emb)




def save_index(vs: FAISS, index_dir: Path):
    index_dir.mkdir(exist_ok=True, parents=True)
    vs.save_local(str(index_dir))
    (index_dir / "stamp.json").write_text(json.dumps({"built_at": int(__import__('time').time())}), encoding="utf-8")




def try_load_index(index_dir: Path) -> Optional[FAISS]:
    if not index_dir.exists():
        return None
    emb = get_embeddings()
    try:
        return FAISS.load_local(str(index_dir), embeddings=emb, allow_dangerous_deserialization=True)
    except Exception:
        return None




def retrieve(vs: FAISS, q: str, k: int) -> List[Document]:
    return vs.similarity_search(q, k=k)




def format_hits(hits: List[Document]) -> str:
    out = []
    for d in hits:
        meta = d.metadata or {}
        src = Path(meta.get("source", "?")).name
        tag = f"[{src}"
        if "page" in meta: tag += f" P{meta['page']}"
        if "slide" in meta: tag += f" S{meta['slide']}"
        tag += "]"
        out.append(tag + "\n" + d.page_content.strip())
    return "\n\n".join(out)