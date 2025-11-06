import hashlib
import time
from typing import Dict, Any, List
import unicodedata
import re
def sha1_of_bytes(data: bytes) -> str:
    h = hashlib.sha1(); h.update(data); return h.hexdigest()


def now_ts() -> int:
    return int(time.time())


def due_wrong(items: List[Dict[str, Any]], now: int | None = None) -> List[Dict[str, Any]]:
    now = now or now_ts()
    due = []
    for it in items:
        box = it.get("box", 1)
        last = it.get("last", it.get("t", now))
        gap_days = {1: 1, 2: 2, 3: 4}.get(box, 1)
        if now - last >= gap_days * 86400:
            due.append(it)
    return due
import hashlib
import time
from typing import Dict, Any, List


def sha1_of_bytes(data: bytes) -> str:
    h = hashlib.sha1(); h.update(data); return h.hexdigest()


def now_ts() -> int:
    return int(time.time())


def due_wrong(items: List[Dict[str, Any]], now: int | None = None) -> List[Dict[str, Any]]:
    now = now or now_ts()
    due = []
    for it in items:
        box = it.get("box", 1)
        last = it.get("last", it.get("t", now))
        gap_days = {1: 1, 2: 2, 3: 4}.get(box, 1)
        if now - last >= gap_days * 86400:
            due.append(it)
    return due

def slugify_name(name: str) -> str:
    """将任意项目名转为仅包含 ascii 字符的安全目录名"""
    name = name.strip()
    if not name:
        return f"proj_{now_ts()}"
    # 全角转半角等归一化
    name = unicodedata.normalize("NFKD", name)
    # 非字母数字用下划线替换
    name_ascii = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    # 防止全被替换空掉
    if not name_ascii.strip("_"):
        name_ascii = f"proj_{now_ts()}"
    return name_ascii[:64]  # 防止太长