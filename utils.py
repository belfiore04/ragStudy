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