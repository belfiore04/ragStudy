import json
from pathlib import Path
from typing import Dict, Any, List
from utils import now_ts


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.meta_path = root / "project.json"
        self.files_dir = root / "files"
        self.index_dir = root / "index"
        self.preview_dir = root / "previews"
        self.chat_path = root / "chats.jsonl"
        self.wrong_path = root / "wrong.jsonl"
        self.meta: Dict[str, Any] = {}


    def exists(self) -> bool:
        return self.meta_path.exists()
    
    
    def load_meta(self):
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
    
    
    def save_meta(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    
    # --- 聊天 ---
    def append_chat(self, record: Dict[str, Any]):
        with open(self.chat_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    
    def load_chats(self, limit: int = 200) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.chat_path.exists():
            return out
        with open(self.chat_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except: # noqa
                    pass
        return out[-limit:]
    
    
    # --- 错题本 ---
    def log_wrong(self, record: Dict[str, Any]):
        with open(self.wrong_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    
    def load_wrong(self) -> List[Dict[str, Any]]:
        if not self.wrong_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with open(self.wrong_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except: # noqa
                    pass
        return out