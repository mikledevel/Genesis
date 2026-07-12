"""
Custom Blocks Store - пользовательские блоки для Pipeline Builder.
"""

import json, os
from typing import Dict, List, Optional
from datetime import datetime


class CustomBlock:
    def __init__(self, name: str, code: str, icon: str = "🔧", category: str = "custom"):
        self.id = f"block_{datetime.now().strftime('%Y%m%d%H%M%S')}_{abs(hash(name)) % 10000:04d}"
        self.name = name
        self.code = code
        self.icon = icon
        self.category = category
        self.author = ""
        self.created_at = datetime.now().isoformat()
        self.usage_count = 0

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "code": self.code,
            "icon": self.icon, "category": self.category,
            "author": self.author, "created_at": self.created_at,
            "usage_count": self.usage_count
        }


class CustomBlockStore:
    def __init__(self):
        os.makedirs("blocks", exist_ok=True)
        self.path = "blocks/store.json"
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"blocks": []}, f, indent=2)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f: return json.load(f)

    def _save(self, data):
        with open(self.path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)

    def create(self, name: str, code: str, icon: str = "🔧", category: str = "custom") -> Dict:
        block = CustomBlock(name, code, icon, category)
        data = self._load()
        data["blocks"].append(block.to_dict())
        self._save(data)
        return block.to_dict()

    def list_blocks(self, category: str = None) -> List[Dict]:
        data = self._load()
        blocks = data["blocks"]
        if category:
            blocks = [b for b in blocks if b["category"] == category]
        return blocks

    def delete_block(self, block_id: str):
        data = self._load()
        data["blocks"] = [b for b in data["blocks"] if b["id"] != block_id]
        self._save(data)

    def get_block(self, block_id: str) -> Optional[Dict]:
        data = self._load()
        for b in data["blocks"]:
            if b["id"] == block_id: return b
        return None