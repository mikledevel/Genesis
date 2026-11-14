"""
Custom Blocks Store - пользовательские блоки для Pipeline Builder.
"""

import json, os
import uuid
from app.db.document_store import DatabaseDocument, legacy_document, document_mutation
from typing import Dict, List, Optional
from datetime import datetime


class CustomBlock:
    def __init__(self, name: str, code: str, icon: str = "🔧", category: str = "custom"):
        self.id = "block_" + uuid.uuid4().hex
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
        self.path = "blocks/store.json"
        self._document = DatabaseDocument("blocks", legacy_document(self.path, {"blocks": []}))

    def _load(self):
        return self._document.read()

    def _save(self, data):
        self._document.replace(data)

    @document_mutation
    def create(self, name: str, code: str, icon: str = "🔧", category: str = "custom", author: str = "") -> Dict:
        block = CustomBlock(name, code, icon, category)
        block.author = author
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

    @document_mutation
    def delete_block(self, block_id: str, owner_user_id: str) -> bool:
        """Only removes the block if owner_user_id created it. Returns False (nothing
        deleted) otherwise - previously this had no ownership check at all, so any
        unauthenticated caller could delete any user's custom block by id."""
        data = self._load()
        before = len(data["blocks"])
        target = next((b for b in data["blocks"] if b["id"] == block_id), None)
        if target is None or target.get("author") != owner_user_id:
            return False
        data["blocks"] = [b for b in data["blocks"] if b["id"] != block_id]
        self._save(data)
        return len(data["blocks"]) < before

    def get_block(self, block_id: str) -> Optional[Dict]:
        data = self._load()
        for b in data["blocks"]:
            if b["id"] == block_id: return b
        return None