"""
Agent Config Store — хранилище конфигураций ИИ-агентов.

Каждый агент = JSON-конфигурация с:
- Системным промптом
- Выбранной моделью
- Инструментами
- Настройками
- Базой знаний
"""

import json, os, uuid
from typing import Dict, List, Any, Optional
from datetime import datetime


class AgentConfig:
    """Конфигурация одного ИИ-агента."""
    
    def __init__(self, name: str, system_prompt: str, model: str = "groq-70b"):
        self.id = f"agent_{uuid.uuid4().hex[:12]}"
        self.name = name
        self.description = ""
        self.system_prompt = system_prompt
        self.greeting = "Привет! Как я могу помочь?"
        self.model = model
        self.temperature = 0.5
        self.max_tokens = 1000
        self.tools: List[str] = []
        self.knowledge_base: List[str] = []
        self.author = ""
        self.rating = 0.0
        self.usage_count = 0
        self.created_at = datetime.now().isoformat()
        self.published = False
        self.price_per_call = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "greeting": self.greeting,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "knowledge_base": self.knowledge_base,
            "author": self.author,
            "rating": self.rating,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
            "published": self.published,
            "price_per_call": self.price_per_call,
        }


class AgentStore:
    """Хранилище агентов."""
    
    AVAILABLE_MODELS = {
        "gpt-oss-120b": {"name": "GPT OSS 120B", "provider": "Groq", "params": "120B", "type": "Reasoning"},
        "gpt-oss-20b": {"name": "GPT OSS 20B", "provider": "Groq", "params": "20B", "type": "Fast"},
        "llama-4-scout": {"name": "Llama 4 Scout", "provider": "Groq", "params": "109B", "type": "Vision"},
        "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B", "provider": "Groq", "params": "70B", "type": "Multilingual"},
        "qwen-3-32b": {"name": "Qwen 3 32B", "provider": "Groq", "params": "32B", "type": "Tool Use"},
        "qwen-3.6-27b": {"name": "Qwen 3.6 27B", "provider": "Groq", "params": "27B", "type": "Vision"},
        "local-7b": {"name": "Qwen 2.5 7B", "provider": "Local", "params": "7B", "type": "Fallback"},
    }
    
    AVAILABLE_TOOLS = [
        {"id": "web_search", "name": "Web Search", "icon": "🌐", "desc": "Поиск в интернете"},
        {"id": "dataset_search", "name": "Dataset Search", "icon": "📊", "desc": "Поиск датасетов"},
        {"id": "code_generation", "name": "Code Gen", "icon": "💻", "desc": "Генерация кода"},
        {"id": "model_training", "name": "Model Training", "icon": "🤖", "desc": "Обучение ML"},
        {"id": "research", "name": "Research", "icon": "🔍", "desc": "Исследование рынка"},
    ]
    
    def __init__(self, db=None):
        self.db = db
        self._init_store()
    
    def _init_store(self):
        os.makedirs("agents", exist_ok=True)
        self.store_path = "agents/store.json"
        if not os.path.exists(self.store_path):
            with open(self.store_path, "w") as f:
                json.dump({"agents": [], "featured": [], "categories": {}}, f, indent=2)
    
    def _load(self) -> Dict:
        with open(self.store_path, "r") as f:
            return json.load(f)
    
    def _save(self, data: Dict):
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_agent(self, config: AgentConfig) -> AgentConfig:
        """Создать нового агента."""
        data = self._load()
        data["agents"].append(config.to_dict())
        self._save(data)
        return config
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Получить агента по ID."""
        data = self._load()
        for a in data["agents"]:
            if a["id"] == agent_id:
                return a
        return None
    
    def list_agents(self, author: str = None, published_only: bool = True) -> List[Dict]:
        """Список агентов с фильтрацией."""
        data = self._load()
        agents = data["agents"]
        if author:
            agents = [a for a in agents if a["author"] == author]
        if published_only:
            agents = [a for a in agents if a["published"]]
        return agents
    
    def search_agents(self, query: str) -> List[Dict]:
        """Поиск по агентам."""
        data = self._load()
        q = query.lower()
        return [a for a in data["agents"] if q in a["name"].lower() or q in a["description"].lower()]
    
    def publish_agent(self, agent_id: str, price: float = 0):
        """Опубликовать агента."""
        data = self._load()
        for a in data["agents"]:
            if a["id"] == agent_id:
                a["published"] = True
                a["price_per_call"] = price
                break
        self._save(data)
    
    def clone_agent(self, agent_id: str, new_author: str) -> Optional[AgentConfig]:
        """Клонировать агента."""
        original = self.get_agent(agent_id)
        if not original:
            return None
        config = AgentConfig(
            name=f"{original['name']} (copy)",
            system_prompt=original["system_prompt"]
        )
        config.author = new_author
        config.description = original["description"]
        config.model = original["model"]
        config.tools = original["tools"]
        return self.create_agent(config)
    
    def get_featured(self) -> List[Dict]:
        """Избранные агенты."""
        data = self._load()
        return [self.get_agent(aid) for aid in data.get("featured", []) if self.get_agent(aid)]
    
    def get_categories(self) -> Dict:
        """Категории агентов."""
        data = self._load()
        return data.get("categories", {})
    
    def add_review(self, agent_id: str, user: str, rating: int, comment: str = ""):
        """Добавить отзыв."""
        data = self._load()
        for a in data["agents"]:
            if a["id"] == agent_id:
                if "reviews" not in a:
                    a["reviews"] = []
                a["reviews"].append({
                    "user": user, "rating": rating, "comment": comment,
                    "date": datetime.now().isoformat()
                })
                # Пересчёт рейтинга
                ratings = [r["rating"] for r in a["reviews"]]
                a["rating"] = round(sum(ratings) / len(ratings), 1)
                break
        self._save(data)
