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
    
    def __init__(self, name: str, system_prompt: str, model: str = "openai/gpt-oss-120b"):
        self.id = f"agent_{uuid.uuid4().hex[:12]}"
        self.name = name
        self.description = ""
        self.system_prompt = system_prompt
        self.greeting = "Hi! How can I help you?"
        self.model = model
        self.temperature = 0.5
        self.max_tokens = 1000
        self.tools: List[str] = []
        self.knowledge_base: List[str] = []
        # If set, this bot can call the predict_with_model tool against this ONE registered
        # ML model (see GenesisAgent._do_bot_predict) - lets a published bot give real
        # predictions instead of only talking about them. Deliberately singular/scoped, not a
        # list - a bot predicting with an arbitrary set of the owner's models would need a much
        # more careful selection/disambiguation story than "describe your bot" currently offers.
        self.linked_model_id: Optional[str] = None
        # If set, a webhook URL POSTed to whenever this bot can't help and hands off to a
        # human (see GenesisAgent._do_request_handoff) - closes the loop on the existing
        # anti-hallucination deflection behavior (bot_builder.py), which tells a bot to say
        # "contact support" but previously had no actual mechanism behind those words. Webhook
        # only for now (Slack/Discord/Zapier all accept a plain JSON POST with no extra setup);
        # email delivery would need an SMTP/transactional-email provider this project doesn't
        # have configured, so it's a natural next step rather than a half-built one now.
        self.handoff_webhook_url: Optional[str] = None
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
            "linked_model_id": self.linked_model_id,
            "handoff_webhook_url": self.handoff_webhook_url,
            "author": self.author,
            "rating": self.rating,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
            "published": self.published,
            "price_per_call": self.price_per_call,
        }


class AgentStore:
    """Хранилище агентов."""
    
    # Model catalog data lives in app.core.models_catalog (single source of truth,
    # verified against console.groq.com/docs/models) - see MODEL_CATALOG / to_dict()
    # there and the /api/agents/models* endpoints in main.py. Not duplicated here.

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
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump({"agents": [], "featured": [], "categories": {}}, f, indent=2)
    
    def _load(self) -> Dict:
        with open(self.store_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _save(self, data: Dict):
        with open(self.store_path, "w", encoding="utf-8") as f:
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
    
    def publish_agent(self, agent_id: str, price: float = 0, owner_user_id: str = None) -> bool:
        """Опубликовать агента. Returns False if the agent doesn't exist or doesn't belong to owner_user_id."""
        data = self._load()
        for a in data["agents"]:
            if a["id"] == agent_id:
                if owner_user_id is not None and a["author"] != owner_user_id:
                    return False
                a["published"] = True
                a["price_per_call"] = price
                self._save(data)
                return True
        return False

    ALLOWED_UPDATE_FIELDS = {"name", "description", "system_prompt", "greeting", "model",
                             "temperature", "max_tokens", "tools", "price_per_call", "published",
                             "linked_model_id", "handoff_webhook_url"}

    def update_agent(self, agent_id: str, updates: Dict, owner_user_id: str) -> Optional[Dict]:
        """Update a bot's settings. Only fields in ALLOWED_UPDATE_FIELDS are applied.
        Returns the updated agent dict, or None if it doesn't exist or isn't owned by owner_user_id."""
        data = self._load()
        for a in data["agents"]:
            if a["id"] == agent_id:
                if a["author"] != owner_user_id:
                    return None
                for k, v in updates.items():
                    if k in self.ALLOWED_UPDATE_FIELDS:
                        a[k] = v
                self._save(data)
                return a
        return None

    def delete_agent(self, agent_id: str, owner_user_id: str) -> bool:
        """Delete a bot. Returns False if it doesn't exist or isn't owned by owner_user_id."""
        data = self._load()
        target = next((a for a in data["agents"] if a["id"] == agent_id), None)
        if not target or target["author"] != owner_user_id:
            return False
        data["agents"] = [a for a in data["agents"] if a["id"] != agent_id]
        self._save(data)
        return True

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
