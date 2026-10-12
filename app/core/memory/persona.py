"""
Persona Core - генерация "ядра личности" ИИ на основе графа знаний.

Раз в N диалогов или по расписанию анализирует накопленные факты
и создаёт сжатый портрет пользователя для системного промпта.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.core.memory.graph import KnowledgeGraph


class PersonaGenerator:
    """
    Генератор персонального профиля.
    
    Сжимает граф знаний в короткий текстовый портрет
    для подачи в системный промпт LLM.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.last_generated: Optional[str] = None
        self.persona_cache: Optional[str] = None
    
    def generate(self, user_id: str = "user") -> str:
        """
        Генерация персоны пользователя.
        
        Returns:
            Текстовый портрет пользователя (200-500 слов)
        """
        facts = self.graph.get_facts_about(user_id, min_confidence=0.4)
        
        if not facts:
            return "Пользователь. Информация пока не собрана."
        
        # Группируем факты по категориям
        persona_parts = {
            "personal": [],
            "preferences": [],
            "goals": [],
            "context": [],
        }
        
        for fact in facts:
            pred = fact["predicate"]
            val = fact["value"]
            conf = fact["confidence"]
            
            if conf < 0.5:
                continue
            
            if pred in ("name", "age", "city", "profession"):
                persona_parts["personal"].append(f"{pred}={val}")
            elif pred in ("likes", "dislikes", "hobby"):
                persona_parts["preferences"].append(f"{pred}={val}")
            elif pred in ("goal", "deadline"):
                persona_parts["goals"].append(f"{pred}={val}")
            else:
                persona_parts["context"].append(f"{pred}={val}")
        
        # Собираем портрет
        lines = [f"# Профиль пользователя (сгенерирован {datetime.now().strftime('%Y-%m-%d %H:%M')})"]
        
        if persona_parts["personal"]:
            lines.append("\n## Личная информация")
            lines.extend(f"- {p}" for p in persona_parts["personal"])
        
        if persona_parts["preferences"]:
            lines.append("\n## Предпочтения")
            lines.extend(f"- {p}" for p in persona_parts["preferences"])
        
        if persona_parts["goals"]:
            lines.append("\n## Цели и планы")
            lines.extend(f"- {p}" for p in persona_parts["goals"])
        
        if persona_parts["context"]:
            lines.append("\n## Контекст")
            lines.extend(f"- {p}" for p in persona_parts["context"][:10])
        
        lines.append(f"\nВсего фактов: {len(facts)}")
        
        persona_text = "\n".join(lines)
        self.persona_cache = persona_text
        self.last_generated = datetime.now().isoformat()
        
        return persona_text
    
    def get_prompt_context(self, max_tokens: int = 500) -> str:
        """
        Возвращает persona в формате для системного промпта.
        """
        if not self.persona_cache:
            self.generate()
        
        # Обрезаем до примерного количества токенов (1 токен ≈ 3 символа)
        max_chars = max_tokens * 3
        if len(self.persona_cache) > max_chars:
            return self.persona_cache[:max_chars] + "..."
        return self.persona_cache
    
    def should_regenerate(self, new_facts_count: int, threshold: int = 10) -> bool:
        """
        Проверка, нужно ли перегенерировать персону.
        """
        return new_facts_count >= threshold
