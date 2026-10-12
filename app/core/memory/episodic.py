"""
Episodic Memory - извлечение событий и фактов из диалогов.

Анализирует поток сообщений, выделяет:
- Явные факты (имя, возраст, город)
- Предпочтения (любит/ненавидит)
- Планы и цели
- Эмоциональные события
- Отношения между сущностями
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Fact:
    """Один извлечённый факт."""
    subject: str
    predicate: str
    object: str
    fact_type: str          # "personal", "preference", "plan", "event", "relationship"
    confidence: float       # 0.0 - 1.0
    source_message: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)


class FactExtractor:
    """
    Извлекает факты из сообщений на естественном языке.
    
    В реальной системе использует LLM для извлечения.
    Сейчас — rule-based экстрактор с паттернами.
    """
    
    # Паттерны для извлечения фактов
    PATTERNS = {
        "personal": [
            (r"(?:меня зовут|я|моё имя)\s+(\w+)", "name"),
            (r"(?:мне|я)\s+(\d+)\s+(?:лет|год)", "age"),
            (r"(?:я живу в|я из|мой город)\s+(\w+)", "city"),
            (r"(?:я работаю|моя профессия|я по профессии)\s+(.+?)(?:\.|,|$)", "profession"),
        ],
        "preference": [
            (r"(?:я люблю|обожаю|нравится)\s+(.+?)(?:\.|,|$)", "likes"),
            (r"(?:я ненавижу|не люблю|терпеть не могу)\s+(.+?)(?:\.|,|$)", "dislikes"),
            (r"(?:моё хобби|я увлекаюсь)\s+(.+?)(?:\.|,|$)", "hobby"),
        ],
        "plan": [
            (r"(?:я хочу|я планирую|моя цель|я собираюсь)\s+(.+?)(?:\.|,|$)", "goal"),
            (r"(?:к\s+(\w+)\s+(?:я|мы)\s+(?:должны|нужно|надо))\s+(.+?)(?:\.|,|$)", "deadline"),
        ],
        "event": [
            (r"(?:сегодня|вчера|на этой неделе)\s+(.+?)(?:\.|,|$)", "emotion"),
            (r"(?:я (?:рад|счастлив|грустный|злой|тревожусь|волнуюсь))(?:\s+из-за)?\s+(.+?)(?:\.|,|$)", "emotion"),
        ],
        "relationship": [
            (r"(?:мой|моя)\s+(\w+)\s+(\w+)", "relation"),
            (r"(?:у меня есть)\s+(\w+)\s*[:-]?\s*(\w+)", "possession"),
        ],
    }
    
    def extract(self, message: str, speaker: str = "user") -> List[Fact]:
        """
        Извлечение фактов из сообщения.
        
        Args:
            message: текст сообщения
            speaker: кто говорит (user/agent)
            
        Returns:
            список извлечённых фактов
        """
        facts = []
        timestamp = datetime.now().isoformat()
        
        for fact_type, patterns in self.PATTERNS.items():
            for pattern, predicate in patterns:
                matches = re.findall(pattern, message, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        obj = " ".join(match)
                        subject = match[0] if len(match) > 0 else speaker
                    else:
                        obj = match
                        subject = speaker
                    
                    fact = Fact(
                        subject=speaker,
                        predicate=predicate,
                        object=obj.strip(),
                        fact_type=fact_type,
                        confidence=self._calculate_confidence(message, pattern),
                        source_message=message[:100],
                        timestamp=timestamp,
                    )
                    facts.append(fact)
        
        return facts
    
    def _calculate_confidence(self, message: str, pattern: str) -> float:
        """Оценка уверенности в факте."""
        # Базовая уверенность
        confidence = 0.7
        
        # Повышаем если есть усиливающие слова
        boosters = ["очень", "точно", "абсолютно", "всегда", "постоянно"]
        if any(b in message.lower() for b in boosters):
            confidence += 0.2
        
        # Понижаем если есть слова сомнения
        doubters = ["наверное", "может быть", "возможно", "кажется"]
        if any(d in message.lower() for d in doubters):
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def merge_facts(self, existing: List[Fact], new_facts: List[Fact]) -> List[Fact]:
        """
        Объединение новых фактов с существующими.
        
        Если факт противоречит старому — понижает уверенность старого,
        если подтверждает — повышает.
        """
        merged = list(existing)
        
        for new_fact in new_facts:
            found = False
            for old_fact in merged:
                if (old_fact.subject == new_fact.subject and
                    old_fact.predicate == new_fact.predicate):
                    # Обновляем существующий факт
                    if old_fact.object == new_fact.object:
                        old_fact.confidence = min(1.0, old_fact.confidence + 0.1)
                    else:
                        old_fact.confidence = max(0.0, old_fact.confidence - 0.2)
                    old_fact.timestamp = new_fact.timestamp
                    found = True
                    break
            
            if not found:
                merged.append(new_fact)
        
        # Удаляем факты с низкой уверенностью
        merged = [f for f in merged if f.confidence >= 0.3]
        
        return merged

