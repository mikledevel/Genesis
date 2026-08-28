"""
Knowledge Graph - граф знаний персонального ИИ.

Хранит факты в виде графа (subject -[predicate]-> object).
Поддерживает:
- Добавление/обновление узлов и связей
- Запросы к графу
- Вычисление важности фактов
- Сериализацию в JSON
"""

import json
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict


class KnowledgeGraph:
    """
    Граф знаний.
    
    Узлы — сущности (пользователь, объекты, места).
    Рёбра — отношения между ними с атрибутами.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.nodes: Dict[str, Dict] = {}       # node_id -> {type, properties, created_at, importance}
        self.edges: List[Dict] = []            # {source, target, predicate, confidence, timestamp}
        
        if storage_path:
            self._load()
    
    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.7, **kwargs):
        """Добавление факта в граф."""
        # Создаём узлы если их нет
        if subject not in self.nodes:
            self.nodes[subject] = {
                "id": subject,
                "type": "entity",
                "properties": {},
                "created_at": datetime.now().isoformat(),
                "importance": 0.5,
            }
        if obj not in self.nodes:
            self.nodes[obj] = {
                "id": obj,
                "type": "entity",
                "properties": {},
                "created_at": datetime.now().isoformat(),
                "importance": 0.5,
            }
        
        # Проверяем, есть ли уже такая связь
        for edge in self.edges:
            if (edge["source"] == subject and 
                edge["target"] == obj and 
                edge["predicate"] == predicate):
                # Обновляем уверенность
                edge["confidence"] = max(edge["confidence"], confidence)
                edge["timestamp"] = datetime.now().isoformat()
                return
        
        # Новая связь
        self.edges.append({
            "source": subject,
            "target": obj,
            "predicate": predicate,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        })
        
        # Повышаем важность узлов
        self.nodes[subject]["importance"] = min(1.0, self.nodes[subject]["importance"] + 0.05)
        self.nodes[obj]["importance"] = min(1.0, self.nodes[obj]["importance"] + 0.03)
    
    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Dict]:
        """Запрос к графу."""
        results = []
        for edge in self.edges:
            if subject and edge["source"] != subject:
                continue
            if predicate and edge["predicate"] != predicate:
                continue
            if obj and edge["target"] != obj:
                continue
            results.append(edge)
        return sorted(results, key=lambda e: e.get("confidence", 0), reverse=True)
    
    def get_facts_about(self, subject: str, min_confidence: float = 0.5) -> List[Dict]:
        """Все факты о субъекте."""
        facts = []
        for edge in self.edges:
            if edge["source"] == subject and edge["confidence"] >= min_confidence:
                facts.append({
                    "predicate": edge["predicate"],
                    "value": edge["target"],
                    "confidence": edge["confidence"],
                })
        return sorted(facts, key=lambda f: f["confidence"], reverse=True)
    
    def get_important_facts(self, top_n: int = 10) -> List[Dict]:
        """Самые важные факты в графе."""
        # Сортируем узлы по важности
        important_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1]["importance"],
            reverse=True,
        )[:top_n]
        
        facts = []
        for node_id, node in important_nodes:
            node_facts = self.get_facts_about(node_id)
            facts.append({
                "entity": node_id,
                "importance": node["importance"],
                "facts": node_facts,
            })
        return facts
    
    def to_dict(self) -> Dict:
        """Сериализация в словарь."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "updated_at": datetime.now().isoformat(),
        }
    
    def save(self):
        """Сохранение в файл."""
        if self.storage_path:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load(self):
        """Загрузка из файла."""
        import os
        if self.storage_path and os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", [])
    
    def get_stats(self) -> Dict:
        """Статистика графа."""
        predicate_counts = defaultdict(int)
        for edge in self.edges:
            predicate_counts[edge["predicate"]] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "top_predicates": dict(sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "avg_confidence": sum(e["confidence"] for e in self.edges) / len(self.edges) if self.edges else 0,
        }
