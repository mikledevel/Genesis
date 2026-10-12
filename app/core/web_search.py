"""
Web Search - поиск в интернете через SerpAPI (Google Search).

Ищет информацию по запросу, анализирует конкурентов,
собирает лучшие практики для ML-проекта.
"""

import json, urllib.request, urllib.parse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    source: str = "google"


class WebSearcher:
    """
    Поисковик для исследования ML-задач.
    
    Использует бесплатный Google Search (через публичный endpoint)
    или SerpAPI для структурированных результатов.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Поиск информации по ML-задаче.
        
        Args:
            query: поисковый запрос
            num_results: количество результатов
            
        Returns:
            список результатов поиска
        """
        results = []
        
        # Способ 1: Бесплатный Google Search (через urllib)
        try:
            url = "https://www.google.com/search?q=" + urllib.parse.quote(query + " machine learning")
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp = urllib.request.urlopen(req, timeout=5)
            html = resp.read().decode('utf-8', errors='ignore')
            
            # Простое извлечение заголовков и ссылок
            import re
            # Ищем ссылки
            links = re.findall(r'href="(https?://[^"]+)"', html)
            # Ищем заголовки
            titles = re.findall(r'<h3[^>]*>(.*?)</h3>', html)
            
            for i, (title, link) in enumerate(zip(titles[:num_results], links[:num_results])):
                if 'google.com' not in link and 'youtube.com' not in link:
                    results.append(SearchResult(
                        title=re.sub(r'<[^>]+>', '', title),
                        link=link,
                        snippet="",
                        source="google"
                    ))
        except Exception as e:
            print(f"Google search error: {e}")
        
        # Способ 2: DuckDuckGo (бесплатный, без API ключа)
        if not results:
            try:
                url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query + " ML") + "&format=json"
                resp = urllib.request.urlopen(url, timeout=5)
                data = json.loads(resp.read())
                
                for item in data.get("Results", [])[:num_results]:
                    results.append(SearchResult(
                        title=item.get("Text", ""),
                        link=item.get("FirstURL", ""),
                        snippet=item.get("Text", ""),
                        source="duckduckgo"
                    ))
            except: pass
        
        return results[:num_results]
    
    def search_competitors(self, task_description: str) -> List[SearchResult]:
        """
        Поиск конкурентов и существующих решений.
        """
        queries = [
            f"{task_description} solution",
            f"{task_description} API",
            f"{task_description} model marketplace",
            f"best {task_description} approach",
        ]
        
        all_results = []
        for query in queries:
            results = self.search(query, 3)
            all_results.extend(results)
        
        # Убираем дубликаты
        seen = set()
        unique = []
        for r in all_results:
            if r.link not in seen:
                seen.add(r.link)
                unique.append(r)
        
        return unique[:8]
    
    def search_datasets_info(self, topic: str) -> List[SearchResult]:
        """
        Поиск информации о лучших датасетах для задачи.
        """
        queries = [
            f"{topic} dataset kaggle",
            f"{topic} dataset machine learning",
            f"best dataset for {topic}",
        ]
        
        all_results = []
        for query in queries:
            results = self.search(query, 3)
            all_results.extend(results)
        
        return all_results[:5]
    
    def format_for_agent(self, results: List[SearchResult], query_type: str = "general") -> str:
        """
        Форматирование результатов для показа агенту.
        """
        if not results:
            return f"Не нашёл информации по запросу. Попробуйте уточнить."
        
        lines = [f"🔍 Результаты исследования ({query_type}):\n"]
        
        for i, r in enumerate(results[:8], 1):
            title = r.title[:100]
            snippet = r.snippet[:150] if r.snippet else r.link[:80]
            lines.append(f"{i}. **{title}**\n   🔗 {r.link}\n   📝 {snippet}\n")
        
        return "\n".join(lines)


class ProjectResearcher:
    """
    Полный цикл исследования ML-проекта.
    
    Проводит:
    1. Web search по теме
    2. Анализ конкурентов
    3. Поиск датасетов
    4. Сбор лучших практик
    """
    
    def __init__(self):
        self.searcher = WebSearcher()
    
    def research_project(self, task_description: str) -> Dict[str, Any]:
        """
        Полное исследование ML-проекта.
        
        Returns:
            словарь с результатами исследования
        """
        return {
            "task": task_description,
            "general_info": self.searcher.search(task_description, 5),
            "competitors": self.searcher.search_competitors(task_description),
            "datasets_info": self.searcher.search_datasets_info(task_description),
            "best_practices": self.searcher.search(f"best practices {task_description}", 5),
        }
    
    def format_research_summary(self, research: Dict[str, Any]) -> str:
        """
        Форматирует полное исследование в читаемый текст.
        """
        lines = [
            f"# Исследование проекта: {research['task']}\n",
            "## 📊 Общая информация",
            self.searcher.format_for_agent(research['general_info'], "general"),
            "\n## 🏢 Конкуренты и существующие решения",
            self.searcher.format_for_agent(research['competitors'], "competitors"),
            "\n## 📦 Рекомендуемые датасеты",
            self.searcher.format_for_agent(research['datasets_info'], "datasets"),
            "\n## 💡 Лучшие практики",
            self.searcher.format_for_agent(research['best_practices'], "best_practices"),
            "\n---\nСледующий шаг: найти и обучить модель на лучшем датасете.",
        ]
        return "\n".join(lines)
