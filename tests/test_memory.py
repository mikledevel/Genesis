import pytest, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.memory.episodic import FactExtractor, Fact
from app.core.memory.graph import KnowledgeGraph
from app.core.memory.persona import PersonaGenerator

class TestFactExtractor:
    @pytest.fixture
    def extractor(self): return FactExtractor()
    
    def test_extract_name(self, extractor):
        facts = extractor.extract("Меня зовут Александр")
        names = [f for f in facts if f.predicate == "name"]
        assert len(names) > 0
        
    def test_extract_age(self, extractor):
        facts = extractor.extract("Мне 30 лет")
        ages = [f for f in facts if f.predicate == "age"]
        assert len(ages) > 0
        
    def test_extract_preference(self, extractor):
        facts = extractor.extract("Я люблю чёрный кофе и ненавижу пробки")
        likes = [f for f in facts if f.predicate == "likes"]
        dislikes = [f for f in facts if f.predicate == "dislikes"]
        assert len(likes) > 0 or len(dislikes) > 0
        
    def test_extract_goal(self, extractor):
        facts = extractor.extract("Я хочу выучить испанский к лету")
        goals = [f for f in facts if f.predicate == "goal"]
        assert len(goals) > 0
        
    def test_extract_emotion(self, extractor):
        facts = extractor.extract("Я очень рад сегодня")
        assert len(facts) > 0
        
    def test_merge_facts(self, extractor):
        old = extractor.extract("Меня зовут Александр")
        new = extractor.extract("Меня зовут Саша")
        merged = extractor.merge_facts(old, new)
        assert len(merged) >= 1


class TestKnowledgeGraph:
    @pytest.fixture
    def graph(self): return KnowledgeGraph()
    
    def test_add_fact(self, graph):
        graph.add_fact("user", "name", "Александр", 0.9)
        assert "user" in graph.nodes
        assert "Александр" in graph.nodes
        
    def test_query(self, graph):
        graph.add_fact("user", "likes", "кофе", 0.8)
        graph.add_fact("user", "dislikes", "пробки", 0.7)
        results = graph.query(subject="user")
        assert len(results) == 2
        
    def test_get_facts_about(self, graph):
        graph.add_fact("user", "name", "Александр", 0.9)
        graph.add_fact("user", "age", "30", 0.6)
        facts = graph.get_facts_about("user")
        assert len(facts) == 2
        
    def test_importance(self, graph):
        graph.add_fact("user", "name", "Александр", 0.9)
        assert graph.nodes["user"]["importance"] > 0.5
        
    def test_get_stats(self, graph):
        graph.add_fact("user", "name", "Александр", 0.9)
        stats = graph.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1


class TestPersonaGenerator:
    @pytest.fixture
    def persona(self):
        g = KnowledgeGraph()
        g.add_fact("user", "name", "Александр", 0.9)
        g.add_fact("user", "age", "30", 0.8)
        g.add_fact("user", "city", "Москва", 0.7)
        g.add_fact("user", "likes", "кофе", 0.8)
        g.add_fact("user", "goal", "выучить Python", 0.6)
        return PersonaGenerator(g)
    
    def test_generate(self, persona):
        text = persona.generate()
        assert "Александр" in text or "name" in text
        
    def test_prompt_context(self, persona):
        text = persona.get_prompt_context(max_tokens=50)
        assert len(text) > 0
