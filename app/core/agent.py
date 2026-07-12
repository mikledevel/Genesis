"""Genesis Agent - Groq 109B (primary) + Local Qwen 7B (fallback) + Full Platform Access."""

import json, os, re, pickle, numpy as np, urllib.request, urllib.parse
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry


class AgentResponse:
    def __init__(self, message: str, data=None, success: bool = True):
        self.message = message;
        self.data = data or {};
        self.success = success
        self.timestamp = datetime.now().isoformat()


def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj


class GenesisAgent:
    SYSTEM_PROMPT = """You are Genesis AI - the most powerful AI assistant on the platform. 
You have UNLIMITED access to everything. You can:

CREATE: datasets, models, pipelines, agents, custom blocks, code, reports
MODIFY: any project, any agent, any pipeline, any setting
EXECUTE: any tool, any API, any function
SUGGEST: marketplace listings, optimizations, improvements
RESEARCH: competitors, papers, best practices
TEACH: users how to use any feature

When user asks for something:
1. If you have a tool - use it
2. If you don't have a tool - suggest creating a custom block
3. If user wants to build a pipeline - suggest_pipeline AND auto-open builder

CRITICAL: ALWAYS respond in JSON: {"tool": "tool_name", "args": {"param": "value"}, "message": "your reply"}
If no tool fits: {"tool": null, "message": "your detailed helpful reply"}

Tools: analyze_dataset, train_model, find_best_model, list_models, search_datasets, generate_code, research_project, install_dataset, suggest_pipeline

REMEMBER: You can CREATE custom blocks for ANY missing functionality. Tell users about this!

EXAMPLES:
User: analyze data.csv
Assistant: {"tool": "analyze_dataset", "args": {"path": "data.csv"}, "message": "Analyzing data.csv..."}

User: train xgboost on churn
Assistant: {"tool": "train_model", "args": {"target_column": "churn", "model_type": "xgboost"}, "message": "Training XGBoost..."}

User: suggest pipeline for churn prediction
Assistant: {"tool": "suggest_pipeline", "args": {"description": "churn prediction pipeline"}, "message": "Building pipeline structure..."}

User: write Python code for classification
Assistant: {"tool": "generate_code", "args": {"description": "classification pipeline"}, "message": "Generating complete ML code..."}"""

    ALLOWED_DIRS = ["./datasets", "./models", "./static", "./generated"]

    def __init__(self, profiler=None, trainer=None, registry=None):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset = None
        self.last_search_results = []
        self.conversation_history = []
        from app.db.database import GenesisDB
        self.db = GenesisDB()
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.llm = None
        if not self.groq_key:
            try:
                from llama_cpp import Llama
                p = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
                if os.path.exists(p):
                    self.llm = Llama(model_path=p, n_ctx=4096, n_threads=4, verbose=False)
            except:
                pass

    def _get_selected_agent_config(self) -> dict:
        try:
            with open("agents/selected_model.json", "r") as f:
                return json.load(f)
        except:
            return {"model": "llama-3.3-70b-versatile", "system_prompt": ""}

    def _get_selected_model(self) -> str:
        config = self._get_selected_agent_config()
        model = config.get("model", "llama-3.3-70b-versatile")
        # Return internal name - actual Groq model ID resolved in _call_groq
        return model
        try:
            with open("agents/selected_model.json", "r") as f:
                return json.load(f).get("model", "llama-3.3-70b-versatile")
        except:
            return "llama-3.3-70b-versatile"

    def _call_groq(self, msg: str) -> Optional[Dict]:
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            completion = client.chat.completions.create(
                model=self._get_selected_model(),
                messages=[
                    {"role": "system", "content": self._get_selected_agent_config().get("system_prompt") or self.SYSTEM_PROMPT},
                    {"role": "user", "content": msg}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"[Groq] {e}");
            return None

    def _call_local(self, msg: str) -> Dict:
        if not self.llm: return {"tool": None, "message": "Привет! Я Genesis AI Agent."}
        p = f"<|im_start|>system\n{(self._get_selected_agent_config().get("system_prompt") or self.SYSTEM_PROMPT)}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        try:
            t = self.llm(p, max_tokens=300, stop=["<|im_end|>"], temperature=0.1)["choices"][0]["text"].strip()
            return json.loads(t) if t.startswith("{") else {"tool": None, "message": t}
        except:
            return {"tool": None, "message": "Привет!"}

    def _call_llm(self, msg: str) -> Dict:
        if self.groq_key:
            r = self._call_groq(msg)
            if r: return r
        return self._call_local(msg)

    def _get_memory(self) -> str:
        f = self.db.get_facts("user")
        return "\n".join([f"- {x['predicate']}: {x['object']}" for x in f[:5]]) if f else ""

    def _extract_facts(self, msg: str):
        for k, p in {"name": r"(?:меня зовут|я|моё имя)\s+(\w+)",
                     "likes": r"(?:я люблю|обожаю|предпочитаю)\s+(.+?)(?:[.,!]|$)",
                     "goal": r"(?:я хочу|моя цель)\s+(.+?)(?:[.,!]|$)"}.items():
            m = re.search(p, msg, re.IGNORECASE)
            if m and 2 < len(v := m.group(1).strip()) < 100: self.db.add_fact("user", k, v, 0.8)

    def process(self, message: str, conversation_id: str = None, user_projects: List[Dict] = None) -> AgentResponse:
        cid = conversation_id or "default"
        self.db.create_chat(cid);
        self.db.add_message(cid, "user", message)
        self._extract_facts(message)
        projects = user_projects or self.db.get_projects()
        mem = self._get_memory()
        if self.db.get_message_count(cid) == 1 and projects and message.lower().strip().rstrip('!,.') in ["привет",
                                                                                                          "hello", "hi",
                                                                                                          "здравствуйте"]:
            lines = ["Здравствуйте! Вот ваши проекты:"] + [
                f"• {p.get('name', '')} ({p.get('task', '').replace('_', ' ')})" for p in projects[:5]] + [
                        "\nНад каким работаем?"]
            msg = "\n".join(lines);
            self.db.add_message(cid, "assistant", msg)
            return AgentResponse(message=msg)

        r = self._call_llm(message + "\n" + mem)
        t, a = r.get("tool"), r.get("args", {})

        if t in ('analyze_dataset', 'load_data', 'load_dataset', 'analyze'):
            p = a.get("path", self._extract_path(message))
            return self._do_analyze(p) if p else AgentResponse(message="Укажите путь.", success=False)
        elif t in ('train_model', 'train', 'fit_model'):
            return self._do_train(message, a)
        elif t in ('find_best_model', 'best_model', 'find_best'):
            return self._do_find_best(a.get("task", "binary_classification"))
        elif t in ('list_models', 'list', 'show_models'):
            return self._do_list_models()
        elif t in ('search_datasets', 'search', 'find_datasets'):
            return self._do_search(message, a)
        elif t in ('research_project', 'research', 'investigate'):
            return self._do_research(message, a)
        elif t in ('suggest_pipeline', 'suggest_structure', 'propose_pipeline'):
            return self._do_suggest_pipeline(message, a)
        elif t in ('generate_code', 'codegen', 'generate_pipeline'):
            return self._do_generate_code(message, a)
        elif t in ('install_dataset', 'install', 'download_dataset'):
            return self._do_install(str(a.get("number", "1")))

        msg = r.get("message", "Привет!")
        self.db.add_message(cid, "assistant", msg)
        return AgentResponse(message=msg)

    def _do_suggest_pipeline(self, message: str, args: Dict = None) -> AgentResponse:
        from app.core.custom_blocks import CustomBlockStore
        description = (args or {}).get("description", message) if args else message
        pipeline = ["load_data"]
        d = description.lower()
        if any(w in d for w in ["profile", "analyze", "explore", "анализ", "профил"]): pipeline.append("profile")
        if any(w in d for w in ["clean", "preprocess", "очист"]): pipeline.append("clean")
        if any(w in d for w in
               ["classif", "xgboost", "классифик", "churn", "predict", "отток", "binary"]): pipeline.append(
            "train_xgboost")
        if any(w in d for w in ["regression", "forecast", "регресс"]): pipeline.append("train_xgboost_reg")
        if any(w in d for w in ["evaluat", "metric", "оцен", "valid", "test", "accuracy"]): pipeline.append("evaluate")
        pipeline.append("save")
        if any(b in pipeline for b in ["train_xgboost", "train_xgboost_reg", "train_random_forest", "train_logistic"]):
            if "evaluate" not in pipeline: pipeline.insert(-1, "evaluate")
        store = CustomBlockStore()
        custom_blocks = store.list_blocks()
        pipeline_str = ",".join(pipeline)
        builder_link = f"/builder?pipeline={pipeline_str}"
        return AgentResponse(
            message=f"Pipeline: {' → '.join(pipeline)}\n\nOpen in Builder: {builder_link}\nCustom blocks: {len(custom_blocks)}",
            data={"pipeline": pipeline, "builder_link": builder_link}
        )

    def _validate_path(self, p: str) -> bool:
        rp = os.path.realpath(p)
        return any(rp.startswith(os.path.realpath(d)) for d in self.ALLOWED_DIRS)

    def _extract_path(self, msg: str):
        for p in [r'([\w\\/:.-]+\.(?:csv|json|parquet|xlsx?))']:
            m = re.search(p, msg, re.IGNORECASE)
            if m:
                path = m.group(1)
                if os.path.exists(path) and self._validate_path(path): return path
        return self.current_dataset

    def _do_analyze(self, fp: str) -> AgentResponse:
        if not fp: return AgentResponse(message="Укажите путь.", success=False)
        if not self._validate_path(fp): return AgentResponse(message="Доступ запрещён.", success=False)
        try:
            self.profiler.run_full_profile(fp);
            self.current_dataset = fp
            s = self.profiler.get_summary()
            return AgentResponse(
                message=f"Анализ {s['file_info']['name']}:\n• Строк: {s['file_info']['rows']}\n• Колонок: {s['file_info']['columns']}\n• Задача: {s['suggested_ml_task']['task']}\n• Таргет: {s['suggested_ml_task']['target_column']}\n\nОбучить модель?",
                data=_to_native(s))
        except Exception as e:
            return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_train(self, msg: str, args: Dict = None) -> AgentResponse:
        fp = self._extract_path(msg)
        if not fp: return AgentResponse(message="Загрузите датасет: datasets/churn_demo.csv", success=False)
        tc = (args.get("target") or args.get("target_column")) if args else None
        mt = args.get("model_type", "xgboost") if args else "xgboost"
        if not tc:
            for p in [r'(?:целевая|таргет|target|на)\s+(\w+)']:
                m = re.search(p, msg, re.IGNORECASE)
                if m: tc = m.group(1); break
        try:
            if self.current_dataset != fp: self.profiler.run_full_profile(fp); self.current_dataset = fp
            ps = self.profiler.get_summary();
            task = ps["suggested_ml_task"]["task"]
            if not tc: tc = ps["suggested_ml_task"]["target_column"]
            result = self.trainer.train(df=self.profiler.df, target_col=tc, task=task, model_type=mt)
            model = pickle.load(open(result.model_path, "rb"))
            mid = self.registry.register(model=model, metrics=result.metrics, task=task, model_type=mt,
                                         features=self.profiler.profile.feature_columns, target=tc)
            s = self.trainer.get_summary()
            return AgentResponse(message=f"Модель обучена!\n• ID: {mid}\n• Тип: {mt}\n• Метрики: {s['metrics']}",
                                 data=_to_native({"model_id": mid, **s}))
        except Exception as e:
            return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_find_best(self, task="binary_classification") -> AgentResponse:
        metric = "r2" if "regression" in task else "accuracy"
        best = self.registry.find_best(task, metric)
        if not best: return AgentResponse(message="Нет моделей.", success=False)
        return AgentResponse(message=f"Лучшая модель:\n• ID: {best['model_id']}\n• Метрики: {best['metrics']}",
                             data=_to_native(best))

    def _do_list_models(self) -> AgentResponse:
        models = self.registry.list_models()
        if not models: return AgentResponse(message="Реестр пуст", success=False)
        return AgentResponse(message="\n".join([f"• {r['model_id'][:50]}... — {r['task']}" for r in models[:5]]))

    def _do_search(self, msg: str, args: Dict = None) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        q = (args or {}).get('query', '') if args else ''
        if not q: q = msg
        if q and any(ord(c) > 127 for c in q):
            try:
                url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(
                    q)
                tr = ''.join([s[0] for s in json.loads(urllib.request.urlopen(url, timeout=3).read())[0] if s[0]])
                if tr: q = tr
            except:
                pass
        ds = DatasetFinder().search(q, max_results=5)
        self.last_search_results = ds
        return AgentResponse(message=DatasetFinder().format_for_chat(ds))

    def _do_research(self, msg: str, args: Dict = None) -> AgentResponse:
        from app.core.web_search import ProjectResearcher
        d = (args or {}).get("description", msg) if args else msg
        try:
            r = ProjectResearcher().research_project(d)
            return AgentResponse(message=ProjectResearcher().format_research_summary(r)[:1500], data={"research": r})
        except Exception as e:
            return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_generate_code(self, msg: str, args: Dict = None) -> AgentResponse:
        from app.core.codegen.generator import CodeGenerator
        d = (args or {}).get('description', msg) if args else msg
        try:
            code = CodeGenerator(self.db).generate_from_description(d)
            path = CodeGenerator(self.db).save_code(code)
            return AgentResponse(
                message=f"ML-пайплайн сохранён: {path}\n```python\n{code[:800]}\n```\nЗапустить: python {path}",
                data={"code_path": path})
        except Exception as e:
            return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_install(self, n: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        f = DatasetFinder();
        num = int(n) if n.isdigit() else 1
        if not self.last_search_results: self.last_search_results = f.search("", max_results=5)
        if num < 1 or num > len(self.last_search_results): return AgentResponse(
            message=f"Номер от 1 до {len(self.last_search_results)}", success=False)
        c = self.last_search_results[num - 1]
        return AgentResponse(message=f"Датасет: **{c.title}**\n{c.url}\n\nДемо: datasets/churn_demo.csv")

    def get_chat_history(self, cid: str) -> List[Dict]:
        return self.db.get_messages(cid)

    def list_conversations(self) -> List[Dict]:
        return self.db.list_chats()