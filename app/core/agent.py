"""Genesis Agent - Groq 109B (primary) + Local Qwen 7B (fallback) + Full Platform Access."""

import json, os, re, pickle, numpy as np, urllib.request, urllib.parse
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry


class AgentResponse:
    def __init__(self, message: str, data=None, success: bool = True, message_id: Optional[int] = None):
        self.message = message;
        self.data = data or {};
        self.success = success
        self.timestamp = datetime.now().isoformat()
        # DB row id of the persisted assistant message, if one was saved - lets the frontend
        # rate this exact message (👍/👎) immediately without a reload/history refetch.
        self.message_id = message_id


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

Tools: analyze_dataset, train_model, find_best_model, list_models, search_datasets, generate_code, research_project, install_dataset, suggest_pipeline, list_marketplace, list_projects, publish_marketplace, create_custom_block, create_bot, create_model, add_skill

REMEMBER: You can CREATE custom blocks for ANY missing functionality. Tell users about this!

add_skill gives one of the user's OWN bots a new real, tested capability (checking a website,
calling a public API) - not just more conversation. Use it when the user asks to add a real
capability to an existing bot (e.g. "add a skill to my SupportBot that checks order status on
myshop.com"). Needs args: {"bot_name": "<the bot's exact name>", "skill_description": "<what
it should do, as specific as possible>"}. If the user is already chatting directly with the
specific bot they mean (not the general assistant), you don't need bot_name - just pass
skill_description. If they haven't said which bot, ask - don't guess.

EXAMPLES:
User: привет
Assistant: {"tool": null, "message": "Привет! Чем могу помочь?"}

User: кто ты?
Assistant: {"tool": null, "message": "Я — Genesis AI, ассистент этой платформы. Помогаю анализировать данные, обучать модели и строить ML-пайплайны."}

User: analyze data.csv
Assistant: {"tool": "analyze_dataset", "args": {"path": "data.csv"}, "message": "Analyzing data.csv..."}

User: train xgboost on churn
Assistant: {"tool": "train_model", "args": {"target_column": "churn", "model_type": "xgboost"}, "message": "Training XGBoost..."}

User: train a model on datasets/churn.csv to predict churn
Assistant: {"tool": "train_model", "args": {"path": "datasets/churn.csv", "target_column": "churn"}, "message": "Training on datasets/churn.csv..."}
IMPORTANT - if the user's message names an actual file (has a .csv/.json/.parquet/.xlsx
extension), ALWAYS extract it into args.path exactly as written, same as analyze_dataset
above - don't rely on the message text being re-parsed later, put the literal path in args.

User: suggest pipeline for churn prediction
Assistant: {"tool": "suggest_pipeline", "args": {"description": "churn prediction pipeline"}, "message": "Building pipeline structure..."}

User: построй полноценную модель для прогноза оттока клиентов
Assistant: {"tool": "create_model", "args": {"description": "модель для прогноза оттока клиентов"}, "message": "Строю и тестирую модель на реальных данных..."}

IMPORTANT - suggest_pipeline vs create_model vs generate_code vs train_model, these are
easily confused, pick carefully:
- suggest_pipeline: user wants to SEE/PLAN the steps (a diagram/description), nothing is
  actually trained. Triggers: "предложи пайплайн", "suggest a pipeline", "what steps would..."
- create_model / train_model / generate_code: user wants a REAL, ACTUALLY TRAINED model
  right now. Triggers: "построй модель", "обучи модель", "train a model", "build a model",
  "спрогнозируй" with an actual dataset in context. When in doubt and a dataset has
  already been analyzed in this conversation, prefer create_model - it actually runs the
  code and registers a working model, which is almost always what "build/train X" means.

User: write Python code for classification
Assistant: {"tool": "generate_code", "args": {"description": "classification pipeline"}, "message": "Generating complete ML code..."}

User: покажи маркетплейс / что есть в маркетплейсе
Assistant: {"tool": "list_marketplace", "args": {}, "message": "Смотрю маркетплейс..."}

User: какие у меня проекты
Assistant: {"tool": "list_projects", "args": {}, "message": "Показываю ваши проекты..."}

User: опубликуй мою последнюю модель в маркетплейс за 0.02 доллара за вызов
Assistant: {"tool": "publish_marketplace", "args": {"price_per_call": 0.02}, "message": "Публикую модель..."}

User: создай блок для отправки email после обучения модели
Assistant: {"tool": "create_custom_block", "args": {"name": "Send Email", "description": "Sends an email notification with training results after the model is saved"}, "message": "Создаю кастомный блок..."}

User: сделай бота поддержки для интернет-магазина обуви, отвечает на вопросы о доставке и возврате
Assistant: {"tool": "create_bot", "args": {"description": "бот поддержки для интернет-магазина обуви, отвечает на вопросы о доставке и возврате"}, "message": "Проектирую и тестирую бота..."}

User: add a skill to my SupportBot that checks order status on myshop.com/orders
Assistant: {"tool": "add_skill", "args": {"bot_name": "SupportBot", "skill_description": "Check the status of an order on myshop.com given an order ID"}, "message": "Building and testing that skill now..."}

IMPORTANT: Only use a tool when the user's message actually asks for that specific action (analyzing a real file, training a real model, building a real pipeline). For greetings, small talk, questions about yourself, or general conversation, ALWAYS respond with {"tool": null, "message": "..."} — never invent a pipeline or dataset action that wasn't requested."""

    ALLOWED_DIRS = ["./datasets", "./models", "./static", "./generated"]
    # See process() - the only tool a SCOPED (published-bot) conversation may invoke, besides
    # plain conversation (None).
    BOT_SCOPED_ALLOWED_TOOLS = {None, "predict_with_model", "predict", "run_model",
                                "request_handoff", "handoff_to_human", "escalate",
                                "run_skill", "call_skill", "use_skill",
                                "add_skill", "add_capability", "create_skill"}
    MAX_SKILLS_PER_BOT = 20  # generous, but not unbounded - see _do_add_skill

    # Internal/legacy short names AND now-deprecated real Groq IDs -> currently recommended real Groq model ID.
    # Groq requires the exact id from https://console.groq.com/docs/models
    # Legacy short aliases -> current verified Groq model ids (see models_catalog.py,
    # the single source of truth this now routes through). llama-4-scout and
    # qwen/qwen3-32b were fully retired by Groq and no longer resolve to anything -
    # they route to the current default rather than a dead model id.
    MODEL_MAP = {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "llama-4-scout": "openai/gpt-oss-120b",  # retired by Groq - no longer callable
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",  # confirmed production, not deprecated
        "qwen-3-32b": "openai/gpt-oss-120b",  # retired by Groq - no longer callable
        "qwen-3.6-27b": "qwen/qwen3.6-27b",
        "groq-70b": "openai/gpt-oss-120b",  # legacy placeholder id used by older marketplace agents
        "local-7b": None,  # not a Groq model - handled by local llama.cpp fallback
        "meta-llama/llama-4-scout-17b-16e-instruct": "openai/gpt-oss-120b",  # retired by Groq
        "qwen/qwen3-32b": "openai/gpt-oss-120b",  # retired by Groq
    }

    def __init__(self, profiler=None, trainer=None, registry=None, agent_override: Optional[Dict] = None, db=None):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset = None
        self.current_user_id = "anonymous"
        self.last_search_results = []
        self.conversation_history = []
        if db is not None:
            self.db = db
        else:
            from app.db.database import GenesisDB
            self.db = GenesisDB()
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.llm = None
        # Optional explicit {model, system_prompt, agent_id, price_per_call, author} override -
        # bypasses the per-user DB "selected model" lookup entirely. Used by external channels
        # (Telegram) that invoke one specific published bot directly and have no logged-in
        # platform user session to read a selection from - avoids writing to (and racing with)
        # users.selected_model_json, which is a single global field per platform account.
        self.agent_override = agent_override
        if not self.groq_key:
            try:
                from llama_cpp import Llama
                p = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
                if os.path.exists(p):
                    self.llm = Llama(model_path=p, n_ctx=4096, n_threads=4, verbose=False)
            except:
                pass

    def _get_selected_agent_config(self) -> dict:
        if self.agent_override:
            return self.agent_override
        try:
            data = self.db.get_selected_model(self.current_user_id)
            return data if data else {"model": "openai/gpt-oss-120b", "system_prompt": ""}
        except Exception:
            return {"model": "openai/gpt-oss-120b", "system_prompt": ""}

    def _get_selected_model(self) -> str:
        config = self._get_selected_agent_config()
        model = config.get("model", "gpt-oss-120b")
        # Return internal name - actual Groq model ID resolved in _resolve_groq_model
        return model

    def _resolve_groq_model(self, internal_name: str) -> str:
        """Translate an internal/catalog model name into the real Groq API model id.
        Names already unknown to MODEL_MAP (e.g. explicit Groq ids) pass through unchanged."""
        if internal_name in self.MODEL_MAP:
            resolved = self.MODEL_MAP[internal_name]
            return resolved or internal_name
        return internal_name

    def _get_effective_system_prompt(self, query: str = "") -> str:
        """Combine the base Genesis AI identity/tool spec with an optional custom
        agent persona. When chatting AS a specific published bot (agent_id present):
          - tool access is SCOPED (see process()/BOT_SCOPED_ALLOWED_TOOLS) - a bot persona
            can converse and optionally predict, but not reach platform-builder tools
          - if it has uploaded knowledge base documents, retrieves the chunks most relevant
            to `query` (see app.core.knowledge_base) and appends them, so it can answer from
            real uploaded facts instead of only ever deflecting on anything not in its prompt
          - if its owner linked an ML model to it, describes that model's expected feature
            names so the underlying LLM knows what to ask the user for and how to name them
            when calling predict_with_model (see _do_bot_predict)
        Both KB retrieval and predict-capability lookups swallow their own failures (never
        break the chat itself over a side-feature lookup issue) but log them, since a
        silently-empty knowledge base or silently-missing predict capability is a real
        regression a bot owner would want to know about."""
        agent_id = self._get_selected_agent_config().get("agent_id")
        custom = self._get_selected_agent_config().get("system_prompt", "").strip()
        if not custom:
            result = self.SYSTEM_PROMPT
        elif agent_id:
            # Bot-scoped: tool access is restricted (see process()), so the "you keep all
            # tools above" framing used for the platform's own assistant would be misleading.
            result = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"PERSONA OVERLAY (apply this personality/tone on top of everything above, "
                f"but you are still Genesis AI, NOT ChatGPT/OpenAI). You are operating as a "
                f"published bot right now, NOT the general Genesis AI platform assistant - you "
                f"can converse normally and, if described below, call predict_with_model, but "
                f"you cannot train models, build other bots, or use any other platform tool "
                f"even if asked; just decline naturally and stay in character if asked to do so."
                f"\n{custom}"
            )
        else:
            result = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"PERSONA OVERLAY (apply this personality/tone on top of everything above, "
                f"but you are still Genesis AI, NOT ChatGPT/OpenAI, and you keep all tools above):\n{custom}"
            )

        if agent_id and query:
            try:
                from app.core.knowledge_base import retrieve_relevant_chunks, format_context_block
                chunks = self.db.get_kb_chunks_for_agent(agent_id)
                if chunks:
                    relevant = retrieve_relevant_chunks(query, chunks)
                    context_block = format_context_block(relevant)
                    if context_block:
                        result = f"{result}\n\n{context_block}"
            except Exception as e:
                print(f"[KB retrieval] agent_id={agent_id} error: {e}")

        if agent_id:
            try:
                from app.core.agent_store import AgentStore
                bot = AgentStore().get_agent(agent_id)
                linked_model_id = (bot or {}).get("linked_model_id")
                if linked_model_id:
                    record = self.registry.get_record(linked_model_id)
                    features = record.get("features", [])
                    target = record.get("target", "result")
                    predict_block = (
                        f"\n\nPREDICTION CAPABILITY: you can call the predict_with_model tool to get a real "
                        f"prediction for '{target}' from this bot's linked ML model. Ask the user for any of "
                        f"these values you don't already have from the conversation, then respond with "
                        f'tool="predict_with_model" and args={{"features": {{...}}}} using EXACTLY these '
                        f"feature names as keys, with numeric values: {', '.join(features)}"
                    )
                    result = f"{result}{predict_block}"
            except Exception as e:
                print(f"[predict capability] agent_id={agent_id} error: {e}")

            # Describe any sandbox-tested skills this bot has - see app/core/skills/. Only
            # test_passed=True, active skills are ever listed (see
            # GenesisDB.list_skills_for_agent), so the LLM is never told about a skill that
            # doesn't actually work.
            try:
                skills = self.db.list_skills_for_agent(agent_id)
                if skills:
                    skill_lines = "\n".join(
                        f'- "{s["name"]}" (id: {s["id"]}): {s["description"]} - '
                        f'params schema: {s["input_schema"] or "any JSON object"}'
                        for s in skills)
                    result = (
                        f"{result}\n\nSKILLS: you have these tested capabilities beyond plain "
                        f"conversation. Call one with tool=\"run_skill\" and "
                        f'args={{"skill_id": "<id above>", "params": {{...}}}} when it '
                        f"would genuinely help answer the user, using params that match its "
                        f"schema. Only call a skill when it's actually relevant - don't force "
                        f"it into unrelated conversation.\n{skill_lines}"
                    )
            except Exception as e:
                print(f"[skills] agent_id={agent_id} error: {e}")

            # Always describe the handoff capability, regardless of whether a webhook is
            # actually configured - the tool itself degrades gracefully (see
            # _do_request_handoff) and gives an honest answer either way, so there's no
            # config to check here first.
            result = (
                f"{result}\n\nHANDOFF CAPABILITY: if you genuinely cannot help - the question "
                f"needs real-time data, account/order-specific details, or anything outside "
                f'what you actually know - call the request_handoff tool with args={{"reason": '
                f'"brief reason"}} INSTEAD OF just saying you can\'t help. This actually '
                f"notifies a real person rather than being empty words. Don't call it for "
                f"things you can genuinely answer yourself or from the knowledge base context above."
            )
        return result

    def _groq_kwargs_for(self, model: str) -> Dict:
        """gpt-oss/qwen3 reasoning models spend completion tokens on internal thinking before
        emitting the final answer - reasoning_effort='low' keeps that overhead small so the
        JSON response actually fits within max_tokens instead of getting cut off mid-thought."""
        if model.startswith("openai/gpt-oss") or model.startswith("qwen/qwen3"):
            return {"reasoning_effort": "low"}
        return {}

    def _log_groq_usage(self, call_site: str, model: str, completion=None, success: bool = True,
                         error: str = None, latency_ms: int = None, error_type: str = None,
                         provider: str = "groq"):
        """Best-effort internal Groq usage/observability logging - see GenesisDB.log_groq_usage
        docstring for why this exists (there was previously no visibility at all into the
        platform's own Groq API cost or reliability, as opposed to the separate
        external-developer /api/models/predict metering). Never lets a logging failure break
        the actual chat/tool call."""
        try:
            usage = getattr(completion, "usage", None) if completion else None
            self.db.log_groq_usage(
                self.current_user_id, call_site, model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                success=success, error=error, latency_ms=latency_ms, error_type=error_type, provider=provider)
        except Exception:
            pass

    def _call_groq(self, msg: str) -> Dict:
        """Calls the Groq API and returns the parsed tool-call JSON.

        Raises a typed LLMError subclass (see app.core.llm_errors) on ANY failure - missing
        key, rate limit, timeout, connection error, non-2xx response, or a response that
        isn't valid JSON. Callers MUST let this propagate rather than catching it and
        returning a canned reply: doing that previously made a Groq outage indistinguishable
        from a real answer (HTTP 200, a hardcoded greeting), which meant nobody could tell a
        provider outage from a working bot, and marketplace buyers were billed for a message
        the model never actually generated.
        """
        import time
        import groq as groq_sdk
        from app.core.llm_errors import LLMConfigError, LLMRateLimitError, LLMTimeoutError, LLMProviderError

        groq_model = self._resolve_groq_model(self._get_selected_model())
        if not self.groq_key:
            raise LLMConfigError("GROQ_API_KEY is not configured")

        start = time.monotonic()
        try:
            from groq import Groq
            from app.config import settings
            client = Groq(api_key=self.groq_key, timeout=settings.groq_request_timeout_seconds,
                          max_retries=settings.groq_max_retries)
            sys_prompt = self._get_effective_system_prompt(msg)
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": msg}
                ],
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
                **self._groq_kwargs_for(groq_model),
            )
        except groq_sdk.RateLimitError as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            print(f"[Groq] model='{groq_model}' rate_limit: {e}")
            self._log_groq_usage("chat_routing", groq_model, success=False, error=str(e),
                                  error_type="rate_limit", latency_ms=latency_ms)
            raise LLMRateLimitError(f"Groq rate-limited this request: {e}") from e
        except groq_sdk.AuthenticationError as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            print(f"[Groq] model='{groq_model}' auth_error: {e}")
            self._log_groq_usage("chat_routing", groq_model, success=False, error=str(e),
                                  error_type="auth", latency_ms=latency_ms)
            raise LLMConfigError(f"Groq rejected our API credentials: {e}") from e
        except groq_sdk.PermissionDeniedError as e:
            # Distinct from AuthenticationError: the API key itself is valid, but this
            # specific model isn't enabled for it - some models on Groq need access granted
            # separately in the console before a key can use them, and a 403 here means that
            # step wasn't done for model_id. Retrying changes nothing, so this maps to
            # LLMConfigError (operator-actionable) like AuthenticationError, not
            # LLMProviderError (which implies "try again later" is a reasonable next step).
            latency_ms = int((time.monotonic() - start) * 1000)
            print(f"[Groq] model='{groq_model}' permission_denied: {e}")
            self._log_groq_usage("chat_routing", groq_model, success=False, error=str(e),
                                  error_type="permission_denied", latency_ms=latency_ms)
            raise LLMConfigError(
                f"This Groq API key doesn't have access to model '{groq_model}' - check model "
                f"access at console.groq.com, or select a different model. ({e})") from e
        except groq_sdk.APITimeoutError as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            print(f"[Groq] model='{groq_model}' timeout: {e}")
            self._log_groq_usage("chat_routing", groq_model, success=False, error=str(e),
                                  error_type="timeout", latency_ms=latency_ms)
            raise LLMTimeoutError(f"Groq request timed out: {e}") from e
        except (groq_sdk.APIConnectionError, groq_sdk.APIStatusError, groq_sdk.APIError) as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            print(f"[Groq] model='{groq_model}' provider_error: {e}")
            self._log_groq_usage("chat_routing", groq_model, success=False, error=str(e),
                                  error_type="provider_error", latency_ms=latency_ms)
            raise LLMProviderError(f"Groq provider error: {e}") from e
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            print(f"[Groq] model='{groq_model}' unexpected_error: {e}")
            self._log_groq_usage("chat_routing", groq_model, success=False, error=str(e),
                                  error_type="unexpected", latency_ms=latency_ms)
            raise LLMProviderError(f"Unexpected error calling Groq: {e}") from e

        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            parsed = json.loads(completion.choices[0].message.content)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            self._log_groq_usage("chat_routing", groq_model, completion, success=False,
                                  error=str(e), error_type="malformed_response", latency_ms=latency_ms)
            raise LLMProviderError(f"Model returned a response that wasn't valid JSON: {e}") from e

        self._log_groq_usage("chat_routing", groq_model, completion, success=True, latency_ms=latency_ms)
        return parsed

    def _call_local(self, msg: str) -> Dict:
        """Uses a locally-loaded GGUF model as a genuine offline fallback for when Groq is
        unavailable. Only takes this path when a real local model was actually loaded at
        startup (self.llm is not None, see __init__) - previously this returned a hardcoded
        "Hi! I am Genesis AI Agent." success message even when self.llm was None, which is
        not a fallback, it's total unavailability, and must be reported as an error rather
        than a fake successful reply."""
        from app.core.llm_errors import LLMConfigError, LLMProviderError
        if not self.llm:
            raise LLMConfigError("No Groq API key is configured and no local model is loaded")
        p = f"<|im_start|>system\n{self._get_effective_system_prompt(msg)}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        try:
            t = self.llm(p, max_tokens=300, stop=["<|im_end|>"], temperature=0.1)["choices"][0]["text"].strip()
            return json.loads(t) if t.startswith("{") else {"tool": None, "message": t}
        except Exception as e:
            raise LLMProviderError(f"Local model inference failed: {e}") from e

    def _call_llm(self, msg: str) -> Dict:
        """Returns the parsed tool-call JSON, or raises an LLMError subclass (see
        app.core.llm_errors) if generation genuinely failed. process() and the
        /api/agent/message route both let this propagate rather than catching it and
        returning a fake success - see the docstrings above for why that matters."""
        from app.core.llm_errors import LLMError
        if self.groq_key:
            try:
                return self._call_groq(msg)
            except LLMError:
                if self.llm:
                    # A real local model is actually loaded - this is a genuine degraded-mode
                    # fallback, not a fake one, so it's fine to try it for real.
                    return self._call_local(msg)
                raise
        return self._call_local(msg)

    def _get_memory(self, user_id: str) -> str:
        f = self.db.get_facts(user_id)
        return "\n".join([f"- {x['predicate']}: {x['object']}" for x in f[:5]]) if f else ""

    def _extract_facts(self, msg: str, user_id: str):
        for k, p in {"name": r"(?:меня зовут|я|моё имя)\s+(\w+)",
                     "likes": r"(?:я люблю|обожаю|предпочитаю)\s+(.+?)(?:[.,!]|$)",
                     "goal": r"(?:я хочу|моя цель)\s+(.+?)(?:[.,!]|$)"}.items():
            m = re.search(p, msg, re.IGNORECASE)
            if m and 2 < len(v := m.group(1).strip()) < 100: self.db.add_fact(user_id, k, v, 0.8)

    def process(self, message: str, conversation_id: str = None, user_projects: List[Dict] = None, user_id: str = None) -> AgentResponse:
        cid = conversation_id or "default"
        uid = user_id or "anonymous"  # only reachable via internal/test calls that bypass the auth-required API layer
        self.current_user_id = uid
        self.db.create_chat(cid, user_id=uid)
        self.db.add_message(cid, "user", message)
        self._extract_facts(message, uid)
        projects = user_projects or self.db.get_projects(user_id=uid)
        mem = self._get_memory(uid)
        if self.db.get_message_count(cid) == 1 and projects and message.lower().strip().rstrip('!,.') in ["привет",
                                                                                                          "hello", "hi",
                                                                                                          "здравствуйте"]:
            lines = ["Hi! Here are your projects:"] + [
                f"• {p.get('name', '')} ({p.get('task', '').replace('_', ' ')})" for p in projects[:5]] + [
                        "\nWhich one do you want to work on?"]
            msg = "\n".join(lines);
            mid = self.db.add_message(cid, "assistant", msg)
            return AgentResponse(message=msg, message_id=mid)

        # Check the account's monthly Groq usage cap BEFORE spending any tokens on this
        # message - see app.core.quota. Protects the platform's own shared GROQ_API_KEY from
        # unbounded per-account cost (bot-builder sessions and codegen fix loops in
        # particular can each be many real Groq calls). Checked here, once, ahead of ANY LLM
        # path (Groq or the local fallback) rather than inside _call_groq itself, so a
        # blocked user gets an honest quota message instead of silently falling through to
        # the local-model fallback.
        from app.core.quota import check_quota, quota_exceeded_message
        allowed, used, limit = check_quota(self.db, uid if uid != "anonymous" else None)
        if not allowed:
            msg = quota_exceeded_message(used, limit)
            mid = self.db.add_message(cid, "assistant", msg)
            return AgentResponse(message=msg, message_id=mid, success=False)

        r = self._call_llm(message + "\n" + mem)
        t, a = r.get("tool"), r.get("args", {})

        # Published-bot conversations get a SCOPED tool surface. A bot persona should only be
        # able to converse (with knowledge-base context automatically injected - see
        # _get_effective_system_prompt) and optionally call predict_with_model if its owner
        # linked a model to it. It must NOT be able to reach platform-builder tools like
        # train_model/create_bot/publish_marketplace just because the underlying LLM call uses
        # the same JSON tool-calling mechanism as the platform's own assistant - those tools act
        # on the CHATTING user's own account, so an unscoped published bot would otherwise let
        # any stranger talking to it train models or publish marketplace listings under their
        # own account while thinking they're just chatting with someone else's support bot.
        agent_id = self._get_selected_agent_config().get("agent_id")
        blocked_tool_attempt = bool(agent_id and t and t not in self.BOT_SCOPED_ALLOWED_TOOLS)
        if agent_id and t not in self.BOT_SCOPED_ALLOWED_TOOLS:
            t = None

        if t in ('analyze_dataset', 'load_data', 'load_dataset', 'analyze'):
            p = a.get("path", self._extract_path(message))
            resp = self._do_analyze(p) if p else AgentResponse(message="Please provide a file path.", success=False)
        elif t in ('train_model', 'train', 'fit_model'):
            resp = self._do_train(message, a)
        elif t in ('retrain_model', 'retrain', 'update_model'):
            resp = self._do_retrain(a.get("model_id", ""), a.get("path", self._extract_path(message)), a)
        elif t in ('find_best_model', 'best_model', 'find_best'):
            resp = self._do_find_best(a.get("task", "binary_classification"))
        elif t in ('list_models', 'list', 'show_models'):
            resp = self._do_list_models()
        elif t in ('search_datasets', 'search', 'find_datasets'):
            resp = self._do_search(message, a)
        elif t in ('research_project', 'research', 'investigate'):
            resp = self._do_research(message, a)
        elif t in ('suggest_pipeline', 'suggest_structure', 'propose_pipeline'):
            resp = self._do_suggest_pipeline(message, a)
        elif t in ('generate_code', 'codegen', 'generate_pipeline'):
            resp = self._do_generate_code(message, a)
        elif t in ('install_dataset', 'install', 'download_dataset'):
            resp = self._do_install(str(a.get("number", "1")))
        elif t in ('list_marketplace', 'marketplace', 'show_marketplace'):
            resp = self._do_list_marketplace(a)
        elif t in ('list_projects', 'projects', 'show_projects'):
            resp = self._do_list_projects()
        elif t in ('publish_marketplace', 'publish_to_marketplace', 'publish_model'):
            resp = self._do_publish_marketplace(a)
        elif t in ('create_custom_block', 'create_block', 'new_block'):
            resp = self._do_create_custom_block(a)
        elif t in ('create_bot', 'build_bot', 'make_bot', 'bot_builder'):
            resp = self._do_create_bot(message, a)
        elif t in ('create_model', 'build_model', 'train_full_model'):
            resp = self._do_create_model(message, a)
        elif t in ('predict_with_model', 'predict', 'run_model'):
            resp = self._do_bot_predict(agent_id, a)
        elif t in ('request_handoff', 'handoff_to_human', 'escalate'):
            resp = self._do_request_handoff(agent_id, cid, message, a)
        elif t in ('run_skill', 'call_skill', 'use_skill'):
            resp = self._do_run_skill(agent_id, a)
        elif t in ('add_skill', 'add_capability', 'create_skill'):
            resp = self._do_add_skill(agent_id, a)
        else:
            if blocked_tool_attempt:
                # The model tried to use a platform-builder tool (train a model, create a
                # bot, publish to the marketplace, etc.) that isn't available while chatting
                # with someone else's published bot - see the scoping note above. Its own
                # "message" text was written WHILE reasoning about that tool call, so it can't
                # be trusted verbatim: an LLM explaining why it can't do something will often
                # reach for the real, structural reason ("I'm a bot someone else built, not
                # the platform assistant"), which leaks implementation details this bot's own
                # persona has no business discussing. A fixed, in-character decline avoids
                # that regardless of what the model was about to say.
                resp = AgentResponse(message="I'm not able to do that here, but I'm happy to help with what I'm actually built for - what do you need?")
            else:
                resp = AgentResponse(message=r.get("message", "Hi! How can I help?"))

        resp.message_id = self.db.add_message(cid, "assistant", resp.message, resp.data)
        return resp

    # Fixed catalog of blocks the Pipeline Builder understands (kept in sync with PBLOCKS in static/index.html)
    PIPELINE_BLOCKS = {
        "load_data": "Load CSV - reads a tabular dataset",
        "load_text": "Load Text - reads a text corpus",
        "load_images": "Load Images - reads an image dataset",
        "profile": "Profile - inspects columns, types, missing values, target candidates",
        "clean": "Clean - handles missing values, duplicates, outliers",
        "scale": "Scale - normalizes/standardizes numeric features",
        "encode": "Encode - converts categorical columns into numeric features",
        "feature_select": "Feature Select - keeps only the most informative columns",
        "tokenize": "Tokenize - splits text into tokens for NLP",
        "embedding": "Embeddings - converts tokens/text into vector representations",
        "augment": "Augment - generates additional training examples (e.g. image augmentation)",
        "train_xgboost": "Train XGBoost - gradient boosted trees for classification",
        "train_xgboost_reg": "Train XGBoost (Regression) - gradient boosted trees for regression",
        "train_random_forest": "Train Random Forest - ensemble of decision trees for classification",
        "train_logistic": "Train Logistic Regression - simple linear classifier",
        "train_svm": "Train SVM - support vector machine classifier",
        "train_knn": "Train KNN - k-nearest-neighbors classifier",
        "train_mlp": "Train Neural Net - small feed-forward neural network",
        "train_cnn": "Train CNN - convolutional network for images",
        "cross_validate": "Cross-Validate - k-fold validation for a more reliable metric estimate",
        "evaluate": "Evaluate - computes accuracy/F1/precision/recall on a held-out set",
        "evaluate_reg": "Evaluate (Regression) - computes RMSE/MAE/R2",
        "confusion_matrix": "Confusion Matrix - visual breakdown of correct vs incorrect predictions per class",
        "save": "Save Model - persists the trained model to the registry",
    }

    def _ai_build_pipeline(self, description: str) -> Optional[tuple]:
        """Ask Groq to pick block ids from the FIXED PIPELINE_BLOCKS catalog for this description,
        and to request NEW custom blocks (by name+description) for anything the catalog doesn't cover.
        Returns (blocks, custom_defs) where custom_defs is a list of {"ref", "name", "description"} for
        steps that need a brand-new block, or None on any failure so the caller falls back to keywords."""
        if not self.groq_key:
            return None
        try:
            from groq import Groq
            from app.config import settings
            client = Groq(api_key=self.groq_key, timeout=settings.groq_request_timeout_seconds,
                          max_retries=settings.groq_max_retries)
            catalog = "\n".join(f'- "{k}": {v}' for k, v in self.PIPELINE_BLOCKS.items())
            prompt = (
                f"Available pipeline blocks (use ONLY these exact ids, in a sensible order):\n{catalog}\n\n"
                f"User wants this ML pipeline: {description}\n\n"
                'Respond ONLY with JSON: {"blocks": ["load_data", "custom:my_step", "...", "save"], '
                '"custom_blocks": [{"ref": "custom:my_step", "name": "My Step", "description": "what it does in plain language"}]}. '
                'Always start with a load_* block and end with "save". Prefer existing catalog blocks. '
                'ONLY invent a "custom:<short_name>" block (and describe it in custom_blocks) when the pipeline '
                'genuinely needs a step that has no equivalent in the catalog above. '
                'Do not mix train_cnn with tabular data, do not add tokenize unless there is text.'
            )
            groq_model = self._resolve_groq_model(self._get_selected_model())
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=800,
                response_format={"type": "json_object"},
                **self._groq_kwargs_for(groq_model),
            )
            parsed = json.loads(completion.choices[0].message.content)
            self._log_groq_usage("pipeline_builder", groq_model, completion, success=True)
            custom_defs = parsed.get("custom_blocks", []) or []
            valid_refs = {c["ref"] for c in custom_defs if "ref" in c and "name" in c}
            blocks = [b for b in parsed.get("blocks", []) if b in self.PIPELINE_BLOCKS or b in valid_refs]
            return (blocks, custom_defs) if blocks else None
        except Exception as e:
            print(f"[Groq pipeline builder] error: {e}")
            self._log_groq_usage("pipeline_builder", self._resolve_groq_model(self._get_selected_model()), success=False, error=str(e))
            return None

    def _keyword_build_pipeline(self, description: str) -> List[str]:
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
        return pipeline

    def _do_suggest_pipeline(self, message: str, args: Dict = None) -> AgentResponse:
        from app.core.custom_blocks import CustomBlockStore
        description = (args or {}).get("description", message) if args else message
        store = CustomBlockStore()
        ai_result = self._ai_build_pipeline(description)
        custom_defs = []
        if ai_result:
            pipeline, custom_defs = ai_result
        else:
            pipeline = self._keyword_build_pipeline(description)

        # Materialize any AI-requested custom blocks as real blocks.CustomBlock rows, and splice
        # their real generated ids into the pipeline in place of the "custom:xxx" placeholder.
        ref_to_real_id = {}
        block_labels = dict(self.PIPELINE_BLOCKS)
        for cdef in custom_defs:
            ref = cdef.get("ref")
            name = cdef.get("name", ref)
            desc = cdef.get("description", "")
            if not ref or ref in ref_to_real_id:
                continue
            created = store.create(
                name=name,
                code=f"# TODO: implement '{name}'\n# {desc}\ndef run(data):\n    raise NotImplementedError({desc!r})",
                icon="🔧", category="custom",
            )
            ref_to_real_id[ref] = created["id"]
            block_labels[created["id"]] = f"{name} (custom) - {desc}"
        pipeline = [ref_to_real_id.get(b, b) for b in pipeline]

        if any(b in pipeline for b in ["train_xgboost", "train_xgboost_reg", "train_random_forest", "train_logistic"]):
            if "evaluate" not in pipeline: pipeline.insert(-1, "evaluate")
        custom_blocks = store.list_blocks()
        short_names = {**{k: v.split(" - ")[0] for k, v in self.PIPELINE_BLOCKS.items()},
                       **{cb["id"]: cb["name"] for cb in custom_blocks}}
        pipeline_str = ",".join(pipeline)
        builder_link = f"/builder?pipeline={pipeline_str}"
        steps_explained = "\n".join(
            f"{i+1}. {block_labels.get(b, b)}" for i, b in enumerate(pipeline)
        )
        return AgentResponse(
            message=(f"Pipeline: {' → '.join(short_names.get(b, b) for b in pipeline)}\n\n"
                     f"What each step does:\n{steps_explained}\n\n"
                     f"Open in Builder: {builder_link}\nCustom blocks used: {len(custom_blocks)}"),
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
        if not fp: return AgentResponse(message="Please provide a file path.", success=False)
        if not self._validate_path(fp): return AgentResponse(message="Access denied: that path isn't in an allowed directory.", success=False)
        try:
            self.profiler.run_full_profile(fp);
            self.current_dataset = fp
            s = self.profiler.get_summary()
            return AgentResponse(
                message=f"Analysis of {s['file_info']['name']}:\n• Rows: {s['file_info']['rows']}\n• Columns: {s['file_info']['columns']}\n• Task: {s['suggested_ml_task']['task']}\n• Target: {s['suggested_ml_task']['target_column']}\n\nTrain a model?",
                data=_to_native(s))
        except Exception as e:
            return AgentResponse(message=f"Error: {e}", success=False)

    def _do_train(self, msg: str, args: Dict = None) -> AgentResponse:
        # Prefer the LLM's own extracted path argument (see the train_model few-shot example
        # above) over re-deriving it from raw text - mirrors analyze_dataset/retrain_model,
        # which already do this. Falls back to regex extraction only if the LLM didn't supply
        # one (e.g. "train xgboost on churn" with no filename in the message at all).
        fp = (args or {}).get("path") or self._extract_path(msg)
        if not fp:
            return AgentResponse(message="Please provide a dataset path, e.g.: datasets/your_file.csv", success=False)
        if not os.path.exists(fp):
            return AgentResponse(message=f"File not found: {fp}", success=False)
        if not self._validate_path(fp):
            return AgentResponse(message="Access denied: path is outside allowed directories (datasets/, models/, static/, generated/).", success=False)
        tc = (args.get("target") or args.get("target_column")) if args else None
        # Default to real AutoML ("auto" now cross-validates xgboost/random_forest/
        # hist_gradient_boosting/logistic-or-linear and fits the winner - see trainer.py)
        # rather than blindly always using xgboost regardless of what actually fits the data best.
        mt = args.get("model_type", "auto") if args else "auto"
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
            # result.model_type is the ACTUAL chosen algorithm (e.g. when mt="auto" picked
            # "hist_gradient_boosting") - register/display that, not the literal "auto" request.
            chosen_type = result.model_type
            # Register the ACTUAL post-encoding columns the saved model expects (result.feature_columns),
            # not the raw pre-encoding profiler columns - those two lists diverge whenever the dataset
            # has categorical features (one-hot encoding changes both the count and the names), and a
            # caller building a predict-time feature vector needs the real thing to get the right shape.
            mid = self.registry.register(model=model, metrics=result.metrics, task=task, model_type=chosen_type,
                                         features=result.feature_columns, target=tc,
                                         metadata={"author": self.current_user_id})
            s = self.trainer.get_summary()
            return AgentResponse(message=f"Model trained!\n• ID: {mid}\n• Type: {chosen_type}\n• Metrics: {s['metrics']}",
                                 data=_to_native({"model_id": mid, **s}))
        except Exception as e:
            return AgentResponse(message=f"Error: {e}", success=False)

    def _do_retrain(self, model_id: str, new_dataset_path: Optional[str], args: Dict = None) -> AgentResponse:
        """Retrain an EXISTING registered model on a new/updated dataset, producing a new
        VERSION in the same model family (see ModelRegistry.register family_id/version)
        rather than an unrelated fresh model - the trained model itself is genuinely new
        (retraining always refits from scratch, there's no incremental/warm-start update
        here), but its lineage is tracked so nothing is lost: get_family_history(family_id)
        still shows every prior version, same idea as prompt version history."""
        if not model_id:
            return AgentResponse(message="Please provide the model_id of the model to retrain.", success=False)
        if not new_dataset_path:
            return AgentResponse(message="Please provide the path to the new/updated dataset.", success=False)
        try:
            old_record = self.registry.get_record(model_id)
        except KeyError:
            return AgentResponse(message=f"Model {model_id} was not found in the registry.", success=False)
        # Ownership check: only the model's original author may retrain it. Records created
        # before ownership tracking existed have no "author" in metadata - treat those as
        # platform-owned/legacy and still allow retrain rather than locking everyone out of
        # models nobody can claim. Anything WITH an author is enforced strictly. Without this,
        # any authenticated user who learns/guesses a model_id could trigger a retrain (real
        # compute cost, and it rewrites the model's version history) on a model they don't own.
        record_author = (old_record.get("metadata") or {}).get("author")
        if record_author and record_author != self.current_user_id:
            return AgentResponse(message="You don't have permission to retrain this model.", success=False)
        if not os.path.exists(new_dataset_path):
            return AgentResponse(message=f"File not found: {new_dataset_path}", success=False)

        task = old_record["task"]
        target = ((args or {}).get("target") or (args or {}).get("target_column")) or old_record["target"]
        # Keep the same algorithm the previous version used by default (a fair like-for-like
        # retrain), unless the caller explicitly asks for a different one or "auto".
        model_type = (args or {}).get("model_type") or old_record["model_type"]

        try:
            self.profiler.run_full_profile(new_dataset_path)
            result = self.trainer.train(df=self.profiler.df, target_col=target, task=task, model_type=model_type)
            model = pickle.load(open(result.model_path, "rb"))
            family_id = old_record.get("family_id", old_record["model_id"])
            new_id = self.registry.register(
                model=model, metrics=result.metrics, task=task, model_type=result.model_type,
                features=result.feature_columns, target=target,
                metadata={**old_record.get("metadata", {}), "retrained_from": model_id,
                         "retrain_dataset": new_dataset_path},
                family_id=family_id, parent_model_id=model_id,
            )
            history = self.registry.get_family_history(family_id)
            new_version = history[0]["version"]
            return AgentResponse(
                message=(f"Model retrained ✅ New version v{new_version} "
                        f"(id: {new_id}); the previous version is kept in the history.\n"
                        f"Metrics: {result.metrics}"),
                data=_to_native({"model_id": new_id, "family_id": family_id,
                                 "version": new_version, "metrics": result.metrics,
                                 "history_length": len(history)}))
        except Exception as e:
            return AgentResponse(message=f"Retraining error: {e}", success=False)

    def _do_bot_predict(self, agent_id: Optional[str], args: Dict) -> AgentResponse:
        """Runs a real prediction using the ONE model this bot's owner linked to it (see
        AgentConfig.linked_model_id) - this is what lets a published bot give an actual answer
        instead of just talking about one. args["features"] must be a dict of {feature_name:
        value} using the model's own registered feature names as keys - the bot's system
        prompt is given that exact list (see _get_effective_system_prompt) so the underlying
        LLM knows what to ask the user for and how to name it."""
        if not agent_id:
            return AgentResponse(message="Predictions are only available in a chat with a specific bot.", success=False)
        from app.core.agent_store import AgentStore
        bot = AgentStore().get_agent(agent_id)
        model_id = (bot or {}).get("linked_model_id")
        if not model_id:
            return AgentResponse(message="This bot isn't linked to any model.", success=False)
        try:
            record = self.registry.get_record(model_id)
        except KeyError:
            return AgentResponse(message="The linked model could no longer be found in the registry.", success=False)

        expected = record.get("features", [])
        features = args.get("features") or {}
        missing = [f for f in expected if f not in features]
        if missing:
            return AgentResponse(
                message=f"I need values for: {', '.join(missing)}. Could you provide them?",
                success=False)
        try:
            row = [float(features[f]) for f in expected]
        except (ValueError, TypeError) as e:
            return AgentResponse(message=f"Invalid value among the provided features: {e}", success=False)

        try:
            import numpy as np
            model = self.registry.get_model(model_id)
            prediction = model.predict(np.array(row).reshape(1, -1))
            pred_value = prediction[0]
            if hasattr(pred_value, "item"):
                pred_value = pred_value.item()
            target = record.get("target", "result")
            return AgentResponse(
                message=f"Prediction ({target}): {pred_value}",
                data=_to_native({"prediction": pred_value, "target": target, "model_id": model_id}))
        except Exception as e:
            return AgentResponse(message=f"Prediction error: {e}", success=False)

    def _do_request_handoff(self, agent_id: Optional[str], conversation_id: str,
                             user_message: str, args: Dict) -> AgentResponse:
        """Fires the webhook configured on this bot (if any) to actually notify a human that
        the bot couldn't help - closes the loop on the existing anti-hallucination deflection
        behavior (bot_builder.py), which previously only ever produced words with nothing
        behind them. Deliberately overrides whatever the LLM itself said (rather than trusting
        its phrasing verbatim) so the user gets an honest, consistent answer about whether a
        real person was actually notified - if no webhook is configured, the bot must NOT
        claim someone will follow up, since no one will."""
        if not agent_id:
            return AgentResponse(message="Handoff to a specialist is only available in a chat with a specific bot.", success=False)
        from app.core.agent_store import AgentStore
        bot = AgentStore().get_agent(agent_id)
        if not bot:
            return AgentResponse(message="Bot not found.", success=False)

        reason = (args or {}).get("reason", "user needs help the bot couldn't provide")
        webhook_url = bot.get("handoff_webhook_url")
        if not webhook_url:
            return AgentResponse(
                message=("I'm not able to check that myself, and support notifications "
                         "aren't configured for this bot yet - please contact the "
                         "seller directly."),
                data={"handoff_attempted": True, "delivered": False, "reason": "no_webhook_configured"})

        from app.core.handoff import send_handoff_webhook, build_handoff_payload
        payload = build_handoff_payload(bot.get("name", "Bot"), agent_id, conversation_id, user_message, reason)
        result = send_handoff_webhook(webhook_url, payload)
        self.db.log_handoff_event(agent_id, conversation_id, user_message, reason,
                                  result["delivered"], result.get("error"))
        if result["delivered"]:
            return AgentResponse(
                message="I've forwarded your question to the support team - they'll get back to you shortly.",
                data={"handoff_attempted": True, "delivered": True})
        return AgentResponse(
            message="I tried to forward your question to the support team, but delivery failed. Please try contacting them directly.",
            data={"handoff_attempted": True, "delivered": False, "reason": result.get("error")})

    def _do_run_skill(self, agent_id: Optional[str], args: Dict) -> AgentResponse:
        """Runs one of this bot's own sandbox-tested skills (see app/core/skills/) with
        params the LLM chose from the conversation. Only ever runs skills that belong to
        THIS bot and are marked test_passed=True (see GenesisDB.list_skills_for_agent,
        which is also what populates the system prompt describing available skills in the
        first place - the LLM can only ever name a skill_id it was actually told about)."""
        if not agent_id:
            return AgentResponse(message="Skills are only available in a chat with a specific bot.", success=False)
        skill_id = (args or {}).get("skill_id", "")
        params = (args or {}).get("params", {}) or {}
        if not skill_id:
            return AgentResponse(message="Please specify which skill to run.", success=False)

        available = {s["id"]: s for s in self.db.list_skills_for_agent(agent_id)}
        if skill_id not in available:
            return AgentResponse(message="That skill isn't available on this bot.", success=False)

        full_skill = self.db.get_skill(skill_id)
        if not full_skill or not full_skill["active"] or not full_skill["test_passed"]:
            return AgentResponse(message="That skill isn't currently available.", success=False)

        if full_skill["action_type"] == "outbound_action":
            # This skill does something to a third party (send a message, post something,
            # etc.) - it never runs directly from a chat turn, no matter who's asking or how
            # the conversation is going. It always creates a pending action that only the
            # bot's OWNER can approve (see /api/pending-actions/{id}/approve in main.py) -
            # this is true even when the owner themself is the one chatting right now; there
            # is deliberately no "skip the approval because it's you" shortcut, so the
            # approval step can't accidentally be bypassed by a future code path.
            action = self.db.create_pending_action(
                skill_id=skill_id, agent_id=agent_id, owner_user_id=full_skill["owner_user_id"],
                params=params, source="chat")
            return AgentResponse(
                message=(f"I'd like to use \"{full_skill['name']}\" with these details: {json.dumps(params)}. "
                         f"Since this skill takes an action outside the platform, it needs your approval first - "
                         f"you can approve or reject it from your dashboard."),
                data={"pending_action_id": action["id"], "requires_approval": True})

        from app.core.skills.skill_generator import SkillGenerator
        outcome = SkillGenerator().run_stored_skill(
            full_skill["code"], params, full_skill["allowed_domains"])
        if not outcome["success"]:
            return AgentResponse(
                message=f"I tried to use the \"{full_skill['name']}\" skill, but it didn't work: {outcome['error']}",
                success=False, data={"skill_id": skill_id, "error": outcome["error"]})
        return AgentResponse(
            message=f"Ran \"{full_skill['name']}\": {json.dumps(outcome['result'])}",
            data={"skill_id": skill_id, "result": outcome["result"]})

    def _do_add_skill(self, agent_id: Optional[str], args: Dict) -> AgentResponse:
        """Adds a new real, sandbox-tested capability to one of the CALLER'S OWN bots -
        the conversational counterpart to /api/skills/create, reachable either while chatting
        directly with an existing bot (agent_id already identifies it) or from the general
        platform assistant by naming the bot (args["bot_name"]).

        Ownership is enforced here regardless of which conversation this came from: this tool
        is listed in BOT_SCOPED_ALLOWED_TOOLS so it CAN be invoked while chatting with any
        published bot (including someone else's, if a stranger tries it against a support bot
        they don't own) - the check below is what actually stops that, not the tool-scoping
        list, which only controls which tools are reachable at all, not who they act on."""
        args = args or {}
        if not self.current_user_id or self.current_user_id == "anonymous":
            return AgentResponse(message="Please sign in to add a skill to a bot.", success=False)

        raw_request = (args.get("skill_description") or args.get("description") or "").strip()
        if not raw_request:
            return AgentResponse(
                message="Tell me what the skill should actually do - what should it check, and on which site or service?",
                success=False)

        from app.core.agent_store import AgentStore
        target_agent_id = agent_id
        if not target_agent_id:
            bot_name = args.get("bot_name", "").strip()
            if not bot_name:
                return AgentResponse(message="Which bot should get this skill? Tell me its name.", success=False)
            my_bots = AgentStore().list_agents(author=self.current_user_id, published_only=False)
            matches = [b for b in my_bots if b.get("name", "").strip().lower() == bot_name.lower()]
            if not matches:
                return AgentResponse(message=f"I couldn't find a bot of yours named \"{bot_name}\".", success=False)
            target_agent_id = matches[0]["id"]

        bot = AgentStore().get_agent(target_agent_id)
        if not bot:
            return AgentResponse(message="That bot doesn't exist.", success=False)
        if bot.get("author") != self.current_user_id:
            return AgentResponse(message="You can only add skills to bots you own.", success=False)

        # This triggers real Groq calls beyond the routing call that got us here (parsing the
        # request, then the generate/test/fix loop) - same discipline as /api/skills/create's
        # _enforce_groq_quota, just inlined since agent.py can't import the main.py route helper.
        from app.core.quota import check_quota, quota_exceeded_message
        allowed, used, limit = check_quota(self.db, self.current_user_id)
        if not allowed:
            return AgentResponse(message=quota_exceeded_message(used, limit), success=False)

        existing = [s for s in self.db.list_skills_for_owner(self.current_user_id) if s["agent_id"] == target_agent_id]
        if len(existing) >= self.MAX_SKILLS_PER_BOT:
            return AgentResponse(
                message=f"\"{bot['name']}\" already has {len(existing)} skills - the limit is "
                        f"{self.MAX_SKILLS_PER_BOT} per bot. Remove one before adding another.",
                success=False)

        from app.core.skills.skill_generator import SkillGenerator, _detect_write_http_method
        gen = SkillGenerator(user_id=self.current_user_id)
        parsed = gen.parse_request(raw_request)
        if not parsed:
            return AgentResponse(
                message="I couldn't understand what this skill should do - try describing it "
                        "more concretely (what it checks, and on which site).",
                success=False)

        domains = parsed.get("allowed_domains") or []
        if not domains:
            return AgentResponse(
                message=f"I need to know which specific website or service this should check - "
                        f"that wasn't clear from \"{raw_request}\". Try naming the site.",
                success=False)

        result = gen.build(parsed.get("description", raw_request), domains)
        action_type = parsed.get("action_type", "read_only")
        if result["code"] and _detect_write_http_method(result["code"]):
            action_type = "outbound_action"  # same defense-in-depth override as everywhere else

        skill = self.db.create_skill(
            agent_id=target_agent_id, owner_user_id=self.current_user_id,
            name=(parsed.get("name") or "skill")[:60], description=parsed.get("description", raw_request),
            code=result["code"] or "", allowed_domains=domains, input_schema={}, action_type=action_type)
        self.db.update_skill_test_result(skill["id"], passed=result["success"], log=json.dumps(result["log"])[:4000])

        if result["success"]:
            approval_note = (" (since this takes an action outside the platform, it'll need your "
                             "approval each time it runs)") if action_type == "outbound_action" else ""
            return AgentResponse(
                message=f"Added \"{skill['name']}\" to \"{bot['name']}\" - tested and working{approval_note}.",
                data={"skill_id": skill["id"], "action_type": action_type})
        return AgentResponse(
            message=f"I tried to build that skill for \"{bot['name']}\", but it didn't pass testing. "
                    f"Try rephrasing what it should do, or check that {', '.join(domains)} is the right site.",
            success=False, data={"skill_id": skill["id"]})

    def _do_find_best(self, task="binary_classification") -> AgentResponse:
        metric = "r2" if "regression" in task else "accuracy"
        best = self.registry.find_best(task, metric)
        if not best: return AgentResponse(message="No models found.", success=False)
        return AgentResponse(message=f"Best model:\n• ID: {best['model_id']}\n• Metrics: {best['metrics']}",
                             data=_to_native(best))

    def _do_list_models(self) -> AgentResponse:
        models = self.registry.list_models()
        if not models: return AgentResponse(message="The registry is empty.", success=False)
        return AgentResponse(message="\n".join([f"• {r['model_id'][:50]}... — {r['task']}" for r in models[:5]]))

    def _do_list_marketplace(self, args: Dict = None) -> AgentResponse:
        from app.core.marketplace import Marketplace
        task = (args or {}).get("task")
        listings = Marketplace().search(task=task) if task else Marketplace().search()
        if not listings:
            return AgentResponse(message="There are no published models in the marketplace yet.", success=False)
        lines = [f"• {l.get('name', l.get('model_id', '?'))} — ${l.get('price_per_call', 0):.3f}/call, "
                 f"★{l.get('rating', 0):.1f}" for l in listings[:8]]
        return AgentResponse(message="Marketplace:\n" + "\n".join(lines), data={"listings": listings[:8]})

    def _do_list_projects(self) -> AgentResponse:
        projects = self.db.get_projects(user_id=self.current_user_id)
        if not projects:
            return AgentResponse(message="No projects yet. Want to create one?", success=False)
        lines = [f"• {p.get('name', '')} ({p.get('task', '').replace('_', ' ')}) — "
                 f"{p.get('model_count', 0)} models" for p in projects[:10]]
        return AgentResponse(message="Your projects:\n" + "\n".join(lines), data={"projects": projects[:10]})

    def _do_publish_marketplace(self, args: Dict) -> AgentResponse:
        """Actually publishes a listing to the marketplace - a real DB write, not a suggestion."""
        from app.core.marketplace import Marketplace
        model_id = args.get("model_id")
        if not model_id:
            recs = self.registry.list_models()
            if not recs: return AgentResponse(message="No trained models available to publish.", success=False)
            model_id = recs[0]["model_id"]
        try:
            listing = Marketplace().publish(
                model_id=model_id,
                name=args.get("name", model_id[:40]),
                description=args.get("description", "Published via Genesis AI Agent"),
                author=args.get("author", "genesis_user"),
                price_per_call=float(args.get("price_per_call", 0.01)),
            )
            return AgentResponse(message=f"Published to marketplace: {listing.get('name', model_id)} "
                                          f"at ${listing.get('price_per_call', 0):.3f}/call.", data=listing)
        except Exception as e:
            return AgentResponse(message=f"Failed to publish: {e}", success=False)

    def _do_create_custom_block(self, args: Dict) -> AgentResponse:
        """Real tool: creates a new block in blocks/store.json (not just a suggestion) so it
        immediately shows up in the Pipeline Builder's block palette."""
        from app.core.custom_blocks import CustomBlockStore
        name = (args or {}).get("name", "").strip()
        description = (args or {}).get("description", "").strip()
        if not name:
            return AgentResponse(message="Please provide a name for the block.", success=False)
        code = (args or {}).get("code") or (
            f"# TODO: implement '{name}'\n# {description}\ndef run(data):\n    raise NotImplementedError({description!r})"
        )
        block = CustomBlockStore().create(
            name=name, code=code, icon=(args or {}).get("icon", "🔧"),
            category=(args or {}).get("category", "custom"),
        )
        return AgentResponse(
            message=f"Created block \"{name}\" (id: {block['id']}).\n{description}\n\n"
                    f"It's already available in the Pipeline Builder - add it to a pipeline.",
            data=block,
        )

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
            return AgentResponse(message=f"Error: {e}", success=False)

    def _do_create_model(self, msg: str, args: Dict = None) -> AgentResponse:
        from app.core.model_builder import ModelBuilder
        if not self.current_dataset:
            return AgentResponse(message="Please load/select a dataset first (analyze a file), and then I can build a model.", success=False)
        description = msg
        target = (args or {}).get('target_column')
        result = ModelBuilder().build(description, dataset_path=self.current_dataset, target_column=target, user_id=self.current_user_id)
        if not result.get("success"):
            return AgentResponse(message=f"Couldn't build a working model: {result.get('error', 'see the attempt log')}", success=False, data={"log": result.get("log", [])})
        model_id = ModelBuilder().register(result["saved_model_path"], result["metrics"], author_user_id=self.current_user_id) if result.get("saved_model_path") else None
        metrics_str = ", ".join(f"{k}: {v}" for k, v in result.get("metrics", {}).items() if isinstance(v, (int, float)))
        msg_text = f"Model trained and tested ✅ ({metrics_str})."
        if model_id:
            msg_text += f" Registered in the registry (id: {model_id}) - you can publish it to the Agent Store/Marketplace."
        return AgentResponse(message=msg_text, data={"model_id": model_id, "metrics": result.get("metrics", {}), "iterations": result.get("iterations", 0)})

    def _do_create_bot(self, msg: str, args: Dict = None) -> AgentResponse:
        from app.core.bot_builder import BotBuilder
        # Use the raw user message, not args.get('description') - the router call that
        # extracted args saw memory context (e.g. "user's name is X") appended to the
        # message, and could echo that into the description, leaking into the bot's name.
        description = msg
        result = BotBuilder(user_id=self.current_user_id).build(description)
        if not result.get("spec"):
            return AgentResponse(message=result.get("error", "Couldn't generate the bot."), success=False)
        spec = result["spec"]
        status = "all tests passed ✅" if result.get("success") else f"some tests failed after {result.get('iterations', 0)} attempts"
        skills = result.get("skills") or []
        skills_note = ""
        if skills:
            passed = [s for s in skills if s.get("test_passed")]
            failed = [s for s in skills if not s.get("test_passed")]
            parts = []
            if passed:
                parts.append(f"{len(passed)} real skill(s) built and tested ({', '.join(s['name'] for s in passed)})")
            if failed:
                parts.append(f"{len(failed)} skill(s) didn't pass testing and won't be included ({', '.join(s['name'] for s in failed)})")
            skills_note = " " + " - ".join(parts) + "."
        return AgentResponse(
            message=f"Bot \"{spec.get('name', 'Bot')}\" is ready - {status}.{skills_note} Check the preview on the right - you can chat with the draft right away or ask for changes.",
            data={"spec": spec, "test_results": result.get("test_results", []), "success": result.get("success", False),
                  "iterations": result.get("iterations", 0), "skills": skills})

    def _do_generate_code(self, msg: str, args: Dict = None) -> AgentResponse:
        d = (args or {}).get('description', msg) if args else msg
        target = (args or {}).get('target_column')
        if self.groq_key:
            from app.core.codegen.ai_generator import AICodeGenerator
            result = AICodeGenerator(self.db, user_id=self.current_user_id).build(d, dataset_path=self.current_dataset, target_column=target)
            if result.get("code"):
                status = "✅ tested, actually runs and works" if result["success"] else f"⚠️ generated, but failed verification after {result.get('iterations', 0)} attempts - check the code, it may need manual fixes"
                msg_text = f"ML pipeline {status}"
                if result.get("saved_path"):
                    msg_text += f"\nSaved to: {result['saved_path']}"
                msg_text += f"\n```python\n{result['code'][:800]}\n```"
                return AgentResponse(message=msg_text, data={"code": result["code"], "success": result["success"], "log": result["log"]})
            # Groq call itself failed (key set but request errored) - fall through to template
        try:
            from app.core.codegen.generator import CodeGenerator
            code = CodeGenerator(self.db).generate_from_description(d)
            path = CodeGenerator(self.db).save_code(code)
            return AgentResponse(
                message=f"ML pipeline saved (template, not verified by running it): {path}\n```python\n{code[:800]}\n```\nRun it with: python {path}",
                data={"code_path": path})
        except Exception as e:
            return AgentResponse(message=f"Error: {e}", success=False)

    def _do_install(self, n: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        f = DatasetFinder();
        num = int(n) if n.isdigit() else 1
        if not self.last_search_results: self.last_search_results = f.search("", max_results=5)
        if num < 1 or num > len(self.last_search_results): return AgentResponse(
            message=f"Pick a number from 1 to {len(self.last_search_results)}", success=False)
        c = self.last_search_results[num - 1]
        return AgentResponse(message=f"Dataset: **{c.title}**\n{c.url}\n\nDemo: datasets/churn_demo.csv")

    def get_chat_history(self, cid: str) -> List[Dict]:
        return self.db.get_messages(cid)

    def list_conversations(self) -> List[Dict]:
        return self.db.list_chats()