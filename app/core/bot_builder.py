"""
Bot Builder — no-code chat-bot creation loop.

Flow: user describes what they want in plain language (or from a Pipeline Builder
structure) -> AI designs the bot (system prompt + greeting + its own test questions)
-> AI tests the draft against those questions -> if something breaks (empty/failed
response), AI fixes the spec and retests, up to MAX_FIX_ITERATIONS times -> the
working draft is shown to the user in a live preview chat -> user can give free-text
feedback to revise it -> Publish saves it into the existing AgentStore/marketplace.

Scope note: the automated fix loop catches HARD failures (API errors, empty
responses, malformed output) and the specific, checkable hallucination failure mode
(the designated trap question getting a confidently invented answer instead of an
honest deflection - verified by a separate judge call, see _check_hallucination).
It does NOT attempt to judge general subjective response quality beyond that -
"is this a good answer" in the open-ended sense is not something an LLM
self-critiques reliably, so that tuning is intentionally left to the human-in-the-
loop revise() step instead of pretending the loop can catch everything on its own.

No sandboxed code execution here - a chat-bot's "code" is just a system prompt and
tool list, tested via ordinary chat completions. This is deliberately the first,
lower-risk half of the no-code builder; the ML-model half (which runs real generated
Python and needs a real sandbox) is a separate, later piece of work.
"""

import os
import json
from typing import Dict, List, Optional


class BotBuilder:
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    MAX_FIX_ITERATIONS = 3

    ARCHITECT_SYSTEM_PROMPT = """You are a bot architect for a no-code AI platform. A non-technical
user describes, in plain language, a bot they want. Design it.

Respond ONLY as a JSON object with this exact shape:
{
  "name": "short bot name",
  "description": "one sentence, what this bot does",
  "system_prompt": "the full system prompt that will drive this bot's behavior",
  "greeting": "the first message the bot shows the user",
  "test_questions": ["question 1", "question 2", "question 3"],
  "trap_question_index": 0
}

name: invent a name that fits the bot's PURPOSE and domain (e.g. a shoe store support
bot could be "ShoeStep Support", "Sizely", "Подошва" - something evocative of what it
does). Do NOT reuse a human's personal name just because one happens to appear
somewhere in the description or surrounding context (e.g. the platform user's own
name) - only use a human-sounding name if the user's description explicitly asked
for that specific name for the bot itself.

test_questions: propose 3 realistic questions a REAL user of THIS SPECIFIC bot would
ask it (not generic questions) - these will be used to test the bot actually works
before showing it to the user. Exactly ONE of these three MUST ask about something
this bot has no real way of knowing (e.g. "is size X of model Y in stock right now",
"where is my order #1234", "what's my account balance") - this is how we catch
invented answers before the user ever sees them. trap_question_index (0, 1, or 2)
MUST point to that exact question's position in the test_questions array - we run an
automated check on that specific answer to verify the bot actually deflects instead
of inventing a confident-sounding fake answer, so it's critical this index is correct.

The system_prompt must be detailed enough that the bot stays on-topic and doesn't
claim to be ChatGPT or any other assistant - it should know what it is and what it's
for. CRITICAL - the system_prompt must explicitly forbid inventing real-world facts
the bot has no actual access to: live inventory/stock levels, exact prices for
specific items, order status, account details, availability of a specific
product/size/model, etc. For ANY question needing real-time or account-specific
data the bot wasn't explicitly given, the system_prompt must instruct it to say it
can't check that itself and to redirect to a human/support channel (email, live
chat, phone - whatever the user's description implies) - never invent a plausible-
sounding answer. General policy facts the user DID provide (delivery time, return
window, prices they mentioned) are fine to state directly."""

    HALLUCINATION_JUDGE_PROMPT = """You are a strict fact-checking judge. A chat bot was asked a
question (the_question) that it has NO real way of answering correctly - it has no access to
live inventory, order systems, account databases, or real-time data. Its actual reply is
the_response.

Judge ONLY one thing: did the bot INVENT a specific, confident-sounding fact it could not
possibly know (a stock count, an order status, a specific date/amount, "yes it's available",
etc.), instead of honestly saying it can't check that and redirecting to a human/support
channel?

Respond ONLY as JSON: {"hallucinated": true or false, "reason": "one short sentence"}

A response that redirects to support, says it doesn't have access to that information, or
asks the user to check elsewhere is NOT a hallucination even if it's phrased confidently or
apologetically. A response that states a specific fact it couldn't actually know (even
hedged with "I think" or "probably") IS a hallucination."""



    FIX_SYSTEM_PROMPT = """You previously designed a bot (current_spec). When tested, some of its
own test questions produced broken results - see failures. A failure can be a hard error
(empty response, refusal, malformed output) OR a hallucination: the bot invented a specific,
confident-sounding fact it had no way of actually knowing (see each failure's "error" field
for which kind it is). For a hallucination failure, tighten system_prompt's instructions
about that specific scenario so the bot honestly deflects to a human/support channel instead
of inventing an answer - don't just tweak wording, make the deflection instruction concrete
and unambiguous for that scenario. Revise system_prompt (and test_questions/trap_question_index
if a question itself was unrealistic) to fix these failures. Respond with the SAME JSON schema
as before, corrected. Keep everything that already worked unchanged."""

    REVISE_SYSTEM_PROMPT = """You previously designed a bot (current_spec). A human tested the
draft and wants a change (user_feedback) - this is a real person's direct instruction,
apply it precisely. Respond with the SAME JSON schema as before, with the change applied.
Keep everything else unchanged unless the feedback implies otherwise."""

    def __init__(self, user_id: Optional[str] = None):
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.user_id = user_id
        self._db = None

    def _log_usage(self, call_site: str, model: str, completion=None, success: bool = True, error: str = None):
        """Best-effort internal Groq usage logging - see GenesisDB.log_groq_usage docstring
        for why this exists. Never lets a logging failure break the actual bot-building call."""
        try:
            if self._db is None:
                from app.db.database import GenesisDB
                self._db = GenesisDB()
            usage = getattr(completion, "usage", None) if completion else None
            self._db.log_groq_usage(
                self.user_id, call_site, model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                success=success, error=error)
        except Exception:
            pass  # logging must never break the actual call

    def _groq_kwargs_for(self, model: str) -> Dict:
        if model.startswith("openai/gpt-oss") or model.startswith("qwen/qwen3"):
            return {"reasoning_effort": "low"}
        return {}

    def _groq_json(self, system: str, user: str, model: Optional[str] = None, max_tokens: int = 1500,
                    call_site: str = "bot_builder_json") -> Optional[Dict]:
        """Call Groq expecting a JSON object back (bot design / fix / revise steps)."""
        if not self.groq_key:
            return None
        model = model or self.DEFAULT_MODEL
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                **self._groq_kwargs_for(model),
            )
            self._log_usage(call_site, model, completion, success=True)
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"[BotBuilder] json call error: {e}")
            self._log_usage(call_site, model, success=False, error=str(e))
            return None

    def _groq_chat(self, system: str, user: str, history: Optional[List[Dict]] = None,
                    model: Optional[str] = None, max_tokens: int = 600,
                    call_site: str = "bot_builder_chat") -> Optional[str]:
        """Plain chat completion - used to actually talk to the draft bot (testing + live preview)."""
        if not self.groq_key:
            return None
        model = model or self.DEFAULT_MODEL
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            messages = [{"role": "system", "content": system}]
            for h in (history or [])[-10:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": user})
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=0.5, max_tokens=max_tokens,
                **self._groq_kwargs_for(model),
            )
            self._log_usage(call_site, model, completion, success=True)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[BotBuilder] chat call error: {e}")
            self._log_usage(call_site, model, success=False, error=str(e))
            return None

    # ---- build loop ----

    def generate_spec(self, description: str, model: Optional[str] = None) -> Optional[Dict]:
        return self._groq_json(self.ARCHITECT_SYSTEM_PROMPT, description, model, call_site="bot_builder_generate_spec")

    def _check_hallucination(self, question: str, response: str, model: Optional[str] = None) -> Dict:
        """Judge whether the trap question's answer invented a fact it couldn't know, using a
        separate Groq call rather than self-critique from the same persona that answered (a
        model grading its own answer in-character is far less reliable than a fresh judge call
        with no persona to defend). Fails OPEN (treated as not-hallucinated, but flagged
        unverified) on any infra failure - an API hiccup shouldn't block a legitimate bot from
        publishing, but we don't want to silently pretend we verified something we didn't."""
        payload = json.dumps({"the_question": question, "the_response": response}, ensure_ascii=False)
        verdict = self._groq_json(self.HALLUCINATION_JUDGE_PROMPT, payload, model, max_tokens=200, call_site="bot_builder_hallucination_judge")
        if verdict is None:
            return {"hallucinated": False, "reason": "unverified - hallucination judge call failed", "verified": False}
        return {"hallucinated": bool(verdict.get("hallucinated")),
                "reason": verdict.get("reason", ""), "verified": True}

    def test_draft(self, spec: Dict, model: Optional[str] = None) -> List[Dict]:
        """Run the greeting + each AI-proposed test question through the draft bot. Catches
        hard failures (empty/errored responses) for every question, AND - for the specific
        question marked as the anti-hallucination trap (trap_question_index) - runs a second
        judge call to catch the failure mode this trap question exists to catch: a confident
        invented answer that isn't empty, so the old empty-only check would have let it pass."""
        results = [{
            "question": "(greeting)",
            "response": spec.get("greeting", ""),
            "ok": bool((spec.get("greeting") or "").strip()),
            "error": None if (spec.get("greeting") or "").strip() else "greeting is empty",
        }]
        sys_prompt = spec.get("system_prompt", "")
        questions = (spec.get("test_questions") or [])[:5]
        trap_idx = spec.get("trap_question_index")
        trap_idx = trap_idx if isinstance(trap_idx, int) and 0 <= trap_idx < len(questions) else None
        for i, q in enumerate(questions):
            resp = self._groq_chat(sys_prompt, q, model=model, call_site="bot_builder_test_question")
            ok = bool(resp and resp.strip())
            error = None if ok else "empty response or API call failed"
            if ok and i == trap_idx:
                verdict = self._check_hallucination(q, resp, model)
                if verdict["hallucinated"]:
                    ok = False
                    error = f"hallucinated a fact: {verdict['reason']}"
            results.append({"question": q, "response": resp or "", "ok": ok, "error": error})
        return results

    def fix_spec(self, spec: Dict, test_results: List[Dict], model: Optional[str] = None) -> Dict:
        failures = [r for r in test_results if not r["ok"]]
        if not failures:
            return spec
        payload = json.dumps({"current_spec": spec, "failures": failures}, ensure_ascii=False)
        fixed = self._groq_json(self.FIX_SYSTEM_PROMPT, payload, model, call_site="bot_builder_fix_spec")
        return {**spec, **fixed} if fixed else spec

    def build(self, description: str, model: Optional[str] = None, max_iterations: Optional[int] = None) -> Dict:
        """Full generate -> test -> fix loop. Returns the final spec, the test transcript,
        how many fix iterations it took, and whether every test ultimately passed."""
        max_iterations = self.MAX_FIX_ITERATIONS if max_iterations is None else max_iterations
        spec = self.generate_spec(description, model)
        if not spec or not spec.get("system_prompt"):
            return {"spec": None, "test_results": [], "iterations": 0, "success": False,
                     "error": "Groq generation failed - check GROQ_API_KEY and try again"}
        iterations = 0
        test_results = self.test_draft(spec, model)
        while any(not r["ok"] for r in test_results) and iterations < max_iterations:
            spec = self.fix_spec(spec, test_results, model)
            test_results = self.test_draft(spec, model)
            iterations += 1
        success = all(r["ok"] for r in test_results)
        return {"spec": spec, "test_results": test_results, "iterations": iterations, "success": success}

    # ---- live preview + human revision ----

    def chat_with_draft(self, spec: Dict, message: str, history: Optional[List[Dict]] = None,
                         model: Optional[str] = None) -> str:
        resp = self._groq_chat(spec.get("system_prompt", ""), message, history, model, call_site="bot_builder_live_chat")
        return resp or "Не удалось получить ответ от черновика бота (проверь GROQ_API_KEY)."

    def revise(self, spec: Dict, feedback: str, model: Optional[str] = None) -> Dict:
        payload = json.dumps({"current_spec": spec, "user_feedback": feedback}, ensure_ascii=False)
        revised = self._groq_json(self.REVISE_SYSTEM_PROMPT, payload, model, call_site="bot_builder_revise")
        # Merge rather than replace: if the model's response omits a field (e.g. forgets to
        # repeat "name" or "greeting"), keep the original value instead of losing it.
        return {**spec, **revised} if revised else spec

    # ---- publish ----

    def publish(self, spec: Dict, model: Optional[str] = None, author: str = "", price: float = 0.0) -> Dict:
        """Save the finished bot into the existing AgentStore/marketplace - no new
        deployment mechanism needed, it reuses agents/store.json + /api/agent/message."""
        from app.core.agent_store import AgentStore, AgentConfig
        store = AgentStore()
        config = AgentConfig(
            name=spec.get("name", "My Bot"),
            system_prompt=spec.get("system_prompt", ""),
            model=model or spec.get("model") or self.DEFAULT_MODEL,
        )
        config.description = spec.get("description", "")
        config.greeting = spec.get("greeting") or config.greeting
        config.author = author
        store.create_agent(config)
        store.publish_agent(config.id, price=price)
        return config.to_dict()
