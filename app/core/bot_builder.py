"""
Bot Builder — no-code chat-bot creation loop.

Flow: user describes what they want in plain language (or from a Pipeline Builder
structure) -> AI designs the bot (system prompt + greeting + its own test questions
+ any real skills it needs, see skills_needed) -> any requested skills are generated
and sandbox-tested (app/core/skills/skill_generator.SkillGenerator) -> AI tests the
draft against its own questions, WITH real access to whichever skills passed their
test -> if something breaks (empty/failed response), AI fixes the spec and retests,
up to MAX_FIX_ITERATIONS times -> the working draft is shown to the user in a live
preview chat (also with real skill access) -> user can give free-text feedback to
revise it -> Publish saves it into the existing AgentStore/marketplace, persisting
any skill that passed testing.

Scope note: the automated fix loop catches HARD failures (API errors, empty
responses, malformed output) and the specific, checkable hallucination failure mode
(the designated trap question getting a confidently invented answer instead of an
honest deflection - verified by a separate judge call, see _check_hallucination).
It does NOT attempt to judge general subjective response quality beyond that -
"is this a good answer" in the open-ended sense is not something an LLM
self-critiques reliably, so that tuning is intentionally left to the human-in-the-
loop revise() step instead of pretending the loop can catch everything on its own.

Skills are generated and tested BEFORE test_draft() runs (see build()'s ordering) -
this matters. Earlier, skills were generated AFTER testing, which meant a test
question about something a skill covers could only "pass" by the model coincidentally
already knowing the answer (e.g. a famous public repo) or by the fix loop nudging the
system prompt toward sounding more confident - neither is a real capability check, and
the second is actively counterproductive (more convincing hallucination, not less).
Generating skills first means test_draft and chat_with_draft can give the model real
tool-calling access to already-verified code, so "the test passed" and "the live
preview actually works" mean the same thing.
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
  "trap_question_index": 0,
  "skills_needed": []
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
this bot has no real way of knowing UNLESS it's covered by a skill you requested
below (e.g. "is size X of model Y in stock right now", "where is my order #1234",
"what's my account balance") - this is how we catch invented answers before the user
ever sees them. trap_question_index (0, 1, or 2)
MUST point to that exact question's position in the test_questions array - we run an
automated check on that specific answer to verify the bot actually deflects instead
of inventing a confident-sounding fake answer, so it's critical this index is correct.

skills_needed: this is what separates a bot that just TALKS about a topic from a bot
that can actually DO something real. If the user's description implies the bot needs
genuine external capability - checking a live website, calling a public API, looking
something up that changes over time - list each such capability as an object:
    {"name": "snake_case_name", "description": "plain-language spec of exactly what
     this capability should do and what inputs/outputs it needs - this is the actual
     instruction an AI will use to write real, tested code, so be specific",
     "allowed_domains": ["example.com"], "action_type": "read_only"}
allowed_domains: the ONLY domain(s) this capability will ever be allowed to contact -
list the real domain(s) implied by the user's request (e.g. "github.com" for a GitHub
bot). If the category naturally has several real sources rather than one (e.g. "find
tenders" implies multiple procurement portals, "check prices" might mean several
retailers), list SEVERAL real, specific domains rather than inventing one generic
site or picking just one arbitrarily - a skill can only ever reach exactly the
domains listed here, so under-listing silently limits what it can actually find, not
just what it's allowed to do. action_type is "read_only" for anything that just checks/fetches information,
or "outbound_action" for anything that would send/post/submit something to a third
party (sending a message, posting a comment, submitting a form) - default to
"read_only" unless the description clearly asks the bot to send or contact something.
If the bot is just a knowledgeable conversational assistant with no need for live
external data (advice, FAQ, brainstorming, general how-to help), skills_needed MUST
be an empty list - do not invent capabilities nobody asked for.

The system_prompt must be detailed enough that the bot stays on-topic and doesn't
claim to be ChatGPT or any other assistant - it should know what it is and what it's
for. CRITICAL - the system_prompt must explicitly forbid inventing real-world facts
the bot has no actual access to: live inventory/stock levels, exact prices for
specific items, order status, account details, availability of a specific
product/size/model, etc., EXCEPT for whatever is genuinely covered by a skill listed
in skills_needed (the bot will be told at runtime, separately, exactly which tested
skills it has available and how to use them - you don't need to describe them again
here, just don't tell the bot it categorically CAN'T do real-time lookups if you're
also requesting a skill that gives it exactly that ability). For ANY question needing
real-time or account-specific data that ISN'T covered by a requested skill, the
system_prompt must instruct it to say it can't check that itself and to redirect to
a human/support channel (email, live chat, phone - whatever the user's description
implies) - never invent a plausible-sounding answer. General policy facts the user
DID provide (delivery time, return window, prices they mentioned) are fine to state
directly."""

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

    @staticmethod
    def _is_tool_use_failed_error(e: Exception) -> bool:
        """Some reasoning models (gpt-oss, qwen3) occasionally try to invoke a tool that was
        never declared, even with no `tools` param sent at all - Groq rejects this server-side
        with a 400 whose body has "code": "tool_use_failed". Checked via the structured error
        body (e.body), not a string match on str(e) - str() on a groq SDK exception only
        returns whatever message argument the SDK happened to construct, which isn't a stable
        contract to match against; body is the actual structured data the API returned."""
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            error = body.get("error") if isinstance(body.get("error"), dict) else {}
            code = error.get("code") or body.get("code")
            if code == "tool_use_failed":
                return True
        return False

    @staticmethod
    def _skills_to_tools_schema(skills: List[Dict]) -> List[Dict]:
        """Converts tested skills into Groq's function-calling tool format. Parameters are
        deliberately a permissive open object rather than a strict per-field schema - skills
        don't currently record a real JSON Schema for their inputs (see SkillGenerator's
        TEST_PARAMS, which is example data, not a schema), so the model is left to infer
        reasonable argument names from the skill's own description, the same way it already
        has to when deciding to call the skill at all."""
        tools = []
        for s in skills:
            tools.append({
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", "") or f"Runs the {s['name']} capability.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
                },
            })
        return tools

    def _groq_json(self, system: str, user: str, model: Optional[str] = None, max_tokens: int = 1500,
                    call_site: str = "bot_builder_json") -> Optional[Dict]:
        """Call Groq expecting a JSON object back (bot design / fix / revise steps)."""
        if not self.groq_key:
            return None
        model = model or self.DEFAULT_MODEL
        try:
            from groq import Groq
            from app.config import settings
            client = Groq(api_key=self.groq_key, timeout=settings.groq_request_timeout_seconds,
                          max_retries=settings.groq_max_retries)
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
            # Some reasoning models (gpt-oss, qwen3) occasionally try to invoke a tool that
            # was never declared, even with no `tools` param sent at all - Groq validates
            # this server-side and rejects it with a 400 tool_use_failed. There's no real
            # fix on our side (it's the model's own generation misbehaving, not a bug in our
            # request), so this is logged distinctly for visibility but otherwise handled the
            # same as any other generation failure: return None, let the caller's normal
            # "generation failed" handling take over.
            if self._is_tool_use_failed_error(e):
                print(f"[BotBuilder] model '{model}' tried to call an undeclared tool instead "
                      f"of returning JSON - treating as a failed generation: {e}")
            else:
                print(f"[BotBuilder] json call error: {e}")
            self._log_usage(call_site, model, success=False, error=str(e))
            return None

    def _groq_chat(self, system: str, user: str, history: Optional[List[Dict]] = None,
                    model: Optional[str] = None, max_tokens: int = 600,
                    call_site: str = "bot_builder_chat", skills: Optional[List[Dict]] = None) -> Optional[str]:
        """Plain chat completion - used to actually talk to the draft bot (testing + live
        preview). When `skills` is given (already-generated, already sandbox-tested code -
        see build()), the model gets REAL tool-calling access to them: if it calls one, we
        execute the actual skill code (through the same SafeFetcher-guarded
        SkillGenerator.run_stored_skill used for a published bot's real skill calls - no
        separate, weaker path for the draft/preview case) and feed the real result back for
        a final natural-language answer. This is what makes "the test passed" and "the live
        preview works" mean the same thing - see this module's docstring."""
        if not self.groq_key:
            return None
        model = model or self.DEFAULT_MODEL
        tested_skills = {s["name"]: s for s in (skills or []) if s.get("test_passed") and s.get("code")}
        tools = self._skills_to_tools_schema(list(tested_skills.values())) if tested_skills else None
        try:
            from groq import Groq
            from app.config import settings
            client = Groq(api_key=self.groq_key, timeout=settings.groq_request_timeout_seconds,
                          max_retries=settings.groq_max_retries)
            messages = [{"role": "system", "content": system}]
            for h in (history or [])[-10:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": user})

            kwargs = {"model": model, "messages": messages, "temperature": 0.5,
                      "max_tokens": max_tokens, **self._groq_kwargs_for(model)}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            completion = client.chat.completions.create(**kwargs)
            self._log_usage(call_site, model, completion, success=True)
            msg = completion.choices[0].message

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content

            # The model asked to use one or more skills - run each for real, then ask for a
            # final answer with the real result(s) in hand. Only one round: the follow-up
            # call omits `tools` entirely, so the model literally cannot try to call
            # anything else there (closing off the same tool_use_failed loop handled below,
            # and guaranteeing this terminates rather than chaining indefinitely).
            from app.core.skills.skill_generator import SkillGenerator
            runner = SkillGenerator()
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in tool_calls],
            })
            for tc in tool_calls:
                skill = tested_skills.get(tc.function.name)
                if not skill:
                    tool_result = {"error": f"Unknown skill: {tc.function.name}"}
                else:
                    try:
                        params = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        params = {}
                    outcome = runner.run_stored_skill(skill["code"], params, skill.get("allowed_domains", []))
                    tool_result = outcome["result"] if outcome["success"] else {"error": outcome["error"]}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)})

            follow_up = client.chat.completions.create(
                model=model, messages=messages, temperature=0.5, max_tokens=max_tokens,
                **self._groq_kwargs_for(model))
            self._log_usage(f"{call_site}_tool_followup", model, follow_up, success=True)
            return follow_up.choices[0].message.content
        except Exception as e:
            if self._is_tool_use_failed_error(e):
                # The model tried to invoke a tool it invented (e.g. a fictional
                # "repo_browser" lookup) instead of just answering in text - this reliably
                # happens on some reasoning models when a question implies checking
                # something in real time (exactly our trap questions and anything
                # GitHub/repo-shaped). We never declared any tools in this request, so this
                # is the model's own confusion, not something a retry with the same prompt
                # tends to fix (confirmed in production logs: it fails the same way across
                # several different retries/questions). The honest, correct answer to "the
                # model wanted to look something up in real time and couldn't" IS a
                # deflection - which is exactly what we want it to say here anyway, so we
                # return that directly instead of erroring out and blocking bot creation.
                print(f"[BotBuilder] model '{model}' tried to call an undeclared tool "
                      f"('{self._extract_attempted_tool_name(e)}') instead of answering in "
                      f"text - returning an honest deflection instead: {e}")
                self._log_usage(call_site, model, success=False, error=str(e))
                return ("I don't have a way to check that in real time, sorry - I can only "
                        "work with what I already know.")
            print(f"[BotBuilder] chat call error: {e}")
            self._log_usage(call_site, model, success=False, error=str(e))
            return None

    @staticmethod
    def _extract_attempted_tool_name(error: Exception) -> str:
        """Best-effort extraction of the fictional tool name from Groq's error body, purely
        for clearer logging - never affects behavior if this can't parse (falls back to a
        generic label)."""
        try:
            import re
            match = re.search(r'"name":\s*"([^"]+)"', str(error))
            return match.group(1) if match else "unknown"
        except Exception:
            return "unknown"

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

    def test_draft(self, spec: Dict, model: Optional[str] = None, skills: Optional[List[Dict]] = None) -> List[Dict]:
        """Run the greeting + each AI-proposed test question through the draft bot. Catches
        hard failures (empty/errored responses) for every question, AND - for the specific
        question marked as the anti-hallucination trap (trap_question_index) - runs a second
        judge call to catch the failure mode this trap question exists to catch: a confident
        invented answer that isn't empty, so the old empty-only check would have let it pass.

        `skills` (already-generated, already sandbox-tested - see build()'s ordering) gives
        the model real tool-calling access during this test, same as chat_with_draft's live
        preview - so a test question covering something a skill does gets a REAL check, not
        a lucky guess from the model's training data (see this module's docstring)."""
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
            resp = self._groq_chat(sys_prompt, q, model=model, call_site="bot_builder_test_question", skills=skills)
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
        """Full generate -> build skills -> test -> fix loop. Returns the final spec, the
        test transcript, how many fix iterations it took, and whether every test ultimately
        passed.

        If the architect decided the bot needs real capabilities (spec["skills_needed"]),
        each one is generated and sandbox-tested FIRST, here (see
        app/core/skills/skill_generator.SkillGenerator - same generate -> sandbox-test ->
        self-correct loop, just for a single function instead of a whole bot spec) - BEFORE
        test_draft() runs, so the test loop can give the model real tool access to whatever
        passed, and a test question about something a skill covers gets a genuine check
        instead of a lucky guess from the model's own training data (see this module's
        docstring for why the ordering matters). This does NOT yet insert anything into the
        skills database or link it to a real bot - there is no real agent_id until the user
        actually clicks Publish (see publish() below), which is what persists any skill that
        passed its test here, tied to the newly-created bot's real id."""
        max_iterations = self.MAX_FIX_ITERATIONS if max_iterations is None else max_iterations
        spec = self.generate_spec(description, model)
        if not spec or not spec.get("system_prompt"):
            return {"spec": None, "test_results": [], "iterations": 0, "success": False,
                     "error": "Groq generation failed - check GROQ_API_KEY and try again"}

        skills_needed = spec.get("skills_needed") or []
        skill_results = []
        if skills_needed and self.groq_key:
            from app.core.skills.skill_generator import SkillGenerator, _detect_write_http_method
            for requested in skills_needed[:5]:  # hard cap - one bot creation shouldn't spawn unbounded Groq calls
                domains = requested.get("allowed_domains") or []
                if not domains:
                    skill_results.append({**requested, "test_passed": False,
                                          "error": "No allowed_domains given - skipped"})
                    continue
                result = SkillGenerator(user_id=self.user_id).build(requested.get("description", ""), domains)
                action_type = requested.get("action_type", "read_only")
                if result["code"] and _detect_write_http_method(result["code"]):
                    action_type = "outbound_action"  # same defense-in-depth override as /api/skills/create
                skill_results.append({
                    "name": requested.get("name", "skill"), "description": requested.get("description", ""),
                    "allowed_domains": domains, "action_type": action_type,
                    "code": result["code"], "test_passed": result["success"], "log": result["log"],
                })
        spec["skills_needed"] = skill_results  # replace the request list with the actual outcome

        iterations = 0
        test_results = self.test_draft(spec, model, skills=skill_results)
        while any(not r["ok"] for r in test_results) and iterations < max_iterations:
            spec = self.fix_spec(spec, test_results, model)
            # fix_spec can only ever edit the conversational fields (prompt/greeting/
            # questions) via the FIX_SYSTEM_PROMPT - it never regenerates skills_needed, so
            # skill_results (already tested above) stays valid and is reused as-is here.
            spec["skills_needed"] = skill_results
            test_results = self.test_draft(spec, model, skills=skill_results)
            iterations += 1
        success = all(r["ok"] for r in test_results)

        return {"spec": spec, "test_results": test_results, "iterations": iterations, "success": success,
                "skills": skill_results}

    # ---- live preview + human revision ----

    def chat_with_draft(self, spec: Dict, message: str, history: Optional[List[Dict]] = None,
                         model: Optional[str] = None) -> str:
        skills = spec.get("skills_needed") or []
        resp = self._groq_chat(spec.get("system_prompt", ""), message, history, model,
                                call_site="bot_builder_live_chat", skills=skills)
        return resp or "Couldn't get a response from the draft bot (check GROQ_API_KEY)."

    def revise(self, spec: Dict, feedback: str, model: Optional[str] = None) -> Dict:
        payload = json.dumps({"current_spec": spec, "user_feedback": feedback}, ensure_ascii=False)
        revised = self._groq_json(self.REVISE_SYSTEM_PROMPT, payload, model, call_site="bot_builder_revise")
        # Merge rather than replace: if the model's response omits a field (e.g. forgets to
        # repeat "name" or "greeting"), keep the original value instead of losing it.
        return {**spec, **revised} if revised else spec

    # ---- publish ----

    def publish(self, spec: Dict, model: Optional[str] = None, author: str = "", price: float = 0.0) -> Dict:
        """Save the finished bot into the existing AgentStore/marketplace - no new
        deployment mechanism needed, it reuses agents/store.json + /api/agent/message.

        Also persists any skill that was generated and passed its sandbox test during
        build() (spec["skills_needed"], now holding outcomes rather than requests - see
        build()'s docstring). Skills that failed their test are NOT saved - a broken skill
        sitting inactive in the database would be dead weight, and GenesisAgent only ever
        offers a bot skills that are both active AND test_passed (see
        GenesisDB.list_skills_for_agent), so an unsaved failed skill and a saved-but-inactive
        one would behave identically anyway; not saving it is simpler and leaves no trace to
        clean up later. The code itself (already generated and tested) is reused as-is here -
        this does NOT call Groq again, so publishing costs nothing extra beyond what build()
        already spent."""
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

        saved_skills = []
        for skill in (spec.get("skills_needed") or []):
            if not skill.get("test_passed") or not skill.get("code"):
                continue
            from app.db.database import GenesisDB
            db = self._db or GenesisDB()
            record = db.create_skill(
                agent_id=config.id, owner_user_id=author,
                name=skill.get("name", "skill"), description=skill.get("description", ""),
                code=skill["code"], allowed_domains=skill.get("allowed_domains", []),
                input_schema={}, action_type=skill.get("action_type", "read_only"))
            db.update_skill_test_result(record["id"], passed=True, log=json.dumps(skill.get("log", []))[:4000])
            saved_skills.append(record["id"])

        result = config.to_dict()
        result["saved_skill_ids"] = saved_skills
        return result
