"""
SkillGenerator - generates a single Python function implementing a bot "skill" (e.g. "check
this site for cheap iPhone listings"), following the exact same generate -> execute ->
validate -> fix loop as AICodeGenerator (app/core/codegen/ai_generator.py), but for a
different contract: instead of a full ML pipeline script, the model writes ONE function,
`run(params)`, and the ONLY way that function can reach the network is a `fetch()` helper
injected into its execution namespace - see app/core/skills/net_guard.py for what that
enforces and why.

Contract the generated code must follow:
    def run(params: dict) -> dict:
        ...
        result = fetch("https://example.com/api", method="GET")
        ...
        return {"some": "json-serializable result"}

`fetch` is a global already available when the function runs - the model must NOT import
`requests`, `urllib`, `socket`, `http.client`, or anything else network-capable; it must NOT
import `os`, `sys`, `subprocess`, or use `eval`/`exec`/`__import__`. This is enforced two ways:
  1. A static AST-based check (_static_safety_check) run BEFORE the code is ever executed,
     as defense-in-depth and a fast rejection for the obvious cases.
  2. Real sandboxing at execution time - see docstring on execute() below. The static check
     alone is NOT the security boundary; a determined adversary can obfuscate a banned import
     past a source-level scan. The actual boundary is that the execution environment has no
     way to reach the network except through the injected, allowlist-and-IP-checked `fetch`.

SAFETY NOTE, same caveat as AICodeGenerator (read that module's docstring too): the
subprocess-based dev fallback used here when Docker isn't available provides NO real process
isolation - it just doesn't give the generated code a network-capable library to import
successfully (it isn't installed as `fetch`, and the static check blocks the obvious raw
alternatives). That is meaningfully weaker than genuine sandboxing and must never be the
execution path for real, multi-tenant traffic - see docker_sandbox.py's DockerSandboxNetworked
(network_mode="none" - the container has no network device at all, so nothing the code does,
however it tries, can reach anywhere except through the file-based bridge to SafeFetcher
running on the host).
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import shutil
from typing import Dict, Optional

from app.core.skills.net_guard import SafeFetcher, SkillNetworkError

BANNED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "http", "httplib",
    "ftplib", "smtplib", "telnetlib", "shutil", "pathlib", "ctypes", "multiprocessing",
    "threading", "asyncio", "importlib",
}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}


def _static_safety_check(code: str) -> Optional[str]:
    """Returns an error message if the code uses anything banned, else None. Uses the AST
    (not string matching) so it isn't fooled by e.g. banned names appearing in a string or
    comment, and correctly catches `import os.path` / `from os import system` variants."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Code has a syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORTS:
                    return f"Import of '{alias.name}' is not allowed - use the provided fetch() function for network access."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_IMPORTS:
                return f"Import from '{node.module}' is not allowed - use the provided fetch() function for network access."
        elif isinstance(node, ast.Call):
            fn_name = None
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr
            if fn_name in BANNED_CALLS:
                return f"Use of '{fn_name}(...)' is not allowed in a skill."

    if "def run(" not in code:
        return "Code must define a function named exactly 'run' - e.g. def run(params):"
    return None


def _detect_write_http_method(code: str) -> bool:
    """Defense in depth for action_type classification: even if a skill's creator declares
    it "read_only" (by mistake, or deliberately trying to dodge the approval gate), a skill
    whose own code passes a write HTTP method to fetch() is treated as an outbound_action
    regardless of what was declared - see main.py's create_skill route, which calls this and
    overrides the declared action_type if it returns True. Scans for the method string
    appearing anywhere as a literal (not just as a fetch() keyword argument specifically) -
    deliberately broad, since a determined attempt to hide this would be through obfuscation
    this simple scan can't fully defeat anyway (see the module docstring's note that the
    static check is defense-in-depth, not the security boundary - the real boundary for
    outbound_action skills is that they never run without going through pending_actions)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True  # can't prove it's safe - treat conservatively as needing approval
    write_methods = {"POST", "PUT", "DELETE", "PATCH"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.upper() in write_methods:
            return True
    return False


class SkillGenerator:
    MAX_ITERATIONS = 3
    EXEC_TIMEOUT = 30  # seconds - skills should be quick lookups, not long-running jobs, but a
    # real multi-request lookup (e.g. checking several tender sites) needs real headroom -
    # raised from 15s after a report of real skills timing out on legitimately slow sites.

    GENERATE_PROMPT = """You are writing a single Python function for a "skill" that a chatbot
will be able to call during a conversation. Given a plain-language description of what the
skill should do AND the exact list of domains it's allowed to contact, write ONE function
with this EXACT signature:

    def run(params: dict) -> dict:
        ...

Rules, all mandatory:
- You may use a function called `fetch(url, method="GET", headers=None, body=None)` which is
  already available as a global - it makes an HTTP(S) request and returns
  {"status": int, "text": str, "headers": dict}. This is the ONLY way to reach the network.
- You will be given "allowed_domains" - fetch() calls to any OTHER domain are rejected before
  they even leave the sandbox, so only build URLs on these domains. If more than one domain
  is given, use ALL of them where relevant (e.g. check several sources and combine or return
  whichever responds) rather than only ever using the first one - the point of being given
  several is to actually cover the topic, not to have unused options.
- Do NOT import requests, urllib, socket, http, os, sys, subprocess, or anything else -
  only fetch() and Python's standard data-handling modules (json, re, math, datetime,
  collections) are available. Any other import will cause the skill to be rejected.
- Do NOT use eval, exec, compile, __import__, or open().
- `run` must return a JSON-serializable dict (str/int/float/bool/list/dict/None values only).
- Wrap EACH fetch() call in its own try/except - if a skill checks multiple domains, one
  source failing (timeout, error status, unreachable) must not stop it from still trying the
  others and returning whatever it did find. Only return an {"error": "..."} dict if every
  source it tried failed.
- Also define, right after the function, a module-level dict called TEST_PARAMS containing
  realistic example arguments for a smoke test - e.g. TEST_PARAMS = {"query": "iphone 13",
  "max_price": 300}. This MUST match the params run() actually expects.

Respond with ONLY the raw Python code (the run function, and TEST_PARAMS), no markdown
fences, no explanation before or after."""

    FIX_PROMPT = """The skill function below failed when actually tested. Fix it.
Respond with ONLY the corrected, complete raw Python code (same rules as before: def run(params),
only the fetch() global and stdlib data modules, a matching TEST_PARAMS dict). Keep everything
that wasn't the cause of the error unchanged - fix only what the error points to."""

    def __init__(self, db=None, user_id: Optional[str] = None, model: Optional[str] = None):
        self.db = db
        self.user_id = user_id
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        self.MODEL = model or "openai/gpt-oss-120b"

    def _log_usage(self, success: bool, completion=None, error: str = None):
        try:
            if self.db is None:
                from app.db.database import GenesisDB
                self.db = GenesisDB()
            usage = getattr(completion, "usage", None) if completion else None
            self.db.log_groq_usage(
                self.user_id, "skill_codegen", self.MODEL,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                success=success, error=error)
        except Exception:
            pass

    def _call_groq(self, system: str, user: str) -> Optional[str]:
        if not self.groq_key:
            return None
        try:
            from app.core.groq_client import call_llm
            completion = call_llm(
                self.groq_key,
                model=self.MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2, max_tokens=1200,
            )
            self._log_usage(True, completion)
            code = completion.choices[0].message.content.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[1] if "\n" in code else code
                if code.endswith("```"):
                    code = code.rsplit("```", 1)[0]
            return code.strip()
        except Exception as e:
            self._log_usage(False, error=str(e))
            return None

    def generate(self, description: str, allowed_domains: Optional[list] = None) -> Optional[str]:
        return self._call_groq(self.GENERATE_PROMPT, json.dumps({
            "description": description, "allowed_domains": allowed_domains or []}))

    def fix(self, code: str, error: str) -> Optional[str]:
        return self._call_groq(self.FIX_PROMPT, json.dumps({"code": code, "error": error}))

    def execute(self, code: str, allowed_domains: list) -> Dict:
        """Runs the skill's TEST_PARAMS through run() once, using whichever sandbox backend
        is available - see _get_sandbox_backend's twin in ai_generator.py for the same
        docker-required-in-production policy, applied here identically."""
        from app.core.codegen.docker_sandbox import docker_available, ensure_image_built
        if docker_available() and ensure_image_built():
            from app.core.codegen.docker_sandbox import DockerSandboxNetworked
            return DockerSandboxNetworked().execute(code, allowed_domains)
        from app.config import settings
        if not settings.debug:
            raise RuntimeError(
                "Docker sandbox is unavailable and debug=False. Refusing to execute "
                "AI-generated, network-capable skill code without isolation in a non-dev "
                "environment. Install and start Docker before enabling skills in production."
            )
        print("[SkillGenerator] Docker not available - falling back to dev-only subprocess "
              "sandbox because debug=True. This is meaningfully less safe (see module "
              "docstring) - install Docker before serving skills to anyone but yourself.")
        return self._execute_subprocess(code, allowed_domains)

    def _execute_subprocess(self, code: str, allowed_domains: list) -> Dict:
        static_error = _static_safety_check(code)
        if static_error:
            return {"success": False, "stdout": "", "stderr": static_error, "returncode": None}

        workdir = tempfile.mkdtemp(prefix="genesis_skill_")
        try:
            harness = f"""
import json, sys
sys.path.insert(0, {json.dumps(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))})
from app.core.skills.net_guard import SafeFetcher
_fetcher = SafeFetcher(allowed_domains={allowed_domains!r})
fetch = _fetcher.fetch

{code}

try:
    result = run(TEST_PARAMS)
    json.dumps(result)  # confirm it's actually JSON-serializable, not just dict-shaped
    print("SKILL_TEST_RESULT:" + json.dumps({{"ok": True, "result": result}}))
except Exception as e:
    print("SKILL_TEST_RESULT:" + json.dumps({{"ok": False, "error": str(e)}}))
"""
            script_path = os.path.join(workdir, "skill.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(harness)
            proc = subprocess.run(
                [sys.executable, script_path], cwd=workdir,
                capture_output=True, text=True, timeout=self.EXEC_TIMEOUT)
            success = "SKILL_TEST_RESULT:" in proc.stdout and '"ok": true' in proc.stdout.lower()
            return {"success": success, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:], "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Execution exceeded {self.EXEC_TIMEOUT}s - likely stuck on a slow/unreachable request.", "returncode": None}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": None}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def run_stored_skill(self, code: str, params: dict, allowed_domains: list) -> Dict:
        """Executes an already-saved, already-tested skill with REAL parameters chosen by the
        LLM during a live conversation (as opposed to execute()/build(), which run the
        skill's own TEST_PARAMS during creation). Re-runs the static safety check even though
        the code passed it at creation time - cheap insurance against a future code path that
        writes to bot_skills without going through build() first, or a row edited directly in
        the database. Uses the same sandbox backend selection (Docker in production, subprocess
        dev-fallback with the same debug-mode-only restriction) as execute().

        Returns {"success": bool, "result": dict|None, "error": str|None} - the shape agent.py
        needs to turn into a chat reply, not the raw stdout/stderr shape execute() returns
        (which is about the codegen loop, not a single live call).
        """
        static_error = _static_safety_check(code)
        if static_error:
            return {"success": False, "result": None, "error": f"Skill failed a safety check and cannot run: {static_error}"}

        from app.core.codegen.docker_sandbox import docker_available, ensure_image_built
        if docker_available() and ensure_image_built():
            from app.core.codegen.docker_sandbox import DockerSandboxNetworked
            raw = DockerSandboxNetworked().execute_with_params(code, params, allowed_domains)
        else:
            from app.config import settings
            if not settings.debug:
                return {"success": False, "result": None,
                        "error": "Docker sandbox is unavailable - refusing to run network-capable skill code without isolation."}
            raw = self._execute_subprocess_with_params(code, params, allowed_domains)

        if not raw["success"]:
            return {"success": False, "result": None, "error": (raw["stderr"] or raw["stdout"] or "Skill execution failed")[:500]}
        try:
            payload = json.loads(raw["stdout"].split("SKILL_RUN_RESULT:", 1)[1].splitlines()[0])
        except Exception:
            return {"success": False, "result": None, "error": "Skill ran but its output couldn't be parsed"}
        if not payload.get("ok"):
            return {"success": False, "result": None, "error": payload.get("error", "Unknown error")}
        return {"success": True, "result": payload.get("result"), "error": None}

    def _execute_subprocess_with_params(self, code: str, params: dict, allowed_domains: list) -> Dict:
        """Same subprocess dev-fallback as _execute_subprocess, but calls run(params) with the
        real, caller-supplied params instead of the code's own TEST_PARAMS, and uses a
        different output marker (SKILL_RUN_RESULT vs SKILL_TEST_RESULT) so a caller can never
        confuse a live run's output with a self-test's."""
        workdir = tempfile.mkdtemp(prefix="genesis_skill_run_")
        try:
            harness = f"""
import json, sys
sys.path.insert(0, {json.dumps(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))})
from app.core.skills.net_guard import SafeFetcher
_fetcher = SafeFetcher(allowed_domains={allowed_domains!r})
fetch = _fetcher.fetch

{code}

_params = {params!r}
try:
    result = run(_params)
    json.dumps(result)
    print("SKILL_RUN_RESULT:" + json.dumps({{"ok": True, "result": result}}))
except Exception as e:
    print("SKILL_RUN_RESULT:" + json.dumps({{"ok": False, "error": str(e)}}))
"""
            script_path = os.path.join(workdir, "skill_run.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(harness)
            proc = subprocess.run(
                [sys.executable, script_path], cwd=workdir,
                capture_output=True, text=True, timeout=self.EXEC_TIMEOUT)
            success = "SKILL_RUN_RESULT:" in proc.stdout
            return {"success": success, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:], "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Execution exceeded {self.EXEC_TIMEOUT}s.", "returncode": None}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": None}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    PARSE_REQUEST_PROMPT = """A user is asking, in plain language, for a new real capability
("skill") to be added to their bot. Extract a structured spec from their request.

Respond ONLY as a JSON object with this exact shape:
{
  "name": "snake_case_short_name",
  "description": "a precise, complete spec of what the skill should do and what inputs/outputs
                   it needs - this becomes the actual instruction used to write real code",
  "allowed_domains": ["example.com"],
  "action_type": "read_only"
}

allowed_domains: the specific real domain(s) implied by the request (e.g. "check eBay
prices" implies "ebay.com"). If the category naturally has several real sources rather
than one (e.g. "find tenders" implies multiple procurement portals), list SEVERAL real,
specific domains rather than just one - a skill can only ever reach exactly the domains
listed here, so under-listing silently limits what it can find. If the request doesn't
clearly imply any real domain/service at all (too vague to know what site or API it
should talk to), return an empty list for allowed_domains instead of guessing - leave
description filled in but domains empty so the caller can ask a clarifying question
rather than building something that contacts the wrong site.

action_type: "outbound_action" if the request would send, post, submit, or contact something/
someone; otherwise "read_only" (checking, fetching, looking something up)."""

    def parse_request(self, raw_request: str) -> Optional[Dict]:
        """Turns a plain-language "add a skill that..." request into the structured fields
        build() needs (domains, action_type) - the same classification step BotBuilder's
        architect does inline when it decides skills_needed for a brand-new bot, factored out
        here so add-a-skill-to-an-existing-bot (see GenesisAgent._do_add_skill) doesn't need
        a whole new bot spec to do it. Returns None on total Groq failure - the caller should
        treat that as "couldn't figure out what you're asking for", not silently build nothing."""
        raw = self._call_groq(self.PARSE_REQUEST_PROMPT, json.dumps({"request": raw_request}))
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or "description" not in parsed:
            return None
        return parsed

    def build(self, description: str, allowed_domains: list, max_iterations: Optional[int] = None) -> Dict:
        max_iterations = self.MAX_ITERATIONS if max_iterations is None else max_iterations
        code = self.generate(description, allowed_domains)
        if not code:
            return {"code": None, "success": False, "log": [], "error": "Groq generation failed - check GROQ_API_KEY"}

        log = []
        success = False
        for i in range(max_iterations + 1):
            static_error = _static_safety_check(code)
            if static_error:
                result = {"success": False, "stdout": "", "stderr": static_error, "returncode": None}
            else:
                result = self.execute(code, allowed_domains)
            log.append({"iteration": i, "success": result["success"],
                        "stdout_tail": result["stdout"][-500:], "stderr_tail": result["stderr"][-500:]})
            if result["success"]:
                success = True
                break
            if i < max_iterations:
                error_for_fix = static_error if static_error else (result["stderr"] or result["stdout"])
                fixed = self.fix(code, error_for_fix)
                if fixed:
                    code = fixed

        return {"code": code, "success": success, "log": log, "iterations": len(log)}
