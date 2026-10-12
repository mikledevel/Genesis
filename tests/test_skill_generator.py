"""
Tests for app.core.skills.skill_generator. These do NOT call Groq (no network dependency on
an LLM) - they test the static safety check and the actual subprocess execution path with
hand-written code standing in for what the LLM would produce, exactly like
tests/test_agent.py mocks the Groq boundary while testing the real dispatch logic around it.

Docker-path (DockerSandboxNetworked) tests are NOT included here - there is no Docker daemon
in the environment these tests were written in. That class was reviewed carefully but not
executed; see its docstring in docker_sandbox.py for what to verify once Docker is available.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.core.skills.skill_generator import SkillGenerator, _static_safety_check


class TestStaticSafetyCheck:
    def test_legitimate_skill_passes(self):
        code = '''
def run(params):
    r = fetch("https://api.example.com/search")
    return {"status": r["status"]}
TEST_PARAMS = {"query": "test"}
'''
        assert _static_safety_check(code) is None

    @pytest.mark.parametrize("bad_import", ["os", "sys", "subprocess", "socket", "requests", "urllib"])
    def test_banned_import_rejected(self, bad_import):
        code = f"import {bad_import}\ndef run(params): return {{}}"
        result = _static_safety_check(code)
        assert result is not None
        assert bad_import in result

    def test_banned_from_import_rejected(self):
        code = "from os import system\ndef run(params): return {}"
        assert _static_safety_check(code) is not None

    @pytest.mark.parametrize("banned_call", ["eval('1')", "exec('1')", "open('/etc/passwd')", "__import__('os')"])
    def test_banned_call_rejected(self, banned_call):
        code = f"def run(params):\n    {banned_call}\n    return {{}}"
        assert _static_safety_check(code) is not None

    def test_missing_run_function_rejected(self):
        code = "def other_name(params): return {}"
        assert _static_safety_check(code) is not None

    def test_syntax_error_rejected_cleanly(self):
        code = "def run(params:\n    return {}"
        result = _static_safety_check(code)
        assert result is not None
        assert "syntax error" in result.lower()


class TestSubprocessExecution:
    """Uses the real subprocess sandbox path (dev-fallback) - see module docstring for why
    this is NOT the production security boundary (that's DockerSandboxNetworked), but it IS
    real, actual code execution, not a mock, and genuinely exercises SafeFetcher end to end."""

    def test_allowed_domain_request_succeeds(self):
        gen = SkillGenerator()
        code = '''
def run(params):
    r = fetch("https://github.com/" + params["path"])
    return {"status": r["status"]}
TEST_PARAMS = {"path": "anthropics"}
'''
        result = gen._execute_subprocess(code, allowed_domains=["github.com"])
        assert result["success"], result["stdout"] + result["stderr"]

    def test_disallowed_domain_request_fails_gracefully(self):
        gen = SkillGenerator()
        code = '''
def run(params):
    r = fetch("https://github.com/anthropics")
    return {"status": r["status"]}
TEST_PARAMS = {}
'''
        result = gen._execute_subprocess(code, allowed_domains=["pypi.org"])
        assert not result["success"]
        assert "only allowed to contact" in result["stdout"]

    def test_code_with_banned_import_never_executes(self):
        gen = SkillGenerator()
        code = "import os\ndef run(params):\n    return {\"files\": os.listdir('/')}\nTEST_PARAMS = {}"
        result = gen._execute_subprocess(code, allowed_domains=["github.com"])
        assert not result["success"]
        assert "not allowed" in result["stderr"]

    def test_non_serializable_return_value_fails(self):
        """run() must return JSON-serializable data - returning something like a set (which
        json.dumps can't handle) must be caught as a failure, not silently accepted."""
        gen = SkillGenerator()
        code = "def run(params):\n    return {\"bad\": {1, 2, 3}}\nTEST_PARAMS = {}"
        result = gen._execute_subprocess(code, allowed_domains=[])
        assert not result["success"]

    def test_exception_in_run_is_caught_not_crashed(self):
        gen = SkillGenerator()
        code = "def run(params):\n    return 1 / 0\nTEST_PARAMS = {}"
        result = gen._execute_subprocess(code, allowed_domains=[])
        assert not result["success"]
        assert "SKILL_TEST_RESULT" in result["stdout"]  # harness caught it, didn't crash

    def test_timeout_is_enforced(self):
        gen = SkillGenerator()
        gen.EXEC_TIMEOUT = 2
        code = "import time\ndef run(params):\n    time.sleep(30)\n    return {}\nTEST_PARAMS = {}"
        # "import time" is allowed (not in BANNED_IMPORTS) - only network/system libs are banned
        result = gen._execute_subprocess(code, allowed_domains=[])
        assert not result["success"]
        assert "timeout" in (result["stderr"] or "").lower() or result["returncode"] is None
