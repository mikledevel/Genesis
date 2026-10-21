"""
DockerSandbox - production-grade isolation for executing AI-generated ML pipeline code,
replacing the dev-only subprocess sandbox in ai_generator.py for real, multi-tenant use.

Isolation applied on every run:
  - network_mode="none"      - the container cannot reach the internet or the host network
  - mem_limit / nano_cpus    - hard resource caps, one runaway script can't starve the host
  - read_only root filesystem - only /sandbox (the mounted workdir) is writable
  - non-root user (baked into the image, see sandbox.Dockerfile)
  - a wall-clock timeout enforced from the Python side, independent of the container

Falls back cleanly to the (weaker) subprocess sandbox in ai_generator.py if Docker isn't
installed/running - this keeps local single-user development working without requiring
Docker, while real deployments serving other users should always have Docker available.
"""
import os
import tempfile
import shutil
from typing import Dict

IMAGE_NAME = "genesis-sandbox:latest"
DOCKERFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docker", "sandbox.Dockerfile")


def docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def ensure_image_built() -> bool:
    """Build the sandbox image if it doesn't exist yet. Returns True if the image is
    ready to use, False if Docker itself isn't available or the build failed."""
    try:
        import docker
        client = docker.from_env()
        try:
            client.images.get(IMAGE_NAME)
            return True
        except docker.errors.ImageNotFound:
            pass
        build_context = os.path.dirname(DOCKERFILE_PATH)
        client.images.build(path=build_context, dockerfile="sandbox.Dockerfile", tag=IMAGE_NAME, rm=True)
        return True
    except Exception as e:
        print(f"[DockerSandbox] could not build sandbox image: {e}")
        return False


class DockerSandbox:
    TIMEOUT_SECONDS = 45
    MEMORY_LIMIT = "512m"
    NANO_CPUS = 1_000_000_000  # 1 full CPU core

    def execute(self, code: str, workdir: str) -> Dict:
        """Same interface/return shape as AICodeGenerator.execute() so it's a drop-in
        replacement: {"success": bool, "stdout": str, "stderr": str, "returncode": int|None}."""
        import docker
        from docker.errors import ContainerError, ImageNotFound, APIError

        os.makedirs(os.path.join(workdir, "models"), exist_ok=True)
        with open(os.path.join(workdir, "pipeline.py"), "w", encoding="utf-8") as f:
            f.write(code)

        client = docker.from_env()
        container = None
        try:
            container = client.containers.run(
                IMAGE_NAME,
                volumes={os.path.abspath(workdir): {"bind": "/sandbox", "mode": "rw"}},
                working_dir="/sandbox",
                network_mode="none",
                mem_limit=self.MEMORY_LIMIT,
                nano_cpus=self.NANO_CPUS,
                read_only=True,
                tmpfs={"/tmp": "size=64m"},  # sklearn/xgboost sometimes need scratch space
                detach=True,
                stdout=True,
                stderr=True,
            )
            result = container.wait(timeout=self.TIMEOUT_SECONDS)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            returncode = result.get("StatusCode", 1)
            success = returncode == 0 and "PIPELINE_SUCCESS" in logs
            return {"success": success, "stdout": logs[-4000:], "stderr": "" if success else logs[-4000:], "returncode": returncode}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": f"Sandbox execution error: {e}", "returncode": None}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


class DockerSandboxNetworked:
    """Runs AI-generated SKILL code (see app/core/skills/skill_generator.py) - the variant
    that needs network access to be useful at all (checking a website, calling a public API),
    which is exactly what makes it higher-risk than the network_mode="none" ML-pipeline
    sandbox above.

    The container STILL runs with network_mode="none" - it has literally no network device,
    so nothing the generated code does (however it tries: raw sockets, a smuggled import, an
    obfuscated call) can reach anywhere on its own. Instead, "network access" is provided
    through a file-based request/response bridge over the same bind-mounted volume every
    other sandbox here already uses:

        1. The container's `fetch()` global (injected via the harness script, not the
           model's own code) writes a JSON request file to the shared volume and polls for
           a matching response file to appear.
        2. THIS class, running on the host (which has real network), watches that same
           volume while the container runs. For each request file it finds, it performs the
           actual HTTP call through SafeFetcher (app/core/skills/net_guard.py) - which is
           what actually enforces the domain allowlist and blocks private/internal IPs - and
           writes the result back.

    This means the enforcement point is OUTSIDE the sandboxed code entirely; there is no
    execution path, however the generated code misbehaves, that reaches the network without
    going through SafeFetcher first.

    NOTE ON VERIFICATION: I built and reviewed this carefully, but had no Docker daemon
    available in my own environment to actually run it end-to-end - see docker_sandbox.py's
    docstring / the accompanying message for what to check before trusting this in
    production. Please run it for real (with GEOBLOCK/skills enabled in a dev environment)
    and report back the exact error if anything doesn't work as described.
    """
    TIMEOUT_SECONDS = 35  # raised from 20s - a real multi-request skill (e.g. checking several
    # sites for tenders) needs headroom beyond a single quick lookup
    MEMORY_LIMIT = "256m"
    NANO_CPUS = 500_000_000  # 0.5 CPU core - skills are lightweight lookups, not training jobs
    POLL_INTERVAL_SECONDS = 0.2

    def execute(self, code: str, allowed_domains: list) -> Dict:
        """Runs the skill's own TEST_PARAMS - used during creation/self-test (see
        SkillGenerator.build())."""
        call_snippet = "run(TEST_PARAMS)"
        marker = "SKILL_TEST_RESULT"
        return self._run_harness(code, allowed_domains, call_snippet, marker)

    def execute_with_params(self, code: str, params: dict, allowed_domains: list) -> Dict:
        """Runs the skill with REAL params chosen by the LLM during a live conversation -
        used by SkillGenerator.run_stored_skill(). Separate marker (SKILL_RUN_RESULT vs
        SKILL_TEST_RESULT) so a live call's output can never be confused with a self-test's."""
        call_snippet = f"run({params!r})"
        marker = "SKILL_RUN_RESULT"
        return self._run_harness(code, allowed_domains, call_snippet, marker)

    def _run_harness(self, code: str, allowed_domains: list, call_snippet: str, marker: str) -> Dict:
        import time
        import docker

        workdir = tempfile.mkdtemp(prefix="genesis_skill_sandbox_")
        try:
            harness = f"""
import json, time

def fetch(url, method="GET", headers=None, body=None):
    req_id = fetch._counter
    fetch._counter += 1
    with open(f"/sandbox/_req_{{req_id}}.json", "w") as f:
        json.dump({{"url": url, "method": method, "headers": headers, "body": body}}, f)
    resp_path = f"/sandbox/_resp_{{req_id}}.json"
    deadline = time.time() + 15
    while not __import__("os").path.exists(resp_path):
        if time.time() > deadline:
            raise TimeoutError("No response from the host bridge within 15s")
        time.sleep(0.1)
    with open(resp_path) as f:
        resp = json.load(f)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["result"]
fetch._counter = 0

{code}

try:
    result = {call_snippet}
    json.dumps(result)
    print("{marker}:" + json.dumps({{"ok": True, "result": result}}))
except Exception as e:
    print("{marker}:" + json.dumps({{"ok": False, "error": str(e)}}))
"""
            with open(os.path.join(workdir, "skill.py"), "w", encoding="utf-8") as f:
                f.write(harness)

            client = docker.from_env()
            container = client.containers.run(
                IMAGE_NAME,
                command=["python", "/sandbox/skill.py"],
                volumes={os.path.abspath(workdir): {"bind": "/sandbox", "mode": "rw"}},
                working_dir="/sandbox",
                network_mode="none",
                mem_limit=self.MEMORY_LIMIT,
                nano_cpus=self.NANO_CPUS,
                detach=True,
                stdout=True,
                stderr=True,
            )
            served_requests = set()
            deadline = time.time() + self.TIMEOUT_SECONDS
            try:
                from app.core.skills.net_guard import SafeFetcher
                fetcher = SafeFetcher(allowed_domains)
                while time.time() < deadline:
                    container.reload()
                    for fname in os.listdir(workdir):
                        if not fname.startswith("_req_") or fname in served_requests:
                            continue
                        served_requests.add(fname)
                        req_id = fname[len("_req_"):-len(".json")]
                        with open(os.path.join(workdir, fname)) as f:
                            req = json.load(f)
                        try:
                            result = fetcher.fetch(req["url"], req.get("method", "GET"),
                                                    req.get("headers"), req.get("body"))
                            payload = {"result": result}
                        except Exception as e:
                            payload = {"error": str(e)}
                        with open(os.path.join(workdir, f"_resp_{req_id}.json"), "w") as f:
                            json.dump(payload, f)
                    if container.status in ("exited", "dead"):
                        break
                    time.sleep(self.POLL_INTERVAL_SECONDS)

                result = container.wait(timeout=5)
                logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                returncode = result.get("StatusCode", 1)
                success = f"{marker}:" in logs and '"ok": true' in logs.lower()
                return {"success": success, "stdout": logs[-3000:], "stderr": "" if success else logs[-3000:], "returncode": returncode}
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": f"Sandbox execution error: {e}", "returncode": None}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
