"""AgentBrowserSession - lets a bot drive a real, isolated headless browser on request
("computer use" style agency - navigate, click, type, read the page, see a screenshot), the
feature discussed as "agency like the big labs have" (Anthropic's computer use, OpenAI's
Operator both work this way: a managed, isolated browser/VM the model controls - NOT direct
access to the end user's own machine, which is a fundamentally different and far riskier
thing this codebase deliberately does NOT do; see the Agent Store trust discussion this was
designed around - an arbitrary published bot getting real access to installers/local files on
whoever installs it would turn the marketplace into a malware distribution channel).

ARCHITECTURE: one container per session, built from docker/agent_browser.Dockerfile, running
docker/agent_browser_harness.py, communicating over a file-based action/result bridge on a
bind-mounted shared directory - same SHAPE as docker_sandbox.py's DockerSandboxNetworked uses
for skills (host writes a request, the container-side process picks it up and writes a
response), but carrying a SEQUENCE of stateful browser actions against one persistent page
instead of one-shot API calls.

SECURITY MODEL - read this before enabling agent_browser_enabled, and see
docker/agent_browser.Dockerfile's comment for more: UNLIKE the skills sandbox (network_mode=
"none", zero real network, every outbound call individually checked against an allowlist by
SafeFetcher on the host), this container has REAL network access, because there is no
practical way to proxy a full browser session - HTML, sub-resources, JS, CSS, redirects,
cookies - through a simple JSON file bridge the way one API call can be proxied. This is a
conscious, documented trade-off, not an oversight:
  - Isolation instead comes from: a single-use, ephemeral container (destroyed at session
    end, never reused across sessions or users), a non-root user inside it, hard wall-clock
    and action-count caps (see settings.agent_browser_max_session_minutes/
    max_actions_per_session), and per-session CPU/memory limits.
  - What this does NOT give you: domain-level allowlisting of where the browser can go. A
    stronger version would run Chromium behind an MITM-capable filtering proxy (TLS
    termination + an allowlist, enforced the way SafeFetcher enforces one for skills) - that
    is real, buildable future work, intentionally not attempted here rather than shipped
    half-working and unverified.
  - Given the above, this should be gated behind a higher trust bar than the rest of the
    product - NOT offered to brand-new free-credit accounts, for instance - until that
    stronger isolation exists. That gating is a product/ops decision to make at the API-key
    or account level; this module only provides the primitive.

NOTE ON VERIFICATION (same honesty as docker_sandbox.py's DockerSandboxNetworked carries):
this was built carefully, reusing this codebase's own established container-isolation
patterns, but there was no Docker daemon available to actually run it end-to-end while
writing it. Before trusting this with real users: build the image (`docker build -f
docker/agent_browser.Dockerfile -t genesis-agent-browser:latest docker/`), run a session for
real, and confirm navigate/click/type/screenshot/extract_text all behave as documented here.
"""
import base64
import json
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

from app.config import settings

IMAGE_NAME = "genesis-agent-browser:latest"
DOCKERFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docker", "agent_browser.Dockerfile")
BUILD_CONTEXT = os.path.dirname(DOCKERFILE_PATH)
SESSIONS_ROOT = os.path.join(tempfile.gettempdir(), "genesis_agent_browser_sessions")

ALLOWED_ACTION_TYPES = {"navigate", "click", "type", "press_key", "scroll", "extract_text", "screenshot"}
RESULT_POLL_INTERVAL_SECONDS = 0.2
RESULT_WAIT_TIMEOUT_SECONDS = 25  # a single action (incl. a slow page navigation) taking
                                   # longer than this is treated as failed, not hung forever
CONTAINER_MEMORY_LIMIT = "768m"  # a real browser needs meaningfully more headroom than the
                                  # 256-512m used by the text-only sandboxes
CONTAINER_NANO_CPUS = 1_000_000_000  # 1 full CPU core


def compute_browser_cost_cents(duration_seconds: float) -> int:
    """Real cost for a session of this duration, in whole cents. Rounds UP (ceil) and
    enforces a 1-cent minimum - same rounding philosophy as public_api_billing.compute_cost_
    cents and for the same reason (a metered charge should never round down to free)."""
    import math
    raw_cents = (duration_seconds / 60.0) * settings.agent_browser_price_cents_per_minute
    return max(1, math.ceil(raw_cents))


def agent_browser_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def ensure_agent_browser_image_built() -> bool:
    try:
        import docker
        client = docker.from_env()
        try:
            client.images.get(IMAGE_NAME)
            return True
        except docker.errors.ImageNotFound:
            pass
        client.images.build(path=BUILD_CONTEXT, dockerfile=os.path.basename(DOCKERFILE_PATH),
                            tag=IMAGE_NAME, rm=True)
        return True
    except Exception as e:
        print(f"[AgentBrowser] could not build agent-browser image: {e}")
        return False


class AgentBrowserError(Exception):
    """Raised for session-lifecycle failures (couldn't start, session not found, session
    expired/over its action cap) - distinct from a single ACTION failing, which is reported
    back as {"success": False, "error": ...} rather than raised, so one bad selector doesn't
    need to be handled as an exception by the caller."""


class AgentBrowserSession:
    """One browser session = one container. Deliberately recoverable from disk/Docker state
    alone (container name + a small metadata.json in the shared workdir), NOT from any
    in-memory Python state - a session's actions normally arrive as separate HTTP requests
    (create session, then N separate "do one action" calls, e.g. one per step of an LLM's
    tool-use loop), which in a real deployment may land on different uvicorn worker
    processes. `resume()` reconstructs a session from just its id, from any process."""

    def __init__(self, session_id: str, workdir: str, container, user_id: str, created_at: float):
        self.session_id = session_id
        self.workdir = workdir
        self.container = container
        self.user_id = user_id
        self.created_at = created_at
        self._next_n = self._count_existing_actions()

    def _count_existing_actions(self) -> int:
        n = 0
        while os.path.exists(os.path.join(self.workdir, f"action_{n}.json")):
            n += 1
        return n

    @classmethod
    def create(cls, user_id: str) -> "AgentBrowserSession":
        import uuid
        import docker

        if not agent_browser_available():
            raise AgentBrowserError("Docker isn't available on this server - the agent browser requires it.")
        if not ensure_agent_browser_image_built():
            raise AgentBrowserError("Could not build/find the agent-browser Docker image.")

        session_id = f"agb_{uuid.uuid4().hex[:20]}"
        workdir = os.path.join(SESSIONS_ROOT, session_id)
        os.makedirs(workdir, exist_ok=True)
        created_at = time.time()
        with open(os.path.join(workdir, "metadata.json"), "w") as f:
            json.dump({"user_id": user_id, "created_at": created_at}, f)

        client = docker.from_env()
        container = client.containers.run(
            IMAGE_NAME,
            name=f"genesis-agent-browser-{session_id}",
            volumes={os.path.abspath(workdir): {"bind": "/session", "mode": "rw"}},
            mem_limit=CONTAINER_MEMORY_LIMIT,
            nano_cpus=CONTAINER_NANO_CPUS,
            # Real network access, deliberately - see this module's docstring for why and
            # what that trades away. Everything else about this container is still locked
            # down (ephemeral, non-root, resource-capped, no host mounts beyond /session).
            network_mode="bridge",
            detach=True,
            remove=False,  # removed explicitly in close(), so a crashed close() can still clean up later
        )
        return cls(session_id, workdir, container, user_id, created_at)

    @classmethod
    def resume(cls, session_id: str) -> "AgentBrowserSession":
        import docker
        workdir = os.path.join(SESSIONS_ROOT, session_id)
        meta_path = os.path.join(workdir, "metadata.json")
        if not os.path.exists(meta_path):
            raise AgentBrowserError(f"No such session: {session_id}")
        with open(meta_path) as f:
            meta = json.load(f)
        client = docker.from_env()
        try:
            container = client.containers.get(f"genesis-agent-browser-{session_id}")
        except docker.errors.NotFound:
            raise AgentBrowserError(f"Session {session_id}'s container is no longer running "
                                    f"(it may have been closed or expired).")
        return cls(session_id, workdir, container, meta["user_id"], meta["created_at"])

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.elapsed_seconds > settings.agent_browser_max_session_minutes * 60

    def perform_action(self, action_type: str, params: Optional[Dict] = None) -> Dict:
        """Returns {"success": bool, "data": {...}, "error": str|None}. Raises
        AgentBrowserError (a session-level problem) rather than returning it in the result
        dict, so callers can tell "this one action failed" (returned dict, keep going) apart
        from "this session can't continue at all" (exception, must stop/close)."""
        if action_type not in ALLOWED_ACTION_TYPES:
            raise AgentBrowserError(f"Unknown action type {action_type!r}. Allowed: {sorted(ALLOWED_ACTION_TYPES)}")
        if self.is_expired:
            raise AgentBrowserError(f"Session exceeded its {settings.agent_browser_max_session_minutes}-minute "
                                    f"limit - close it and start a new one.")
        if self._next_n >= settings.agent_browser_max_actions_per_session:
            raise AgentBrowserError(f"Session reached its {settings.agent_browser_max_actions_per_session}-action limit.")

        n = self._next_n
        action_path = os.path.join(self.workdir, f"action_{n}.json")
        tmp_path = action_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"type": action_type, "params": params or {}}, f)
        os.rename(tmp_path, action_path)  # atomic, so the harness never sees a half-written action

        result_path = os.path.join(self.workdir, f"result_{n}.json")
        waited = 0.0
        while not os.path.exists(result_path):
            time.sleep(RESULT_POLL_INTERVAL_SECONDS)
            waited += RESULT_POLL_INTERVAL_SECONDS
            if waited >= RESULT_WAIT_TIMEOUT_SECONDS:
                self._next_n += 1  # still consumes an action slot even on timeout - a wedged
                                    # action must not let a caller retry it forever for free
                return {"success": False, "data": {}, "error": "Action timed out waiting for a response "
                                                                "from the browser container."}

        with open(result_path) as f:
            result = json.load(f)
        self._next_n += 1

        if action_type == "screenshot" and result.get("success"):
            shot_path = os.path.join(self.workdir, result["data"].get("screenshot_file", ""))
            if os.path.exists(shot_path):
                with open(shot_path, "rb") as f:
                    result["data"]["screenshot_base64"] = base64.b64encode(f.read()).decode("ascii")
        return result

    def close(self) -> Dict:
        """Best-effort graceful shutdown (ask the harness to exit via a real action, which
        matters for Chromium to release its resources cleanly) then always force-remove the
        container regardless, so a session can never be left running/billing silently even if
        the graceful path fails. Returns {"duration_seconds": float} for billing - see the
        API route, which converts this into an actual charge via GenesisDB.charge_for_usage."""
        duration = self.elapsed_seconds
        try:
            # Write the shutdown action DIRECTLY (not via perform_action) - close() must be
            # best-effort even on an already-expired or at-its-action-cap session, and
            # perform_action would raise AgentBrowserError in exactly those cases instead of
            # letting teardown proceed.
            n = self._next_n
            action_path = os.path.join(self.workdir, f"action_{n}.json")
            with open(action_path, "w") as f:
                json.dump({"type": "shutdown", "params": {}}, f)
        except Exception:
            pass
        try:
            self.container.stop(timeout=5)
        except Exception:
            pass
        try:
            self.container.remove(force=True)
        except Exception:
            pass
        try:
            shutil.rmtree(self.workdir, ignore_errors=True)
        except Exception:
            pass
        return {"duration_seconds": duration}

    def list_screenshots(self) -> List[str]:
        """Filenames of every screenshot taken so far this session, oldest first - lets a
        caller review the session's visual history without re-requesting screenshots."""
        if not os.path.isdir(self.workdir):
            return []
        return sorted(f for f in os.listdir(self.workdir) if f.startswith("screenshot_") and f.endswith(".png"))
