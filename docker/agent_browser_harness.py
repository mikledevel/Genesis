"""Runs INSIDE the agent-browser container (see agent_browser.Dockerfile). Controls one
persistent Playwright browser context for the lifetime of a session, driven by action files
the HOST writes to the shared volume mounted at /session - this is the same file-bridge shape
app/core/codegen/docker_sandbox.py's DockerSandboxNetworked uses for skills (host writes a
request, this process picks it up, writes a response), adapted for a sequence of stateful
browser actions instead of one-shot HTTP calls.

Protocol (all files in /session, the bind-mounted shared volume):
  action_<n>.json  (host writes)   {"type": "navigate"|"click"|"type"|"screenshot"|
                                     "extract_text"|"scroll"|"press_key"|"shutdown",
                                     "params": {...}}
  result_<n>.json  (this writes)   {"success": bool, "data": {...}, "error": str|None}
  screenshot_<n>.png (this writes, only for the "screenshot" action type)

Actions are processed strictly in order (n = 0, 1, 2, ...) and one failing action does NOT
crash the session - the error is reported back in that action's result file, and the harness
keeps polling for the next one, so the caller (see app/core/agent_browser.py) can see what
went wrong and decide what to do next, the same way a real "the agent sees an error and
adjusts" loop works.

MAX_ACTIONS below is a defense-in-depth internal cap - the host already enforces its own
session action-count limit (see AgentBrowserSession.MAX_ACTIONS_PER_SESSION) before it even
writes another action file, so this should never actually bind in normal operation; it exists
so a host-side bug can't turn into an unbounded loop running inside the container.
"""
import json
import os
import sys
import time

SESSION_DIR = "/session"
POLL_INTERVAL_SECONDS = 0.2
ACTION_WAIT_TIMEOUT_SECONDS = 120  # how long to wait for the NEXT action before giving up
                                     # and exiting - the host didn't send anything in that
                                     # time, which only happens if the host process died
MAX_ACTIONS = 200
DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
NAV_TIMEOUT_MS = 20_000
ACTION_TIMEOUT_MS = 10_000


def _wait_for_file(path, timeout_seconds):
    waited = 0.0
    while not os.path.exists(path):
        time.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS
        if waited >= timeout_seconds:
            return False
    return True


def _write_result(n, success, data=None, error=None):
    path = os.path.join(SESSION_DIR, f"result_{n}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"success": success, "data": data or {}, "error": error}, f)
    os.rename(tmp_path, path)  # atomic from the host's point of view - never sees a half-written file


def main():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(viewport=DEFAULT_VIEWPORT)
        context.set_default_timeout(ACTION_TIMEOUT_MS)
        context.set_default_navigation_timeout(NAV_TIMEOUT_MS)
        page = context.new_page()

        n = 0
        while n < MAX_ACTIONS:
            action_path = os.path.join(SESSION_DIR, f"action_{n}.json")
            if not _wait_for_file(action_path, ACTION_WAIT_TIMEOUT_SECONDS):
                break  # host went quiet - exit cleanly rather than poll forever

            try:
                with open(action_path) as f:
                    action = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                _write_result(n, False, error=f"could not read action file: {e}")
                n += 1
                continue

            atype = action.get("type")
            params = action.get("params") or {}

            if atype == "shutdown":
                _write_result(n, True, data={"shut_down": True})
                break

            try:
                if atype == "navigate":
                    resp = page.goto(params["url"], wait_until="domcontentloaded")
                    _write_result(n, True, data={"url": page.url,
                                                  "status": resp.status if resp else None,
                                                  "title": page.title()})
                elif atype == "click":
                    page.click(params["selector"])
                    _write_result(n, True, data={"clicked": params["selector"]})
                elif atype == "type":
                    page.fill(params["selector"], params.get("text", ""))
                    _write_result(n, True, data={"typed_into": params["selector"]})
                elif atype == "press_key":
                    page.keyboard.press(params["key"])
                    _write_result(n, True, data={"pressed": params["key"]})
                elif atype == "scroll":
                    dy = params.get("amount", 600) * (1 if params.get("direction", "down") == "down" else -1)
                    page.mouse.wheel(0, dy)
                    _write_result(n, True, data={"scrolled": dy})
                elif atype == "extract_text":
                    selector = params.get("selector")
                    text = page.inner_text(selector) if selector else page.inner_text("body")
                    # Cap what comes back - a full page's text can be enormous, and this is
                    # meant for an LLM to read as context, not a scraping/export tool.
                    _write_result(n, True, data={"text": text[:8000], "truncated": len(text) > 8000})
                elif atype == "screenshot":
                    shot_path = os.path.join(SESSION_DIR, f"screenshot_{n}.png")
                    page.screenshot(path=shot_path, full_page=params.get("full_page", False))
                    _write_result(n, True, data={"screenshot_file": f"screenshot_{n}.png",
                                                  "url": page.url})
                else:
                    _write_result(n, False, error=f"unknown action type: {atype!r}")
            except Exception as e:
                # One bad selector/timeout must not kill the whole session - report the error
                # and keep going, same as a real agent loop recovering from a failed step.
                _write_result(n, False, error=f"{type(e).__name__}: {e}")

            n += 1

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last-resort: if something fails before/outside the per-action loop (browser
        # wouldn't launch, playwright itself errored), surface it as result_0 so the host
        # isn't left waiting forever for a file that will never come.
        print(f"[harness] fatal error: {type(e).__name__}: {e}", file=sys.stderr)
        _write_result(0, False, error=f"harness failed to start: {type(e).__name__}: {e}")
        sys.exit(1)
