"""Runs tests/ui/ui_test.js (jsdom) against the real static/index.html: no modal visible on entry,
Markdown rendering + XSS escaping, live step checklist from an SSE stream, error/retry, stop button,
slash commands, model card and the bot window. Skipped unless Node and `npm install` in tests/ui are available."""
import os, shutil, subprocess
import pytest

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


@pytest.mark.skipif(shutil.which("node") is None or not os.path.isdir(os.path.join(UI_DIR, "node_modules", "jsdom")),
                    reason="needs Node.js and `cd tests/ui && npm install`")
def test_agent_ui_in_jsdom():
    r = subprocess.run(["node", "ui_test.js"], cwd=UI_DIR, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, "\n".join(l for l in r.stdout.splitlines() if l.startswith("FAIL")) or r.stdout[-1500:] + r.stderr[-800:]
