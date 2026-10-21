"""Run the REAL app on 127.0.0.1:8000 with call_llm replaced by a keyword router (no network, no keys).
Use it to smoke-test everything except the LLM itself:
    SECRET_KEY=test DEBUG=true GROQ_API_KEY=gsk_placeholder python3 scripts/fake_llm_launch.py
    python3 scripts/live_smoke.py          # in a second terminal
WARNING: it writes genesis.db / registry / uploads into the project folder - use a copy."""
import json, os, re, sys
from types import SimpleNamespace
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)                      # the app mounts ./static and keeps genesis.db / registry/ here
sys.path.insert(0, ROOT)
import app.core.groq_client as gc

def fake_call_llm(api_key, *, model, messages, **kw):
    user = messages[-1]["content"]
    low = user.lower()
    m = re.search(r"[\w/.\-]+\.csv", user)
    path = m.group(0) if m else None
    if "forecast" in low and path:
        out = {"tool": "forecast_model", "args": {"path": path, "horizon": 14, "horizon_unit": "day"}, "message": "Forecasting..."}
    elif "train" in low and path:
        out = {"tool": "train_model", "args": {"path": path}, "message": "Training..."}
    else:
        out = {"tool": None, "message": "ok (fake llm)"}
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(out)))], usage=usage)

gc.call_llm = fake_call_llm
import uvicorn
from app.main import app
uvicorn.run(app, host="127.0.0.1", port=8000)
