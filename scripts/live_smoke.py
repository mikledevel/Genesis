"""End-to-end smoke test over real HTTP against a running server (see scripts/fake_llm_launch.py).
Covers auth, upload, streamed training + forecast, registry, API-key predict, bots, pages."""
import json, random, time, sys, io
import httpx

B = "http://127.0.0.1:8000"
c = httpx.Client(base_url=B, timeout=280)
results = []

def rec(name, ok, note=""):
    results.append((name, ok, note))
    print(("PASS " if ok else "FAIL ") + name + (f"  | {note}" if note else ""), flush=True)

def stream(msg, cid=None, attachments=None):
    body = {"message": msg}
    if cid: body["conversation_id"] = cid
    if attachments: body["attachments"] = attachments
    events = []
    t = time.time()
    with c.stream("POST", "/api/agent/message/stream", json=body, headers=H) as r:
        status = r.status_code
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return status, events, time.time() - t

# --- data
random.seed(1)
rows = ["customer_id,age,tenure_months,monthly_spend,support_calls,churned"]
for i in range(600):
    age = random.randint(18, 70); ten = random.randint(1, 72)
    spend = round(random.uniform(10, 120), 2); calls = random.randint(0, 9)
    p = 0.15 + 0.06 * calls - 0.004 * ten + (0.1 if spend > 90 else 0)
    rows.append(f"{1000+i},{age},{ten},{spend},{calls},{1 if random.random() < p else 0}")
churn_csv = "\n".join(rows).encode()

import math
ts = ["date,sales"]
import datetime
d0 = datetime.date(2025, 1, 1)
for i in range(300):
    v = 100 + 0.2 * i + 15 * math.sin(2 * math.pi * i / 7) + random.gauss(0, 3)
    ts.append(f"{d0 + datetime.timedelta(days=i)},{v:.2f}")
ts_csv = "\n".join(ts).encode()

# 1 register / me / login
email = f"e2e{int(time.time())}@test.io"
r = c.post("/api/auth/register", json={"email": email, "password": "secret12", "name": "E2E"})
rec("register", r.status_code == 200 and "token" in r.json(), f"{r.status_code}")
tok = r.json().get("token", "")
H = {"Authorization": f"Bearer {tok}"}
r = c.get("/api/auth/me", headers=H)
rec("auth/me", r.status_code == 200, f"{r.status_code}")
r = c.post("/api/auth/login", json={"email": email, "password": "secret12"})
rec("login", r.status_code == 200, f"{r.status_code}")
r = c.post("/api/auth/login", json={"email": email, "password": "wrong"})
rec("login wrong password -> 401", r.status_code == 401, f"{r.status_code}")

# 2 llm status
r = c.get("/api/llm/status", headers=H)
rec("llm/status", r.status_code == 200, f"{r.status_code} {r.text[:160]}")

# 3 upload
r = c.post("/api/agent/upload", headers=H, files={"file": ("churn.csv", churn_csv, "text/csv")})
rec("upload csv", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
up = r.json() if r.status_code == 200 else {}
r2 = c.post("/api/agent/upload", headers=H, files={"file": ("ts.csv", ts_csv, "text/csv")})
rec("upload timeseries csv", r2.status_code == 200, f"{r2.status_code}")
up2 = r2.json() if r2.status_code == 200 else {}
r3 = c.post("/api/agent/upload", headers=H, files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
rec("upload .exe rejected (415)", r3.status_code == 415, f"{r3.status_code}")
r4 = c.post("/api/agent/upload", files={"file": ("x.csv", b"a,b\n1,2", "text/csv")})
rec("upload without auth rejected", r4.status_code in (401, 403), f"{r4.status_code}")
print("upload payload:", json.dumps(up)[:300])
att = up.get("id") or up.get("attachment_id")
att2 = up2.get("id") or up2.get("attachment_id")

# 4 plain chat (no LLM key -> expect a clean error, not a 500)
st, ev, dt = stream("hello", cid=f"e2e-hello-{int(time.time())}")
last = ev[-1] if ev else {}
rec("chat stream terminates with final/error", last.get("type") in ("final", "error"), f"http={st} last={json.dumps(last, ensure_ascii=False)[:200]} {dt:.1f}s")

st0, ev0, _ = stream("hello without id")
l0 = ev0[-1] if ev0 else {}
rec("chat WITHOUT conversation_id works for a new user", l0.get("type") == "final", f"last={json.dumps(l0)[:160]}")

# 5 train via chat
st, ev, dt = stream("Train a model to predict churned from the attached file", cid=f"e2e-train-{int(time.time())}", attachments=[att] if att else None)
types = [e.get("type") for e in ev]
last = ev[-1] if ev else {}
rec("train: stream ends cleanly", last.get("type") in ("final", "error"), f"http={st} types={sorted(set(types))} {dt:.1f}s")
print("TRAIN LAST:", json.dumps(last, ensure_ascii=False, default=str)[:1800])
print("TRAIN EVENTS:", [(e.get("type"), e.get("id") or e.get("step"), e.get("status")) for e in ev][:30])
mid = None
if last.get("type") == "final":
    data = last.get("data") or {}
    print("TRAIN DATA KEYS:", list(data.keys())[:20])
    mid = data.get("model_id") or (data.get("result") or {}).get("model_id")

# 6 forecast via chat
st, ev, dt = stream("Forecast sales for the next 14 days", cid=f"e2e-fc-{int(time.time())}", attachments=[att2] if att2 else None)
types = [e.get("type") for e in ev]
last = ev[-1] if ev else {}
rec("forecast: stream ends cleanly", last.get("type") in ("final", "error"), f"http={st} types={sorted(set(types))} {dt:.1f}s")
print("FORECAST LAST:", json.dumps(last, ensure_ascii=False, default=str)[:1800])

# 7 registry
r = c.get("/api/registry/models", headers=H)
rec("registry/models", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
models = r.json() if r.status_code == 200 else []
if isinstance(models, dict):
    models = models.get("models", [])
print("registry count:", len(models))

# 8 api key + predict
r = c.post("/api/keys/create", headers=H, json={})
rec("keys/create", r.status_code == 200, f"{r.status_code} {r.text[:120]}")
key = (r.json() or {}).get("key") or (r.json() or {}).get("api_key") if r.status_code == 200 else None
mid2 = mid
if mid2 and key:
    rp = c.post(f"/api/models/{mid2}/predict", headers={"X-API-Key": key},
                json={"data": {"age": 40, "tenure_months": 5, "monthly_spend": 95.0, "support_calls": 7}})
    rec("model predict via API key", rp.status_code == 200, f"{rp.status_code} {rp.text[:240]}")
else:
    rec("model predict via API key", False, "no model in registry or no key")

# 9 bot create/publish/mine/delete
r = c.post("/api/agents/create", headers=H, json={"name": "E2E Bot", "system_prompt": "You answer about cats only.", "description": "t"})
rec("agents/create", r.status_code == 200, f"{r.status_code}")
aid = r.json().get("id") if r.status_code == 200 else None
if aid:
    r = c.post(f"/api/agents/{aid}/publish", headers=H, json={"price": 0})
    rec("agents/publish", r.status_code == 200, f"{r.status_code} {r.text[:100]}")
    r = c.get("/api/agents/mine", headers=H)
    rec("agents/mine lists it", r.status_code == 200 and aid in r.text, f"{r.status_code}")
    r = c.get("/api/agents", headers=H)
    rec("agents (marketplace) lists published", r.status_code == 200 and aid in r.text, f"{r.status_code}")
    r = c.get(f"/api/agents/{aid}/analytics", headers=H)
    rec("agents/analytics", r.status_code == 200, f"{r.status_code}")

# 10 bot-builder test draft (needs LLM)
r = c.post("/api/bot-builder/generate", headers=H, json={"description": "A tutor bot for school kids"})
rec("bot-builder/generate (LLM down -> clean error, not 500)", r.status_code < 500 or r.status_code in (502, 503, 504), f"{r.status_code} {r.text[:160]}")

# 11 pages / misc
for p in ["/", "/static/index.html"]:
    r = c.get(p)
    rec(f"GET {p}", r.status_code == 200, f"{r.status_code} {len(r.content)}B")
r = c.get("/api/agent/conversations", headers=H)
rec("conversations list", r.status_code == 200, f"{r.status_code} {r.text[:120]}")
r = c.get("/api/pipeline/templates", headers=H)
rec("pipeline/templates", r.status_code == 200, f"{r.status_code}")
r = c.get("/openapi.json")
rec("openapi.json", r.status_code == 200, f"{r.status_code}")

print()
bad = [x for x in results if not x[1]]
print(f"{len(results)-len(bad)}/{len(results)} passed")
for n, _, note in bad:
    print("  FAILED:", n, "|", note)
