import urllib.request, json, os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise SystemExit("Set DEEPSEEK_API_KEY in your environment or .env file before running this script.")

d = json.dumps({
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Say hi in JSON format"}],
    "max_tokens": 100
}).encode()
r = urllib.request.Request("https://api.deepseek.com/v1/chat/completions", data=d, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
})
try:
    resp = json.loads(urllib.request.urlopen(r, timeout=30).read())
    print("OK:", resp["choices"][0]["message"]["content"][:200])
except Exception as e:
    print("Error:", e)
