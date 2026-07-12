import urllib.request, json
d = json.dumps({
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Say hi in JSON format"}],
    "max_tokens": 100
}).encode()
r = urllib.request.Request("https://api.deepseek.com/v1/chat/completions", data=d, headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-8a799db2c0f8483ca71441e802c37149"
})
try:
    resp = json.loads(urllib.request.urlopen(r, timeout=30).read())
    print("OK:", resp["choices"][0]["message"]["content"][:200])
except Exception as e:
    print("Error:", e)
