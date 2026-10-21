"""Check every configured LLM provider with one tiny real request:

    python -m app.core.llm_check

Prints, per provider, whether its key works, the model used and the latency - or the exact error
type/message - so a wrong key, a wrong model name or an exhausted quota is found in seconds
instead of at 2 a.m. when the failover is actually needed. Never prints a key. Exit code 0 if at
least one provider answered, 1 otherwise. Uses the same settings (.env) and the same provider
plan as the running app, but ignores the cooldown state."""
import sys
import time

from app.config import settings
from app.core import groq_client

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
HINTS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}


def main() -> int:
    plan = groq_client.provider_plan(settings.groq_api_key, DEFAULT_GROQ_MODEL)
    configured = {name for name, _, _ in plan}
    print(f"Provider order: {' -> '.join(groq_client.provider_order())}\n")
    ok_count = 0
    for name in groq_client.provider_order():
        if name not in configured:
            print(f"  {groq_client.PRETTY[name]:<11} not configured (set {HINTS[name]} in .env)")
    for name, model, client in plan:
        t0 = time.monotonic()
        try:
            c = groq_client._create(client, name, model, [{"role": "user", "content": "Reply with the single word: ok"}],
                                    {"max_tokens": 64, "temperature": 0}, None)
            text = (c.choices[0].message.content or "").strip().replace("\n", " ")[:40]
            print(f"  {groq_client.PRETTY[name]:<11} OK     {model}  {time.monotonic() - t0:.1f}s  reply={text!r}")
            ok_count += 1
        except Exception as e:
            print(f"  {groq_client.PRETTY[name]:<11} FAILED {model}  {type(e).__name__}: {str(e)[:220]}")
    print(f"\n{ok_count} of {len(plan)} configured model(s) answered.")
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
