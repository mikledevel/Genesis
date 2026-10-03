from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Genesis AI"
    debug: bool = False
    database_url: str = ""
    redis_url: str = ""
    # Both currently unused - no code path reads either of these (verified: no
    # `.llm_api_key` or `.deepseek_api_key` reference anywhere in app/). Harmless to leave
    # set to "", but setting them does nothing today; see OPENROUTER_API_KEY below for the
    # fallback provider that IS actually wired up.
    llm_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""
    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""
    secret_key: str = "change-me"
    # Comma-separated list of allowed frontend origins in production, e.g.
    # "https://genesis-ai.app,https://www.genesis-ai.app". Empty means "reflect no origin"
    # (CORSMiddleware below falls back to "*" only when debug=True, for local development).
    allowed_origins: str = ""
    # Platform-wide default monthly Groq token cap per user, checked BEFORE each internal
    # Groq call is made (see app.core.quota) - protects the platform's own shared
    # GROQ_API_KEY from unbounded per-user cost, now that real usage is actually tracked
    # (see GenesisDB.log_groq_usage). 0 or negative disables the cap entirely (unlimited).
    # A per-user override can be set in the DB (users.monthly_groq_token_limit) for cases
    # that need a different limit than this platform default.
    default_monthly_groq_token_limit: int = 2_000_000
    # Timeout for every outbound Groq call across the codebase (agent.py, bot_builder.py,
    # ai_generator.py, skill_generator.py). Split into a connect timeout (how long to wait to
    # even reach the provider - a real outage/DNS failure should fail fast) and a read timeout
    # (how long to wait for the actual completion once connected - generation is legitimately
    # slow sometimes, especially for the heavier multi-step flows like bot-builder/revise or
    # skill generation, which chain several Groq calls internally).
    #
    # HISTORY: this used to be a single flat 20s timeout with max_retries=1. That was too
    # aggressive - real, eventually-successful completions have been observed taking up to
    # ~70s (see docs/notes), so a flat 20s cap was killing legitimate-but-slow requests, not
    # just genuine outages, and the SDK's built-in retry on timeout meant a doomed request
    # could silently eat 2x20s before finally failing. Net effect from the user's side: "works
    # fine, then randomly times out" on longer prompts. Now: fail fast on real unreachability,
    # give real generations a realistic ceiling, and don't retry a timeout invisibly - one
    # predictable wait, then a clear error the user can just resend.
    groq_connect_timeout_seconds: float = 10.0
    groq_read_timeout_seconds: float = 90.0
    # Separate, more generous read timeout for calls known to generate large structured
    # output in one shot (a full bot spec, a skill's code, a trained model's generated
    # training script) rather than a short conversational reply. Observed in production
    # logs: a single bot-builder "revise" call (one JSON completion, no internal loop) took
    # ~69s on its own - the standard 90s read timeout has margin for that, but not much, and
    # a bigger revision or a slower moment on the provider's end could still clip it.
    groq_read_timeout_heavy_seconds: float = 150.0
    groq_max_retries: int = 0
    # Optional automatic fallback provider. When set, every outbound LLM call (see
    # app.core.groq_client.call_llm) that fails on Groq for ANY reason - bad/missing key,
    # rate limit, timeout, capacity, provider outage - is automatically retried once via
    # OpenRouter (an OpenAI-compatible router that can serve the same open-weight models,
    # e.g. openai/gpt-oss-120b, from ~20 different backing providers) before the failure is
    # allowed to reach the user. This is what turns "Groq is at capacity" from a user-facing
    # timeout into an invisible retry. Leave empty to disable - behavior is then identical
    # to Groq-only (unchanged). Get a key at https://openrouter.ai/keys.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Public developer API (POST /api/v1/chat/completions) pricing. Prices are per ONE
    # MILLION tokens, in whole US cents, so they can be expressed as plain integers rather
    # than fractions-of-a-cent floats: 50 -> $0.50/1M input tokens, 200 -> $2.00/1M output
    # tokens. This is roughly a 3x markup over this platform's own raw Groq cost for
    # openai/gpt-oss-120b (~$0.15/$0.75 per 1M as billed to us) - enough margin to cover
    # infrastructure, support, and the OpenRouter fallback (which can cost more per call than
    # Groq on some models), without being the kind of markup that makes a developer's own
    # unit economics not work. Revisit if the underlying provider pricing moves meaningfully.
    #
    # Every billed call has a minimum charge of public_api_min_charge_cents (1 cent) even if
    # the token-based cost would round to less - avoids a "the first ~2000 tokens are free"
    # loophole from integer-cent rounding on tiny requests, and matches how most metered APIs
    # charge (a per-request floor plus token overage), rather than showing users a "$0.00
    # charged" line on every short chat message.
    public_api_input_price_cents_per_million: int = 50
    public_api_output_price_cents_per_million: int = 200
    public_api_min_charge_cents: int = 1
    public_api_default_model: str = "openai/gpt-oss-120b"
    # Hard ceiling on the `max_tokens` a caller can request in one call - this is what the
    # PRE-charge (see app/main.py's chat_completions_v1) is computed against, so it's also
    # what caps the platform's worst-case exposure on one request before a refund is issued
    # for whatever wasn't actually used.
    public_api_max_tokens_cap: int = 4096
    # Public HTTPS base URL this server is reachable at, used to register the Telegram
    # webhook (see app.core.telegram_integration / the /api/agents/{id}/telegram/* routes in
    # main.py). Telegram requires a real public HTTPS URL - it cannot deliver to localhost.
    # Left empty by default (e.g. local dev); a bot can still be connected and tested via the
    # built-in test-chat endpoint without this set, it just won't receive live messages until
    # this is configured and the bot is reconnected.
    public_base_url: str = ""
    # Outbound email (password reset, email verification). If smtp_host is empty, emails are
    # printed to the server log instead of sent - safe default for local dev, but means
    # password reset/verification links only work for real in production once this is set.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "Genesis AI <no-reply@genesis-ai.app>"
    # Error monitoring (https://sentry.io) - optional. Empty = disabled, errors just print()
    # to the log as before.
    sentry_dsn: str = ""
    # Geo-blocking - see app/core/geo_block.py for the full detection/fail-open policy and
    # the important limitation notes on region-level accuracy for the Ukraine oblasts.
    # Country list: Russia, Belarus, Cuba, Iran, North Korea, Syria - the jurisdictions under
    # comprehensive OFAC sanctions programs. geoblock_ua_regions handles the Ukrainian
    # territories that are sanctioned at the region (not country) level: Crimea, the
    # so-called "DPR"/"LPR" (Donetsk and Luhansk), and the Zaporizhzhia/Kherson oblasts added
    # to OFAC's scope in 2022 - the rest of Ukraine is NOT blocked.
    geoblock_enabled: bool = True
    geoblock_countries: str = "RU,BY,CU,IR,KP,SY"
    geoblock_ua_regions: str = "Crimea,Donetsk,Luhansk,Zaporizhzhia,Kherson"
    # Prints, for every non-exempt request, exactly what IP/country/region the middleware
    # resolved and why it allowed or blocked - turn on temporarily to diagnose "it isn't
    # blocking who I expect" in a real deployment (most often caused by a reverse proxy not
    # forwarding X-Forwarded-For, so every request looks like it's coming from the proxy's
    # own private IP - see the SKIPPING log line for exactly that case). Leave off in normal
    # operation; it logs client IPs on every request, which you don't want as a permanent
    # habit even though none of this is more sensitive than any standard access log.
    geoblock_debug_log: bool = False
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
