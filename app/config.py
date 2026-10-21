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
    default_monthly_groq_token_limit: int = 300_000
    # Comma-separated e-mails of the platform owners (override with ADMIN_EMAILS in .env). An account gets the admin
    # panel and an unlimited token quota ONLY if its e-mail is in this list AND the e-mail is VERIFIED (a verification
    # link went to that mailbox) - never on the e-mail string alone, because anyone can type any address at
    # registration. Empty = nobody is an admin. Before selling or sharing the project, replace the default.
    admin_emails: str = "efremovmisa295@gmail.com"
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
    # OpenRouter model(s) to ask for when Groq fails. Empty = the SAME model id the caller sent to
    # Groq (works for e.g. openai/gpt-oss-120b). For a free-only key set e.g.
    # OPENROUTER_MODEL=openai/gpt-oss-120b:free . Comma-separated = tried in order.
    openrouter_model: str = ""

    # Second fallback: Google Gemini through its OpenAI-compatible endpoint. Key from
    # https://aistudio.google.com/apikey . GEMINI_MODEL may be a comma-separated list.
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-flash-latest"

    # Provider chain (see app.core.groq_client.call_llm). A provider without a key is skipped.
    llm_provider_order: str = "groq,openrouter,gemini"
    # After a provider fails it is skipped for this long (rate limit / timeout / outage); a bad or
    # forbidden key is skipped for llm_auth_cooldown_seconds. 0 disables the cooldown entirely.
    llm_cooldown_seconds: int = 60
    # A model the provider says doesn't exist / was decommissioned is skipped this long.
    llm_missing_model_cooldown_seconds: int = 3600
    # Groq rate limits are PER MODEL, so when the requested model (usually openai/gpt-oss-120b) is
    # rate-limited or down, the other Groq models below are tried - in this order - BEFORE the
    # chain moves on to OpenRouter / Gemini. Ids come from app/core/models_catalog.py (verified
    # against Groq's docs). Add llama-3.1-8b-instant at the end for a last-resort weak model.
    # Empty = no same-provider fallback. Models that can't do what the request needs (e.g. vision)
    # are skipped automatically.
    groq_fallback_models: str = "llama-3.3-70b-versatile,openai/gpt-oss-20b,qwen/qwen3.6-27b"
    llm_auth_cooldown_seconds: int = 900

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

    # Free starter balance granted once a new account verifies its email (not at raw
    # registration - tying it to email verification is a meaningfully higher bar against
    # farming free credits with throwaway addresses than granting it instantly would be,
    # though with no payment-method-based KYC at all it's not a complete defense). In cents:
    # 200 = $2.00, enough for a real first try of chat/bot-builder/model-builder without a
    # card on file - the whole point being that someone with a good idea and no money yet
    # (the motivating case for this feature) can actually use the product before paying for
    # anything.
    signup_credit_cents: int = 200

    # Agent browser ("computer use" style agency - a bot drives a real, isolated headless
    # browser on request). See app/core/agent_browser.py's module docstring for the full
    # architecture and - importantly - its security trade-offs relative to the skills
    # sandbox. Off by default: this is the least-tested, highest-risk-surface feature in the
    # codebase (real network access from inside the container, by necessity - a browser has
    # to fetch real pages), and should be turned on only after someone has actually run a
    # session against it with a real Docker daemon and reviewed the result - see that file's
    # verification note.
    agent_browser_enabled: bool = False
    # Chat agent: how many previous messages of the conversation the LLM sees (0 = none), and
    # how many tool calls it may chain per user message. 1 = single step (old behaviour); raise
    # to 3-5 to let it chain e.g. analyze -> train -> publish. Each extra step is one more LLM
    # call counted against the user's quota, and published bots never chain regardless.
    agent_history_messages: int = 10
    agent_max_steps: int = 1
    # Priced per minute of wall-clock session time (not per action) since the cost driver is
    # a whole container + browser process running, not individual actions - closer to how
    # real cloud compute is billed. 5 cents/minute = $3/hour; charged at session close based
    # on actual duration (post-paid, like real cloud compute), with a minimum-balance gate at
    # session START so a near-empty account can't begin something it can't pay for.
    agent_browser_price_cents_per_minute: int = 5
    agent_browser_min_balance_cents: int = 50
    # Hard caps, both for cost-bounding and resource hygiene - an abandoned/forgotten session
    # must not run (and bill) forever, and a runaway action loop must not either.
    agent_browser_max_session_minutes: int = 10
    agent_browser_max_actions_per_session: int = 60
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
    # Resend (https://resend.com) - the simplest way to send real e-mail. Put the key ONLY in .env (RESEND_API_KEY), never in
    # code or chat. If set it is used instead of SMTP. email_from must be an address on a domain you verified in Resend
    # (resend.com/domains); the sandbox sender onboarding@resend.dev delivers only to your own Resend account address.
    # Empty email_from falls back to smtp_from.
    resend_api_key: str = ""
    email_from: str = ""
    # At most this many verification / password-reset e-mails per address per hour (stops e-mail bombing of a victim and
    # protects the provider's daily quota).
    email_per_address_per_hour: int = 3
    # Bot protection (see app/core/antibot.py). Cloudflare Turnstile is free: create a widget at dash.cloudflare.com ->
    # Turnstile and put both keys in .env. The captcha is on only when BOTH are set.
    turnstile_site_key: str = ""
    turnstile_secret_key: str = ""
    antibot_min_form_seconds: float = 3.0        # a human needs a few seconds to fill the form; scripts do not
    antibot_block_disposable: bool = True        # refuse throw-away e-mail services at registration
    disposable_email_domains_extra: str = ""     # comma-separated extra domains to refuse
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
