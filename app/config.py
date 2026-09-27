from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Genesis AI"
    debug: bool = False
    database_url: str = ""
    redis_url: str = ""
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
    # Request timeout (seconds) and retry count for every outbound Groq call across the
    # codebase (agent.py, bot_builder.py, ai_generator.py, skill_generator.py). The Groq SDK
    # previously ran with its own defaults nowhere overridden - when the key is missing,
    # invalid, or the provider is unreachable, that let a single call hang for well over a
    # minute (observed: 60-70s) before finally surfacing as a 502/503/504 to the caller,
    # tying up a worker the whole time. A short explicit timeout + a single retry makes a
    # genuine provider outage fail fast and predictably instead.
    groq_request_timeout_seconds: float = 20.0
    groq_max_retries: int = 1
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
    # Geo-blocking - see app/core/geo_block.py for the full detection/fail-open policy.
    # Off by default so cloning/forking this repo doesn't silently block anyone; the actual
    # product decision to block Russia is applied via .env, not hardcoded here.
    geoblock_enabled: bool = False
    geoblock_countries: str = "RU"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
