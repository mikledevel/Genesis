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
    # Platform-wide default monthly Groq token cap per user, checked BEFORE each internal
    # Groq call is made (see app.core.quota) - protects the platform's own shared
    # GROQ_API_KEY from unbounded per-user cost, now that real usage is actually tracked
    # (see GenesisDB.log_groq_usage). 0 or negative disables the cap entirely (unlimited).
    # A per-user override can be set in the DB (users.monthly_groq_token_limit) for cases
    # that need a different limit than this platform default.
    default_monthly_groq_token_limit: int = 2_000_000
    # Public HTTPS base URL this server is reachable at, used to register the Telegram
    # webhook (see app.core.telegram_integration / the /api/agents/{id}/telegram/* routes in
    # main.py). Telegram requires a real public HTTPS URL - it cannot deliver to localhost.
    # Left empty by default (e.g. local dev); a bot can still be connected and tested via the
    # built-in test-chat endpoint without this set, it just won't receive live messages until
    # this is configured and the bot is reconnected.
    public_base_url: str = ""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
