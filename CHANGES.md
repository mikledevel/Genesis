# Lead-dev pass — Aug 24, 2026

## Fixed (correctness / security)

1. **Broken error responses (`app/main.py`)** — 5 endpoints returned `{"error": ...}, 404/401/429/500`
   as a bare tuple. FastAPI serializes that as a JSON array with an HTTP **200**, so callers
   checking the status code never saw the failure. Replaced with `raise HTTPException(...)`.
   Affected: `/api/models/{id}/predict`, `/api/agents/{id}`, `/api/agents/{id}/clone`.

2. **API key sent as a query parameter** on `/api/models/{model_id}/predict` — leaks into
   server logs, proxies, and browser history. Now read from the `X-API-Key` header, with the
   old query param kept as a deprecated fallback so existing integrations don't break.

3. **`secret_key` could silently stay `"change-me"` in production** — every JWT would be
   forgeable. The app now refuses to start if `debug=False` and `secret_key` is still the
   placeholder.

4. **Unisolated code-execution fallback** (`app/core/codegen/ai_generator.py`) — when Docker
   isn't available, the AI-generated ML pipeline code fell back to a bare `subprocess.run`
   with no network/resource isolation, and would do so silently in any environment. It's now
   gated on `debug=True`; in production it raises instead of running untrusted code unsandboxed.

## Repo hygiene

- `.gitignore` now covers `generated/`, `genesis.db`, `chats.json`, `projects.json`,
  `.pytest_cache/` — these were dev-local artifacts sitting in the repo.
- Removed the ~100 stale auto-generated pipeline scripts under `generated/`.

## Not yet done (see prior conversation for full rationale)

- Frontend: `static/index.html` is a 3,700-line single-file monolith. Recommend splitting into
  components once a build step is introduced — didn't do this here since it's a structural
  rewrite, not a bug fix, and deserves its own pass with tests around each extracted piece.
- SQLite → Postgres migration path for real concurrent-write scale.
- Rate limiting at the FastAPI layer (currently only the Groq-token quota is enforced;
  nothing throttles raw request volume per user/IP).

---

# Round 2 — same day, continued

## Added

1. **Request-volume rate limiting** (`app/core/rate_limit.py`) — new middleware, no new
   dependency. Two tiers: a tight 10/min/IP limit on `/api/auth/login` and
   `/api/auth/register` (brute-force / credential-stuffing / signup-spam protection), and a
   looser 120/min limit on the rest of `/api/*`, keyed by bearer token when present so users
   behind the same NAT/proxy don't throttle each other. In-memory sliding window — correct for
   the current single-process deployment; documented in the module docstring as needing a
   Redis-backed store (via the already-declared but unused `settings.redis_url`) if this ever
   runs as multiple workers/processes. Unit-tested the core sliding-window logic directly
   (bypassing the unavailable `starlette` import in this sandbox) — 10-then-block-then-recover
   behavior confirmed correct.
2. **SQLite hardened for concurrency** (`app/db/database.py`) — `_connect()` now sets
   `journal_mode=WAL` (readers no longer block writers) and `busy_timeout=5000` (collisions
   retry for 5s instead of raising `database is locked` immediately). Verified against a real
   throwaway SQLite file: PRAGMA values confirmed applied, and a full `GenesisDB` smoke test
   (create user, create chat, add message, read back) passed with the change in place. This is
   a mitigation, not a substitute for Postgres — see below.
3. **`docs/POSTGRES_MIGRATION.md`** — concrete migration plan (SQLAlchemy Core, not raw
   dialect-specific SQL, not a full ORM rewrite; batch order; what to test at each step).
   Written as a plan rather than executed code because this sandbox has no network access and
   no Postgres instance to actually run a migration against — shipping a rewrite of a 68-method
   data layer that I could only syntax-check, never execute, would trade a real known
   limitation (SQLite's single-writer ceiling, now mitigated) for an unverified one.

## Verification notes for this round
No `fastapi`/`starlette` installed in this sandbox and no network access to install them, so
the middleware itself could only be syntax-checked (`py_compile`), not run end-to-end through
a real ASGI request. I unit-tested its core rate-limiting logic in isolation (extracted and
exec'd independent of the starlette import) to at least verify the algorithm, and picked
middleware ordering (CORS outermost, wrapping the rate limiter) by reasoning through Starlette's
documented last-added-is-outermost behavior rather than by running it — flagging this
explicitly rather than presenting it as test-verified when it isn't. Worth an actual `TestClient`
smoke test before this goes to production.
