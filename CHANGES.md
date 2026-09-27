# Round 4 — Aug 29, 2026 — Frontend redesign (pre-auth landing, auth UX, design system)

## Note on this round's starting point
This zip's `CHANGES.md` and `app/core/models_catalog.py` reflect the state through **Round 2**
only — the Round 3 Groq-deprecation fixes described in prior conversation (`deprecated`/
`replacement_id` fields, `resolve_active_model()`, the `MODEL_MAP` identity-mapping fix) are
**not present in this codebase**. Flagging this explicitly rather than silently reconciling it,
since it means whichever copy has those fixes should be treated as the canonical one going
forward — worth checking before deploying either.

## Critical bug found and fixed before any redesign work started

1. **The entire frontend `<script>` block was one missing line away from being dead code**
   (`static/index.html`) — the closing brace of `handleEmailLinkParams()` was immediately
   followed by the *body* of `doRegister()`, with no `async function doRegister() {` line
   ever declared. A `<script>` tag with a syntax error doesn't execute at all in a browser —
   so **no JS on the page ran**: no `checkAuth()`, no login, no registration, no panel
   navigation, nothing. Confirmed via `node --check` (`SyntaxError: await is only valid in
   async functions...`), then fixed by restoring the missing function declaration and
   re-verified clean. This should ship immediately regardless of anything else in this round.

2. **Two other modals silently shared DOM ids** (`np-name`/`np-desc` used by both the Prompt
   Lab "new variant" modal and the "New Project" modal). `getElementById` always resolves to
   the first match in the DOM, so `createProject()` was reading whatever was typed in the
   Prompt Lab modal (usually nothing) instead of the actual New Project form — new projects
   were silently created with an empty name/description. Renamed the New Project modal's
   fields to `projn-name`/`projn-desc` and updated `createProject()` to match.

## Redesign — biggest ask: users no longer have to log in to see the product

- **New pre-auth landing/explore experience** (`#landing-screen`) replaces the old behavior of
  immediately blocking the whole page behind a full-screen login form. Flow is now
  Discover → Explore → Try → Register:
  - Hero + feature overview describing the three real product surfaces (Agent Builder,
    AutoML, Marketplace).
  - A **live "Explore" section** backed by real, currently-public endpoints
    (`/api/marketplace/stats`, `/api/marketplace/listings`, `/api/agents/models`) — actual
    marketplace listings and the real Groq model catalog, not placeholder content. When the
    marketplace has zero listings (it does, right now — `marketplace/listings.json` is empty),
    the empty state says so honestly instead of faking data, and the model catalog still gives
    real content to browse. Clicking any card opens the register flow.
  - A concrete "how it works" section and a closing CTA.
- **Auth is now a dismissable overlay, not a wall.** `#auth-screen` is opened from explicit
  "Sign in" / "Start building free" buttons, closable via an X button, backdrop click, or
  Escape, and the landing page stays mounted underneath it.
- **Registration is now a 2-step flow** instead of one flat form: step 1 (name/email/password,
  with live password-strength meter and show/hide toggle) → step 2, an optional "what are you
  building?" quick-pick (agent / AutoML / marketplace / just exploring) that's stored client-side
  and used to open the right panel on first login — a real personalization, not a cosmetic one.
  A step-3 success screen confirms the account and that a verification email was sent before
  handing off into the workspace.
- Added password visibility toggles to login, register, and the "set new password" reset modal
  (none existed before).
- Inline field-level validation (red border + message) on the register form instead of only a
  single error line at the bottom.

## Design system
- Added `Inter`/`Inter Tight` for real typographic hierarchy (headings vs. body), a motion
  token set (`--ease`, `--dur`, `--dur-fast`) used for modal/auth transitions, larger radius
  tokens for the new marketing surfaces, and a dark-mode-aware accent gradient.
- Raised `--txt3`'s contrast in light mode (`#9C9CA0` → `#75757A`) — the old value sat under
  3:1 against white for text that size, which fails WCAG AA for normal text.
- Added global `:focus-visible` rings, a skip-to-content link, and `aria-label`s on every
  icon-only nav button (they previously had empty `title=""` attributes and were invisible to
  screen readers/keyboard users beyond a hover tooltip).

## Responsive layer (there was none before — zero `@media` queries in the whole app)
- Sidebar nav collapses to a bottom tab bar under 640px.
- The agent conversation sidebar becomes an off-canvas drawer (toggle button in the topbar)
  under 820px instead of permanently eating horizontal space.
- Stat/feature grids collapse from 3–4 columns down to 2 then 1 as width shrinks; modals cap
  at 94vw; the auth box and landing sections get mobile-appropriate padding.
- This covers the shell and the highest-traffic surfaces (landing, auth, nav, dashboard grids).
  It does **not** yet cover every specialized builder screen (pipeline canvas, bot preview
  dock, registry table) — those need their own mobile pass, noted below.

## Verification
No `fastapi`/browser/network access in this sandbox, so verification was: `node --check` on
the full extracted `<script>` block (clean), a CSS brace-balance check (470/470), a duplicate-id
sweep across the whole file (now zero real duplicates — the one remaining `chat-empty` "duplicate"
is a static placeholder replaced wholesale by a JS template at runtime, so only one ever exists
in the live DOM), and manual review of every edited region. **Not** verified: actual rendering,
click-through, or responsive behavior in a real browser — this is CSS/JS reasoned through, not
visually tested, and deserves a real browser pass before shipping to production.

## Not yet done
- `static/builder/index.html` (the standalone pipeline-builder page) wasn't touched — it's a
  separate, already-reasonably-styled dark-mode tool; a design-system pass to match the new
  tokens is a reasonable follow-up but was out of scope to keep this round's blast radius
  contained to the main shell.
- Per-panel responsive treatment (pipeline canvas, registry table overflow, bot preview dock)
  beyond the shell-level fixes above.
- Reconciling this codebase with whichever copy has the Round 3 Groq-deprecation fixes (see
  note at the top) before either goes to production.

---

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
