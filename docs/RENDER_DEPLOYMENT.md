# GenesisAI on Render Free

The `render.yaml` Blueprint explicitly selects `plan: free`. It runs the FastAPI
application with one Uvicorn worker, production checks enabled and a generated
JWT signing secret. The startup script derives the public URL and CORS origin
from `RENDER_EXTERNAL_URL`, which Render provides automatically.

## Required configuration

Connect the deployment branch on GitHub to a Render Blueprint. Set these secret
environment variables in Render, never in Git:

- `GROQ_API_KEY`: the provider key used by the AI features.
- `RESEND_API_KEY` and `EMAIL_FROM`: a verified sender for account verification
  and password reset emails. SMTP variables from `.env.example` are an alternative.
- `ADMIN_EMAILS`: the owner's email address; admin access requires verification.

Keep `SECRET_KEY` stable across subsequent deployments. Optional fallback provider
keys and other settings are documented in `.env.example`. Payment keys and Docker
browser execution are not required to launch the demo.

## Free-tier limitations

The local filesystem is ephemeral. Set `DATABASE_URL` to a persistent PostgreSQL
database for accounts, configuration documents, billing, conversations, queues and
supported file artifacts. Without it, SQLite and local files can be lost on restart
or redeploy. Existing local files must be transferred explicitly. See
[PostgreSQL configuration](POSTGRES.md) and [migration and backups](BACKEND_HARDENING.md).
Heavy model training can exceed a small instance's memory budget. Generated code
and external model execution require an isolated Docker runner; those operations
fail closed in production when it is unavailable. Free hosting is intended for a demo.

Keep `SECRET_KEY` and `INTEGRATION_ENCRYPTION_KEY` stable. Startup runs
`scripts/migrate.py` before serving requests, then disables automatic DDL. Configure
`TRUSTED_PROXY_CIDRS` for actual proxy peers only. Use `/ready` as the deployment
readiness check; `/health` checks process liveness.

Deploy source without `.env`, `genesis.db`, local agent stores, model artifacts,
uploaded datasets or generated files. A clean instance initializes empty stores.
Verify `/health`, `/`, `/builder`, registration and an AI request after deployment.

See [Render Free](https://render.com/docs/free) and
[FastAPI deployment](https://render.com/docs/deploy-fastapi).
