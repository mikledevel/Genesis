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

Render Free sleeps after 15 minutes without requests and has 512 MB RAM. SQLite,
accounts, bot configurations, uploads and trained models on the local filesystem
are lost when the instance restarts or redeploys. This configuration is a demo,
not durable hosting for customer data. `DATABASE_URL` is currently unused by the
SQLite implementation, so setting a PostgreSQL URL alone does not fix persistence.
Heavy model training can exceed the free instance's memory budget. Generated code
execution needs a separate Docker daemon and is unavailable in this configuration.

Deploy source without `.env`, `genesis.db`, local agent stores, model artifacts,
uploaded datasets or generated files. A clean instance initializes empty stores.
Verify `/health`, `/`, `/builder`, registration and an AI request after deployment.

See [Render Free](https://render.com/docs/free) and
[FastAPI deployment](https://render.com/docs/deploy-fastapi).
