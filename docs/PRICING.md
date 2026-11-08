# GenesisAI plans: what is enforced, what it costs, what is still manual

Source of truth in code: `app/core/plans.py` (change numbers there, nothing else). Public list: `GET /api/plans`.
Everything in the table is checked by the server; nothing is promised that is not enforced.

| | Free | Pro ($19/mo) | Business ($79/mo) |
|---|---|---|---|
| LLM tokens / month | 300,000 | 5,000,000 | 25,000,000 |
| Bots | 2 | 10 | 50 |
| Saved models | 3 | 25 | 100 |
| API keys | 1 | 5 | 20 |
| Calls per API key (cap on a new key) | 100 | 5,000 | 50,000 |
| Dataset upload size | 5 MB | 25 MB | 25 MB (platform cap) |
| Knowledge-base documents per bot | 3 | 25 | 100 |
| Knowledge-base text per bot (characters, all documents together) | 200,000 | 5,000,000 | 25,000,000 |
| Knowledge-base passages per answer (what the bot gets to read for each question; the bot owner's plan decides) | 3 | 6 | 8 |
| Active scheduled jobs | 1 | 10 | 50 |
| A/B tests (stopped ones count until deleted) | 2 | 20 | 100 |
| Saved prompts | 10 | 100 | 500 |
| AI-written custom code models (sandbox) | no | yes | yes |
| Server training jobs per month (train, forecast, retrain, improve, create model: each start counts; the count restarts on the 1st) | 0 (no server training) | 40 | 300 |
| Model improvement search depth (`improve_model`) | - | balanced | thorough |
| Training on the platform's servers (train, forecast, retrain, **improve a weak model**, create model) | no | yes | yes |
| Training on your own computer (`scripts/train_local.py`, zip from `GET /api/local-training/bundle`) | yes | yes | yes |

Counts (documents, jobs, tests, prompts) are checked when something is CREATED; deleting frees the slot. A server training job counts when
it starts, even if it later fails on bad input; training on your own computer is never counted. A knowledge-base upload is refused when it
would take the bot over its text limit, and a file that is larger than four bytes per allowed character is refused without being read whole.

Why server training is paid: it spends the platform's CPU and memory (tuning can run minutes per model), the one cost that
grows with every model a user trains. Training on the user's own machine costs the platform nothing, so it stays free: the zip
carries the same trainer the server uses, and the data and the model never leave the user's computer. A locally trained model can be brought back: the platform cuts the training file and keeps the test rows (`/api/local-training/prepare`), the user
trains on train.csv, uploads model.pkl, and the platform scores it itself before registering it (free on every plan, saved-model slots count). Such a model can be listed on the marketplace by its author; the listing shows the platform-measured
scores and the label "trained on the owner's own computer, not by the platform".

Business is shown on the public list and can be purchased through Stripe Checkout.
Owner accounts (verified e-mail in `ADMIN_EMAILS`) have no token or count limits. A per-user token override set in the
admin panel beats the plan (0 = unlimited, empty = plan default).

## Why these numbers (cost side)
LLM price of the default model, checked Oct 2026: gpt-oss-120b on Groq, $0.15 per 1M input and $0.60 per 1M output tokens.
Assume 70% input / 30% output = about $0.29 per 1M tokens blended; $0.60 is the worst case (all output).

| Plan | Token cost at full use (blended / worst) | Price | Payment fees (assumed 2.9% + $0.30) | Left before hosting, CPU, support, tax |
|---|---|---|---|---|
| Free | $0.09 / $0.18 | $0 | - | -$0.09 (marketing cost) |
| Pro | $1.43 / $3.00 | $19 | $0.85 | about $16.7 / $15.2 |
| Business | $7.13 / $15.00 | $79 | $2.59 | about $69 / $61 |

Not in these numbers: hosting, bandwidth, CPU time of AutoML jobs, VAT, support time, and free accounts that never pay.
Most accounts will not use their full quota, so the real cost per account is lower than the table.

## Why these prices (demand side): a hypothesis, not a fact
$19 is a guess for a small shop that plans stock from its data. Nobody has paid yet, so the first job is to find out:
ask pilot users what they pay today for the same job (a spreadsheet freelancer, an analyst hour, a tool) and price against
that. Also test the one-off paid report ($100-300 per forecast, see `scripts/make_report.py`) next to the subscription.

## What is manual today
- Stripe subscriptions use hosted Checkout for Pro and Business, with subscription webhooks updating `users.plan`.
  The deployer must configure `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, both monthly price IDs, and
  `PUBLIC_BASE_URL` with the public HTTPS origin. Register `POST {PUBLIC_BASE_URL}/api/payments/webhook` in Stripe
  and subscribe it to `checkout.session.completed` and `customer.subscription.created`, `updated`, `deleted`,
  `paused`, and `resumed`. Checkout stays hidden until the required keys, prices, and production HTTPS URL exist.
  GenesisAI currently does not collect tax; Stripe Tax and automatic tax are not enabled.
- Marketplace balance (cents) is separate money between users and authors. Plans do not touch it and admins get no free balance.
- Before launch: the LLM provider key must be on a paid tier. One third-party pricing index lists a free-tier ceiling of
  about 200K tokens/day for gpt-oss-120b on Groq: check your own console, it would cap the whole platform.

## Not built (never put on a pricing page yet)
team seats, forecast report export from the web app (the report exists as a script), priority support, white-label bots.

## Owner / admin
`ADMIN_EMAILS` (default in `app/config.py`: the owner's address; override in `.env`; replace before selling or sharing the
project). Admin = that e-mail AND verified. An owner e-mail is verified only from a logged-in session of the same account
(mailbox + password together). Register the owner account FIRST, before the site is public. If registration says "already
exists" for your own address, treat it as an attack: use "forgot password" (it voids the other person's password and
sessions), log in, then verify. The admin panel manages plans, token limits and suspensions only; it cannot read anyone's
data, impersonate, or mint balance; every change is in the audit log.
