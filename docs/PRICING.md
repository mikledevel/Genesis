# GenesisAI plans: what is enforced, what it costs, what is still manual

Source of truth in code: `app/core/plans.py` (change numbers there, nothing else). Public list: `GET /api/plans`.
Everything in the table is checked by the server; nothing is promised that is not enforced.

| | Free | Pro ($19/mo) | Business ($79/mo, by arrangement) |
|---|---|---|---|
| LLM tokens / month | 300,000 | 5,000,000 | 25,000,000 |
| Bots | 2 | 10 | 50 |
| Saved models | 3 | 25 | 100 |
| API keys | 1 | 5 | 20 |
| Calls per API key (cap on a new key) | 100 | 5,000 | 50,000 |
| Dataset upload size | 5 MB | 25 MB | 25 MB (platform cap) |
| AI-written custom code models (sandbox) | no | yes | yes |

Business is hidden from the public list until it has something Pro does not; an admin can still assign it.
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
- No automatic subscription checkout. Sell by invoice / payment link, then set the plan in the admin panel (`/admin`).
  Stripe is wired only for pay-per-use balance top-ups; a subscription webhook that sets `users.plan` is the next step
  and was NOT built or tested (no Stripe keys or network in the dev sandbox).
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
