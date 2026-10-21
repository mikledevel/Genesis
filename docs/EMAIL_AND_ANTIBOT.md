# E-mail (Resend) and bot protection: setup and limits

## 1. Real e-mail through Resend
Everything the platform e-mails goes through one function (`app/core/email_sender.py`): verification, password reset and the
scheduler's notifications ("skill waits for your approval", "job paused", "job update"). With `RESEND_API_KEY` set they are
really sent; without it nothing is sent and the message (with its link) is printed to the server log.

1. resend.com -> API Keys -> create a key. A key limited to "Sending access" is enough: the platform never calls anything else.
2. resend.com -> Domains -> add a domain you own (a subdomain such as `mail.yourdomain.com` is recommended), add the DNS records
   Resend shows (SPF, DKIM), wait until it says verified.
3. `.env` (never in code, never in chat; if a key was ever pasted anywhere, revoke it and make a new one):
   ```
   RESEND_API_KEY=re_...
   EMAIL_FROM=Genesis AI <no-reply@mail.yourdomain.com>
   PUBLIC_BASE_URL=https://your-real-site
   ```
   `PUBLIC_BASE_URL` matters: the links in the mails are built from it; empty = `http://127.0.0.1:8000`, which is useless to users.
4. Restart, log in as the owner, open `/admin`, press "Send a test e-mail to me". The panel shows the provider's real answer
   (e.g. `HTTP 403: domain not verified`).

Until a domain is verified Resend only lets you send from `onboarding@resend.dev`, and only to the address of your own Resend
account: fine to see it work (`EMAIL_FROM=Genesis AI <onboarding@resend.dev>`), useless for real users.
Limits worth knowing (Resend docs/guides, Oct 2026; check your dashboard): the free plan is 100 e-mails/day and 3,000/month, and the
API allows only a few requests per second (sources say 2 to 10). One signup = one e-mail, so about 100 signups a day is the free
ceiling. Failed sends are retried (429 / 5xx / network: 3 attempts, honouring Retry-After); a 4xx (bad key, unverified domain) is not.
Verification and reset mails are sent from a background thread: a slow provider never stalls the server, and the response time never
reveals whether an address has an account. A failure is logged, never shown to the user, and never breaks registration.

## 2. What protects registration and password reset
| Layer | Stops | Does NOT stop |
|---|---|---|
| Hidden honeypot field | naive form-filling scripts | anything that talks to the API directly |
| Minimum form time (3 s) | instant submits from the page | a script that omits or fakes the field |
| Disposable e-mail list | well-known throw-away services | new ones (the list is partial; add more in `DISPOSABLE_EMAIL_DOMAINS_EXTRA`), real mailboxes |
| Cloudflare Turnstile (server-verified) | the bulk of automated signups; this is the real gate | a person (or a paid human farm) |
| One mailbox = one account (`a.b+x@gmail.com` is `ab@gmail.com`) | farming free tokens with aliases | different real mailboxes |
| 3 verification / reset mails per address per hour | e-mail bombing a victim, burning the Resend quota | - |
| Existing: 10 register/login tries per minute per IP, e-mail verified before any signup credit | brute force from one address | a distributed attack |

Turnstile is free. dash.cloudflare.com -> Turnstile -> Add site (your domain) -> copy both keys:
```
TURNSTILE_SITE_KEY=...      # public, sent to the browser
TURNSTILE_SECRET_KEY=...    # secret, server only
```
The captcha is on only when BOTH are set (one alone = off, with a warning in the log). If Cloudflare cannot be reached the request
is refused with "try again" (fail closed), not waved through. For local testing Cloudflare provides dummy keys: site key
`1x00000000000000000000AA` with secret `1x0000000000000000000000000000000AA` always pass, `2x0000000000000000000000000000000AA`
always fails. **Never put the dummy secret in production: it accepts any token.**

Not covered: login (only the per-IP rate limit protects it; a captcha after repeated failures would be the next step), the
resend-verification and chat endpoints (they need a session), and a human who solves the captcha by hand.
