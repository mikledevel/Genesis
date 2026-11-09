# Direct USDT payments on TRON

GenesisAI accepts native USDT (TRC20) on TRON mainnet to the owner's configured
receiving address. Pro is listed at 19 USDT and Business at 79 USDT per 30 days.
Each invoice has a unique fractional discount below 0.01 USDT. The exact six-decimal
amount is displayed before payment and must arrive in full, excluding wallet/exchange
withdrawal fees. Invoice amounts are never reused for the same receiving address.

## Customer flow

1. Register and verify your email, then open Settings → Plan.
2. Choose Pro or Business. Reopening an unpaid invoice returns the same invoice.
3. Scan the receiving-address QR code or copy the address. Select USDT on TRON
   (TRC20) in your wallet, enter the exact invoice amount, and send once. The QR
   contains the address only; it does not prefill the token or amount.
4. Leave the payment window open. The server detects the transfer and activates
   access automatically after blockchain confirmation. No TXID entry is required.
   An optional TXID field is available under Advanced for recovery.
5. A verified transfer grants 30 days. Renewing the same plan adds 30 days to the
   current access end time. Renewals are manual, and switching plans waits until
   the current paid period ends.

The payment window is one hour. A payment made during that window can still be
confirmed after the window closes. Late, short, rounded, wrong-token, or wrong-chain
payments are not automatically credited; support must investigate the original
TXID before requesting any additional transfer.

## Verification and storage

The server uses the mainnet `https://api.trongrid.io/walletsolidity/` API. Only a
solidified successful receipt containing a Transfer event from the official USDT
contract `TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t` with the exact recipient, amount, and
invoice time window is accepted. A user-provided TXID alone grants nothing.
An invoice belongs to one authenticated account, and a transaction can credit only
one invoice. Database transactions and user row locks prevent replay and races.

No private key, seed phrase, signing, broadcasts, custody, or automatic refunds are
implemented. USDT payments grant plan access and do not top up marketplace balances.
Fiat conversion and local business/tax obligations remain separate from this flow.

Set these server-only variables on Render:

- `DATABASE_URL`: dedicated private PostgreSQL login, TLS enabled.
- `PUBLIC_BASE_URL`: public HTTPS origin.
- `CRYPTO_PAYMENTS_ENABLED=true`
- `CRYPTO_USDT_TRON_ADDRESS`: confirmed receiving address.
- `STRIPE_SUBSCRIPTIONS_ENABLED=false` while Stripe onboarding is incomplete.
- `TRONGRID_API_KEY`: recommended for production quotas; sent only in the
  `TRON-PRO-API-KEY` request header. Free unauthenticated mainnet reads were verified
  during setup, but can be throttled. A node outage fails closed, blocks new invoice
  creation, and leaves existing transfers available for a later verification retry.

Use a free TronGrid account/API key for reliable production traffic. The official
[TronGrid access guidance](https://developers.tron.network/docs/exchangewallet-integrate-with-the-tron-network)
requires an API key for production traffic. Never share wallet private keys.

The first release caps allocation at 9,999 never-reused invoice amounts per listed
price and receiving address. Exhaustion blocks new invoices before any payment is
requested; monitor allocation and plan a receiving-address transition before that
limit. Per-user creation is capped at five new invoices per day. Verification is
throttled to one attempt per invoice per ten seconds. Recent history shows ten invoices.

Sources: [Tether contract](https://tether.to/en/supported-protocols/),
[TRON confirmation semantics](https://developers.tron.network/reference/gettransactioninfobyid-1).

## Automatic detection

A separate lightweight monitor scans unpaid invoices every 30 seconds while the
application runs. The payment window also checks every 20 seconds while visible.
Both paths share atomic database throttling and cached TRON history lookups.
Discovery uses confirmed, incoming USDT index records with complete bounded
pagination; each candidate is independently verified using its solidified receipt.
Index metadata alone cannot grant access. No private keys or third-party checkout
accounts are needed. A customer still authorizes the transfer in their own wallet.

The monitor revisits invoices from the last seven days. Older invoices can still be
checked from the payment window or with the optional transaction-ID recovery flow.
The service generates QR SVGs locally and exposes them only to the invoice owner.
No invoice/address is sent to an external QR image service.

Render Free can sleep when idle. Monitoring resumes when the application wakes;
PostgreSQL keeps unpaid invoices and payments intact. Opening the payment window
triggers automatic recovery, so the owner does not manually approve transfers.

History API: https://developers.tron.network/reference/get-trc20-transaction-info-by-account-address
