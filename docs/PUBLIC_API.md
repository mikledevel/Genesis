# Genesis AI Public API

A token-metered chat completions API for integrating Genesis AI into your own product or
backend. Shaped like OpenAI's Chat Completions API on purpose - if you already have code
using the `openai` or `groq` Python/JS SDKs, you can usually point it at Genesis AI by
changing only the base URL and API key.

## Getting an API key

1. Sign in to Genesis AI.
2. Go to **Settings → API Keys** (or `POST /api/keys/create` if you're integrating
   programmatically) and create a key. Keys look like `gen-xxxxxxxxxxxxxxxxxxxxxxxx`.
3. Add funds to your account balance: **Settings → Balance**, or `POST
   /api/payments/checkout`. The public API bills against this same balance - there's no
   separate subscription or invoice.

## Authentication

Send your key either way:

```
Authorization: Bearer gen-xxxxxxxxxxxxxxxxxxxxxxxx
```

or

```
X-API-Key: gen-xxxxxxxxxxxxxxxxxxxxxxxx
```

The `Authorization: Bearer` form is what the official OpenAI and Groq SDKs send by default,
so it's the one to use if you're pointing an existing SDK client here (see examples below).

## Endpoint: chat completions

```
POST /api/v1/chat/completions
```

**Request body**

| Field         | Type            | Required | Notes                                                        |
|---------------|-----------------|----------|----------------------------------------------------------------|
| `messages`    | array           | yes      | `[{"role": "user"/"system"/"assistant", "content": "..."}]`   |
| `model`       | string          | no       | Defaults to `openai/gpt-oss-120b`. See **Models** below.      |
| `max_tokens`  | integer         | no       | Defaults to 1024. Hard cap: 4096.                             |
| `temperature` | number          | no       | Defaults to 0.7.                                              |

**Example**

```bash
curl https://your-genesis-instance.example.com/api/v1/chat/completions \
  -H "Authorization: Bearer gen-xxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 100
      }'
```

**Using the official OpenAI Python SDK instead of raw HTTP:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="gen-xxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://your-genesis-instance.example.com/api/v1",
)
resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=100,
)
print(resp.choices[0].message.content)
```

**Response**

```json
{
  "id": "genesis-8f2a1c9d3e4b5a6f7c8d9e0f",
  "object": "chat.completion",
  "created": 1735689600,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "Hello there!"}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
  "genesis_billing": {"charged_cents": 1, "balance_cents_remaining": 4999}
}
```

`choices`, `usage`, and the top-level `id`/`object`/`model` fields follow the OpenAI chat
completions shape. `genesis_billing` is a Genesis-specific extra field (any OpenAI-compatible
client just ignores fields it doesn't recognize) showing exactly what this call cost and what
you have left, since billing here is prepaid-balance-based rather than a monthly invoice.

## Models

| Model id                     | Notes                                   |
|-------------------------------|------------------------------------------|
| `openai/gpt-oss-120b`         | Default. Best general-purpose quality.   |
| `openai/gpt-oss-20b`          | Smaller/faster, lower cost per token isn't different (see Pricing) but responses are typically quicker. |
| `llama-3.3-70b-versatile`     | Alternative general-purpose model.       |
| `qwen/qwen3.6-27b`            | Alternative general-purpose model.       |

Requesting any other model id returns `400 Bad Request` with the list of supported ids, rather
than an opaque failure from the underlying provider.

`GET /api/v1/models` returns this same list programmatically (and doubles as a cheap way to
confirm your key is valid before wiring up real calls).

## Pricing

Billed per token, per call - no subscription, no minimum spend beyond the 1-cent-per-call
floor below.

| | Per 1M tokens |
|---|---|
| Input (prompt) | $0.50 |
| Output (completion) | $2.00 |

Every billed call has a **1-cent minimum**, even if the token-based cost would round to less -
most short chat messages will land on this minimum rather than a fraction of a cent.

**How you're actually charged:** before generating a response, we place a hold equal to the
worst-case cost of your request (your prompt's estimated size, plus `max_tokens` as if fully
used). Once the real response is generated, we true that up to the *real* cost based on actual
token usage and refund the difference - so setting a generous `max_tokens` "just in case"
doesn't cost you extra; you only ever pay for tokens actually used. `genesis_billing.charged_cents`
in every response is the final, real amount for that call.

**Examples at default pricing:**

| Request | Approx. cost |
|---|---|
| Short chat message (~50 in, ~100 out tokens) | 1¢ (minimum) |
| Medium request (~2,000 in, ~1,000 out tokens) | ~3¢ |
| Long document analysis (~20,000 in, ~2,000 out tokens) | ~14¢ |

## Errors

| Status | Meaning |
|---|---|
| 400 | Malformed request (empty `messages`, unsupported `model`, invalid `max_tokens`) |
| 401 | Missing or invalid API key |
| 402 | Insufficient balance for this call's worst-case cost - top up and retry |
| 451 | Request blocked for export-control/sanctions reasons (see below) |
| 502 / 504 | The AI provider failed to generate a response - you were NOT charged (full refund is automatic); retry |

## Availability restrictions

In compliance with applicable export control and sanctions requirements, Genesis AI - the
public API included - is not available to users connecting from Russia, Belarus, Cuba, Iran,
North Korea, Syria, or the Crimea, Donetsk, Luhansk, Zaporizhzhia, or Kherson regions of
Ukraine. Requests from these locations receive `451 Unavailable For Legal Reasons`. See
`app/core/geo_block.py` for the detection approach and its known limitations if you're
operating your own deployment of this platform.

## Checking your usage

```
GET /api/v1/usage
```

Authenticate with your normal platform login (`Authorization: Bearer <session token>`, the
same one the web app uses - not an API key) to see aggregate spend and token counts across
all of your API keys, plus your most recent individual calls.
