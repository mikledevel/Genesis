"""Typed exceptions for LLM-provider failures.

Previously, any failure calling the Groq API (missing key, rate limit, timeout,
network error, malformed response) was swallowed and replaced with a hardcoded
"successful" reply (see GenesisAgent._call_local before this fix). That made a
provider outage indistinguishable from a real answer to both the end user and to
anyone monitoring the platform, and meant marketplace buyers could be charged for
a message the model never actually generated.

These exception classes let call sites (agent.py) and route handlers (main.py)
tell these failure modes apart and respond appropriately - e.g. a rate limit
should probably be retried by the client, a config error should page an operator,
and none of them should ever be billed to a user as a successful generation.
"""


class LLMError(Exception):
    """Base class for all LLM-provider related failures."""
    #: Default HTTP status code a route handler should map this to.
    http_status = 502


class LLMConfigError(LLMError):
    """API key/credentials are missing or invalid. This is an operator-facing
    configuration problem, not something the end user can fix by retrying."""
    http_status = 503


class LLMRateLimitError(LLMError):
    """The provider rejected the request due to rate limiting (ours or theirs)."""
    http_status = 429


class LLMTimeoutError(LLMError):
    """The request to the provider timed out."""
    http_status = 504


class LLMProviderError(LLMError):
    """A generic upstream provider failure: 5xx response, connection error,
    or a response that couldn't be parsed as the expected JSON tool call."""
    http_status = 502
