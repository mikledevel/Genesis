"""
SafeFetcher - the ONLY way AI-generated skill code is allowed to reach the network.

This exists because a skill (e.g. "check this site for iPhone listings") needs real internet
access to do anything useful, but the code that provides that access was written by an LLM
from a natural-language request, run on a multi-tenant platform, on behalf of a user we've
never vetted. That combination - untrusted, AI-generated, network-capable code - is exactly
the SSRF (server-side request forgery) threat model: without this guard, a "skill" could be
tricked (deliberately or as an accidental side effect of a vague prompt) into making the
platform's own server fetch http://169.254.169.254/ (cloud metadata - often leaks real
credentials), http://localhost:PORT/ (the platform's own internal services), or an internal
company IP the operator never intended to expose to the internet.

Two independent checks, both required:
  1. Hostname allowlist - the skill's creator (at generation time) declares which domain(s)
     it's allowed to talk to (e.g. "ebay.com"). Anything else is rejected before a single
     byte is sent, regardless of what the IP resolves to.
  2. Resolved-IP blocklist - even for an ALLOWED hostname, we resolve it ourselves and refuse
     to connect if it resolves to a private/loopback/link-local/reserved address. This closes
     the "DNS rebinding" gap: a hostname can look innocent in an allowlist but be configured
     (by an attacker, or just by misconfiguration) to resolve to an internal IP.

This module provides the logic used by BOTH the real Docker sandbox (via the file-based
bridge in app/core/codegen/docker_sandbox.py - the container itself has no network device at
all; this class runs on the HOST and answers requests on the container's behalf) and, for
local single-developer testing without Docker, an in-process wrapper injected directly into
the generated code's execution namespace.
"""
import ipaddress
import socket
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from urllib.parse import urlparse

MAX_RESPONSE_BYTES = 2_000_000  # 2MB - skills read pages/API responses, not download files
REQUEST_TIMEOUT_SECONDS = 15
ALLOWED_SCHEMES = {"http", "https"}


class SkillNetworkError(Exception):
    """Raised for both policy violations (domain not allowed, private IP) and real network
    failures (timeout, DNS failure, connection refused). Skill code should treat this as "the
    fetch didn't work" and handle it gracefully, exactly like any other network error - it
    deliberately does NOT distinguish "you're not allowed" from "the server is down" in the
    exception type, so a skill can't be used to probe which internal hosts exist/are
    reachable based on different error messages (that would itself be an SSRF-adjacent
    information leak)."""
    pass


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable - refuse rather than guess
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified
        # Explicit belt-and-suspenders for the single most common real-world SSRF target:
        # cloud provider instance metadata, which serves real credentials over plain HTTP
        # with no auth to anything that can reach it.
        or ip_str.startswith("169.254.")
    )


def _resolve_all_ips(hostname: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(hostname, None)
        return list({info[4][0] for info in infos})
    except socket.gaierror:
        return []


def is_domain_allowed(hostname: str, allowed_domains: List[str]) -> bool:
    """A hostname is allowed if it exactly matches an allowlist entry, or is a subdomain of
    one (e.g. allowlisting "ebay.com" also permits "www.ebay.com" and "api.ebay.com") - but
    NOT the reverse (allowlisting "api.ebay.com" does not grant "ebay.com" or "evil-ebay.com",
    and "notebay.com" does not match "ebay.com" - the check is on dot-separated labels, not a
    raw substring, specifically to avoid that class of bypass)."""
    hostname = hostname.lower().rstrip(".")
    for allowed in allowed_domains:
        allowed = allowed.lower().rstrip(".")
        if hostname == allowed or hostname.endswith("." + allowed):
            return True
    return False


class SafeFetcher:
    def __init__(self, allowed_domains: List[str]):
        self.allowed_domains = allowed_domains

    def fetch(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
              body: Optional[str] = None) -> Dict:
        """Returns {"status": int, "text": str, "headers": dict} or raises SkillNetworkError.
        Deliberately synchronous and simple - skills are meant to make a handful of requests
        to check/gather data, not stream large payloads or hold long-lived connections."""
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            raise SkillNetworkError(f"URL scheme must be http or https, got: {parsed.scheme!r}")
        if not parsed.hostname:
            raise SkillNetworkError("URL has no hostname")
        if not is_domain_allowed(parsed.hostname, self.allowed_domains):
            raise SkillNetworkError(
                f"This skill is only allowed to contact: {', '.join(self.allowed_domains)}")

        resolved_ips = _resolve_all_ips(parsed.hostname)
        if not resolved_ips:
            raise SkillNetworkError(f"Could not resolve host: {parsed.hostname}")
        if any(_is_blocked_ip(ip) for ip in resolved_ips):
            # Fires for genuine DNS rebinding attempts AND for the mundane case of a
            # developer accidentally allowlisting "localhost" or similar - both should fail
            # the same way.
            raise SkillNetworkError(f"Host {parsed.hostname} resolves to a disallowed address")

        req = urllib.request.Request(
            url, method=method.upper(),
            data=body.encode("utf-8") if body else None,
            headers={**(headers or {}), "User-Agent": "GenesisAI-Skill/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise SkillNetworkError(f"Response exceeded the {MAX_RESPONSE_BYTES} byte limit")
                return {
                    "status": resp.status,
                    "text": raw.decode("utf-8", errors="replace"),
                    "headers": dict(resp.headers),
                }
        except urllib.error.HTTPError as e:
            # A real HTTP error response (404, 500, etc.) is useful information for a skill
            # (e.g. "this listing was removed") - return it rather than raising, matching how
            # a normal HTTP client library behaves.
            return {"status": e.code, "text": e.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace"), "headers": dict(e.headers or {})}
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            raise SkillNetworkError(f"Request failed: {e}")
