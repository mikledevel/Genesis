"""
Tests for app.core.skills.net_guard - the SSRF-prevention boundary for skill code.

These are the most important tests in the whole skills feature: everything else (codegen
quality, the sandbox bridge, billing) is a normal engineering bug if wrong. A gap here is a
security hole that lets any user's "skill" reach internal infrastructure.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from unittest.mock import patch
from app.core.skills.net_guard import SafeFetcher, SkillNetworkError, is_domain_allowed, _is_blocked_ip


class TestDomainAllowlist:
    def test_exact_match_allowed(self):
        assert is_domain_allowed("ebay.com", ["ebay.com"])

    def test_subdomain_of_allowed_domain_is_allowed(self):
        assert is_domain_allowed("api.ebay.com", ["ebay.com"])
        assert is_domain_allowed("www.ebay.com", ["ebay.com"])

    def test_unrelated_domain_rejected(self):
        assert not is_domain_allowed("evil.com", ["ebay.com"])

    def test_similar_looking_domain_not_matched_as_substring(self):
        """The classic bypass attempt: "notebay.com" contains "ebay.com" as a substring but
        is a completely different, unrelated domain and must NOT match."""
        assert not is_domain_allowed("notebay.com", ["ebay.com"])
        assert not is_domain_allowed("ebay.com.evil.com", ["ebay.com"])

    def test_parent_domain_not_granted_by_subdomain_allowlist(self):
        """Allowlisting a specific subdomain does not grant its parent domain."""
        assert not is_domain_allowed("ebay.com", ["api.ebay.com"])

    def test_case_insensitive(self):
        assert is_domain_allowed("EBAY.COM", ["ebay.com"])

    def test_trailing_dot_normalized(self):
        assert is_domain_allowed("ebay.com.", ["ebay.com"])


class TestBlockedIPs:
    @pytest.mark.parametrize("ip", [
        "127.0.0.1",        # loopback
        "192.168.1.1",      # private
        "10.0.0.5",         # private
        "172.16.0.1",       # private
        "169.254.169.254",  # cloud metadata - the single most common real-world SSRF target
        "169.254.1.1",      # link-local generally
        "0.0.0.0",          # unspecified
        "224.0.0.1",        # multicast
    ])
    def test_dangerous_ips_blocked(self, ip):
        assert _is_blocked_ip(ip)

    @pytest.mark.parametrize("ip", [
        "8.8.8.8",       # Google DNS - genuinely public
        "1.1.1.1",       # Cloudflare DNS - genuinely public
        "93.184.216.34", # example.com's real IP - genuinely public
    ])
    def test_public_ips_allowed(self, ip):
        assert not _is_blocked_ip(ip)

    def test_unparseable_ip_is_blocked(self):
        assert _is_blocked_ip("not-an-ip-address")


class TestSafeFetcherPolicyEnforcement:
    """These test the policy layer WITHOUT making real network calls - mocking DNS
    resolution to simulate the DNS-rebinding scenario specifically."""

    def test_disallowed_domain_rejected_before_any_dns_lookup(self):
        fetcher = SafeFetcher(allowed_domains=["ebay.com"])
        with patch("app.core.skills.net_guard._resolve_all_ips") as mock_resolve:
            with pytest.raises(SkillNetworkError, match="only allowed to contact"):
                fetcher.fetch("https://evil.com/steal-data")
            mock_resolve.assert_not_called()  # domain check must short-circuit before DNS

    def test_non_http_scheme_rejected(self):
        fetcher = SafeFetcher(allowed_domains=["ebay.com"])
        with pytest.raises(SkillNetworkError, match="scheme"):
            fetcher.fetch("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        fetcher = SafeFetcher(allowed_domains=["ebay.com"])
        with pytest.raises(SkillNetworkError, match="scheme"):
            fetcher.fetch("ftp://ebay.com/file")

    def test_allowed_domain_that_resolves_to_private_ip_is_blocked(self):
        """The DNS-rebinding case: hostname is on the allowlist, but resolves to an internal
        address (either an attack, or a misconfigured/compromised DNS record) - must still
        be blocked."""
        fetcher = SafeFetcher(allowed_domains=["internal-tool.example.com"])
        with patch("app.core.skills.net_guard._resolve_all_ips", return_value=["10.0.0.5"]):
            with pytest.raises(SkillNetworkError, match="disallowed address"):
                fetcher.fetch("https://internal-tool.example.com/data")

    def test_cloud_metadata_ip_blocked_even_if_domain_were_somehow_allowed(self):
        fetcher = SafeFetcher(allowed_domains=["metadata.google.internal"])
        with patch("app.core.skills.net_guard._resolve_all_ips", return_value=["169.254.169.254"]):
            with pytest.raises(SkillNetworkError, match="disallowed address"):
                fetcher.fetch("http://metadata.google.internal/computeMetadata/v1/")

    def test_unresolvable_host_raises_cleanly(self):
        fetcher = SafeFetcher(allowed_domains=["ebay.com"])
        with patch("app.core.skills.net_guard._resolve_all_ips", return_value=[]):
            with pytest.raises(SkillNetworkError, match="Could not resolve"):
                fetcher.fetch("https://ebay.com/listings")

    def test_multiple_resolved_ips_all_checked_not_just_first(self):
        """A hostname can resolve to several IPs (round-robin DNS) - if ANY of them is
        private, the request must be blocked, not just checked against the first one."""
        fetcher = SafeFetcher(allowed_domains=["ebay.com"])
        with patch("app.core.skills.net_guard._resolve_all_ips", return_value=["93.184.216.34", "127.0.0.1"]):
            with pytest.raises(SkillNetworkError, match="disallowed address"):
                fetcher.fetch("https://ebay.com/listings")
