"""
Tests for app.core.handoff - webhook delivery for human handoff requests. Network calls are
mocked; this tests the URL validation, payload shape, and defensive error handling (a dead
webhook must never raise).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import urllib.error
from unittest.mock import patch, MagicMock
from app.core.handoff import is_plausible_webhook_url, send_handoff_webhook, build_handoff_payload


class TestUrlValidation:
    def test_valid_https_url(self):
        assert is_plausible_webhook_url("https://hooks.slack.com/services/xxx") is True

    def test_valid_http_url(self):
        assert is_plausible_webhook_url("http://example.com/webhook") is True

    def test_empty_string_rejected(self):
        assert is_plausible_webhook_url("") is False

    def test_none_rejected(self):
        assert is_plausible_webhook_url(None) is False

    def test_no_scheme_rejected(self):
        assert is_plausible_webhook_url("hooks.slack.com/services/xxx") is False

    def test_non_string_rejected(self):
        assert is_plausible_webhook_url(12345) is False


class TestPayloadBuilding:
    def test_payload_has_expected_fields(self):
        payload = build_handoff_payload("ShopBot", "agent_1", "conv_1", "where's my order?", "can't check order status")
        assert payload["event"] == "handoff_requested"
        assert payload["bot_name"] == "ShopBot"
        assert payload["agent_id"] == "agent_1"
        assert payload["conversation_id"] == "conv_1"
        assert payload["user_message"] == "where's my order?"
        assert payload["reason"] == "can't check order status"
        assert "timestamp" in payload


class TestWebhookDelivery:
    def test_successful_delivery(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda self: mock_resp
        mock_resp.__exit__ = lambda *a: None
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = send_handoff_webhook("https://example.com/hook", {"a": "b"})
        assert result["delivered"] is True
        assert result["error"] is None

    def test_204_no_content_counts_as_success(self):
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_resp.__enter__ = lambda self: mock_resp
        mock_resp.__exit__ = lambda *a: None
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = send_handoff_webhook("https://example.com/hook", {"a": "b"})
        assert result["delivered"] is True

    def test_http_error_never_raises(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                "url", 404, "Not Found", {}, None)):
            result = send_handoff_webhook("https://example.com/hook", {"a": "b"})
        assert result["delivered"] is False
        assert "404" in result["error"]

    def test_network_error_never_raises(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            result = send_handoff_webhook("https://example.com/hook", {"a": "b"})
        assert result["delivered"] is False
        assert "connection refused" in result["error"]

    def test_unexpected_exception_never_raises(self):
        with patch("urllib.request.urlopen", side_effect=RuntimeError("something weird")):
            result = send_handoff_webhook("https://example.com/hook", {"a": "b"})
        assert result["delivered"] is False
        assert "something weird" in result["error"]
