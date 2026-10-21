"""
Human handoff delivery for published bots.

Closes a real gap in the existing anti-hallucination design (bot_builder.py): a bot's system
prompt tells it to say "I can't check that, contact support" when it hits the edge of what it
actually knows - but until now, nothing was actually behind those words. This module is what
makes the handoff real: when a bot's owner has configured a webhook URL, hitting a handoff
moment POSTs a plain JSON payload there (Slack incoming webhooks, Discord webhooks, Zapier,
Make.com, and most helpdesk tools all accept this with zero extra setup).

Email delivery is a natural next step but not built here - it would need an SMTP or
transactional-email provider (SendGrid, Postmark, SES, etc.) this project doesn't have
configured, and half-implementing that without a way to actually verify delivery would be
worse than being upfront that it isn't there yet.

Uses urllib (matching the existing HTTP client convention in web_search.py/agent.py) rather
than adding a new dependency for a single POST call.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Optional, Dict

TIMEOUT = 10


def is_plausible_webhook_url(url: str) -> bool:
    """Basic sanity check before ever storing a webhook URL - not a full validator, just
    enough to reject obvious garbage (empty string, no scheme) at input time rather than
    only discovering it's broken the first time a real handoff tries to fire."""
    if not url or not isinstance(url, str):
        return False
    return url.startswith("http://") or url.startswith("https://")


def send_handoff_webhook(webhook_url: str, payload: Dict) -> Dict:
    """POSTs the handoff payload as JSON. Returns {"delivered": bool, "error": str|None} -
    never raises, since a dead/misconfigured webhook must never be allowed to break the
    actual chat response the user is waiting on."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            # 2xx is success; most webhook receivers (Slack, Discord, Zapier) return 200 or 204
            if 200 <= resp.status < 300:
                return {"delivered": True, "error": None}
            return {"delivered": False, "error": f"HTTP {resp.status}"}
    except urllib.error.HTTPError as e:
        return {"delivered": False, "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"delivered": False, "error": f"Network error: {e.reason}"}
    except Exception as e:
        return {"delivered": False, "error": str(e)}


def build_handoff_payload(bot_name: str, agent_id: str, conversation_id: str,
                           user_message: str, reason: str) -> Dict:
    return {
        "event": "handoff_requested",
        "bot_name": bot_name,
        "agent_id": agent_id,
        "conversation_id": conversation_id,
        "user_message": user_message,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
