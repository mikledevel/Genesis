"""
Telegram Bot integration - connect a GenesisAI bot to a real Telegram bot via a BotFather token.

Talks to the official Telegram Bot API (https://core.telegram.org/bots/api, verified live
against the docs before writing this - see method-by-method comments below) directly over
HTTPS via urllib, matching the rest of the codebase's HTTP client convention (see
web_search.py / agent.py) rather than adding a new SDK dependency for a handful of calls.

Flow used by the /api/telegram/* endpoints in main.py:
  1. get_me(token)                 -> validates the token, returns bot identity + privacy mode
  2. set_webhook(...)              -> registers our webhook endpoint with Telegram
  3. import_bot_settings(...)      -> pulls existing commands/description/menu/avatar/admin
                                       rights FROM Telegram so they show up in the GenesisAI UI
  4. send_message(...)             -> used by both the real webhook handler and the built-in
                                       "test chat" feature (which talks to the bot exactly like
                                       a real Telegram user would, without opening Telegram)

Every method raises TelegramAPIError with Telegram's own description on failure rather than
swallowing it - callers decide how to log/surface that.
"""
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

API_URL = "https://api.telegram.org/bot{token}/{method}"
FILE_URL = "https://api.telegram.org/file/bot{token}/{file_path}"
TIMEOUT = 15


class TelegramAPIError(Exception):
    def __init__(self, method: str, description: str, error_code: Optional[int] = None):
        self.method = method
        self.description = description
        self.error_code = error_code
        super().__init__(f"Telegram API {method} failed ({error_code}): {description}")


def _call(token: str, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Low-level POST to https://api.telegram.org/bot<token>/<method>. The Bot API accepts
    application/json for every method (per 'Making requests' in the official docs)."""
    url = API_URL.format(token=token, method=method)
    body = json.dumps(params or {}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Telegram still returns a JSON body with ok=False + description on 4xx (e.g. bad token) -
        # that's the useful error, not the raw HTTP status.
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            raise TelegramAPIError(method, f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise TelegramAPIError(method, f"Network error: {e.reason}")

    if not data.get("ok"):
        raise TelegramAPIError(method, data.get("description", "Unknown error"), data.get("error_code"))
    return data.get("result")


# ---- Identity / webhook lifecycle ----

def get_me(token: str) -> Dict:
    """Validates the token. can_read_all_group_messages reflects whether privacy mode is
    DISABLED - this is the only privacy-mode signal the Bot API itself exposes (absent/False
    means privacy mode is ON, the BotFather default); there is no separate 'get privacy mode'
    call, confirmed against the official User object docs."""
    return _call(token, "getMe")


def set_webhook(token: str, url: str, secret_token: str, drop_pending_updates: bool = False) -> bool:
    return _call(token, "setWebhook", {
        "url": url,
        "secret_token": secret_token,
        "drop_pending_updates": drop_pending_updates,
        "allowed_updates": ["message"],
    })


def delete_webhook(token: str, drop_pending_updates: bool = False) -> bool:
    return _call(token, "deleteWebhook", {"drop_pending_updates": drop_pending_updates})


def get_webhook_info(token: str) -> Dict:
    """Returns WebhookInfo: url, pending_update_count, last_error_date, last_error_message, etc.
    Used for the real-time connection-status check and to detect when Telegram has silently
    stopped delivering (last_error_message set) so we know to auto-reconnect."""
    return _call(token, "getWebhookInfo")


# ---- Import existing BotFather configuration ----

def get_my_commands(token: str) -> List[Dict]:
    return _call(token, "getMyCommands") or []


def set_my_commands(token: str, commands: List[Dict]) -> bool:
    return _call(token, "setMyCommands", {"commands": commands})


def get_my_description(token: str) -> str:
    result = _call(token, "getMyDescription")
    return (result or {}).get("description", "")


def get_my_short_description(token: str) -> str:
    result = _call(token, "getMyShortDescription")
    return (result or {}).get("short_description", "")


def get_chat_menu_button(token: str) -> Dict:
    return _call(token, "getChatMenuButton") or {}


def get_my_default_administrator_rights(token: str) -> Dict:
    return _call(token, "getMyDefaultAdministratorRights") or {}


def get_bot_avatar_file_id(token: str, bot_id: int) -> Optional[str]:
    """Bots are Telegram users too (User.is_bot=true), so a bot's own avatar is fetched the
    same way as any user's profile photo: getUserProfilePhotos(user_id=<the bot's own id>)."""
    result = _call(token, "getUserProfilePhotos", {"user_id": bot_id, "limit": 1})
    photos = (result or {}).get("photos")
    if not photos:
        return None
    return photos[0][-1]["file_id"]  # largest available size


def get_file_url(token: str, file_id: str) -> Optional[str]:
    """getFile resolves a file_id to a downloadable path so the frontend can render it directly."""
    result = _call(token, "getFile", {"file_id": file_id})
    file_path = (result or {}).get("file_path")
    if not file_path:
        return None
    return FILE_URL.format(token=token, file_path=file_path)


def import_bot_settings(token: str, bot_id: int) -> Dict:
    """Pulls everything BotFather already has configured for this bot so the GenesisAI UI can
    show/mirror it instead of the user re-typing it. Each sub-call is best-effort: an empty
    BotFather field (e.g. no commands set yet) is a normal state, not an import failure, so one
    failing call doesn't abort the whole import."""
    imported = {"commands": [], "description": "", "short_description": "",
                "menu_button": {}, "avatar_url": None, "default_admin_rights": {}}
    for key, fn, arg in [
        ("commands", get_my_commands, token),
        ("description", get_my_description, token),
        ("short_description", get_my_short_description, token),
        ("menu_button", get_chat_menu_button, token),
        ("default_admin_rights", get_my_default_administrator_rights, token),
    ]:
        try:
            imported[key] = fn(arg)
        except TelegramAPIError:
            pass
    try:
        file_id = get_bot_avatar_file_id(token, bot_id)
        if file_id:
            imported["avatar_url"] = get_file_url(token, file_id)
    except TelegramAPIError:
        pass
    return imported


# ---- Messaging (used by both the live webhook and the in-app test chat) ----

def send_message(token: str, chat_id, text: str, parse_mode: Optional[str] = None) -> Dict:
    params = {"chat_id": chat_id, "text": text}
    if parse_mode:
        params["parse_mode"] = parse_mode
    return _call(token, "sendMessage", params)


def send_chat_action(token: str, chat_id, action: str = "typing") -> bool:
    return _call(token, "sendChatAction", {"chat_id": chat_id, "action": action})
