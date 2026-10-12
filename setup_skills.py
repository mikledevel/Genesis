#!/usr/bin/env python3
"""
Genesis AI - skill setup script.

Configure your account, your bot, and the list of skills you want it to have in the CONFIG
section below, then run:

    python3 setup_skills.py

It will:
  1. Log in (or register, if the account doesn't exist yet)
  2. Find your bot by name (or create it, if it doesn't exist yet)
  3. Create every skill listed in SKILLS below for that bot, one at a time
  4. Print a clear pass/fail summary at the end, with the self-correction log for any skill
     that failed its sandbox test, so you can see what the AI tried and where it broke

Safe to re-run: it reuses your existing account/bot if they already exist rather than
duplicating them. Skills ARE created fresh each run, though - if you re-run with the same
skill descriptions, you'll get duplicate skill entries. Delete old ones first (see
list_and_delete_skills() below, or just use the app) if you want a clean re-run.
"""
import json
import sys
import urllib.request
import urllib.error

# ============================== CONFIG - edit this ==============================

BASE_URL = "http://127.0.0.1:8000"

EMAIL = "test@example.com"
PASSWORD = "password123"
NAME = "Test"  # only used if the account doesn't exist yet and needs to be registered

BOT_NAME = "GitHubHelper"  # must match an existing bot's name exactly, or a new one is created
BOT_SYSTEM_PROMPT = "You help users with GitHub - checking repos, and answering questions."

# Add as many skills here as you want. Each one becomes a separate real API call to
# /api/skills/create - the AI writes and sandbox-tests a real function for each.
#   name:            short label (shown in the dashboard)
#   description:     plain-language instructions for what the skill should do - this is the
#                     actual prompt the AI uses to write the code, so be specific
#   allowed_domains: the ONLY domain(s) this skill's code will ever be allowed to contact -
#                     see app/core/skills/net_guard.py for why this is enforced server-side,
#                     not just a suggestion
#   action_type:      "read_only" (checks/fetches data, runs automatically) or
#                     "outbound_action" (sends/posts something to a third party - always
#                     requires your approval before it actually runs, see main.py's
#                     /api/pending-actions routes)
SKILLS = [
    {
        "name": "check_github_repo",
        "description": "Check if a GitHub repository exists and return its status. "
                        "Params should include 'repo' as 'owner/name', e.g. 'anthropics/claude-code'.",
        "allowed_domains": ["github.com"],
        "action_type": "read_only",
    },
    # Example of a second skill - uncomment and edit to add it:
    # {
    #     "name": "check_pypi_package",
    #     "description": "Check if a Python package exists on PyPI and return its latest version.",
    #     "allowed_domains": ["pypi.org"],
    #     "action_type": "read_only",
    # },
]

# =============================== end of config ===================================


def _request(method, path, token=None, body=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"detail": str(e)}
    except urllib.error.URLError as e:
        print(f"\nCouldn't reach {url} - is the server running? ({e})")
        sys.exit(1)


def login_or_register() -> str:
    status, data = _request("POST", "/api/auth/login", body={"email": EMAIL, "password": PASSWORD})
    if status == 200:
        print(f"Logged in as {EMAIL}")
        return data["token"]

    print(f"Login failed ({data.get('detail', status)}) - registering a new account instead...")
    status, data = _request("POST", "/api/auth/register",
                             body={"email": EMAIL, "password": PASSWORD, "name": NAME})
    if status != 200:
        print(f"Registration also failed: {data.get('detail', status)}")
        sys.exit(1)
    print(f"Registered and logged in as {EMAIL}")
    return data["token"]


def find_or_create_bot(token: str) -> str:
    status, data = _request("GET", "/api/agents/mine", token=token)
    if status == 200:
        for agent in data.get("agents", []):
            if agent.get("name") == BOT_NAME:
                print(f"Found existing bot \"{BOT_NAME}\" (id: {agent['id']})")
                return agent["id"]

    print(f"No existing bot named \"{BOT_NAME}\" found - creating it...")
    status, data = _request("POST", "/api/agents/create", token=token, body={
        "name": BOT_NAME, "system_prompt": BOT_SYSTEM_PROMPT, "description": BOT_NAME,
    })
    if status != 200:
        print(f"Failed to create bot: {data.get('detail', status)}")
        sys.exit(1)
    agent_id = data.get("agent", {}).get("id") or data.get("id")
    print(f"Created bot \"{BOT_NAME}\" (id: {agent_id})")
    return agent_id


def create_skill(token: str, agent_id: str, skill: dict) -> bool:
    print(f"\nCreating skill \"{skill['name']}\" ({skill['action_type']}, domains: {skill['allowed_domains']})...")
    print("  This calls Groq to write and test real code - can take 10-30 seconds.")
    status, data = _request("POST", "/api/skills/create", token=token, body={
        "agent_id": agent_id,
        "name": skill["name"],
        "description": skill["description"],
        "allowed_domains": skill["allowed_domains"],
        "action_type": skill["action_type"],
    })
    if status != 200:
        print(f"  FAILED to call the API: {data.get('detail', status)}")
        return False

    if data.get("test_success"):
        print(f"  \u2705 Passed its sandbox test - ready to use in chat with \"{BOT_NAME}\".")
        return True

    log = data.get("log") or []
    if not log:
        # SkillGenerator.build() returns an empty log when generation itself never produced
        # any code to test (e.g. GROQ_API_KEY missing/invalid) - there's nothing to show per
        # iteration, but there IS a top-level reason worth surfacing.
        print(f"  \u274c Generation never produced code to test - check GROQ_API_KEY in your .env "
              f"and that the server can reach Groq's API.")
        return False

    print(f"  \u274c Did not pass its sandbox test after self-correction attempts.")
    for entry in log:
        status_icon = "\u2705" if entry.get("success") else "\u274c"
        print(f"    iteration {entry.get('iteration')}: {status_icon}")
        stderr_tail = (entry.get("stderr_tail") or "").strip()
        if stderr_tail:
            print(f"      error: {stderr_tail[:300]}")
    return False


def main():
    token = login_or_register()
    agent_id = find_or_create_bot(token)

    results = []
    for skill in SKILLS:
        results.append((skill["name"], create_skill(token, agent_id, skill)))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results:
        print(f"  {'\u2705' if ok else '\u274c'} {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} skills ready.")
    if passed:
        print(f"Go chat with \"{BOT_NAME}\" in the app now and ask it something that would "
              f"use one of these skills.")


if __name__ == "__main__":
    main()
