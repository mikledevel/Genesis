# Workspace interface

The existing monochrome UI now uses a labelled desktop sidebar, a compact tablet rail,
and four mobile navigation destinations. More opens the complete tool list, search,
templates and sign-out. Search also opens with Ctrl+K / Cmd+K; Tab browses results,
Enter opens the first result and Escape closes the native modal dialog.

Eight editable starter prompts cover bots, data, code and evaluation/planning. Selecting
one fills the composer without calling the AI API. Replacing an existing draft requires
confirmation. Pipeline blocks can be added by touch or keyboard as well as drag/drop.

Conversation pins and up to 50 text drafts live in browser localStorage, namespaced by
Genesis user ID. They do not sync between devices. Clearing browser storage removes them.
Attachments are not persisted in drafts. A new unsent message has its own draft; saved
conversations have independent drafts. Storage failures are reported to the user.

Profile names now persist through authenticated PATCH /api/auth/profile. Password changes
require the existing password, atomically replace its hash, revoke old sessions and return
a replacement session token. The frontend keeps JWTs as tokens, not serialized user objects.
New passwords are limited to bcrypt's 72 UTF-8 bytes. Change-password requests share the
authentication rate limit.

## Validation

- Account settings API: tests/test_account_settings.py, including ownership, invalid input,
  password verification, other-account isolation and old-session revocation.
- Browser interaction regression: tests/browser/workspace_smoke.js, passed as filename to
  Playwright browser_run_code. Requires an isolated localhost server, a signed-in test
  account and two chat fixtures (ui-chat-alpha: Customer support launch;
  ui-chat-beta: Dataset quality review). Never runs against production.
- Layout inspection: all ten navigation panels at 320, 768, 1024 and 1440 pixels.
- Visual review: overview, chat, templates, mobile settings and the mobile pipeline builder.
