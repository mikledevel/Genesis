# Genesis AI: product interface

## Scope and audit

The application uses FastAPI with static HTML, CSS and JavaScript. This change retains that stack and the existing API contracts. No production dependencies were added.

The original overview repeated actions across a greeting, equal cards and shortcuts; key data appeared below them. Inter was used for both UI and data. Emoji icons, large radii and inconsistent inline styling obscured the hierarchy. Registry rows were pointer-only, with no text search, sorting or multi-selection. Several API failures appeared as empty datasets or zero counts.

## Tokens

`static/design-tokens.css` is shared by the workspace, standalone builder and admin.

| Role | Value |
|---|---|
| Page / surface | `#181917` |
| Navigation | `#1C1D19` |
| Raised surface / hover | `#1E1F1C` / `#272823` |
| Main / secondary / muted text | `#EEEDE7` / `#BBBDB2` / `#A0A397` |
| Accent / text on accent | `#D6BC82` / `#201C13` |
| Success / warning / error | `#A5C5A0` / `#DFC18A` / `#F0A49B` |
| Spacing | 4, 8, 12, 16, 24, 32, 48px |
| Radii | 4px controls, 6px surfaces, 8px dialogs, 12px exceptional containers |
| Motion | 180ms feedback, 220ms state changes; `cubic-bezier(.2,.8,.2,1)` |
| Elevation | Flat workspace; shadow only for menus, dialogs and tooltips |

Manrope Variable is the UI face; IBM Plex Mono renders data, IDs and code. System fallbacks remain usable when Google Fonts is unavailable. These fonts and the existing Tabler icon font require network access. Pipeline stages use local custom SVG paths from `ui-icons.js`.

## Layout and interaction

- Overview separates account counters, task entry points, recent work and activity. On wide screens, activity occupies a narrower secondary column; below 900px content stacks.
- Navigation groups workspace, building/evaluation and distribution. Mobile retains four destinations and a complete tools dialog.
- Registry filters by ID, model type, target and task; sorts by date or ID. Model buttons support keyboard access. Test metrics are preferred over training metrics when available.
- Selection persists across text filters within the loaded dataset. Task changes refetch the server-filtered list and clear selection. Select-all changes only visible rows. Export includes all selected model records, including selected rows hidden by filters. Reloading clears selection. The existing registry endpoint returns at most 10 recent records per task; search and export operate on the loaded records. Server pagination is outside this UI change. JSON exports contain metadata, not model binaries.
- Cmd/Ctrl+K searches tools and conversations. Up/Down cycle through results; Enter opens; Escape dismisses. Native dialogs preserve focus trapping and restore focus.
- Tooltips support focus and pointer hover and dismiss on Escape, blur or scrolling.

## States

| Component | States |
|---|---|
| Buttons | Default, hover, focus-visible, active, disabled; existing async controls retain progress text |
| Registry | Skeleton loading with aria-busy, populated, filtered empty, initial empty, inline error with retry |
| Selection | Empty/disabled export, partial/indeterminate, all visible selected |
| Dashboard | Skeleton counters, loaded data, true empty data, unavailable sections with retry |
| Command palette | Empty query, filtered results, keyboard focus, no results, closed |
| Inputs | Default, focus-visible, disabled, aria-invalid styling |

Reduced-motion removes transitions and animations. Text and state labels accompany color. The skip link targets the signed-in workspace after login. Registry uses native checkboxes, labelled filters and a table caption.

## Validation

- `node tests/ui/ui_test.js`: existing chat, attachments, Markdown, model and forecast result regressions.
- `node tests/ui/antibot_test.js`: registration and reset form regressions.
- `node tests/ui/admin_test.js`: admin behavior and CSP checks.
- `tests/browser/product_ui.js`: run through Playwright browser_run_code against a read-only local static server on 127.0.0.1:8768, mapping `/` to `/static/index.html` and `/builder` to `/static/builder/index.html`. The test intercepts API requests; model fixtures are read from the existing `registry/index.json`. It never invokes training, real payments, e-mail delivery or a production API.
- Browser checks cover 10 navigation panels at 320, 768, 1024 and 1440px; internal body overflow, command keyboard navigation, search, selection, downloaded JSON, empty state, simulated 503 and retry, and reduced motion.
- Axe WCAG A/AA checks passed for Overview, Registry and the command palette. This is an automated check of those surfaces, not a full manual screen-reader audit of the entire application.

Screenshots in `artifacts/ui-redesign` show the isolated UI fixture; its conversations and empty activity are test data. Production uses the existing live endpoints.

## Liquid Glass beta

An optional CSS appearance layer is enabled by default for this preview. Glass β in the public header and Overview, or Settings → General → Appearance, switches it off. The preference is stored as `genesis_glass_beta` in localStorage and shared with the standalone Pipeline Builder on the same origin. Storage errors do not prevent toggling; an accessible status explains when the preference cannot persist.

`static/glass-beta.css` scopes the effect to `html[data-glass=beta]`. Navigation, topbars, composer and dialogs use translucent surfaces, static inner highlights and backdrop blur. Tables stay opaque. This is a web treatment inspired by Liquid Glass, not native optical refraction. Browsers without backdrop-filter receive solid surfaces; prefers-reduced-transparency removes blur. Existing reduced-motion behavior remains in force. No extra JS animation, pointer tracking, canvas or production dependencies were added.

The mobile overview now places task entry points and recent conversations before counters. Models use a target/type label, version and short ID; full IDs remain in details and tooltips. Overview metrics identify the score being shown. These labels come from existing records and do not rename or alter the saved models.

Validation: existing chat and antibot regressions pass. Browser checks confirmed toggle state, Space activation, persistence after reload, human-name search and row selection. At 390×844 in the isolated fixture, task entry begins at y=259 (previous audit: y=470); page and internal overview have no horizontal overflow. Screenshots are under `artifacts/glass-beta/`.
