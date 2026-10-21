// jsdom checks for static/admin.html + admin.js: nothing from the server is ever interpreted as HTML, 403/401 are
// explained, and the plan / limit / suspend controls call the right endpoints. Run: cd tests/ui && npm install && node admin_test.js
const { JSDOM } = require('jsdom');
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '../../static/admin.html'), 'utf8');
const js = fs.readFileSync(path.join(__dirname, '../../static/admin.js'), 'utf8');
let failures = 0;
let finished = false;
process.on('exit', (c) => { if (!finished && c === 0) { console.log('FAIL the test ended before its last check ran (a promise never settled)'); process.exitCode = 1; } });
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) failures++; };
const tick = (ms = 30) => new Promise(r => setTimeout(r, ms));

const EVIL = '<img src=x onerror="window.__pwned=1">';
const overview = { users: 2, verified: 1, disabled: 0, by_plan: { free: 1, pro: 1, business: 0 }, signups_7d: 2, signups_30d: 2,
  balance_liability_cents: 250, email: { provider: 'resend', from: 'x', key_set: true }, captcha_enabled: false, tokens_this_month: { prompt: 10, completion: 5, total: 15 }, llm_cost_estimate_usd: 0.0123, list_price_mrr_usd: 19 };
const users = { total: 2, limit: 25, offset: 0, users: [
  { id: 'u_evil', email: EVIL + '@x.io', name: EVIL, plan: 'free', email_verified: true, disabled: false, created_at: 't', balance_cents: 0, token_limit_override: null, tokens_used_this_month: 1234, is_admin: false },
  { id: 'u_owner', email: 'owner@x.io', name: 'Owner', plan: 'free', email_verified: true, disabled: false, created_at: 't', balance_cents: 0, token_limit_override: 0, tokens_used_this_month: 0, is_admin: true } ] };
const audit = { actions: [{ id: 1, admin_id: 'u_owner', action: 'set_plan', target_id: 'u_evil', detail: { email: EVIL, to: 'pro' }, created_at: 't' }] };

function boot({ token = 'tok', handler }) {
  const calls = [];
  const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost:8000/admin' });
  const w = dom.window;
  if (token) w.localStorage.setItem('genesis_token_v1', token);
  w.fetch = (url, opts = {}) => {
    calls.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : undefined, auth: (opts.headers || {}).Authorization });
    const r = handler(url, opts);
    return Promise.resolve({ ok: r.status < 400, status: r.status, json: () => Promise.resolve(r.body) });
  };
  w.eval(js);
  return { w, calls, $: id => w.document.getElementById(id) };
}
const good = (url, opts) => {
  if (url.startsWith('/api/admin/overview')) return { status: 200, body: overview };
  if (url.startsWith('/api/admin/users') && (opts.method || 'GET') === 'GET') return { status: 200, body: users };
  if (url.startsWith('/api/admin/audit')) return { status: 200, body: audit };
  return { status: 200, body: { status: 'ok' } };
};

(async () => {
  // ---- no token: told to log in, no request is made
  let t = boot({ token: null, handler: good }); await tick();
  ok(t.calls.length === 0 && /Log in on the main site/.test(t.$('msg').textContent) && t.$('app').hidden, 'no token: explains, makes no request');

  // ---- 403 and 401 are explained, the panel stays hidden
  t = boot({ handler: () => ({ status: 403, body: { detail: 'Admin only' } }) }); await tick();
  ok(t.$('app').hidden && /not an admin/.test(t.$('msg').textContent), '403: "not an admin", panel hidden');
  t = boot({ handler: () => ({ status: 401, body: { detail: 'Invalid or expired token' } }) }); await tick();
  ok(t.$('app').hidden && /expired/.test(t.$('msg').textContent), '401: shows the server message');

  // ---- happy path
  t = boot({ handler: good }); await tick();
  const { w, $ } = t;
  ok(!$('app').hidden, 'panel shown for an admin');
  ok(t.calls.every(c => c.auth === 'Bearer tok'), 'every request carries the session token');
  ok($('cards').children.length === 12 && /resend/.test($('cards').textContent) && /Bot check/.test($('cards').textContent) && /Free \/ Pro \/ Business/.test($('cards').textContent) && $('cards').textContent.includes('$19.00'), 'overview cards rendered');
  ok($('users').querySelectorAll('tr').length === 2 && $('count').textContent === '1-2 of 2', 'two user rows and the counter');

  // ---- XSS: server text is text
  ok(!w.__pwned && $('users').querySelector('img') === null && $('audit').querySelector('img') === null, 'HTML in names / e-mails / audit detail creates no elements and runs no code');
  ok($('users').textContent.includes(EVIL), 'the hostile name is shown literally as text');

  // ---- statuses and limits
  const rows = $('users').querySelectorAll('tr');
  ok(/admin/.test(rows[1].textContent) && /unlimited/.test(rows[1].textContent) && !/Suspend/.test(rows[1].textContent), 'admin row: tagged, unlimited, cannot be suspended from the UI');
  ok(/plan default/.test(rows[0].textContent) && /1,234/.test(rows[0].textContent), 'plan default + token usage formatted');

  // ---- plan change
  const sel = rows[0].querySelector('select'); sel.value = 'pro'; sel.dispatchEvent(new w.Event('change')); await tick();
  const planCall = t.calls.find(c => c.method === 'POST' && c.url.endsWith('/plan'));
  ok(planCall && planCall.url === '/api/admin/users/u_evil/plan' && planCall.body.plan === 'pro', 'plan select posts the new plan to the right user');

  // ---- token limit: number, 0, empty; garbage is refused client-side
  const limitBtn = () => $('users').querySelectorAll('tr')[0].querySelector('button');
  const post = (pred) => t.calls.filter(c => c.method === 'POST' && c.url.endsWith('/token-limit')).filter(pred || (() => true));
  w.prompt = () => '5000'; limitBtn().click(); await tick();
  ok(post(c => c.body.limit === 5000).length === 1, 'limit 5000 posted');
  w.prompt = () => '0'; limitBtn().click(); await tick();
  ok(post(c => c.body.limit === 0).length === 1, 'limit 0 (unlimited) posted');
  w.prompt = () => ''; limitBtn().click(); await tick();
  ok(post(c => c.body.limit === null).length === 1, 'empty -> null (plan default) posted');
  const before = post().length;
  w.prompt = () => '12abc'; limitBtn().click(); await tick();
  w.prompt = () => '-3'; limitBtn().click(); await tick();
  ok(post().length === before && /whole number/.test($('msg').textContent), 'garbage and negative limits never reach the server');

  // ---- suspend asks for confirmation
  const offBtn = () => $('users').querySelectorAll('tr')[0].querySelectorAll('button')[1];
  w.confirm = () => false; offBtn().click(); await tick();
  ok(!t.calls.some(c => c.url.endsWith('/disable')), 'suspend cancelled: nothing sent');
  w.confirm = () => true; offBtn().click(); await tick();
  const dis = t.calls.find(c => c.url.endsWith('/disable'));
  ok(dis && dis.url === '/api/admin/users/u_evil/disable' && dis.body.disabled === true, 'suspend confirmed: posted for the right user');

  // ---- search is debounced and escaped in the URL
  const n0 = t.calls.length;
  $('q').value = 'a&b=c'; $('q').dispatchEvent(new w.Event('input')); $('q').dispatchEvent(new w.Event('input')); await tick(400);
  const searches = t.calls.slice(n0).filter(c => c.url.startsWith('/api/admin/users?'));
  ok(searches.length === 1 && searches[0].url.includes('q=a%26b%3Dc'), 'search: one debounced request, query URL-encoded');

  // ---- test e-mail button: calls the endpoint, shows the real outcome as plain text
  const mailHandler = (url, opts) => url.endsWith('/email/test')
    ? { status: 200, body: { ok: false, provider: 'resend', from: 'x', to: 'owner@x.io', detail: 'HTTP 403: domain <b>not</b> verified' } } : good(url, opts);
  const tm = boot({ handler: mailHandler }); await tick();
  tm.$('emailTest').click(); await tick();
  ok(tm.calls.some(c => c.method === 'POST' && c.url === '/api/admin/email/test'), 'the test e-mail button posts to /api/admin/email/test');
  ok(/FAILED - resend -> owner@x.io: HTTP 403: domain <b>not<\/b> verified/.test(tm.$('info').textContent) && tm.$('info').querySelector('b') === null && !tm.$('emailTest').disabled,
     'a failed delivery is reported literally (no markup interpreted) and the button is re-enabled');

  // ---- the page itself
  ok(/Content-Security-Policy/.test(html) && /script-src 'self'/.test(html) && !/<script>[^<]/.test(html), 'strict CSP and no inline script');
  finished = true;
  console.log(failures ? failures + ' FAILED' : 'ALL ADMIN UI CHECKS PASSED');
  process.exit(failures ? 1 : 0);
})();
