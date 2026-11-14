// jsdom checks for the bot protection in static/index.html: honeypots exist and are out of reach of people, the form timer and
// captcha token are sent, the Turnstile script loads ONLY when the server has a site key, and a refusal while requesting a
// password reset is shown as a refusal (never as "we sent the link").  Run: cd tests/ui && npm install && node antibot_test.js
const { JSDOM } = require('jsdom');
const fs = require('fs'), path = require('path');
let html = fs.readFileSync(path.join(__dirname, '../../static/index.html'), 'utf8').replace('<script src="/static/ui-icons.js?v=20261110"></script>', '<script>' + fs.readFileSync(path.join(__dirname, '../../static/ui-icons.js'), 'utf8') + '</script>');
for (const script of ['appearance','model-display']) {
  html = html.replace('<script src="/static/' + script + '.js?v=20261110-beta1"></script>', '<script>' + fs.readFileSync(require('path').join(__dirname, '../../static/' + script + '.js'), 'utf8') + '</script>');
}
let failures = 0;
let finished = false;
process.on('exit', (c) => { if (!finished && c === 0) { console.log('FAIL the test ended before its last check ran (a promise never settled)'); process.exitCode = 1; } });
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) failures++; };
const tick = (ms = 40) => new Promise(r => setTimeout(r, ms));

function boot({ siteKey = '', handler }) {
  const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost:8000/' });
  const w = dom.window; w.TextDecoder = TextDecoder;
  const calls = [];
  w.fetch = (url, opts = {}) => {
    calls.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : undefined });
    const r = url.endsWith('/config/public') ? { status: 200, body: { turnstile_site_key: siteKey } } : handler(url, opts);
    return Promise.resolve({ ok: r.status < 400, status: r.status, json: () => Promise.resolve(r.body) });
  };
  return { w, calls, $: id => w.document.getElementById(id) };
}
const lastTo = (calls, suffix) => [...calls].reverse().find(c => c.url.endsWith(suffix));
const fillRegister = ($) => { $('r-name').value = 'Ann'; $('r-email').value = 'ann@shop.test'; $('r-password').value = 'secret12'; };

(async () => {
  // ---------------------------------------------------------------- markup
  let t = boot({ handler: () => ({ status: 200, body: {} }) }); await tick(80);
  for (const id of ['r-hp', 'reset-hp']) {
    const el = t.$(id);
    ok(el && el.getAttribute('tabindex') === '-1' && el.getAttribute('autocomplete') === 'off' && el.closest('.ab-hp') && el.closest('.ab-hp').getAttribute('aria-hidden') === 'true',
       `${id}: honeypot input, tabindex -1, autocomplete off, inside an aria-hidden .ab-hp`);
  }
  ok(!/name="(website|url|email)"/.test(t.$('r-hp').outerHTML), 'honeypot name is not one a browser autofills (website/url/email)');
  ok(/\.ab-hp\{position:absolute;left:-9999px/.test(html), 'the honeypot is pushed off-screen by CSS');
  ok(t.$('ts-register') && t.$('ts-reset'), 'captcha containers exist for registration and password reset');

  // ---------------------------------------------------------------- no site key: nothing is loaded, fields still travel
  t = boot({ siteKey: '', handler: () => ({ status: 400, body: { detail: 'Please use a permanent e-mail address: temporary-mail services are not accepted.' } }) }); await tick(80);
  t.w.switchAuthTab('register'); fillRegister(t.$);
  t.w.eval("_formShownAt.register = Date.now() - 8000");
  await t.w.mountTurnstile('ts-register');
  ok(![...t.w.document.scripts].some(s => /challenges\.cloudflare\.com/.test(s.src)), 'no site key: the Turnstile script is NOT loaded');
  await t.w.doRegister(); await tick();
  const reg = lastTo(t.calls, '/auth/register').body;
  ok(reg.website === '' && reg.captcha_token === '' && reg.form_elapsed_ms >= 8000 && reg.form_elapsed_ms < 60000, 'register sends honeypot (empty), elapsed form time and an empty token');
  ok(/permanent e-mail/.test(t.$('register-err-2').textContent), 'a server refusal is shown to the person');
  ok(!t.$('register-btn').disabled, 'the button is usable again after a refusal');

  // a bot that fills the hidden field: the value is sent as-is, the server rejects it
  t.$('r-hp').value = 'http://spam.example'; await t.w.doRegister(); await tick();
  ok(lastTo(t.calls, '/auth/register').body.website === 'http://spam.example', 'a filled honeypot is forwarded untouched (the server decides)');

  // ---------------------------------------------------------------- with a site key: script injected once, widget rendered, token forwarded
  t = boot({ siteKey: 'SITE123', handler: () => ({ status: 400, body: { detail: 'Please complete the human check and try again.' } }) }); await tick(80);
  const m1 = t.w.mountTurnstile('ts-register'), m2 = t.w.mountTurnstile('ts-reset'); await tick();
  const scripts = [...t.w.document.scripts].filter(s => /challenges\.cloudflare\.com\/turnstile\/v0\/api\.js/.test(s.src));
  ok(scripts.length === 1 && /render=explicit/.test(scripts[0].src), 'site key present: the Turnstile script is injected exactly once (two widgets asked for it)');

  const rendered = []; let resets = 0;
  t.w.turnstile = { render: (el, o) => { rendered.push([el.id, o.sitekey]); return 'wid' + rendered.length; }, getResponse: () => 'tok-xyz', reset: () => { resets++; } };
  scripts[0].onload(); await Promise.all([m1, m2]);                       // what a browser does when the script has arrived
  await t.w.mountTurnstile('ts-register'); await t.w.mountTurnstile('ts-reset');
  ok(rendered.length === 2 && rendered.every(r => r[1] === 'SITE123') && new Set(rendered.map(r => r[0])).size === 2, 'each widget is rendered once, with the public site key (asking again renders nothing new)');
  t.w.switchAuthTab('register'); fillRegister(t.$); t.w.eval("_formShownAt.register = Date.now() - 9000");
  await t.w.doRegister(); await tick();
  ok(lastTo(t.calls, '/auth/register').body.captcha_token === 'tok-xyz', 'the captcha token goes with the registration');
  ok(resets === 1 && /human check/.test(t.$('register-err-2').textContent), 'after a refusal the widget is reset (a token works once) and the reason shown');

  // ---------------------------------------------------------------- password reset: refusals are not "sent"
  t = boot({ siteKey: '', handler: () => ({ status: 400, body: { detail: 'Please complete the human check and try again.' } }) }); await tick(80);
  t.$('l-email').value = 'ann@shop.test'; t.w.openPasswordResetRequest(); await tick();
  await t.w.submitPasswordResetRequest(); await tick();
  const msg = t.$('reset-req-msg');
  ok(/human check/.test(msg.textContent) && !/we've sent/i.test(msg.textContent), 'a refused reset request shows the refusal, NOT the "we sent a link" text');
  const rb = lastTo(t.calls, '/auth/password-reset/request').body;
  ok(rb.email === 'ann@shop.test' && rb.website === '' && typeof rb.form_elapsed_ms === 'number', 'the reset request carries the e-mail, honeypot and form time');

  t = boot({ handler: () => ({ status: 429, body: { detail: 'Rate limit exceeded' } }) }); await tick(80);
  t.$('reset-req-email').value = 'ann@shop.test'; await t.w.submitPasswordResetRequest(); await tick();
  ok(/Too many requests/.test(t.$('reset-req-msg').textContent), '429: a clear "too many requests" message');

  t = boot({ handler: () => ({ status: 200, body: { status: 'if_account_exists_email_sent' } }) }); await tick(80);
  t.$('reset-req-email').value = 'nobody@shop.test'; await t.w.submitPasswordResetRequest(); await tick();
  ok(/If an account exists for that email, we've sent a reset link/.test(t.$('reset-req-msg').textContent), 'success keeps the enumeration-safe wording');

  finished = true;
  console.log(failures ? failures + ' FAILED' : 'ALL ANTIBOT UI CHECKS PASSED');
  process.exit(failures ? 1 : 0);
})();
