// Owner admin panel. Every piece of text that comes from the server (names, e-mails, audit details) is put in with
// textContent - never parsed as markup - and the page has a CSP that allows only this file as script.
(function () {
  'use strict';
  var AUTH_KEY = 'genesis_token_v1', PAGE = 25;
  var state = { offset: 0, q: '', plans: ['free', 'pro', 'business'] };
  var $ = function (id) { return document.getElementById(id); };

  function el(tag, text, cls) {
    var n = document.createElement(tag);
    if (text !== undefined && text !== null) n.textContent = String(text);
    if (cls) n.className = cls;
    return n;
  }
  function tokenHeader() {
    var t = null;
    try { t = localStorage.getItem(AUTH_KEY); } catch (e) { /* storage blocked */ }
    return t ? { 'Authorization': 'Bearer ' + t } : null;
  }
  function say(text) { $('msg').textContent = text || ''; }

  function api(method, path, body) {
    var h = tokenHeader();
    if (!h) return Promise.reject({ status: 0, detail: 'Log in on the main site first, then reload this page.' });
    var opts = { method: method, headers: Object.assign({ 'Content-Type': 'application/json' }, h) };
    if (body !== undefined) opts.body = JSON.stringify(body);
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) throw { status: r.status, detail: data.detail || ('Error ' + r.status) };
        return data;
      });
    });
  }
  function fail(e) {
    if (e && (e.status === 401 || e.status === 403 || e.status === 0)) {
      $('app').hidden = true;
      say(e.status === 403 ? 'This account is not an admin (the e-mail must be an owner e-mail and verified).' : e.detail);
    } else say((e && e.detail) || 'Something went wrong');
  }

  function usd(n) { return '$' + Number(n || 0).toFixed(2); }
  function tokens(n) { return Number(n || 0).toLocaleString('en-US'); }

  function renderCards(o) {
    var box = $('cards'); box.textContent = '';
    var rows = [
      ['Users', o.users], ['Verified', o.verified], ['Suspended', o.disabled], ['Signups 7d', o.signups_7d], ['Signups 30d', o.signups_30d],
      ['Free / Pro / Business', [o.by_plan.free, o.by_plan.pro, o.by_plan.business].join(' / ')],
      ['List-price MRR', usd(o.list_price_mrr_usd)], ['Tokens this month', tokens(o.tokens_this_month.total)],
      ['LLM cost (estimate)', usd(o.llm_cost_estimate_usd)], ['Balances owed', usd(o.balance_liability_cents / 100)],
      ['E-mail provider', o.email ? o.email.provider + (o.email.provider === 'dev' ? ' (nothing is sent)' : '') : '?'],
      ['Bot check (captcha)', o.captcha_enabled ? 'on' : 'off']
    ];
    rows.forEach(function (r) { var c = el('div', null, 'card'); c.appendChild(el('b', r[1])); c.appendChild(el('span', r[0])); box.appendChild(c); });
  }

  function userRow(u) {
    var tr = el('tr');
    tr.appendChild(el('td', u.email)); tr.appendChild(el('td', u.name));
    var tdPlan = el('td'), sel = el('select'); sel.setAttribute('aria-label', 'Plan of ' + u.email);
    state.plans.forEach(function (p) { var op = el('option', p); op.value = p; if (p === u.plan) op.selected = true; sel.appendChild(op); });
    sel.addEventListener('change', function () {
      api('POST', '/api/admin/users/' + encodeURIComponent(u.id) + '/plan', { plan: sel.value }).then(load).catch(function (e) { fail(e); load(); });
    });
    tdPlan.appendChild(sel); tr.appendChild(tdPlan);
    tr.appendChild(el('td', tokens(u.tokens_used_this_month)));
    tr.appendChild(el('td', u.is_admin ? 'unlimited (owner)' : (u.token_limit_override === null || u.token_limit_override === undefined ? 'plan default' : (u.token_limit_override === 0 ? 'unlimited' : tokens(u.token_limit_override)))));
    tr.appendChild(el('td', usd(u.balance_cents / 100)));
    var tdSt = el('td');
    tdSt.appendChild(el('span', u.email_verified ? 'verified' : 'unverified', 'tag'));
    if (u.is_admin) tdSt.appendChild(el('span', ' admin', 'tag admin'));
    if (u.disabled) tdSt.appendChild(el('span', ' suspended', 'tag off'));
    tr.appendChild(tdSt);
    var tdAct = el('td');
    var limitBtn = el('button', 'Token limit'); limitBtn.type = 'button';
    limitBtn.addEventListener('click', function () {
      var v = window.prompt('Monthly token limit for ' + u.email + ':\n  a number = that many tokens\n  0 = unlimited\n  empty = the plan default', '');
      if (v === null) return;
      v = v.trim();
      var val = v === '' ? null : Number(v);
      if (val !== null && (!Number.isInteger(val) || val < 0)) { say('Enter a whole number, 0, or leave it empty.'); return; }
      api('POST', '/api/admin/users/' + encodeURIComponent(u.id) + '/token-limit', { limit: val }).then(load).catch(fail);
    });
    tdAct.appendChild(limitBtn);
    if (!u.is_admin) {
      var offBtn = el('button', u.disabled ? 'Enable' : 'Suspend', u.disabled ? '' : 'danger'); offBtn.type = 'button';
      offBtn.addEventListener('click', function () {
        if (!u.disabled && !window.confirm('Suspend ' + u.email + '? They are logged out immediately.')) return;
        api('POST', '/api/admin/users/' + encodeURIComponent(u.id) + '/disable', { disabled: !u.disabled }).then(load).catch(fail);
      });
      tdAct.appendChild(document.createTextNode(' ')); tdAct.appendChild(offBtn);
    }
    tr.appendChild(tdAct);
    return tr;
  }

  function renderUsers(data) {
    var body = $('users'); body.textContent = '';
    data.users.forEach(function (u) { body.appendChild(userRow(u)); });
    var from = data.total ? data.offset + 1 : 0;
    $('count').textContent = from + '-' + (data.offset + data.users.length) + ' of ' + data.total;
    $('prev').disabled = data.offset <= 0;
    $('next').disabled = data.offset + data.users.length >= data.total;
  }

  function renderAudit(actions) {
    var body = $('audit'); body.textContent = '';
    actions.forEach(function (a) {
      var tr = el('tr');
      tr.appendChild(el('td', a.created_at)); tr.appendChild(el('td', a.action)); tr.appendChild(el('td', (a.detail && a.detail.email) || a.target_id || ''));
      tr.appendChild(el('td', JSON.stringify(a.detail || {})));
      body.appendChild(tr);
    });
  }

  function load() {
    say('');
    var qs = '?limit=' + PAGE + '&offset=' + state.offset + (state.q ? '&q=' + encodeURIComponent(state.q) : '');
    return Promise.all([api('GET', '/api/admin/overview'), api('GET', '/api/admin/users' + qs), api('GET', '/api/admin/audit?limit=20')])
      .then(function (r) { $('app').hidden = false; renderCards(r[0]); renderUsers(r[1]); renderAudit(r[2].actions); })
      .catch(fail);
  }

  $('emailTest').addEventListener('click', function () {
    var btn = $('emailTest'), info = $('info');
    btn.disabled = true; info.textContent = 'Sending...';
    api('POST', '/api/admin/email/test').then(function (r) {
      info.textContent = (r.ok ? 'OK' : 'FAILED') + ' - ' + r.provider + ' -> ' + r.to + (r.detail ? ': ' + r.detail : '');
    }).catch(function (e) { info.textContent = ''; fail(e); }).then(function () { btn.disabled = false; });
  });

  var timer = null;
  $('q').addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { state.q = $('q').value.trim(); state.offset = 0; load(); }, 250);
  });
  $('prev').addEventListener('click', function () { state.offset = Math.max(0, state.offset - PAGE); load(); });
  $('next').addEventListener('click', function () { state.offset += PAGE; load(); });
  window.__adminLoad = load;       // test hook
  load();
})();
