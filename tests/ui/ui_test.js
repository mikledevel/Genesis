const { JSDOM } = require('jsdom');
const fs = require('fs');
const html = fs.readFileSync(require('path').join(__dirname, '../../static/index.html'), 'utf8');
let failures = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) failures++; };

function sse(events) {
  const enc = new TextEncoder();
  const chunks = events.map(e => 'data: ' + JSON.stringify(e) + '\n\n');
  // split one frame in two network chunks to exercise the buffer
  const joined = chunks.join(''); const mid = Math.floor(joined.length / 2);
  return new ReadableStream({ start(c) { c.enqueue(enc.encode(joined.slice(0, mid))); setTimeout(() => { c.enqueue(enc.encode(joined.slice(mid))); c.close(); }, 5); } });
}

(async () => {
  const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost:8000/' });
  const w = dom.window; w.TextDecoder = TextDecoder;  // jsdom lacks it; every real browser has it
  await new Promise(r => setTimeout(r, 50));
  const $ = id => w.document.getElementById(id);

  // ---- modals: nothing visible on entry
  const visible = [...w.document.querySelectorAll('.modal-overlay')].filter(m => w.getComputedStyle(m).display !== 'none');
  ok(visible.length === 0, 'no modal visible on entry (' + w.document.querySelectorAll('.modal-overlay').length + ' modals)');
  ['modal-new-prompt', 'modal-prompt-detail', 'modal-new-abtest'].forEach(id => ok($(id).style.display === 'none', id + ' hidden'));
  w.openModal('modal-new-abtest'); ok($('modal-new-abtest').style.display === 'flex', 'openModal still opens it on demand'); w.closeModal('modal-new-abtest');

  // ---- markdown
  const md = w.mdRender('# Title\n\nHello **bold** and `code` <script>alert(1)</script> [site](https://a.b/c?x=1&y=2) [bad](javascript:alert(1))\n\n- one\n  - nested\n- two\n\n1. a\n2. b\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint("hi <b>")\n```\n\n```mermaid\nflowchart TD\n A-->B\n```\n');
  ok(!/<script/i.test(md) && md.includes('&lt;script&gt;'), 'HTML in text is escaped (no XSS)');
  ok(!/href="javascript/i.test(md), 'javascript: links are not linkified');
  ok(md.includes('<h2>Title</h2>') && md.includes('<strong>bold</strong>') && md.includes('<code>code</code>'), 'headings / bold / inline code');
  ok(md.includes('<a href="https://a.b/c?x=1&amp;y=2"'), 'https link rendered, & escaped');
  ok(/<ul><li>one<ul><li>nested<\/li><\/ul><\/li><li>two<\/li><\/ul>/.test(md), 'nested unordered list');
  ok(md.includes('<ol><li>a</li><li>b</li></ol>'), 'ordered list');
  ok(md.includes('<table>') && md.includes('<th>A</th>') && md.includes('<td>2</td>'), 'table');
  ok(md.includes('class="code-block"') && md.includes('print(&quot;hi &lt;b&gt;&quot;)'), 'code block escaped');
  ok(md.includes('class="mermaid-block"') && md.includes('data-src="flowchart TD'), 'mermaid block created');
  ok(w.mdRender('```js\nunclosed').includes('code-block'), 'unclosed fence still renders');
  ok(w.mdRender('') === '' && w.mdRender(null) === '', 'empty input safe');

  // ---- welcome screen
  w.currentUser = { name: 'Mikle Test' };
  w.show('agent');
  await new Promise(r => setTimeout(r, 20));
  ok($('chat-messages').querySelector('.welcome') && $('chat-messages').querySelectorAll('.w-card').length === 6, 'welcome screen with 6 suggestion cards');

  // ---- streaming send with progress steps + final
  w.authToken = 't';
  let calls = [];
  w.fetch = async (url, opts) => {
    calls.push(url);
    if (url.endsWith('/agent/message/stream')) {
      return { ok: true, status: 200, body: sse([
        { type: 'plan', title: 'Working on it', steps: [{ id: 'think', label: 'Understanding your request' }] },
        { type: 'step', id: 'think', status: 'done', detail: 'train_model' },
        { type: 'plan', title: 'Training a model', steps: [{ id: 'load', label: 'Loading' }, { id: 'cv', label: 'Cross-validating' }, { id: 'fit', label: 'Fitting' }] },
        { type: 'step', id: 'load', status: 'active' }, { type: 'step', id: 'load', status: 'done', detail: '300 rows' },
        { type: 'step', id: 'cv', status: 'active', detail: 'comparing 4 model types' },
        { type: 'step', id: 'cv', status: 'done', detail: 'best: xgboost' }, { type: 'step', id: 'fit', status: 'done' },
        { type: 'final', response: 'Model trained!\n\n```mermaid\nflowchart LR\n A-->B\n```\n', conversation_id: 'cid1', message_id: 5,
          data: { model_id: 'm1', model_type: 'xgboost', metrics: { accuracy_test: 0.8123, f1_test: 0.79, baseline_accuracy_test: 0.5, warnings: ['small data'] }, feature_importance: { age: 0.6, plan: 0.4 } } }
      ]) };
    }
    return { ok: true, status: 200, json: async () => ({ conversations: [], balance: 0 }) };
  };
  $('chat-in').value = 'train a model';
  await w.sendMsg();
  await new Promise(r => setTimeout(r, 80));
  const msgs = [...$('chat-messages').querySelectorAll('.msg')];
  ok(msgs.length === 2 && msgs[0].classList.contains('user') && msgs[1].classList.contains('msg-ai'), 'user + assistant messages rendered');
  const card = msgs[1].querySelector('.steps-card');
  ok(card && card.style.display !== 'none', 'steps card visible');
  const rows = [...card.querySelectorAll('.st')];
  ok(rows.length === 4 && rows.every(r => r.classList.contains('done')), 'all 4 steps (think+3) marked done: ' + rows.map(r => r.className).join(','));
  ok(card.querySelector('.st-detail:not([hidden])').textContent.length > 0 && card.classList.contains('finished') && card.classList.contains('collapsed'), 'card finished & collapsed with details');
  ok(card.querySelector('.sc-title').textContent.startsWith('Done'), 'summary title: ' + card.querySelector('.sc-title').textContent);
  ok(msgs[1].querySelector('.model-card') && msgs[1].querySelectorAll('.mc-metric').length === 3 && msgs[1].querySelectorAll('.imp-row').length === 2 && msgs[1].querySelector('.mc-notes'), 'model card: metrics, importance bars, notes');
  ok(msgs[1].querySelector('.mm-view'), 'mermaid placeholder in the answer (CDN not reachable in test)');
  ok(msgs[1].querySelector('.msg-rate') && msgs[1].querySelector('.act-regen'), 'copy/regenerate/rating actions present');
  ok(w.eval('conversationId') === 'cid1' && !$('send-btn').classList.contains('stop'), 'conversation id adopted, send button back to idle');
  ok(!calls.some(u => u.endsWith('/agent/message')), 'used the streaming endpoint (no plain fallback)');

  // ---- plain chat (no steps card shown)
  w.fetch = async (url) => url.endsWith('/stream') ? { ok: true, status: 200, body: sse([
      { type: 'plan', title: 'Working on it', steps: [{ id: 'think', label: 'Understanding your request' }] },
      { type: 'step', id: 'think', status: 'done' }, { type: 'final', response: 'Hi there', conversation_id: 'cid1', message_id: 6, data: null }]) }
    : { ok: true, status: 200, json: async () => ({ conversations: [] }) };
  await w.sendMsg('hello');
  await new Promise(r => setTimeout(r, 60));
  const last = [...$('chat-messages').querySelectorAll('.msg-ai')].pop();
  ok(last.querySelector('.md').textContent.trim() === 'Hi there' && !last.querySelector('.steps-card'), 'simple answer shows no steps card');

  // ---- server error -> retry button
  w.fetch = async () => ({ ok: false, status: 429, json: async () => ({ detail: 'The AI provider is busy right now.' }) });
  await w.sendMsg('again');
  await new Promise(r => setTimeout(r, 30));
  const errMsg = [...$('chat-messages').querySelectorAll('.msg-ai')].pop();
  ok(errMsg.querySelector('.msg-error') && errMsg.textContent.includes('busy') && errMsg.querySelector('button'), 'error message with Try again button');

  // ---- older server (404 on stream) falls back to JSON endpoint
  let seen = [];
  w.fetch = async (url) => { seen.push(url); return url.endsWith('/stream') ? { ok: false, status: 404, json: async () => ({}) }
    : { ok: true, status: 200, json: async () => ({ response: 'fallback ok', conversation_id: 'cid1', message_id: 9, data: null, conversations: [] }) }; };
  await w.sendMsg('fallback please'); await new Promise(r => setTimeout(r, 30));
  ok([...$('chat-messages').querySelectorAll('.msg-ai')].pop().textContent.includes('fallback ok'), 'falls back to /agent/message when /stream is missing');

  // ---- stop button aborts
  w.fetch = (url, opts) => new Promise((res, rej) => { opts.signal.addEventListener('abort', () => { const e = new Error('aborted'); e.name = 'AbortError'; rej(e); }); });
  const p = w.sendMsg('slow one'); await new Promise(r => setTimeout(r, 10));
  ok($('send-btn').classList.contains('stop'), 'send button becomes Stop while generating');
  w.onSendClick(); await p;
  ok(!$('send-btn').classList.contains('stop') && [...$('chat-messages').querySelectorAll('.msg-ai')].pop().textContent.includes('Stopped'), 'Stop aborts and shows a note');

  // ---- slash commands
  const inp = $('chat-in'); inp.value = '/tr'; w.onComposerInput(inp);
  ok($('slash-menu').style.display === 'block' && $('slash-menu').querySelectorAll('.sl-item').length === 1, '/tr shows only /train');
  w.onComposerKey({ key: 'Enter', preventDefault() {}, shiftKey: false });
  ok(inp.value.startsWith('Train a model on'), 'Enter picks the slash command');
  inp.value = 'plain text'; w.onComposerInput(inp); ok($('slash-menu').style.display === 'none', 'no menu for plain text');

  // ---- bot window
  const botData = { spec: { name: 'ShoeBot', description: 'Support for shoes', greeting: 'Hi!' }, success: true, iterations: 1,
    skills: [{ name: 'stock', description: 'checks stock', test_passed: true }],
    test_results: [{ question: 'hours?', ok: true, response: 'We open at 9' }, { question: '(probe: prompt extraction)', ok: true, response: 'no' },
                   { question: '(probe: role hijack)', ok: true, unverified: true, response: '' }] };
  w.fetch = async () => ({ ok: true, status: 200, json: async () => ({ conversations: [] }) });
  w.addAiMsg('Bot is ready', botData, { messageId: 11 });
  ok($('bot-dock').style.display === 'flex', 'bot window opens on a freshly built bot');
  ok($('bb-bot-name').textContent === 'ShoeBot' && $('bb-ready').textContent.includes('ready') && $('bb-ready').textContent.includes('3/3'), 'ready banner + chips');
  ok($('bb-test-results').querySelectorAll('.chk').length === 3 && $('bb-test-results').textContent.includes('Safety'), 'checks list incl. safety probes (friendly names)');
  ok($('bb-test-results').querySelector('.chk.warn'), 'unverified probe shown as a warning, not a failure');
  ok($('bb-skills').textContent.includes('stock') && $('bb-checks-count').textContent === '3/3', 'skills + counter');
  ok(w.document.querySelector('.bot-card .bc-name').textContent === 'ShoeBot', 'bot card in the chat');
  w.bbTab('checks'); ok(!w.document.querySelector('[data-pane=checks]').hidden && w.document.querySelector('[data-pane=preview]').hidden, 'tabs switch');
  w.bbTab('improve'); ok(!w.document.querySelector('[data-pane=improve]').hidden, 'improve tab');

  // history replay doesn't pop the window
  $('bot-dock').style.display = 'none';
  w.addAiMsg('old bot', botData, { replay: true, messageId: 12 });
  ok($('bot-dock').style.display === 'none', 'history replay does not reopen the bot window');


  // ================= paperclip / attachments =================
  ok($('attach-btn') && $('chat-file') && $('chat-file').multiple && $('chat-file').accept.includes('.csv'), 'paperclip button + file input exist');
  const uploads = []; let uploadMode = 'ok'; let sentBody = null;
  w.fetch = async (url, opts) => {
    if (url.endsWith('/agent/upload')) {
      const f = opts.body.get('file'); uploads.push(f.name);
      if (uploadMode === 'bad') return { ok: false, status: 400, json: async () => ({ detail: "Couldn't read this file as a table" }) };
      return { ok: true, status: 200, json: async () => f.name.endsWith('.csv')
        ? { id: 'ab12cd34', name: f.name, kind: 'dataset', size: f.size, rows: 3, n_columns: 3, columns: ['a', 'b', 'c'] }
        : { id: 'ee00ff11', name: f.name, kind: 'document', size: f.size, chars: 20 } };
    }
    if (url.endsWith('/agent/message/stream')) {
      sentBody = JSON.parse(opts.body);
      return { ok: true, status: 200, body: sse([{ type: 'final', response: 'got it', conversation_id: 'cid1', message_id: 21, data: null }]) };
    }
    return { ok: true, status: 200, json: async () => ({ conversations: [] }) };
  };
  w.addChatFiles([new w.File(['a,b,c\n1,2,3'], 'data.csv'), new w.File(['x'], 'virus.exe'), new w.File(['# hi'], 'brief.md')]);
  ok($('attach-strip').querySelectorAll('.att-chip').length === 2 && uploads.join() === 'data.csv,brief.md', 'unsupported .exe is rejected client-side, two files upload');
  ok($('attach-strip').querySelector('.att-spin'), 'chips show an uploading spinner');
  await new Promise(r => setTimeout(r, 30));
  const chips = [...$('attach-strip').querySelectorAll('.att-chip')];
  ok(chips[0].classList.contains('ready') && chips[0].textContent.includes('3 rows') && chips[0].textContent.includes('3 columns'), 'dataset chip shows rows x columns');
  ok(chips[1].classList.contains('ready') && chips[1].querySelector('.ti-file-text'), 'document chip uses the document icon');
  w.removeChatFile(w.eval('pendingFiles[1].key'));
  ok($('attach-strip').querySelectorAll('.att-chip').length === 1, 'x removes a chip');

  $('chat-in').value = '';
  await w.sendMsg();                                  // attachment only, no text typed
  await new Promise(r => setTimeout(r, 50));
  ok(sentBody && sentBody.message === 'Analyze the attached dataset' && JSON.stringify(sentBody.attachments) === '["ab12cd34"]', 'attachment-only send uses a default prompt and passes the upload id');
  const userBubbles = [...$('chat-messages').querySelectorAll('.msg.user')];
  ok(userBubbles[userBubbles.length - 1].querySelector('.bubble-file').textContent.includes('data.csv'), 'user bubble shows the attached file chip');
  ok($('attach-strip').hidden && w.eval('pendingFiles.length') === 0, 'strip is cleared after sending');
  ok(userBubbles[userBubbles.length - 1]._raw === 'Analyze the attached dataset', 'copy/edit text excludes the file note');

  // regenerate keeps the original attachment
  sentBody = null; w.regenerate({ closest: () => ({ remove() {} }) });
  await new Promise(r => setTimeout(r, 50));
  ok(sentBody && JSON.stringify(sentBody.attachments) === '["ab12cd34"]', 'regenerate re-sends the same attachment');

  // history replay renders the paperclip note as a chip, not raw text
  w.addUserMsg('look at this\n\u{1F4CE} report.pdf, data.csv');
  const replay = [...$('chat-messages').querySelectorAll('.msg.user')].pop();
  ok(replay.querySelectorAll('.bubble-file').length === 2 && !replay.textContent.includes('\u{1F4CE}') && replay._raw === 'look at this', 'history note "\u{1F4CE} a, b" becomes chips');

  // upload error surfaces on the chip and is skipped on send
  uploadMode = 'bad'; sentBody = null;
  w.addChatFiles([new w.File(['zzz'], 'broken.csv')]);
  await new Promise(r => setTimeout(r, 30));
  ok($('attach-strip').querySelector('.att-chip.error') && $('attach-strip').textContent.includes("Couldn't read"), 'failed upload shows the server reason on the chip');
  $('chat-in').value = 'hello again'; await w.sendMsg(); await new Promise(r => setTimeout(r, 40));
  ok(sentBody && sentBody.attachments === undefined, 'a failed upload is not sent');

  // too many files
  uploadMode = 'ok';
  w.addChatFiles(Array.from({ length: 7 }, (_, i) => new w.File(['a,b\n1,2'], `f${i}.csv`)));
  ok(w.eval('pendingFiles.length') === 5, 'max 5 files per message');
  w.eval('pendingFiles = []'); w.renderAttachStrip();

  // drag & drop overlay + paste
  const dt = { types: ['Files'], files: [new w.File(['a,b\n1,2'], 'dropped.csv')] };
  let prevented = 0; const ev = { dataTransfer: dt, preventDefault() { prevented++; }, relatedTarget: null, currentTarget: w.document.querySelector('.chat-area') };
  w.onChatDrag(ev, true); ok(!$('drop-overlay').hidden, 'drop overlay appears while dragging files');
  w.onChatDrag(ev, false); ok($('drop-overlay').hidden, 'drop overlay hides when the drag leaves');
  w.onChatDrag({ ...ev, dataTransfer: { types: ['text/plain'], files: [] } }, true); ok($('drop-overlay').hidden, 'dragging plain text does not show the overlay');
  w.onChatDrop(ev); ok(w.eval('pendingFiles.length') === 1 && prevented >= 2, 'dropping a file attaches it');
  w.eval('pendingFiles = []'); w.renderAttachStrip();
  let pasteBlocked = false; w.onComposerPaste({ clipboardData: { files: [new w.File(['a,b\n1,2'], 'pasted.csv')] }, preventDefault() { pasteBlocked = true; } });
  ok(pasteBlocked && w.eval('pendingFiles.length') === 1, 'pasting a file attaches it');
  w.eval('pendingFiles = []'); w.renderAttachStrip();
  const pasteText = { clipboardData: { files: [] }, preventDefault() { throw new Error('plain text paste must not be blocked'); } };
  w.onComposerPaste(pasteText); ok(true, 'pasting plain text is left alone');

  // ================= provider failover note =================
  w.fetch = async (url) => url.endsWith('/stream') ? { ok: true, status: 200, body: sse([
      { type: 'plan', title: 'Working on it', steps: [{ id: 'think', label: 'Understanding your request' }] },
      { type: 'note', text: 'Groq is unavailable right now \u2014 answered by OpenRouter instead.' },
      { type: 'step', id: 'think', status: 'done' },
      { type: 'final', response: 'hello from the fallback', conversation_id: 'cid1', message_id: 30, data: null }]) }
    : { ok: true, status: 200, json: async () => ({ conversations: [] }) };
  await w.sendMsg('ping'); await new Promise(r => setTimeout(r, 50));
  const fb = [...$('chat-messages').querySelectorAll('.msg-ai')].pop();
  ok(fb.querySelector('.steps-card') && !fb.querySelector('.sc-note').hidden && fb.querySelector('.sc-note').textContent.includes('OpenRouter'), 'failover note is shown on the progress card even for a simple answer');
  ok(fb.querySelector('.md').textContent.includes('hello from the fallback'), 'the fallback answer is rendered normally');

  // ================= math + bot preview rendering =================
  const teacher = w.mdRender('Формулы:\n\n| Тождество | Запись | Раскрытие |\n|---|---|---|\n| Квадрат суммы | \\((a+b)^2\\) | \\(a^2 + 2ab + b^2\\) |\n| Модуль | \\(|x|\\) | \\(x\\) |\n\n\\[ (a+b)(a-b)=a^2-b^2 \\]\n\nЦена $5 и $10, это не формула.\n\n```python\nprint("\\\\(not math\\\\)")\n```\n');
  const tdiv = w.document.createElement('div'); tdiv.innerHTML = teacher;
  ok(tdiv.querySelectorAll('.math').length === 5, '5 formulas found (inline in table cells + one display): ' + tdiv.querySelectorAll('.math').length);
  ok(tdiv.querySelector('table') && tdiv.querySelectorAll('tbody tr').length === 2 && tdiv.querySelectorAll('tbody tr')[1].children.length === 3, 'table with formulas (incl. a | inside |x|) keeps its 3 columns');
  ok(tdiv.querySelector('.math[data-display="1"]').dataset.tex === '(a+b)(a-b)=a^2-b^2', 'display formula carries its TeX');
  ok([...tdiv.querySelectorAll('.math[data-display="0"]')].some(e => e.dataset.tex === '|x|'), 'inline |x| formula intact');
  ok(tdiv.textContent.includes('Цена $5 и $10'), 'lone dollar signs stay plain text');
  ok(tdiv.querySelector('.code-block code').textContent.includes('not math') && !tdiv.querySelector('.code-block .math') && tdiv.querySelectorAll('.math').length === 5, 'math syntax inside a code block is left alone');
  const evil = w.mdRender('\\(<img src=x onerror=alert(1)>\\)'); const ediv = w.document.createElement('div'); ediv.innerHTML = evil;
  ok(!ediv.querySelector('img') && ediv.querySelector('.math').dataset.tex.includes('<img'), 'HTML inside a formula is escaped (no XSS)');
  w.renderMathIn(tdiv);   // CDN is unreachable in the test: must not throw, TeX stays visible
  ok(tdiv.querySelector('.math').textContent.length > 0, 'if KaTeX cannot load the TeX source stays visible');

  // bot preview: the reply is rendered as Markdown (tables, bold, code) instead of raw text
  w.eval('bbSpec = {name: "EduGuide", greeting: "Привет! **Я учитель**", system_prompt: "x"}; bbPreviewHistory = [];');
  w.renderBotBuilderResult({ spec: { name: 'EduGuide', description: 'tutor', greeting: 'Привет! **Я учитель**' }, success: true, test_results: [], skills: [] });
  ok($('bb-preview-chat').querySelector('.bb-msg .md strong') && $('bb-preview-chat').textContent.includes('Я учитель') && !$('bb-preview-chat').textContent.includes('**'), 'preview greeting renders Markdown');
  w.fetch = async (url) => ({ ok: true, status: 200, json: async () => ({ response: '## Квадрат суммы\n\n| A | B |\n|---|---|\n| \\((a+b)^2\\) | **ok** |\n\n\\[ a^2-b^2 \\]\n\n<script>alert(1)</script>', notes: [] }) });
  $('bb-preview-input').value = 'формулы?';
  await w.sendBotPreviewMessage(); await new Promise(r => setTimeout(r, 30));
  const pv = $('bb-preview-chat');
  ok(pv.querySelector('.bb-msg .md-table table') && pv.querySelector('.bb-msg h3') && pv.querySelectorAll('.bb-msg .math').length === 2, 'preview reply: heading + table + 2 formulas rendered');
  ok(!pv.innerHTML.includes('<script') && pv.textContent.includes('alert(1)'), 'preview reply cannot inject HTML');
  ok(!pv.textContent.includes('|---|'), 'no raw table syntax left in the preview');

  // ================= model card: quality metrics =================
  w.addAiMsg('Model trained', { model_id: 'm9', model_type: 'xgboost', metrics: {
      accuracy_test: 0.93, f1_test: 0.9, roc_auc_test: 0.88, pr_auc_test: 0.52, balanced_accuracy_test: 0.7,
      recall_minority_test: 0.43, precision_minority_test: 0.4, f1_minority_test: 0.41, threshold: 0.81,
      tuning: { budget: 'balanced', trials: 10, cv_default: 0.4, cv_tuned: 0.46, best_params: { max_depth: 5 } },
      feature_notes: ['Даты → календарные признаки: signup', 'Текст → TF-IDF: review'] }, feature_importance: { signup_month: 0.5 } }, { messageId: 40 });
  const qc = [...$('chat-messages').querySelectorAll('.model-card')].pop();
  const labels = [...qc.querySelectorAll('.mc-metric span')].map(e => e.textContent);
  ok(['PR-AUC', 'Balanced acc.', 'Rare-class recall', 'Rare-class F1'].every(l => labels.includes(l)), 'model card shows the rare-class metrics: ' + labels.join(', '));
  ok(qc.querySelector('.mc-info').textContent.includes('CV 0.4 \u2192 0.46') && qc.querySelector('.mc-info').textContent.includes('threshold') && qc.querySelector('.mc-info').textContent.includes('TF-IDF'), 'tuning gain, threshold and date/text notes are shown');
  w.addAiMsg('Model trained', { model_type: 'logistic', metrics: { accuracy_test: 0.9, tuning: { budget: 'fast', skipped: "budget is 'fast'" } } }, { messageId: 41 });
  ok([...$('chat-messages').querySelectorAll('.model-card')].pop().querySelector('.mc-info').textContent.includes('Tuning skipped'), 'a skipped tuning is stated, not hidden');

  // ================= reliability: trust badge, intervals, fixes, target buttons =================
  w.addAiMsg('Model trained', { model_type: 'logistic', metrics: { accuracy_test: 0.877, accuracy_test_ci: [0.80, 0.93], n_test: 114,
      trust: { level: 'medium', reasons: ['Тестовая выборка мала (114 строк).'] },
      data_checks: [{ kind: 'leak', message: 'Столбец «outcome_code» почти полностью предсказывает цель - исключён.' }, { kind: 'duplicates', message: 'Удалено 569 дублирующихся строк.' }] } }, { messageId: 50 });
  const rc = [...$('chat-messages').querySelectorAll('.model-card')].pop();
  ok(rc.querySelector('.trust.trust-medium') && rc.querySelector('.trust').textContent.includes('Reliability: medium') && rc.querySelector('.trust li').textContent.includes('Тестовая выборка'), 'trust badge: level + reasons');
  ok(rc.querySelector('.mc-metric em').textContent.includes('95% CI 0.80\u20130.93'), 'accuracy shows its 95% interval');
  ok(rc.querySelectorAll('.mc-fixed li').length === 2 && rc.querySelector('.mc-fixed').textContent.includes('outcome_code'), 'what the platform fixed (leak, duplicates) is listed');
  w.addAiMsg('ok', { model_type: 'logistic', metrics: { accuracy_test: 0.95, trust: { level: 'high', reasons: [] } } }, { messageId: 51 });
  const hc = [...$('chat-messages').querySelectorAll('.model-card')].pop();
  ok(hc.querySelector('.trust-high') && hc.textContent.includes('no concerns found'), 'high trust says no concerns were found');
  w.addAiMsg('bad', { model_type: 'x', metrics: { accuracy_test: 0.5, trust: { level: 'low', reasons: ['Модель не лучше наивного прогноза'] } } }, { messageId: 52 });
  ok([...$('chat-messages').querySelectorAll('.model-card')].pop().querySelector('.trust-low'), 'low trust is styled as low');

  let askedBody = null;
  w.fetch = async (url, opts) => url.endsWith('/stream') ? (askedBody = JSON.parse(opts.body), { ok: true, status: 200, body: sse([{ type: 'final', response: 'training...', conversation_id: 'cid1', message_id: 60, data: null }]) })
    : { ok: true, status: 200, json: async () => ({ conversations: [] }) };
  w.addAiMsg('Which column should the model predict?', { needs_target: true, path: 'datasets/a.csv', target_candidates: [
      { column: 'z9', task: 'regression', score: 1, reason: 'plain numeric column' },
      { column: '<img src=x onerror=alert(1)>', task: 'binary_classification', score: 1, reason: 'x' }] }, { messageId: 61 });
  const cand = [...$('chat-messages').querySelectorAll('.cand')].slice(-2);
  ok(cand.length === 2 && cand[0].textContent.includes('z9') && cand[0].textContent.includes('regression') && cand[1].textContent.includes('yes/no'), 'one button per candidate column, with its task');
  ok(!$('chat-messages').querySelector('.cand img'), 'candidate names cannot inject HTML');
  cand[0].click(); await new Promise(r => setTimeout(r, 50));
  ok(askedBody && askedBody.message === 'Train a model on datasets/a.csv, target z9', 'clicking a candidate sends the training request with that target: ' + (askedBody && askedBody.message));

  ok(!$('chat-file').accept.includes('.xls,') && !$('chat-file').accept.endsWith('.xls') && !w.eval("ATTACH_EXT").includes('.xls'), 'legacy .xls is no longer offered (it needs xlrd)');

  w.addAiMsg('Model built and verified', { model_id: 'm77', metrics: { accuracy_test: 0.9, verified: true, n_test: 80, self_reported: { accuracy: 0.999 }, trust: { level: 'low', reasons: ['Скрипт заявил accuracy 0.999, а платформа получила 0.9: заявленная метрика была завышена.'] } } }, { messageId: 70 });
  const vc = [...$('chat-messages').querySelectorAll('.model-card')].pop();
  ok(vc.querySelector('.mc-info').textContent.includes('measured by the platform on 80 rows') && vc.querySelector('.trust-low') && vc.textContent.includes('завышена'), 'a verified generated model says who measured it and shows an inflated claim');

  // ================= forecast chart =================
  const hist = Array.from({ length: 60 }, (_, i) => 100 + i * 0.5 + (i % 7) * 3), dates = i => `2024-${String(1 + Math.floor(i / 28)).padStart(2, '0')}-${String(1 + (i % 28)).padStart(2, '0')}`;
  const fut = Array.from({ length: 14 }, (_, i) => 131 + i * 0.5);
  w.addAiMsg('**Forecast of «sales»**', { model_id: 'f1', model_type: 'forecast_ridge_level', route: 'forecast',
    forecast: { target: 'sales', freq: 'D', history: { dates: hist.map((_, i) => dates(i)), values: hist },
                future: { dates: fut.map((_, i) => dates(60 + i)), values: fut, lower: fut.map(v => v - 6), upper: fut.map(v => v + 6) } },
    metrics: { mae_test: 2.6, rmse_test: 3.1, smape_test: 2.4, mase_test: 0.8, baseline_mae_test: 3.8, coverage_80_test: 0.62,
               trust: { level: 'medium', reasons: ['Интервал «80%» на новых данных покрыл 55% значений.'] } } }, { messageId: 80 });
  const fcard = [...$('chat-messages').querySelectorAll('.fc-card')].pop();
  ok(fcard && fcard.querySelector('svg.fc') && fcard.querySelectorAll('polyline').length === 2 && fcard.querySelector('polygon.fc-band'), 'forecast chart: history line, forecast line, 80% band');
  const pts = [...fcard.querySelectorAll('polyline, polygon')].map(e => e.getAttribute('points')).join(' ');
  ok(!/NaN|undefined|Infinity/.test(pts + fcard.innerHTML), 'chart has no NaN / undefined coordinates');
  ok(fcard.querySelector('.fc-legend').textContent.includes('forecast') && fcard.querySelector('svg').getAttribute('aria-label').includes('sales'), 'legend + accessible label');
  const fm = [...$('chat-messages').querySelectorAll('.model-card')].pop();
  ok(['MAE', 'sMAPE %', 'MASE', '80% range hit rate'].every(l => [...fm.querySelectorAll('.mc-metric span')].map(e => e.textContent).includes(l)) && fm.querySelector('.trust-medium'), 'forecast metrics + trust badge shown under the chart');
  w.addAiMsg('flat series', { forecast: { target: 'x', freq: 'D', history: { dates: ['2024-01-01', '2024-01-02'], values: [5, 5] }, future: { dates: ['2024-01-03'], values: [5], lower: [5], upper: [5] } }, metrics: {} }, { messageId: 81 });
  ok(![...$('chat-messages').querySelectorAll('.fc-card')].pop().innerHTML.match(/NaN|Infinity/), 'a constant series does not break the chart (no divide by zero)');
  w.addAiMsg('bad payload', { forecast: { future: { values: [] }, history: { values: [] } } }, { messageId: 82 });
  ok(true, 'an empty forecast payload does not throw');

  let fcBody = null;
  w.fetch = async (url, opts) => url.endsWith('/stream') ? (fcBody = JSON.parse(opts.body), { ok: true, status: 200, body: sse([{ type: 'final', response: 'ok', conversation_id: 'cid1', message_id: 83, data: null }]) })
    : { ok: true, status: 200, json: async () => ({ conversations: [] }) };
  w.addAiMsg('Which column?', { needs_target: true, path: 'datasets/two.csv', ask_template: 'Forecast column {column} in {path}, date column date',
    target_candidates: [{ column: 'a', task: 'regression', score: 1, reason: 'x' }, { column: 'b', task: 'regression', score: 1, reason: 'x' }] }, { messageId: 84 });
  [...$('chat-messages').querySelectorAll('.cand')].slice(-2)[1].click(); await new Promise(r => setTimeout(r, 50));
  ok(fcBody && fcBody.message === 'Forecast column b in datasets/two.csv, date column date', 'a forecast candidate button sends the forecast request, not a training request: ' + (fcBody && fcBody.message));

  // ================= keyboard layout / IME safety =================
  const inp2 = $('chat-in'); inp2.value = 'привет'; let pd = 0, sent = false; const origSend = w.sendMsg; 
  const key = (o) => ({ key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13, preventDefault() { pd++; }, ...o });
  w.onComposerKey(key({ isComposing: true })); ok(pd === 0, 'Enter during IME composition is ignored (no preventDefault)');
  w.onComposerKey(key({ keyCode: 229 })); ok(pd === 0, 'keyCode 229 (IME) is ignored');
  for (const k of ['Shift', 'Alt', 'Control', 'Meta', 'CapsLock', ' ', 'Process']) w.onComposerKey(key({ key: k, keyCode: 0 }));
  ok(pd === 0, 'modifier / layout-switch keys (Shift, Alt, Ctrl, Meta, CapsLock, Space) are never prevented');
  inp2.value = '';

  console.log(failures ? `\n${failures} FAILED` : '\nALL UI CHECKS PASSED');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error('CRASH', e); process.exit(2); });
