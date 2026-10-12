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
