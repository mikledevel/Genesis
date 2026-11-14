/* Small UI companion: no API credentials, payment logic, or remote dependencies. */
window.GenesisWorkspace = (() => {
  const pages = [
    ['dashboard','Overview','Your workspace, activity and recent conversations','ti-layout-dashboard'],
    ['agent','AI Agent','Analyze data, build bots and write code','ti-message-chatbot'],
    ['projects','Projects','Organize your work','ti-folder'],
    ['builder','Pipeline Builder','Build a visual machine learning pipeline','ti-box-align-left'],
    ['registry','Model Registry','Browse trained models','ti-database'],
    ['store','Agent Store','Explore and manage AI agents','ti-robot'],
    ['marketplace','Marketplace','Discover published models','ti-building-store'],
    ['memory','Memory','Explore your knowledge graph','ti-brain'],
    ['promptlab','Prompt Lab','Compare and improve prompts','ti-flask'],
    ['settings','Settings','Profile, API keys, plans and usage','ti-settings']
  ];
  const templates = [
    {title:'Customer support bot',category:'Bots',icon:'ti-headset',description:'Define tone, scope, escalation and answers.',prompt:'Help me design a customer support bot for [business]. Ask me about the audience, knowledge sources and escalation rules before building it.'},
    {title:'Dataset quality review',category:'Data',icon:'ti-table',description:'Inspect missing values, targets and data leakage.',prompt:'Analyze my attached dataset. Review missing values, duplicates, potential data leakage and suitable target columns. Explain the findings before training anything.'},
    {title:'Train and compare models',category:'Data',icon:'ti-chart-dots',description:'Plan a baseline, validation and model comparison.',prompt:'Help me train a model on my attached dataset to predict [target]. First confirm the target, task type and validation approach, then compare suitable baseline models.'},
    {title:'Sales assistant',category:'Bots',icon:'ti-messages',description:'Turn product information into a helpful assistant.',prompt:'Help me design a sales assistant for [product or service]. Ask about product information, qualification questions, tone and when to hand over to a human.'},
    {title:'Workflow diagram',category:'Planning',icon:'ti-hierarchy-3',description:'Map steps, decisions and handoffs.',prompt:'Create a Mermaid diagram for [workflow]. Identify the main actors, decisions and error paths. Ask me for missing details first.'},
    {title:'Find a public dataset',category:'Data',icon:'ti-search',description:'Explore data sources for your next experiment.',prompt:'Help me find a public dataset for [problem]. Consider licensing, target availability, size and freshness. Explain which sources I should verify.'},
    {title:'Python coding assistant',category:'Code',icon:'ti-code',description:'Draft a solution with clear assumptions and tests.',prompt:'Help me write Python code for [task]. Clarify the inputs, expected output and constraints, then propose a small implementation with meaningful checks.'},
    {title:'Evaluate an AI agent',category:'Planning',icon:'ti-checklist',description:'Create test cases for reliability and tool failures.',prompt:'Help me create an evaluation plan for [agent]. Include realistic tasks, flaky tool calls, state persistence, unsafe requests and measurable success criteria.'}
  ];
  let owner = null, draftTimer = null, storageWarning = false;
  let searchActions = [], searchIndex = -1, category = 'All';
  const el = id => document.getElementById(id);
  const key = name => owner ? 'genesis_workspace_' + owner + '_' + name : null;
  function read(name, fallback) {
    try { const value = JSON.parse(localStorage.getItem(key(name))); return value ?? fallback; } catch { return fallback; }
  }
  function write(name, value) {
    if (!owner) return;
    try { localStorage.setItem(key(name), JSON.stringify(value)); return true; }
    catch { if (!storageWarning) { storageWarning = true; toast('Browser storage is unavailable. Drafts and pins cannot be saved on this device.','error'); } }
  }
  function pinnedIds() { const ids = read('pins', []); return Array.isArray(ids) ? ids.filter(x => typeof x === 'string') : []; }
  function isPinned(id) { return pinnedIds().includes(id); }
  function togglePin(id) {
    write('pins', isPinned(id) ? pinnedIds().filter(x => x !== id) : [...pinnedIds(), id]);
    renderConversations(); refreshRecent();
  }
  function sortedConversations(items) {
    const pins = new Set(pinnedIds());
    return [...items].sort((a,b) => Number(pins.has(b.id)) - Number(pins.has(a.id)));
  }
  function draftId() { return conversationId || 'new'; }
  function drafts() { const d = read('drafts', {}); return d && typeof d === 'object' && !Array.isArray(d) ? d : {}; }
  function persistDraft() {
    clearTimeout(draftTimer);
    if (!owner) return;
    const all = drafts(), value = el('chat-in').value;
    if (value) all[draftId()] = {text:value.slice(0,50000),updated:Date.now()}; else delete all[draftId()];
    const bounded = Object.fromEntries(Object.entries(all).sort((a,b) => (b[1]?.updated||0)-(a[1]?.updated||0)).slice(0,50));
    const saved = write('drafts', bounded);
    updateDraftStatus(value ? (saved ? 'Draft saved on this device' : 'Draft could not be saved') : '');
  }
  function queueDraft() { clearTimeout(draftTimer); draftTimer = setTimeout(persistDraft,350); }
  function updateDraftStatus(text) {
    const status = el('draft-status'); if (!status) return;
    status.classList.toggle('visible',Boolean(text));
    status.querySelector('span').textContent = text;
  }
  function restoreDraft() {
    clearTimeout(draftTimer);
    const saved = drafts()[draftId()];
    el('chat-in').value = typeof saved?.text === 'string' ? saved.text : '';
    autoResize(el('chat-in'));
    updateDraftStatus(saved?.text ? 'Draft restored from this device' : '');
  }
  function clearDraft() { el('chat-in').value = ''; persistDraft(); autoResize(el('chat-in')); el('chat-in').focus(); }
  function removeDraft(id) { const all = drafts(); delete all[id || 'new']; write('drafts', all); updateDraftStatus(''); }
  function initialize() {
    owner = currentUser.id;
    storageWarning = false;
    el('workspace-greeting').textContent = 'Workspace overview';
    document.querySelector('.skip-link').href = '#workspace-main';
    el('chat-in').value = '';
    conversationId = null;
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active',p.id === 'panel-dashboard'));
    document.querySelectorAll('#app>nav .nav-btn').forEach(b => b.classList.toggle('active',b.id === 'nb-dashboard'));
    renderWelcome(); renderConversations(); restoreDraft(); navigation('dashboard'); refreshRecent();
  }
  function logout() {
    persistDraft(); clearTimeout(draftTimer); owner = null;
    document.querySelector('.skip-link').href = '#landing-main';
    document.querySelectorAll('.workspace-dialog[open]').forEach(d => d.close());
  }
  function navigation(id) {
    document.querySelectorAll('#app>nav .nav-btn').forEach(b => {
      if (b.id === 'nb-' + id) b.setAttribute('aria-current','page'); else b.removeAttribute('aria-current');
    });
    const extra = !['dashboard','agent','projects'].includes(id);
    el('nb-workspace-more').classList.toggle('active',extra);
    el('nb-workspace-more').setAttribute('aria-label',extra ? 'More tools, ' + (pages.find(p => p[0] === id)?.[1] || id) : 'More tools');
    document.querySelector('.agent-sidebar')?.classList.remove('open');
    el('agent-sidebar-toggle')?.setAttribute('aria-expanded','false');
  }
  function openDialog(id) {
    if (!currentUser) return;
    document.querySelectorAll('.workspace-dialog[open]').forEach(d => d.close());
    el(id).showModal();
  }
  function closeDialog(id) { el(id).close(); }
  function openSearch() { openDialog('workspace-search'); el('workspace-query').value = ''; renderSearch(); el('workspace-query').focus(); }
  function resultButton(title,description,icon,action) {
    const button = document.createElement('button'); button.type = 'button';
    const i = document.createElement('i'); i.className = 'ti ' + icon; i.setAttribute('aria-hidden','true');
    const text = document.createElement('span'); text.textContent = title;
    if (description) { const small = document.createElement('small'); small.textContent = description; text.appendChild(small); }
    button.append(i,text); button.addEventListener('click',action); return button;
  }
  function group(container,title) { const label = document.createElement('div'); label.className = 'workspace-result-group'; label.textContent = title; container.appendChild(label); }
  function renderSearch() {
    const q = el('workspace-query').value.trim().toLowerCase(), list = el('workspace-results');
    list.replaceChildren(); searchActions = []; searchIndex = -1;
    function add(title,description,icon,action) {
      const run = () => { closeDialog('workspace-search'); action(); };
      searchActions.push(run); list.appendChild(resultButton(title,description,icon,run));
    }
    group(list,'Workspace');
    pages.filter(p => p.slice(1,3).join(' ').toLowerCase().includes(q)).forEach(p => add(p[1],p[2],p[3],() => show(p[0])));
    if (!q || 'templates starter tasks'.includes(q)) add('Starter templates','Choose an editable starting point','ti-template',openTemplates);
    const chats = sortedConversations(allConvos).filter(c => (c.title || 'Untitled').toLowerCase().includes(q)).slice(0,6);
    if (chats.length) { group(list,'Conversations'); chats.forEach(c => add(c.title || 'Untitled',isPinned(c.id) ? 'Pinned conversation' : 'Continue conversation','ti-message',() => { show('agent'); loadChat(c.id); })); }
    el('command-count').textContent = searchActions.length + ' results';
    if (!searchActions.length) { const empty = document.createElement('p'); empty.className='workspace-empty'; empty.textContent='No matches. Try a tool name or a conversation title.'; list.appendChild(empty); }
  }
  function openMore() {
    const list = el('workspace-more-results'); list.replaceChildren();
    const active = document.querySelector('.panel.active')?.id;
    pages.forEach(p => {
      const button = resultButton(p[1],p[2],p[3],() => { closeDialog('workspace-more'); show(p[0]); });
      if (active === 'panel-' + p[0]) button.setAttribute('aria-current','page'); list.appendChild(button);
    });
    list.appendChild(resultButton('Search workspace','Find tools and conversations','ti-search',openSearch));
    list.appendChild(resultButton('Starter templates','Browse editable task prompts','ti-template',openTemplates));
    list.appendChild(resultButton('Sign out','End this session','ti-logout',() => { closeDialog('workspace-more'); doLogout(); }));
    openDialog('workspace-more');
  }
  function openTemplates() { category='All'; el('template-query').value=''; renderTemplates(); openDialog('workspace-templates'); }
  function renderTemplates() {
    const filters = el('template-filters'); filters.replaceChildren();
    ['All','Bots','Data','Planning','Code'].forEach(value => {
      const b = document.createElement('button'); b.type='button'; b.textContent=value; b.setAttribute('aria-pressed',category===value);
      b.onclick=() => { category=value; renderTemplates(); }; filters.appendChild(b);
    });
    const q = el('template-query').value.trim().toLowerCase(), list = el('template-results'); list.replaceChildren();
    templates.filter(t => (category==='All' || t.category===category) && (t.title+' '+t.description).toLowerCase().includes(q)).forEach(t => {
      list.appendChild(resultButton(t.title,t.description,t.icon,() => useTemplate(t)));
    });
    if (!list.children.length) { const p=document.createElement('p'); p.className='workspace-empty'; p.textContent='No templates match. Try another category or search.'; list.appendChild(p); }
  }
  function useTemplate(template) {
    if (sending) { toast('Wait for the current response before starting another task.','info'); return; }
    if (el('chat-in').value.trim() && !confirm('Replace the current draft with this template?')) return;
    document.querySelectorAll('.workspace-dialog[open]').forEach(d => d.close());
    show('agent'); el('chat-in').value = template.prompt; autoResize(el('chat-in')); persistDraft(); el('chat-in').focus();
    const start = template.prompt.indexOf('['), end = template.prompt.indexOf(']',start);
    if (start>=0 && end>start) el('chat-in').setSelectionRange(start,end+1);
    toast('Template ready. Edit the details, then send.');
  }
  function addBlock(id) {
    if (!PBLOCKS[id]) return;
    pipeline.push(id); renderPipeline();
    if (document.getElementById('modal-pipeline-canvas')) renderPipelineInModal();
    toast('Added ' + PBLOCKS[id].name);
  }
  function refreshRecent() {
    const list = el('workspace-recent-list'); if (!list) return; list.replaceChildren();
    sortedConversations(allConvos).slice(0,5).forEach(c => {
      list.appendChild(resultButton(c.title || 'Untitled',isPinned(c.id) ? 'Pinned' : 'Conversation',isPinned(c.id)?'ti-pin':'ti-message',() => { show('agent'); loadChat(c.id); }));
    });
    if (!list.children.length) {
      const p=document.createElement('p'); p.className='workspace-empty'; p.textContent='Your conversations will appear here. Start with a template or open the AI Agent.'; list.appendChild(p);
    }
  }
  document.addEventListener('keydown',event => {
    if (!owner) return;
    if (event.key === 'Escape') {
      const dialog = document.querySelector('.workspace-dialog[open]');
      if (dialog) { event.preventDefault(); dialog.close(); return; }
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase()==='k') { event.preventDefault(); openSearch(); }
    const dialog = el('workspace-search');
    if (!dialog.open || !dialog.contains(event.target)) return;
    const buttons = [...el('workspace-results').querySelectorAll('button')];
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!buttons.length) return;
      const current = buttons.indexOf(document.activeElement);
      searchIndex = current < 0 ? (event.key === 'ArrowDown' ? 0 : buttons.length - 1) :
        (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
      buttons.forEach((button,index) => button.classList.toggle('command-active',index === searchIndex));
      buttons[searchIndex].focus(); buttons[searchIndex].scrollIntoView({block:'nearest'});
    }
    if (event.key === 'Enter' && event.target === el('workspace-query') && searchActions.length) {
      event.preventDefault(); searchActions[0]();
    }
  });
  let tooltipTarget = null;
  function hideTooltip() {
    const tip = el('ui-tooltip');
    if (tooltipTarget) {
      tooltipTarget.removeAttribute('aria-describedby');
      if (tooltipTarget.dataset.tooltip) { tooltipTarget.title = tooltipTarget.dataset.tooltip; delete tooltipTarget.dataset.tooltip; }
    }
    tooltipTarget = null; if (tip) tip.hidden = true;
  }
  function showTooltip(event) {
    const target = event.target.closest('[title], [data-tooltip]');
    if (!target || target === tooltipTarget) return;
    hideTooltip();
    const text = target.getAttribute('title') || target.dataset.tooltip;
    if (!text) return;
    let tip = el('ui-tooltip');
    if (!tip) { tip = document.createElement('div'); tip.id = 'ui-tooltip'; tip.setAttribute('role','tooltip'); document.body.append(tip); }
    tooltipTarget = target; target.dataset.tooltip = text; target.removeAttribute('title');
    target.setAttribute('aria-describedby','ui-tooltip'); tip.textContent = text; tip.hidden = false;
    const rect = target.getBoundingClientRect();
    tip.style.left = Math.max(12,Math.min(rect.left,innerWidth-tip.offsetWidth-12)) + 'px';
    tip.style.top = Math.max(8,Math.min(rect.bottom+8,innerHeight-tip.offsetHeight-8)) + 'px';
  }
  document.addEventListener('pointerover',showTooltip);
  document.addEventListener('focusin',showTooltip);
  document.addEventListener('pointerout',event => { if (tooltipTarget && !tooltipTarget.contains(event.relatedTarget)) hideTooltip(); });
  document.addEventListener('focusout',hideTooltip);
  document.addEventListener('keydown',event => { if (event.key === 'Escape') hideTooltip(); });
  document.addEventListener('scroll',hideTooltip,true);
  document.addEventListener('DOMContentLoaded',() => {
    [['nb-dashboard','Workspace'],['nb-builder','Build'],['nb-marketplace','Publish'],['nb-memory','Knowledge & evaluation']].forEach(([id,title]) => {
      const label = document.createElement('div'); label.className = 'nav-group-label'; label.textContent = title; el(id)?.before(label);
    });
    document.querySelectorAll('#pipeline-blocks [data-block]').forEach(item => { item.querySelector('span').innerHTML = GenesisIcons.block(item.dataset.block); });
    document.querySelectorAll('#app .topbar-btn').forEach(button => { if (!button.title) button.title = button.textContent.trim(); });
  });
  window.addEventListener('pagehide',persistDraft);
  document.addEventListener('visibilitychange',() => { if (document.hidden) persistDraft(); });
  return {initialize,logout,navigation,openSearch,renderSearch,openMore,openTemplates,renderTemplates,closeDialog,
    isPinned,togglePin,sortedConversations,refreshRecent,addBlock,persistDraft,queueDraft,restoreDraft,clearDraft,removeDraft};
})();
