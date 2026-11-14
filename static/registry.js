/* Реестр: фильтры и выбор работают только с загруженными метаданными. */
window.GenesisRegistry = (() => {
  /** @typedef {{model_id:string, task?:string, model_type?:string, target?:string, features?:string[], metrics?:Object<string, unknown>, created_at?:string,name?:string,version?:number}} RegistryModel */
  /** @type {RegistryModel[]} */
  let models = [];
  const selected = new Set();
  let request = 0;
  let state = 'idle';
  const el = id => document.getElementById(id);
  function visible() {
    const query = el('registry-query').value.trim().toLowerCase();
    const sort = el('registry-sort').value;
    return models.filter(m => (regTask === 'all' || m.task === regTask) &&
      [m.model_id, m.model_type, m.target, GenesisModels.label(m)].some(v => String(v || '').toLowerCase().includes(query)))
      .sort((a, b) => sort === 'name' ? a.model_id.localeCompare(b.model_id) :
        String(sort === 'oldest' ? a.created_at || '' : b.created_at || '').localeCompare(String(sort === 'oldest' ? b.created_at || '' : a.created_at || '')));
  }
  function selection(items = visible()) {
    el('registry-selection').textContent = selected.size + ' selected';
    el('registry-export').disabled = selected.size === 0;
    const all = el('registry-select-all');
    const count = items.filter(m => selected.has(m.model_id)).length;
    all.disabled = !items.length || state !== 'ready';
    all.checked = count > 0 && count === items.length;
    all.indeterminate = count > 0 && count < items.length;
  }
  function message(title, detail, retry = false) {
    const cell = document.createElement('td'); cell.colSpan = 8;
    const box = document.createElement('div'); box.className = 'table-state';
    const heading = document.createElement('strong'); heading.textContent = title;
    const text = document.createElement('p'); text.textContent = detail;
    box.append(heading, text);
    if (retry) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'btn btn-secondary';
      button.textContent = 'Retry loading'; button.onclick = load; box.append(button);
    }
    cell.append(box); const row = document.createElement('tr'); row.append(cell); el('reg-body').replaceChildren(row);
  }
  function render() {
    if (state !== 'ready') return;
    const items = visible(), body = el('reg-body'); body.replaceChildren();
    el('reg-count').textContent = items.length + ' of ' + models.length + ' loaded';
    selection(items);
    if (!items.length) {
      message(models.length ? 'No matching models' : 'No trained models yet', models.length ? 'Change the search or task filter to see more models.' : 'Open the Agent to analyze a dataset and train your first model.');
      return;
    }
    items.forEach(model => {
      const row = document.createElement('tr'); row.classList.toggle('is-selected', selected.has(model.model_id));
      const checkCell = document.createElement('td'); checkCell.className = 'selection-cell';
      const check = document.createElement('input'); check.type = 'checkbox'; check.checked = selected.has(model.model_id);
      check.setAttribute('aria-label', 'Select ' + model.model_id);
      check.onchange = () => { if (check.checked) selected.add(model.model_id); else selected.delete(model.model_id); row.classList.toggle('is-selected',check.checked); selection(); };
      checkCell.append(check); row.append(checkCell);
      const idCell = document.createElement('td');
      const link = document.createElement('button'); link.type = 'button'; link.className = 'model-detail-link';
      const name = document.createElement('span'); name.className = 'model-friendly-name'; name.textContent = GenesisModels.label(model);
      const identity = document.createElement('span'); identity.className = 'model-short-id'; identity.textContent = GenesisModels.identity(model);
      link.append(name,identity); link.title = model.model_id; link.onclick = () => showModelDetail(model.model_id);
      idCell.append(link); row.append(idCell);
      const metrics = model.metrics || {};
      const key = ['accuracy_test','r2_test','accuracy','r2'].find(key => typeof metrics[key] === 'number' && Number.isFinite(metrics[key]));
      const metric = key ? [key,metrics[key]] : Object.entries(metrics).find(([, value]) => typeof value === 'number' && Number.isFinite(value));
      const values = [String(model.task || '—').replaceAll('_', ' '), model.model_type || '—', metric ? metric[0] + ': ' + metric[1].toFixed(3) : '—', String((model.features || []).length), model.target || '—', model.created_at?.slice(0,10) || '—'];
      values.forEach((value,index) => { const cell = document.createElement('td'); cell.textContent = value; if ([2,3,5].includes(index)) cell.className = 'mono'; row.append(cell); });
      body.append(row);
    });
  }
  async function load() {
    const epoch = ++request; state = 'loading'; selected.clear(); models = []; selection([]);
    el('reg-count').textContent = 'Loading models…';
    el('reg-body').setAttribute('aria-busy','true');
    el('reg-body').innerHTML = Array.from({length:5}, () => '<tr>' + '<td><span class="ui-skeleton"></span></td>'.repeat(8) + '</tr>').join('');
    try {
      const url = API + '/registry/models' + (regTask === 'all' ? '' : '?task=' + encodeURIComponent(regTask));
      const response = await apiFetch(url);
      if (!response.ok) throw new Error('Registry request failed');
      const data = await response.json();
      if (!Array.isArray(data.models)) throw new Error('Invalid model list');
      if (epoch !== request) return;
      models = data.models.filter(m => m && typeof m.model_id === 'string');
      state = 'ready'; render();
    } catch {
      if (epoch !== request) return;
      state = 'error'; el('reg-count').textContent = 'Unavailable';
      message('Models could not be loaded', 'Check your connection and try again. Your models have not changed.', true);
    } finally { if (epoch === request) el('reg-body').removeAttribute('aria-busy'); }
  }
  function selectAll(checked) {
    visible().forEach(m => { if (checked) selected.add(m.model_id); else selected.delete(m.model_id); }); render();
  }
  function exportSelected() {
    const data = models.filter(m => selected.has(m.model_id)); if (!data.length) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'genesis-models.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000); toast('Exported metadata for ' + data.length + ' models','success');
  }
  return {load,render,selectAll,exportSelected};
})();
