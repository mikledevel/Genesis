/* Прогнозный отчёт: явное подтверждение данных и экспорт готового результата. */
window.GenesisReports = (() => {
  let dialog, uploaded, payload, ownerId, busy = false;
  let language = localStorage.getItem('genesis-report-language') || (navigator.language.startsWith('ru') ? 'ru' : 'en');
  const ru = {
    'The metric needs numeric values. For decimal-comma CSV, save as XLSX or export with decimal points.':'Метрика должна содержать числа. Если CSV использует десятичную запятую, сохраните файл как XLSX или экспортируйте числа с точкой.',
    'Forecast report':'Прогнозный отчёт','Close':'Закрыть','Recent reports':'Мои отчёты',
    'Choose a column':'Выберите столбец','Date column':'Столбец с датой','Metric column':'Столбец с метрикой',
    '1. Sales table (CSV or XLSX)':'1. Таблица продаж (CSV или XLSX)','2. Date':'2. Дата','3. Metric':'3. Метрика',
    'Forecast periods':'Горизонт в периодах','Duplicate dates':'Повторяющиеся даты','Sum per period':'Сумма за период',
    'Average per period':'Среднее за период','Check data':'Проверить данные','Confirm and build report':'Подтвердить и построить отчёт',
    'Choose a file to begin.':'Выберите файл, чтобы начать.',
    'Upload a table, confirm the date and metric, then review the checks. A one-off report and review is priced by agreement; no subscription is purchased here.':'Загрузите таблицу, укажите дату и метрику, затем проверьте данные. Цена разового отчёта с разбором согласуется отдельно. Здесь вы не покупаете подписку.',
    'Use aggregated sales only. Do not include customer names, emails or payment information. We cannot account for unknown promotions or stock-outs.':'Нужны только агрегированные продажи. Удалите имена клиентов, email и платёжные данные. Прогноз не учитывает неизвестные акции и отсутствие товара.',
    'Uploading and checking the table…':'Загружаем и проверяем таблицу…',
    'Confirm your columns before starting. No training has run.':'Подтвердите столбцы. Обучение ещё не запускалось.',
    'Choose both columns and a whole horizon from 1 to 365.':'Выберите оба столбца и целый горизонт от 1 до 365.',
    'Checking frequency, gaps and available history…':'Проверяем частоту, пропуски и длину истории…',
    'Ready to run. This starts one server forecast job. Accuracy and calibrated intervals are not guaranteed.':'Можно запускать. Будет выполнен один серверный прогноз. Точность и калибровка диапазонов не гарантируются.',
    'The file passes the checks. Server execution is not enabled for this account. Arrange the one-off report and review with the pilot organiser; no money has been charged.':'Таблица прошла проверку. Серверный запуск для этого аккаунта пока не включён. Согласуйте разовый отчёт с организатором пилота. Деньги не списаны.',
    'The report is being prepared. You can close this window and reopen the job from Recent reports.':'Отчёт готовится. Окно можно закрыть и открыть результат через «Мои отчёты».',
    'Saved report · downloading does not train a new model.':'Сохранённый отчёт. Скачивание не запускает обучение заново.',
    'Only the first 30 columns are shown. For this pilot, upload a table with just the date and metric.':'Показаны первые 30 столбцов. Для пилота оставьте в таблице только дату и нужную метрику.',
    'No forecast jobs yet.':'Прогнозов пока нет.', 'Report EN':'Отчёт EN','Forecast CSV':'Прогноз CSV',
    'Could not load recent reports':'Не удалось загрузить отчёты', 'Upload failed. Please retry.':'Не удалось загрузить файл. Попробуйте ещё раз.',
    'Check failed. Please retry.':'Не удалось проверить данные. Попробуйте ещё раз.',
    'Many periods have zero values. Sparse sales require special care; do not use this as an inventory purchase plan.':'Во многих периодах продажи нулевые. Такой ряд требует отдельной проверки. Не используйте этот прогноз как план закупок.'
  };
  function tr(value) {
    if (language !== 'ru' || typeof value !== 'string') return value;
    if (ru[value]) return ru[value];
    const rows=value.match(/^(\d+) rows\./);
    if(rows)return `${rows[1]} строк. Подтвердите столбцы. Горизонт измеряется в периодах ваших данных.`;
    const preview=value.match(/^(\d+) periods · frequency (.+) · (\d+) gaps · (\d+) duplicate dates$/);
    if(preview)return `${preview[1]} периодов · частота ${preview[2]} · пропусков: ${preview[3]} · повторных дат: ${preview[4]}`;
    const short=value.match(/^More history or a shorter horizon is needed\. Minimum (\d+) periods; maximum horizon (\d+)\.$/);
    if(short)return `Нужна более длинная история или меньший горизонт. Минимум ${short[1]} периодов, максимальный горизонт ${short[2]}.`;
    return value;
  }
  const text = (tag, value) => { const el = document.createElement(tag); el.textContent = tr(value); return el; };
  function apiError(data, fallback) {
    if (typeof data.detail === 'string') return tr(data.detail);
    if (typeof data.detail?.message === 'string') return tr(data.detail.message);
    if (Array.isArray(data.detail)) return data.detail.slice(0, 3).map(item => item.msg).filter(Boolean).join('. ') || tr(fallback);
    return tr(fallback);
  }
  function button(label, handler) {
    const el = text('button', label); el.type = 'button'; el.className = 'btn btn-secondary'; el.onclick = handler; return el;
  }
  async function download(jobId, lang, format, source, kind) {
    source.disabled = true;
    try {
      const response = await apiFetch(API + (kind === 'quote' ? '/report-quotes/' : '/jobs/') + encodeURIComponent(jobId) + '/report?lang=' + lang + '&format=' + format);
      if (!response.ok) throw new Error(apiError(await response.json(), 'Could not download the report.'));
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href = url; link.download = 'forecast-' + lang + '.' + format;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { toast(error.message || 'Report download failed', 'error'); }
    finally { source.disabled = false; }
  }
  function addDownloads(box, jobId, kind = 'job') {
    const row = document.createElement('div'); row.className = 'report-downloads';
    row.append(text('p', 'Saved report · downloading does not train a new model.'));
    for (const [label, lang, format] of [['Report EN', 'en', 'html'], ['Отчёт RU', 'ru', 'html'], ['Forecast CSV', 'en', 'csv']]) {
      const btn = button(label, () => download(jobId, lang, format, btn, kind)); row.append(btn);
    }
    box.append(row);
  }
  function field(label, element) {
    const wrapper = text('label', label); wrapper.className = 'report-field'; wrapper.append(element); return wrapper;
  }
  function select(columns, label) {
    const el = document.createElement('select'); el.setAttribute('aria-label', tr(label));
    const empty = text('option', 'Choose a column'); empty.value = ''; el.append(empty);
    for (const column of columns) { const option = document.createElement('option'); option.textContent = column; option.value = column; el.append(option); }
    return el;
  }
  function open() {
    if (dialog && ownerId !== currentUser?.id) { dialog.remove(); dialog = null; uploaded = null; payload = null; }
    ownerId = currentUser?.id;
    if (dialog) { dialog.showModal(); return; }
    dialog = document.createElement('dialog'); dialog.className = 'workspace-dialog report-dialog';
    dialog.setAttribute('aria-labelledby', 'report-title');
    const head = document.createElement('div'); head.className = 'workspace-dialog-head';
    const title = text('h2', 'Forecast report'); title.id = 'report-title';
    head.append(title, button('Close', () => dialog.close())); dialog.append(head);
    const content = document.createElement('div'); content.className = 'report-content'; dialog.append(content);
    const languageButton = button(language === 'en' ? 'Русский' : 'English', () => {
      language = language === 'en' ? 'ru' : 'en'; localStorage.setItem('genesis-report-language',language);
      dialog.close(); dialog.remove(); dialog=null; open();
    });
    content.append(languageButton);
    content.append(text('p', 'Upload a table, confirm the date and metric, then review the checks. A one-off report and review is priced by agreement; no subscription is purchased here.'));
    content.append(text('p', 'Use aggregated sales only. Do not include customer names, emails or payment information. We cannot account for unknown promotions or stock-outs.'));
    const file = document.createElement('input'); file.type = 'file'; file.accept = '.csv,.tsv,.xlsx';
    content.append(field('1. Sales table (CSV or XLSX)', file));
    const settings = document.createElement('div'); content.append(settings);
    const status = text('p', 'Choose a file to begin.'); status.setAttribute('role', 'status'); content.append(status);
    const output = document.createElement('div'); content.append(output);
    const setStatus = value => { status.textContent = tr(value); };
    file.onchange = async () => {
      if (busy || !file.files[0]) return;
      busy = true; file.disabled = true; uploaded = null; payload = null; settings.replaceChildren(); output.replaceChildren();
      languageButton.disabled = true;
      setStatus('Uploading and checking the table…');
      try {
        const form = new FormData(); form.append('file', file.files[0]);
        const response = await apiFetch(API + '/agent/upload', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok || !data.dataset_id) throw new Error(apiError(data, 'The table could not be uploaded.'));
        uploaded = data;
        const date = select(data.columns, 'Date column'), target = select(data.columns, 'Metric column');
        const horizon = document.createElement('input'); horizon.type = 'number'; horizon.min = '1'; horizon.max = '365'; horizon.value = '14';
        const agg = document.createElement('select');
        for (const [name, value] of [['Sum per period', 'sum'], ['Average per period', 'mean']]) {
          const option = text('option', name); option.value = value; agg.append(option);
        }
        settings.append(text('p', `${data.rows} rows. Confirm which columns to use. Forecast horizon is in periods of your data.`),
          field('2. Date', date), field('3. Metric', target), field('Forecast periods', horizon), field('Duplicate dates', agg));
        if (data.n_columns > data.columns.length) settings.append(text('p', 'Only the first 30 columns are shown. For this pilot, upload a table with just the date and metric.'));
        const check = button('Check data', async () => {
          payload = { dataset_id: uploaded.dataset_id, date_column: date.value, target_column: target.value, horizon: Number(horizon.value), agg: agg.value };
          output.replaceChildren();
          if (!payload.date_column || !payload.target_column || !Number.isInteger(payload.horizon) || payload.horizon < 1 || payload.horizon > 365) {
            setStatus('Choose both columns and a whole horizon from 1 to 365.'); return;
          }
          check.disabled = true; setStatus('Checking frequency, gaps and available history…');
          try {
            const preview = await apiFetch(API + '/forecasts/preview', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
            const result = await preview.json();
            if (!preview.ok) throw new Error(apiError(result, 'Data check failed.'));
            output.append(text('p', `${result.points} periods · frequency ${result.frequency} · ${result.gaps} gaps · ${result.duplicates} duplicate dates`));
            for (const warning of result.warnings) output.append(text('p', warning));
            if (!result.eligible) { setStatus(`More history or a shorter horizon is needed. Minimum ${result.minimum_points} periods; maximum horizon ${result.maximum_horizon}.`); return; }
            if (!result.can_run) { setStatus('The file passes the checks. Server execution is not enabled for this account. Arrange the one-off report and review with the pilot organiser; no money has been charged.'); return; }
            setStatus('Ready to run. This starts one server forecast job. Accuracy and calibrated intervals are not guaranteed.');
            const confirmed = {...payload};
            const start = button('Confirm and build report', async () => {
              start.disabled = true;
              try {
                const response = await apiFetch(API + '/jobs', { method:'POST', headers:{'Content-Type':'application/json', 'Idempotency-Key': start.dataset.requestId || (start.dataset.requestId = crypto.randomUUID())}, body:JSON.stringify({kind:'forecast', payload:confirmed}) });
                const job = await response.json();
                if (!response.ok) throw new Error(apiError(job, 'The forecast could not be started.'));
                output.replaceChildren(); setStatus('The report is being prepared. You can close this window and reopen the job from Recent reports.');
                GenesisJobs.attach(output, job.job_id);
              } catch (error) { setStatus(typeof error.message === 'string' ? error.message : 'Please check the data and retry.'); start.disabled = false; }
            });
            output.append(start);
          } catch (error) { setStatus(error.message || 'Check failed. Please retry.'); }
          finally { check.disabled = false; }
        });
        settings.append(check); setStatus('Confirm your columns before starting. No training has run.');
      } catch (error) { setStatus(error.message || 'Upload failed. Please retry.'); }
      finally { busy = false; file.disabled = false; }
    };
    const recent = button('Recent reports', async () => {
      recent.disabled = true;
      try {
        const response = await apiFetch(API + '/jobs');
        if (!response.ok) throw new Error('Could not load recent reports');
        const jobs = (await response.json()).jobs.filter(job => job.kind === 'forecast');
        output.replaceChildren();
        for (const job of jobs) output.append(button(new Date(job.created_at * 1000).toLocaleString() + ' · ' + job.status, () => { output.replaceChildren(); GenesisJobs.attach(output, job.id); }));
        if (!jobs.length) output.append(text('p', 'No forecast jobs yet.'));
      } catch (error) { setStatus(error.message); }
      finally { recent.disabled = false; }
    });
    content.append(button(language === 'ru' ? 'Мой счёт и отчёт' : 'My quote and report', () => GenesisReportQuotes.open()),recent); document.body.append(dialog); dialog.showModal();
  }
  return { open, addDownloads };
})();
