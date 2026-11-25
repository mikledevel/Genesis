/* Разовая услуга: цена задаётся организатором, оплату подтверждает сервер. */
window.GenesisReportQuotes = (() => {
  const element = (tag, value) => { const node=document.createElement(tag); if(value) node.textContent=value; return node; };
  const button = (label, handler) => { const node=element('button',label);node.type='button';node.className='btn btn-secondary';node.onclick=handler;return node; };
  const field = (label,input) => { const node=element('label',label);node.className='report-field';node.append(input);return node; };
  async function open() {
    const owner=currentUser?.id;
    if(!owner)return;
    document.getElementById('report-quotes-dialog')?.remove();
    const ru=(localStorage.getItem('genesis-report-language') || navigator.language).startsWith('ru');
    const t=(r,e)=>ru?r:e;
    const dialog=element('dialog');dialog.id='report-quotes-dialog';dialog.className='workspace-dialog report-dialog';
    dialog.setAttribute('aria-labelledby','quote-title');
    const head=element('div');head.className='workspace-dialog-head';
    const title=element('h2',t('Мой счёт и отчёт','My quote and report'));title.id='quote-title';
    head.append(title,button(t('Закрыть','Close'),()=>dialog.close()));
    const content=element('div');content.className='report-content';
    const status=element('p');status.setAttribute('role','status');
    const list=element('div');content.append(element('p',t('Разовый отчёт с разбором. Цена согласуется с организатором. Подписка не требуется.','One report and review, priced by agreement with the organiser. No subscription is required.')),status,list);
    dialog.append(head,content);document.body.append(dialog);dialog.showModal();
    const active=()=>currentUser?.id===owner && dialog.isConnected;
    async function request(path,options) {
      if(!active())throw new Error(t("Аккаунт изменился. Откройте окно заново.","Account changed. Reopen this window."));
      const response=await apiFetch(API+path,options);
      const data=await response.json();
      if(!active())throw new Error(t('Аккаунт изменился. Откройте окно заново.','Account changed. Reopen this window.'));
      if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:t('Не удалось выполнить запрос. Повторите позже.','Request failed. Please retry later.'));
      return data;
    }
    const post=(path,data={},key)=>request(path,{method:'POST',headers:{'Content-Type':'application/json',...(key?{'Idempotency-Key':key}:{})},body:JSON.stringify(data)});
    async function action(source,fn) {
      source.disabled=true;
      try{await fn();}catch(error){if(active())status.textContent=error.message;}
      finally{source.disabled=false;}
    }
    const names={quoted:t('Цена согласована','Quoted'),paid:t('Оплачено, отчёт готовится','Paid, report being prepared'),ready:t('Отчёт готов','Report ready'),refund_pending:t('Возврат обрабатывается','Refund pending'),refunded:t('Оплата возвращена','Refunded')};
    async function refresh() {
      const data=await request('/report-quotes');list.replaceChildren();
      status.textContent=data.payments_enabled?'':t('Карточная оплата пока не активирована. Оплата не списывается.','Card payment is not activated yet. No charge is made.');
      if(!data.quotes.length)list.append(element('p',t('Счёта пока нет. Согласуйте цену с организатором и обновите список.','No quote yet. Agree a price with the organiser, then refresh.')));
      for(const quote of data.quotes) {
        const box=element('section');box.className='answer-source';
        box.append(element('h3',quote.title),element('p',new Intl.NumberFormat(ru?'ru':'en',{style:'currency',currency:quote.currency}).format(quote.amount_cents/100)+' · '+(names[quote.status]||quote.status)));
        if(quote.status==='quoted') {
          const expired=quote.expires_at*1000<=Date.now();
          const pay=button(t('Оплатить картой','Pay by card'),()=>action(pay,async()=>{
            const checkout=await post('/report-quotes/'+encodeURIComponent(quote.id)+'/checkout');
            const target=new URL(checkout.url);
            if(target.protocol!=='https:'||target.hostname!=='checkout.stripe.com')throw new Error(t('Недопустимая ссылка оплаты','Invalid payment link'));
            window.location.assign(target.href);
          }));pay.disabled=!data.payments_enabled||expired;box.append(pay);
          if(expired)box.append(element('p',t('Срок согласованной цены истёк. Обратитесь к организатору.','Quote expired. Contact the organiser.')));
        }
        if(quote.report_available)GenesisReports.addDownloads(box,quote.id,'quote');
        list.append(box);
      }
    }
    const reload=button(t('Обновить','Refresh'),()=>action(reload,refresh));content.append(reload);
    try {
      const me=await request('/auth/me');
      await refresh();
      if(me.user?.is_admin) {
        const admin=element('section');admin.append(element('h3',t('Организатору','Organiser')));
        const email=element('input');email.type='email';email.autocomplete='off';
        const customer=element('p'),quotes=element('div');let buyer=null;
        const find=button(t('Найти участника','Find participant'),()=>action(find,async()=>{
          buyer=null;customer.textContent='';quotes.replaceChildren();
          const data=await request('/admin/users?q='+encodeURIComponent(email.value.trim())+'&limit=200');
          const match=data.users.find(user=>user.email.toLowerCase()===email.value.trim().toLowerCase()&&!user.disabled);
          if(!match)throw new Error(t('Нужен точный email зарегистрированного участника.','Enter the exact email of a registered participant.'));
          buyer=match.id;customer.textContent=match.email;await customerQuotes();
        }));
        const name=element('input');name.maxLength=200;name.value=t('Прогнозный отчёт с разбором','Forecast report and review');
        const amount=element('input');amount.type='text';amount.inputMode='decimal';amount.placeholder='99.00';
        const currency=element('select');for(const code of ['usd','eur','gbp']){const option=element('option',code.toUpperCase());option.value=code;currency.append(option);}
        let pending=null;
        const issue=button(t('Выставить согласованный счёт','Issue agreed quote'),()=>action(issue,async()=>{
          const value=amount.value.trim().replace(',','.');
          if(!buyer||!/^\d{1,5}(\.\d{1,2})?$/.test(value))throw new Error(t('Выберите участника и укажите цену с точностью до копеек.','Choose a participant and enter a price with up to two decimal places.'));
          const [whole,fraction='']=value.split('.');const cents=Number(whole)*100+Number(fraction.padEnd(2,'0'));
          if(cents<100||cents>1000000)throw new Error(t('Цена должна быть от 1 до 10 000.','Price must be between 1 and 10,000.'));
          const body={user_id:buyer,title:name.value.trim(),amount_cents:cents,currency:currency.value};
          const signature=JSON.stringify(body);if(!pending||pending.signature!==signature)pending={signature,key:crypto.randomUUID()};
          await post('/admin/report-quotes',body,pending.key);pending=null;await customerQuotes();
        }));
        async function customerQuotes() {
          const selected=buyer;const data=await request('/admin/report-quotes/'+encodeURIComponent(selected));
          if(buyer!==selected)return;quotes.replaceChildren();
          for(const quote of data.quotes) {
            const row=element('section');row.className='answer-source';row.append(element('p',quote.title+' · '+quote.amount_cents/100+' '+quote.currency.toUpperCase()+' · '+(names[quote.status]||quote.status)));
            const prefix='/admin/report-quotes/'+encodeURIComponent(selected)+'/'+encodeURIComponent(quote.id);
            if(quote.status==='paid') {
              const job=element('select');
              const empty=element('option',t('Выберите готовый прогноз','Choose a completed forecast'));empty.value='';job.append(empty);
              const jobs=await request('/jobs');
              for(const saved of jobs.jobs.filter(item=>item.kind==='forecast'&&item.status==='succeeded')) {
                const option=element('option',new Date(saved.created_at*1000).toLocaleString(ru?'ru':'en')+' · '+saved.id.slice(0,8));option.value=saved.id;job.append(option);
              }
              const preview=element('div');
              job.onchange=()=>{preview.replaceChildren();if(job.value)GenesisReports.addDownloads(preview,job.value);};
              const deliver=button(t('Выдать выбранный отчёт участнику','Deliver selected report'),()=>action(deliver,async()=>{
                if(!job.value)throw new Error(t('Сначала выберите и проверьте прогноз.','Choose and review a forecast first.'));
                await post(prefix+'/fulfill',{job_id:job.value});await customerQuotes();
              }));row.append(field(t('Ваш готовый прогноз: проверьте файл перед выдачей','Your completed forecast: review the file before delivery'),job),preview,deliver);
            }
            if(['paid','ready','refund_pending'].includes(quote.status)) {
              const refund=button(t('Вернуть полную оплату','Refund full payment'),()=>action(refund,async()=>{
                if(!window.confirm(t('Вернуть клиенту полную сумму этого счёта?','Refund this quote in full?')))return;
                await post(prefix+'/refund');await customerQuotes();
              }));row.append(refund);
            }
            quotes.append(row);
          }
        }
        admin.append(field('Email',email),find,customer,field(t('Услуга','Service'),name),field(t('Согласованная цена','Agreed price'),amount),field(t('Валюта','Currency'),currency),issue,quotes);content.append(admin);
      }
    } catch(error){if(active())status.textContent=error.message;}
  }
  function handleReturn() {
    const url=new URL(window.location.href);
    if(!url.searchParams.has('report_payment'))return;
    url.searchParams.delete('report_payment');history.replaceState(null,'',url.pathname+url.search+url.hash);open();
  }
  return {open,handleReturn};
})();
