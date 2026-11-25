const {JSDOM}=require('jsdom');
const fs=require('fs'),path=require('path'),assert=require('assert');
const dom=new JSDOM('<!doctype html><body></body>',{runScripts:'outside-only',url:'https://example.test/'});
const w=dom.window;
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(){this.open=false;};
w.currentUser={id:'buyer'};w.API='/api';w.localStorage.setItem('genesis-report-language','ru');
let admin=false, enabled=false, requests=[], quoteStatus='quoted', download;
w.GenesisReports={addDownloads:(box,id,kind)=>{download={id,kind};}};
w.apiFetch=async(url,options={})=>{
 requests.push([url,options]);
 const data=url.endsWith('/auth/me')?{user:{is_admin:admin}}:url.includes('/admin/users')?{users:[{id:'customer',email:'person@example.test',disabled:false}]}:
 url==='/api/admin/report-quotes'&&options.method==='POST'?{quote_id:'issued'}:
 {payments_enabled:enabled,quotes:[{id:'quote',title:'<img src=x onerror=alert(1)>',amount_cents:9900,currency:'usd',status:quoteStatus,expires_at:Date.now()/1000+3600,report_available:quoteStatus==='ready'}]};
 return {ok:true,json:async()=>data};
};
w.eval(fs.readFileSync(path.join(__dirname,'../../static/report-quotes.js'),'utf8'));
const buttons=()=>[...w.document.querySelectorAll('button')];
const find=label=>buttons().find(b=>b.textContent===label);
(async()=>{
 await w.GenesisReportQuotes.open();
 assert(find('Оплатить картой').disabled,'provider activation is required');
 assert(!find('Выставить согласованный счёт'),'ordinary customer cannot issue a quote');
 assert(!w.document.querySelector('img'),'quote titles must be escaped');
 assert(!requests.some(([,o])=>o.method==='POST'),'opening dialog must never charge');
 quoteStatus='ready';await w.GenesisReportQuotes.open();assert.deepStrictEqual(download,{id:'quote',kind:'quote'});
 admin=true;quoteStatus='quoted';await w.GenesisReportQuotes.open();
 w.document.querySelector('input[type=email]').value='person@example.test';await find('Найти участника').onclick();
 const inputs=[...w.document.querySelectorAll('input')];inputs.find(input=>input.placeholder==='99.00').value='123,45';
 await find('Выставить согласованный счёт').onclick();
 const issued=requests.find(([url,o])=>url==='/api/admin/report-quotes'&&o.method==='POST');
 assert(JSON.parse(issued[1].body).amount_cents===12345,'price must use exact integer minor units');
 assert(issued[1].headers['Idempotency-Key']);
 w.currentUser={id:'other'};admin=false;await w.GenesisReportQuotes.open();assert(!find('Выставить согласованный счёт'));
 console.log('ALL QUOTE UI CHECKS PASSED');dom.window.close();
})().catch(error=>{console.error(error);process.exitCode=1;dom.window.close();});
