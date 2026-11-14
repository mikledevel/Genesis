// Изолированная проверка интерфейса: API перехвачен, исходные модели читаются из локального реестра.
async (page) => {
  const origin = 'http://127.0.0.1:8768';
  const assert = (value, message) => { if (!value) throw new Error(message); };
  const registry = await (await page.request.get(origin + '/registry/index.json')).json();
  const models = Object.values(registry);
  const chats = [{id:'ui-chat-alpha',title:'Customer support launch'},{id:'ui-chat-beta',title:'Dataset quality review'}];
  const errors = [];
  page.on('pageerror',error => errors.push(error.message));
  let failRegistry = false;
  await page.route('**/api/**',async route => {
    const path = new URL(route.request().url()).pathname.replace('/api','');
    let data = {};
    if (path === '/auth/login' || path === '/auth/me') data = path.endsWith('login') ? {token:'local-ui-fixture',user:{id:'ui-review',name:'UI Review',email:'ui@example.test',plan:'free'}} : {id:'ui-review',name:'UI Review',email:'ui@example.test',plan:'free'};
    if (path === '/registry/models') {
      if (failRegistry) return route.fulfill({status:503,json:{detail:'Fixture outage'}});
      const task = new URL(route.request().url()).searchParams.get('task');
      data = {models:task ? models.filter(m => m.task === task) : models};
    }
    if (path.startsWith('/registry/models/')) data = models.find(m => m.model_id === decodeURIComponent(path.split('/').pop())) || {};
    if (path === '/registry/stats') data = {total_models:models.length};
    if (path === '/agent/conversations') data = {conversations:chats};
    if (path === '/memory/facts') data = {facts:[]};
    if (path === '/marketplace/stats') data = {total_listings:0,total_transactions:0,by_task:{}};
    if (path === '/marketplace/listings') data = {listings:[]};
    if (path === '/agents/models') data = {models:[]};
    if (path === '/dashboard/activity') data = {activity:[]};
    if (path === '/projects') data = {projects:[]};
    if (path === '/plans') data = {plans:[]};
    return route.fulfill({json:data});
  });
  await page.route('**/health',route => route.fulfill({json:{status:'healthy'}}));
  await page.addInitScript(() => { if (location.origin === 'http://127.0.0.1:8768') localStorage.removeItem('genesis_token_v1'); });
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(origin);
  await page.screenshot({path:'/tmp/genesis-landing.png',fullPage:true});
  await page.getByRole('button',{name:'Sign in',exact:true}).first().click();
  await page.locator('#l-email').fill('ui@example.test');
  await page.locator('#l-password').fill('local-ui-fixture');
  await page.locator('#login-btn').click();
  await page.locator('#app').waitFor({state:'visible'});
  await page.locator('#d-models').filter({hasText:String(models.length)}).waitFor();
  await page.screenshot({path:'/tmp/genesis-overview.png',fullPage:true});
  await page.keyboard.press('Control+k');
  await page.locator('#workspace-query').fill('registry');
  await page.keyboard.press('ArrowDown');
  assert(await page.locator('#workspace-results button').first().evaluate(e => e === document.activeElement),'Command keyboard focus failed');
  await page.keyboard.press('Enter');
  await page.locator('#reg-body .model-detail-link').first().waitFor();
  await page.locator('#registry-query').fill('churn');
  const visible = await page.locator('#reg-body .model-detail-link').count();
  assert(visible > 0,'Registry search returned no models');
  await page.locator('#registry-select-all').check();
  assert(await page.locator('#registry-selection').innerText() === visible + ' selected','Bulk selection failed');
  assert(await page.locator('#registry-export').isEnabled(),'Export disabled for selection');
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#registry-export').click();
  const download = await downloadPromise;
  assert(download.suggestedFilename() === 'genesis-models.json','Export filename incorrect');
  const stream = await download.createReadStream(); let exported = '';
  for await (const chunk of stream) exported += chunk.toString();
  assert(JSON.parse(exported).length === visible,'Exported metadata does not match selection');
  await page.locator('#registry-query').fill('no-model-with-this-name');
  assert(await page.getByText('No matching models',{exact:true}).isVisible(),'Filtered empty state absent');
  await page.locator('#registry-query').fill('');
  await page.locator('#reg-task-sel').selectOption('regression');
  await page.getByText('No trained models yet',{exact:true}).waitFor();
  assert(await page.locator('#registry-export').isDisabled(),'Task change retained stale selection');
  await page.locator('#reg-task-sel').selectOption('all');
  await page.locator('#reg-body .model-detail-link').first().waitFor();
  await page.screenshot({path:'/tmp/genesis-registry.png',fullPage:true});
  failRegistry = true;
  await page.locator('#nb-dashboard').click();
  await page.locator('#nb-registry').click();
  await page.getByText('Models could not be loaded',{exact:true}).waitFor();
  failRegistry = false;
  await page.getByRole('button',{name:'Retry loading'}).click();
  await page.locator('#reg-body .model-detail-link').first().waitFor();
  const widths = [320,768,1024,1440];
  const panels = ['dashboard','agent','projects','builder','registry','store','marketplace','memory','promptlab','settings'];
  const overflow = [];
  for (const width of widths) {
    await page.setViewportSize({width,height:900});
    for (const panel of panels) {
      if (width <= 640 && !['dashboard','agent','projects'].includes(panel)) {
        await page.locator('#nb-workspace-more').click();
        await page.locator('#workspace-more-results button').filter({hasText:({builder:'Pipeline Builder',registry:'Model Registry',store:'Agent Store',marketplace:'Marketplace',memory:'Memory',promptlab:'Prompt Lab',settings:'Settings'})[panel]}).click();
      } else await page.locator('#nb-' + panel).click();
      assert(await page.locator('#panel-' + panel).isVisible(),'Panel not visible: ' + panel);
      const spills = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth || [...document.querySelectorAll('.panel.active .panel-body')].some(e => e.scrollWidth > e.clientWidth + 1));
      if (spills) overflow.push(width + ':' + panel);
    }
  }
  await page.setViewportSize({width:320,height:900});
  await page.locator('#nb-dashboard').click();
  await page.screenshot({path:'/tmp/genesis-mobile.png',fullPage:true});
  await page.emulateMedia({reducedMotion:'reduce'});
  assert(await page.locator('.workspace-task').first().evaluate(e => getComputedStyle(e).transitionDuration) === '0s','Reduced motion ignored');
  assert(!overflow.length,'Horizontal page overflow: ' + overflow.join(', '));
  assert(!errors.length,'Browser errors: ' + errors.join('; '));
  return {status:'passed',models:models.length,checks:'command keyboard, search, selection, export, empty, error/retry, 40 panel/viewport combinations, reduced motion',screenshots:['/tmp/genesis-landing.png','/tmp/genesis-overview.png','/tmp/genesis-registry.png','/tmp/genesis-mobile.png']};
}
