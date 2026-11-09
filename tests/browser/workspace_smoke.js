// Run with Playwright's browser_run_code tool against the isolated local UI server.
// Requires a signed-in local fixture account with ui-chat-alpha and ui-chat-beta.
async (page) => {
  if (!/^http:\/\/(127\.0\.0\.1|localhost):/.test(page.url())) throw new Error('Local fixture server only');
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  await page.setViewportSize({width:1440,height:900});
  await page.locator('#nb-agent').click();
  await page.locator('[data-cid="ui-chat-alpha"]').click();
  await page.locator('#chat-in').fill('Alpha draft stays with Alpha');
  await page.locator('[data-cid="ui-chat-beta"]').click();
  await page.locator('#chat-in').fill('Beta draft stays with Beta');
  await page.locator('[data-cid="ui-chat-alpha"]').click();
  assert(await page.locator('#chat-in').inputValue()==='Alpha draft stays with Alpha','Draft leaked across chats');
  await page.keyboard.press('Control+k');
  await page.locator('#workspace-query').fill('dataset');
  assert((await page.locator('#workspace-results').innerText()).includes('Dataset quality review'),'Chat search failed');
  await page.keyboard.press('Escape');
  assert(!await page.locator('#workspace-search').evaluate(e=>e.open),'Escape did not close search');
  await page.getByRole('button',{name:'Clear draft',exact:true}).click();
  const messagesBefore = await page.locator('.msg').count();
  await page.getByRole('button',{name:'Starter templates',exact:true}).click();
  await page.getByRole('button',{name:'Bots',exact:true}).click();
  assert(await page.locator('#template-results button').count()===2,'Template category filter failed');
  await page.locator('#template-results button').filter({hasText:'Customer support bot'}).click();
  assert((await page.locator('#chat-in').inputValue()).includes('[business]'),'Template not editable');
  assert(await page.locator('.msg').count()===messagesBefore,'Template sent a message without approval');
  await page.reload();
  await page.locator('#app').waitFor({state:'visible'});
  await page.locator('#nb-agent').click();
  await page.locator('[data-cid="ui-chat-alpha"]').click();
  assert((await page.locator('#chat-in').inputValue()).includes('[business]'),'Draft did not survive reload');
  await page.setViewportSize({width:320,height:740});
  assert(await page.locator('#app>nav .nav-btn:visible').count()===4,'Mobile navigation is overcrowded');
  await page.locator('#nb-workspace-more').click();
  await page.locator('#workspace-more-results button').filter({hasText:'Pipeline Builder'}).click();
  await page.getByRole('button',{name:/Add pipeline blocks/}).click();
  await page.locator('#pipeline-blocks [data-block="load_data"]').click();
  await page.getByRole('button',{name:/Add pipeline blocks/}).click();
  assert(await page.locator('#pipeline-canvas .pblock-canvas').count()===1,'Touch block selection failed');
  assert(await page.locator('#pipeline-canvas').evaluate(e=>e.clientWidth)>250,'Mobile workbench is squeezed');
  await page.locator('#nb-workspace-more').click();
  await page.locator('#workspace-more-results button').filter({hasText:'Settings'}).click();
  assert(await page.locator('.settings-body').evaluate(e=>e.clientWidth)===320,'Mobile settings are squeezed');
  assert(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),'Page overflows horizontally');
  return {status:'passed', checks:12};
}
