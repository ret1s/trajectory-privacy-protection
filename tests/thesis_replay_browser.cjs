// Optional end-to-end regression: NODE_PATH must resolve a local playwright installation.
// Run against the stdlib server. Uses an isolated headless browser, never a user profile.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {chromium}=require('playwright');
(async()=>{
 const executablePath=process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
 const browser=await chromium.launch({executablePath,headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050}});
 const errors=[],requests=[];const out='build/thesis_demo_20261111';fs.mkdirSync(out,{recursive:true});
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 page.on('request',r=>requests.push(r.url()));
 const visible=selector=>page.locator(selector).isVisible();
 const value=selector=>page.locator(selector).textContent();
 const select=async(selector,v)=>{await page.selectOption(selector,v);};
 const ready=async()=>{await page.waitForFunction(()=>document.getElementById('loading').hidden);assert.equal(await visible('#error'),false);};
 try{
  await page.goto(process.env.DEMO_URL||'http://127.0.0.1:5011');await ready();
  assert.equal(await page.locator('#session option').count(),40);
  assert.equal(await page.locator('#draw option').count(),3);
  assert.equal(await page.locator('#method option').count(),6);
  assert.equal(await value('#event-title'),'Tạo tham chiếu đầu tiên');
  assert.equal(await value('#geo-spent'),'0,01 m⁻¹');
  await page.screenshot({path:out+'/replay.png',fullPage:true});
  const wire=await value('#wire'),before=requests.length;
  for(const purpose of ['fastest_travel','within_radius','minimum_detour','nearest_distance']){
   await select('#purpose',purpose);assert.equal(await value('#wire'),wire);
  }
  for(const cache of ['known','current','session'])await select('#cache',cache);
  for(const category of await page.locator('#category option').evaluateAll(ns=>ns.map(n=>n.value)))await select('#category',category);
  assert.equal(requests.length,before,'Local request changes must not issue network requests');
  await page.click('#next');assert.equal(await value('#time'),'00:20');
  assert.equal(await value('#geo-spent'),'0,01 m⁻¹');
  assert.equal(await value('#q-spent'),'0,5000 / 1');
  await page.click('#fit-gps');await select('#perspective','server');
  for(const id of ['#local-panel','#pipeline','#diagnostics','#fit-gps'])assert.equal(await visible(id),false,id);
  assert.equal(await value('#event-title'),'Packet công khai');
  assert.equal(await page.locator('#fit-all').getAttribute('class'),'active');
  await page.screenshot({path:out+'/server.png',fullPage:true});
  await select('#perspective','analysis');
  for(const m of ['raw','geo_i_Z_k1','transprotect_markov','semantic_poi','fake_queries','geoi_bq']){
   await select('#method',m);assert.ok((await value('#wire')).includes('coordinates'));
   assert.equal(await visible('#diagnostics'),m==='geoi_bq');
  }
  await page.click('#tour-toggle');for(let i=0;i<5;i++)await page.click('#tour-next');assert.equal(await value('#time'),'10:00');
  await page.click('#tour-close');
  for(const id of ['cache-029_day7--draw3','cache-048_day8--draw2']){
   const url=new URL(page.url());url.searchParams.set('sample',id);url.searchParams.set('step','9');await page.goto(url.href);await ready();
   assert.equal(await page.locator('#draw').inputValue(),id);assert.equal(await value('#time'),'03:00');
  }
  await page.click('[data-tab="results"]');
  assert.equal(await page.locator('#privacy-table tr').count(),6);
  assert.equal(await value('#macro-result'),'99,595%');
  for(const s of ['S1','S2','S3','S9','S10']){await select('#scenario',s);assert.equal(await page.locator('#privacy-table tr').count(),6);}
  for(const s of ['S4','S5','S6','S7','S8']){await select('#extra-scenario',s);assert.equal(await page.locator('#extra-table tr').count(),6);}
  await page.screenshot({path:out+'/results.png',fullPage:true});
  await page.click('[data-tab="proof"]');await page.locator('#prior').evaluate(el=>{el.value='90';el.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.equal(await value('#prior-value'),'90%');assert.ok(Number((await value('#posterior-value')).replace('%','').replace(',','.'))>95);
  await page.screenshot({path:out+'/proof.png',fullPage:true});
  await page.click('.brand');await page.waitForFunction(()=>!document.getElementById('replay').hidden);
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile horizontal overflow');
  await page.screenshot({path:out+'/mobile.png',fullPage:true});
  await page.click('#play');await page.waitForFunction(()=>document.getElementById('time').textContent!=='03:00');await page.click('#play');
  assert.deepEqual(errors,[]);
  console.log('PASS: replay, 6 arms, local-only ranking, LBS view, samples, 10 scenarios, proof, mobile, playback');
  fs.writeFileSync(out+'/browser_checks.json',JSON.stringify({status:'pass',errors,viewport:[1440,1050],mobile:[390,844],network_requests:requests.length},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
