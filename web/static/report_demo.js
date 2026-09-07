/* Offline road replay: ground truth is fetched only after evaluator opt-in. */
'use strict';
const $ = id => document.getElementById(id);
const names = {unprotected:'Không bảo vệ',uniform_dummy:'Dummy đều (đối chứng đơn giản)',rem_anchor_only:'Chỉ neo REM',transprotect_adaptation:'TransProtect · thích nghi',anotherme_adaptation:'AnotherMe · thích nghi',semantic_correlation_local_adaptation:'Semantic correlation · thích nghi',geo_i_anchored_dummy:'Neo Geo-I + dummy · thử nghiệm'};
names.dls_graph_adaptation='DLS · không học sâu · thích nghi';
names.geo_i_anchored_dummy_road='Neo Geo-I + dummy · ràng buộc đường';
let overview, roads, run, evaluation, privateRow, timer, requestId=0;
let zoom=1, pan=[0,0], drag=null;
const canvas=$('map'), ctx=canvas.getContext('2d');
async function get(url){const r=await fetch(url);if(!r.ok)throw new Error(`HTTP ${r.status}: ${url}`);return r.json();}
function options(el, entries, selected){el.replaceChildren();for(const [value,label] of entries){const o=document.createElement('option');o.value=value;o.textContent=label;el.append(o);}if(selected!==undefined)el.value=selected;}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Phát';}
function num(v,d=1){return v===null||v===undefined?'—':Number(v).toFixed(d);}
function percent(v){return v==null?'—':num(v*100)+'%';}
function syncRuns(){
  stop(); const selected=overview.runs.filter(r=>r.scenario===$('scenario').value&&r.method===$('method').value&&r.k===Number($('k').value));
  options($('run'),selected.map((r,i)=>[r.id,`Lượt ${i+1}${r.status==='ok'?'':r.status==='not_applicable'?' · không áp dụng':' · lỗi thực thi'}`]));
  $('scenario-info').textContent=overview.catalogue.find(s=>s.id===$('scenario').value)?.name||'';
  if(selected.length)loadRun().catch(showError);
  else {run=null;privateRow=null;$('play').disabled=true;draw();}
}
function showError(e){$('run-status').textContent=e.message;}
async function loadRun(){
  const token=++requestId;stop();const data=await get('/api/report-demo/runs/'+$('run').value);if(token!==requestId)return;
  run=data; privateRow=evaluation?.rows.find(r=>r.id===run.id);zoom=1;pan=[0,0];
  $('step').max=Math.max(0,(run.public?.events.length||0)-1);$('step').value=0;
  $('play').disabled=!run.public;$('run-status').textContent=run.status==='ok'?(run.method==='anotherme_adaptation'?'Tham chiếu ngoại tuyến: dùng cả đoạn hành trình.':''):(run.status==='not_applicable'?'Không áp dụng cho loại đoạn này.':'Không có đầu ra hợp lệ.');
  updateMetrics();draw();
}
function updateMetrics(){
  $('metrics').replaceChildren();if(!$('evaluator').checked||!privateRow?.metrics)return;
  const m=privateRow.metrics;
  for(const text of [`Đối thủ: ${privateRow.attacker_selected||'—'}`,`Sai số so với SUMO: ${num(m.location_mae_m)} m`,`Đoán trong 100 m: ${percent(m.location_hit_100m)}`,`Giữ POI đúng: ${percent(m.poi_recall_at_k)}`,`Chuyển tiếp đúng đồ thị: ${percent(m.directed_track_validity)}`,`Điểm/yêu cầu: ${num(m.coordinates_per_request,0)}`]){const d=document.createElement('div');d.textContent=text;$('metrics').append(d);}
}
function table(){
  $('summary').replaceChildren();for(const r of evaluation.summary){const tr=document.createElement('tr');
    for(const v of [r.scenario,names[r.method]||r.method,r.k,r.not_applicable===r.attempted?'Không áp dụng':`${r.completed}/${r.attempted}`,num(r.location_mae_m),percent(r.location_hit_100m),percent(r.poi_delivered_recall_at_k),num(r.coordinates_per_request,0)]){const td=document.createElement('td');td.textContent=v;tr.append(td);} $('summary').append(tr);
  }
}
function draw(){
  const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;ctx.scale(ratio,ratio);ctx.fillStyle='#f6f8f5';ctx.fillRect(0,0,w,h);
  if(!run?.public){$('public-record').textContent='Không có dữ liệu công bố.';$('step-info').textContent='';return;}
  const events=run.public.events, index=Number($('step').value), all=events.flatMap(e=>e.candidates.map(c=>[c.lon,c.lat]));
  const truth=$('evaluator').checked?privateRow?.truth:null;if(truth)all.push(...truth.map(p=>[p[1],p[0]]));
  const lon0=all.reduce((s,p)=>s+p[0],0)/all.length, lat0=all.reduce((s,p)=>s+p[1],0)/all.length, cosine=Math.cos(lat0*Math.PI/180);
  const local=p=>[(p[0]-lon0)*111320*cosine,(p[1]-lat0)*111320];const bounds=all.map(local);
  const xs=bounds.map(p=>p[0]),ys=bounds.map(p=>p[1]);const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const scale=Math.min((w-100)/Math.max(600,maxx-minx),(h-100)/Math.max(600,maxy-miny))*zoom;
  const screen=p=>{const q=local(p);return [w/2+(q[0]-(minx+maxx)/2)*scale+pan[0],h/2-(q[1]-(miny+maxy)/2)*scale+pan[1]];};
  function line(points,color,width){if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{const a=screen(p);if(i)ctx.lineTo(...a);else ctx.moveTo(...a);});ctx.stroke();}
  function dot(point,color,r){const a=screen(point);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...a,r,0,2*Math.PI);ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke();}
  for(const edge of roads)line(edge,'#b7c2bc',1);
  const tracks=new Map();for(const e of events.slice(0,index+1)){for(const c of e.candidates){if(!tracks.has(c.candidate_id))tracks.set(c.candidate_id,[]);tracks.get(c.candidate_id).push([c.lon,c.lat]);}}
  for(const points of tracks.values())line(points,'#65a2d1',2);
  for(const points of tracks.values())if(points.length===1)dot(points[0],'#9cbfda',2);
  if(truth){line(truth.slice(0,index+1).map(p=>[p[1],p[0]]),'#172d34',3);dot([truth[index][1],truth[index][0]],'#172d34',6);}
  for(const c of events[index].candidates)dot([c.lon,c.lat],'#2164b4',5);
  if(truth&&privateRow?.attack_xy){const manifest=evaluation.manifests.find(m=>m.seed===privateRow.seed);const p=privateRow.attack_xy[index];const metres=6371000*Math.PI/180;dot([p[0]/(metres*Math.cos(manifest.projection_lat0*Math.PI/180)),p[1]/metres],'#cf3428',6);}
  $('step-info').textContent=`Bước ${index+1}/${events.length} · t = ${events[index].timestamp_s} s`;
  $('public-record').textContent=JSON.stringify(events[index],null,2);
  // Fixed 100-m scale in the same local projection used for drawing.
  ctx.strokeStyle='#465b4c';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(20,h-25);ctx.lineTo(20+100*scale,h-25);ctx.stroke();ctx.fillStyle='#465b4c';ctx.font='12px system-ui';ctx.fillText('100 m',20,h-32);
}
$('evaluator').addEventListener('change',async()=>{try{if($('evaluator').checked&&!evaluation)evaluation=await get('/api/report-demo/evaluation');privateRow=evaluation?.rows.find(r=>r.id===run.id);$('results').hidden=!$('evaluator').checked;$('visibility').textContent=$('evaluator').checked?'Có ground truth. Chỉ dành cho trình bày/đánh giá; không cấp cho đối thủ.':'Chỉ hiển thị dữ liệu được công bố.';if(evaluation)table();updateMetrics();draw();}catch(e){$('evaluator').checked=false;showError(e);}});
for(const id of ['scenario','method','k'])$(id).addEventListener('change',syncRuns);
$('run').addEventListener('change',()=>loadRun().catch(showError));$('step').addEventListener('input',draw);
$('play').addEventListener('click',()=>{if(timer){stop();return;}if(Number($('step').value)>=Number($('step').max))$('step').value=0;$('play').textContent='Dừng';timer=setInterval(()=>{if(Number($('step').value)>=Number($('step').max)){stop();return;}$('step').value=Number($('step').value)+1;draw();},700);});
$('reset').addEventListener('click',()=>{stop();$('step').value=0;zoom=1;pan=[0,0];draw();});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(30,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,...pan];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(drag){pan=[drag[2]+e.clientX-drag[0],drag[3]+e.clientY-drag[1]];draw();}});for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>drag=null);
window.addEventListener('resize',draw);
(async()=>{[overview,roads]=await Promise.all([get('/api/report-demo'),get('/api/report-demo/roads')]);options($('scenario'),overview.catalogue.filter(s=>s.status==='runnable').map(s=>[s.id,`${s.id} · ${s.name}`]),'S3');options($('method'),Object.entries(names),'geo_i_anchored_dummy');options($('k'),[...new Set(overview.runs.map(r=>r.k))].map(k=>[k,String(k)]));$('evaluator').disabled=!overview.can_evaluate;for(const text of overview.limitations){const li=document.createElement('li');li.textContent=text;$('limitations').append(li);}for(const s of overview.catalogue){const el=document.createElement('span');el.textContent=`${s.id}: ${s.status==='runnable'?'đã chạy':'đặc tả'}`;el.title=s.name;$('coverage').append(el);}syncRuns();})().catch(showError);
