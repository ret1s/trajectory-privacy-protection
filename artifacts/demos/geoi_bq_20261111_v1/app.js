import {prefixEvents,centroid,localAnswer,qAllowance,qPrefix,posteriorUpper,meters} from './logic.mjs';

const $=id=>document.getElementById(id), fmt=(n,d=0)=>n===null||n===undefined?'N/A':Number(n).toLocaleString('vi-VN',{minimumFractionDigits:d,maximumFractionDigits:d});
const percent=(n,d=2)=>n===null||n===undefined?'N/A':fmt(n*100,d)+'%';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const categoryLabels={cafe:'Cà phê',restaurant:'Nhà hàng',hospital:'Bệnh viện',pharmacy:'Nhà thuốc',fuel:'Nhiên liệu',charging_station:'Trạm sạc',parking:'Bãi đỗ',bank:'Ngân hàng',atm:'ATM'};
const purposeKeys=['macro','nearest_distance','fastest_travel','within_radius','minimum_detour'];
const S={index:null,map:null,results:null,session:null,run:null,step:0,method:'geoi_bq',perspective:'analysis',
 purpose:'nearest_distance',category:null,cache:'session',playing:false,timer:null,speed:1,stage:0,more:20,tour:0,
 camera:null,zoomMode:'all',points:[],drag:null};
const loads=new Map();let loadVersion=0;
async function get(path){if(!loads.has(path))loads.set(path,fetch(path).then(r=>{if(!r.ok)throw Error(`Không tải được ${path} (${r.status})`);return r.json();}).catch(e=>{loads.delete(path);throw e;}));return loads.get(path);}
function fail(e){$('error').hidden=false;$('error').textContent='Không thể mở replay: '+e.message;$('loading').hidden=true;document.body.classList.remove('busy');stop();}
function option(value,label){const n=document.createElement('option');n.value=value;n.textContent=label;return n;}
function rowClass(method){return method==='geoi_bq'?' class="proposed"':'';}
function methodLabel(method){return S.results.methods.find(m=>m.id===method)?.label||method;}
function metricCell(mean,ci,d=2,scale=100){return `<b>${fmt(mean*scale,d)}</b>${ci?`<small>[${ci.map(x=>fmt(x*scale,d)).join('; ')}]</small>`:''}`;}

async function selectRun(id,reset=true){
 const version=++loadVersion;stop();document.body.classList.add('busy');$('loading').hidden=false;
 const native=S.index.sessions.find(s=>s.draws.includes(id));if(!native)throw Error('Sample không có trong TEST index');
 try{
  const [session,run]=await Promise.all([get(`data/sessions/${native.id}.json`),get(`data/runs/${id}.json`)]);
  if(version!==loadVersion)return;
  S.session=session;S.run=run;if(reset)S.step=0;S.more=20;
  $('session').value=native.id;$('draw').replaceChildren(...native.draws.map((d,i)=>option(d,`Draw ${i+1}`)));$('draw').value=id;
  $('map-title').textContent=`${native.family} · ${native.id.endsWith('day7')?'Chuyến ngày 7':'Chuyến ngày 8'}`;
  fitAll();render();$('error').hidden=true;
 }catch(e){if(version===loadVersion)fail(e);}
 finally{if(version===loadVersion){document.body.classList.remove('busy');$('loading').hidden=true;}}
}
function updateURL(){if(!S.run)return;const u=new URL(location.href);u.searchParams.set('sample',S.run.id);u.searchParams.set('step',S.step);u.searchParams.set('method',S.method);history.replaceState(null,'',u);}
function setTab(name){if(!['replay','results','proof'].includes(name))name='replay';if(name!=='replay')stop();for(const b of document.querySelectorAll('.tab')){const active=b.dataset.tab===name;b.classList.toggle('active',active);b.setAttribute('aria-selected',active);}for(const p of document.querySelectorAll('.tab-panel'))p.hidden=p.id!==name;history.replaceState(null,'','#'+name);if(name==='replay')requestAnimationFrame(drawMap);}
function stop(){clearInterval(S.timer);S.timer=null;S.playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','Phát hành trình');}
function play(){if(S.playing){stop();return;}if(S.step===30)S.step=0;S.playing=true;$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Tạm dừng');render();S.timer=setInterval(()=>{if(S.step>=30){stop();return;}S.step++;S.more=20;render();if(S.step===30)stop();},1100/S.speed);}
function stepTo(index){stop();S.step=Math.max(0,Math.min(30,Number(index)||0));S.more=20;render();}
function current(){const t=S.session.times[S.step],events=prefixEvents(S.run.streams[S.method],t);return {t,events,event:events.at(-1),ledger:S.run.ledger[S.step],cert:S.run.certificates[S.step],gps:S.session.truth[S.step]};}

function render(){if(!S.run)return;
 const {t,event,events,ledger,cert,gps}=current(),bq=S.method==='geoi_bq',server=S.perspective==='server';
 document.body.classList.toggle('server-view',server);
 document.querySelectorAll('[data-private]').forEach(n=>n.hidden=server);
 $('pipeline').hidden=server||!bq;$('diagnostics').hidden=server||!bq;$('show-anchor').closest('label').hidden=server||!bq;
 $('time').textContent=clock(t);$('timeline').value=S.step;$('previous').disabled=S.step===0;$('next').disabled=S.step===30;
 $('k-now').textContent=event.q.length;$('packets-now').textContent=events.length;
 $('map-badge').textContent=`${methodLabel(S.method)} · ${event.q.length} ${bq?'Q':'tọa độ'} / packet`;
 $('frame-badge').textContent=bq?`Đoạn ${cert.block} · khung ${Math.floor(t%60/20)+1}/3`:`t = ${fmt(t)} s`;
 const contracts={raw:'Control không bảo vệ: tọa độ gửi chính là GPS thật.',geo_i_Z_k1:'Control nền tảng: công bố trực tiếp tham chiếu REM-Z đã lưu.',transprotect_markov:'Markov adaptation: một tọa độ thay thế. Predictor khác paper gốc.',semantic_poi:'OSM/empirical adaptation: tập gồm điểm thật và bốn dummy, không gắn nhãn real/fake.',fake_queries:'Road/history adaptation: giữ toàn bộ packet xen thêm và đếm đủ chi phí.',geoi_bq:'Chỉ Q đi lên server. GPS, Z, belief và nhu cầu thật giữ ở local.'};
 $('contract-note').textContent=contracts[S.method];
 if(bq){
  const branch=ledger.branch;
  $('event-title').textContent=branch==='fresh'?(t===0?'Tạo tham chiếu đầu tiên':'Tạo Z mới sau noisy test'):branch==='reuse'?'Giữ tham chiếu Z cũ':'Phát khung đã chọn';
  $('event-copy').textContent=branch==='fresh'?(t===0?'REM dùng GPS để lấy Z trên mạng đường. Chọn đoạn Q đầu và phát khung thứ nhất.':'Noisy test quyết định tạo lại Z. Đầu đoạn mới: cập nhật belief và chọn chung ba khung Q.'):branch==='reuse'?'Có kiểm tra GPS riêng tư, nhưng giữ Z. Planner vẫn có thể chọn một đoạn Q mới.':'Không có private read mới ở mốc này. Khung tiếp theo thuộc đoạn đã commit; không lấy mẫu lại đoạn Q.';
  const spent=ledger.spent_units*.01;$('geo-spent').textContent=fmt(spent,2)+' m⁻¹';$('geo-meter').style.width=(spent/.23*100)+'%';
  $('q-spent').textContent=fmt(cert.reserved_prefix_epsilon_Q,4)+' / 1';$('q-meter').style.width=(cert.reserved_prefix_epsilon_Q*100)+'%';
  $('q-allowance').textContent=`Đoạn ${cert.block}: εQ,j = ${fmt(qAllowance(cert.block),5)}. Khung sau không tốn một allowance mới.`;
  const [lat,lon]=S.run.streams.geo_i_Z_k1[S.step].coordinates[0];$('anchor-detail').textContent=`Z nội bộ: ${fmt(lat,5)}, ${fmt(lon,5)} · +${ledger.cost_units} unit tại mốc này`;
 }else{$('event-title').textContent=methodLabel(S.method);$('event-copy').textContent=contracts[S.method];}
 if(server){$('event-title').textContent='Packet công khai';$('event-copy').textContent=contracts[S.method];}
 $('attacker-error').textContent=fmt(meters(gps,centroid(event.q)),1)+' m';
 $('fit-gps').hidden=server;if(server&&S.zoomMode==='gps')fitAll();
 renderWire(events,event,t);if(!server)renderLocal();if(bq&&!server)renderStage();
 if(S.zoomMode==='gps')fitGPS();drawMap();updateURL();
}

function renderWire(events,event,t){
 const schema={timestamp_s:event.t,coordinates:event.coordinates,cover_request:{categories:S.map.categories,response_l:60}};
 $('wire').textContent=JSON.stringify(schema,null,2);$('wire-count').textContent=`${event.q.length} tọa độ`;
 $('packet-history').replaceChildren(...[...events].reverse().map(e=>{const d=document.createElement('div');d.textContent=`${fmt(e.t,2)} s`;const span=document.createElement('span');span.textContent=`${e.q.length} tọa độ`;d.append(span);return d;}));
 $('wire-description').textContent=S.method==='geoi_bq'?'Tọa độ Q, clock và cover request. GPS, Z, belief và purpose/category thật không nằm trong bản tin này.':S.method==='raw'?'Control Raw gửi GPS thật. Nhu cầu POI vẫn giữ local bởi cover wrapper chung.':S.method==='geo_i_Z_k1'?'Control REM-Z công bố trực tiếp Z. Nhu cầu POI giữ local bởi cover wrapper chung.':'Bản tin thực của adaptation đã lưu; không gắn nhãn thành viên thật/giả. Nhu cầu POI dùng cover wrapper chung.';
}
function renderLocal(){
 const available=S.run.masks[S.method][S.step][S.cache];
 const order=S.session.orders[S.step][S.purpose][S.category];
 const local=localAnswer(order,available),scores=S.session.scores[S.step][S.purpose];
 $('poi-count').textContent=fmt(available.length);$('answer-count').textContent=fmt(local.answer.length);$('recall').textContent=percent(local.recall,1);
 const unit=S.purpose==='fastest_travel'?'s':'m';
 $('score-unit').textContent=S.purpose==='fastest_travel'?'Thời gian (s)':S.purpose==='minimum_detour'?'Vòng đường thêm (m)':'Khoảng cách đường (m)';
 $('service-note').textContent=S.purpose==='minimum_detour'?'Xếp theo chi phí vòng đường tới đích local đã khai báo trong workload. Đích này không được gửi lên LBS.':S.purpose==='within_radius'?'Lọc trong 1.000 m theo mạng đường có hướng; không có reference hợp lệ thì Recall là N/A.':'Lọc theo loại POI và sắp xếp tại GPS hiện tại trên cùng mạng đường có hướng.';
 const list=local.answer.slice(0,S.more),refs=new Set(local.reference);
 $('poi-list').innerHTML=list.length?list.map((i,n)=>`<tr><td>${n+1}</td><td>${escape(categoryLabels[S.map.pois[i].category]||S.map.pois[i].category)}<small class="poi-id">${escape(S.map.pois[i].id)}</small></td><td>${fmt(scores[i],1)} ${unit}</td><td>${refs.has(i)?'<span class="yes">✓ Có</span>':'—'}</td></tr>`).join(''):`<tr><td colspan="4">${order.length?'Chưa nhận được POI phù hợp ở mốc này.':'Không có POI tham chiếu phù hợp với bộ lọc này · Recall N/A.'}</td></tr>`;
 $('more-pois').hidden=list.length>=local.answer.length;$('more-pois').textContent=`Xem thêm · ${list.length}/${local.answer.length}`;
 $('missing-pois').textContent=local.missing.length?`${local.missing.length}/${local.reference.length} POI reference @5 chưa có trong tập đã nhận: ${local.missing.map(i=>S.map.pois[i].id).join(', ')}.`:local.reference.length?`Đã nhận đủ ${local.reference.length}/${local.reference.length} POI reference @5.`:'N/A: reference rỗng; không tính thành 0% hoặc 100%.';
}
function renderStage(){
 const {t,ledger,cert}=current(),sampled=t%60===0;
 const active=[ledger.private_read,ledger.private_read,sampled,true,true];
 document.querySelectorAll('[data-stage]').forEach(b=>{const i=Number(b.dataset.stage);b.classList.toggle('active',active[i]);b.classList.toggle('selected',i===S.stage);});
 $('pipeline-note').textContent=sampled?'Đầu đoạn · chọn chung ba khung':'Trong đoạn · phát từ action đã commit';
 const texts=[`Mốc ${t}s: ${ledger.private_read?(ledger.branch==='fresh'?'REM tạo Z':'noisy test giữ Z'):'không có private read mới'}. Mỗi unit u = 0,01 m⁻¹. Chi phí +${ledger.cost_units} unit; tổng ${ledger.spent_units}/23. Việc giữ hay đổi cũng được tính chi phí vì test dùng GPS.`,
 'Belief gần đúng được dự đoán theo chuyển động và cập nhật từ protected history. Replay này không lưu vector belief đầy đủ, nên demo không dựng một heatmap giả hoặc gọi score planner là xác suất GPS.',
 `Đoạn ${cert.block}: thư viện ${cert.public_library_size} ứng viên. Belief đổi điểm số nhưng không cắt support. εQ,j = ${fmt(qAllowance(cert.block),6)}, α = ${cert.alpha}; cận likelihood ratio một lần là ${cert.exact_likelihood_ratio_upper}.`,
 'Tại cùng clock, phát một khung đã commit. Mỗi Q hỏi mọi loại POI ở L60. Các frame còn lại và retry cùng bản tin là hậu xử lý; không phải ba lần chọn đoạn mới.',
 'Hợp metadata đã nhận, loại trùng theo POI ID, dùng cache hợp lệ rồi lọc/sắp xếp theo GPS và nhu cầu thật tại local. Danh sách đầy đủ được giữ; Recall@5 chỉ là đánh giá.'];
 $('stage-detail').textContent=texts[S.stage];
 if(S.stage===2&&cert.integer_scores){const title=document.createElement('div');title.className='micro';title.textContent='Điểm lượng tử hoá của các ứng viên (không phải xác suất chọn, không phải Recall thực).';const strip=document.createElement('div');strip.className='score-strip';for(const value of cert.integer_scores){const bar=document.createElement('i');bar.style.height=Math.max(2,value/65536*100)+'%';bar.title=fmt(value/65536,5);strip.append(bar);}$('stage-detail').append(title,strip);}
}

function fitAll(){if(!S.map)return;let xmin=Infinity,ymin=Infinity,xmax=-Infinity,ymax=-Infinity;for(const road of S.map.roads)for(const [x,y]of road){xmin=Math.min(xmin,x);ymin=Math.min(ymin,y);xmax=Math.max(xmax,x);ymax=Math.max(ymax,y);}for(const p of S.map.pois){const [x,y]=p.xy;xmin=Math.min(xmin,x);ymin=Math.min(ymin,y);xmax=Math.max(xmax,x);ymax=Math.max(ymax,y);}S.camera={x:(xmin+xmax)/2,y:(ymin+ymax)/2,span:Math.max(xmax-xmin,ymax-ymin)*1.08};S.zoomMode='all';$('fit-all').classList.add('active');$('fit-gps').classList.remove('active');}
function fitGPS(){const gps=S.session.truth[S.step];S.camera={x:gps[0],y:gps[1],span:1600};S.zoomMode='gps';$('fit-gps').classList.add('active');$('fit-all').classList.remove('active');}
function projection(){const rect=$('map').getBoundingClientRect(),scale=Math.min(rect.width,rect.height)/S.camera.span;return {w:rect.width,h:rect.height,scale,to:p=>[rect.width/2+(p[0]-S.camera.x)*scale,rect.height/2-(p[1]-S.camera.y)*scale]};}
function drawMap(){if(!S.map||!S.run||$('replay').hidden||!S.camera)return;
 const canvas=$('map'),ctx=canvas.getContext('2d'),{w,h,scale,to}=projection(),dpr=devicePixelRatio||1;if(!w||!h)return;
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 ctx.strokeStyle='#a0bec329';ctx.lineWidth=.75;ctx.beginPath();for(const road of S.map.roads){const first=to(road[0]);ctx.moveTo(...first);for(let j=1;j<road.length;j++)ctx.lineTo(...to(road[j]));}ctx.stroke();
 const {event,events,gps}=current(),server=S.perspective==='server',bq=S.method==='geoi_bq';S.points=[];
 function dot(p,color,label,kind,r=5){const [x,y]=to(p);if(x<0||x>w||y<0||y>h)return false;ctx.fillStyle=color;ctx.strokeStyle='#142d3b';ctx.lineWidth=2;ctx.beginPath();if(kind==='anchor'){ctx.moveTo(x,y-r-1);ctx.lineTo(x+r+1,y);ctx.lineTo(x,y+r+1);ctx.lineTo(x-r-1,y);ctx.closePath();}else ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.stroke();if(label){ctx.font='600 10px -apple-system, sans-serif';ctx.fillStyle=color;ctx.fillText(label,x+9,y-8);}S.points.push({x,y,label:kind==='poi'?label||'POI':label,kind,p});return true;}
 if(!server&&$('show-pois').checked){ctx.globalAlpha=.65;for(const i of S.run.masks[S.method][S.step][S.cache])dot(S.map.pois[i].xy,'#ad92d2','', 'poi',2);ctx.globalAlpha=1;}
 const colors=['#6ae0c7','#8fbbff','#baa4e8','#e8b77d','#e7a0ad'];
 // Slot history is public. Dashed chords describe order, not a routed road path.
 for(let k=0;k<event.q.length;k++){ctx.beginPath();ctx.strokeStyle=colors[k%colors.length]+'65';ctx.lineWidth=1;ctx.setLineDash([3,4]);events.forEach((e,i)=>{const p=to(e.q[k]);if(i)ctx.lineTo(...p);else ctx.moveTo(...p);});ctx.stroke();ctx.setLineDash([]);}
 if(!server&&$('show-truth').checked){ctx.strokeStyle='#eaf6ec';ctx.lineWidth=2;ctx.beginPath();S.session.truth.slice(0,S.step+1).forEach((p,i)=>i?ctx.lineTo(...to(p)):ctx.moveTo(...to(p)));ctx.stroke();dot(gps,'#ffffff','GPS','truth',6);}
 if(!server&&bq&&$('show-anchor').checked)dot(S.run.anchors[S.step],'#efb05f','Z','anchor',6);
 let visible=0;event.q.forEach((p,i)=>{if(dot(p,colors[i%colors.length],event.q.length===1?(S.method==='raw'?'X':'Q'):`Q${i+1}`,'query',5))visible++;});
 if($('show-attacker').checked)dot(centroid(event.q),'#f49cae','Centroid S1','attacker',5);
 $('offscreen').hidden=visible===event.q.length;$('offscreen').textContent=`${event.q.length-visible}/${event.q.length} tọa độ công bố ngoài khung`;
 let length=500;while(length*scale>140)length/=2;while(length*scale<50)length*=2;const px=length*scale;ctx.strokeStyle='#aac4cc';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(20,h-28);ctx.lineTo(20+px,h-28);ctx.moveTo(20,h-31);ctx.lineTo(20,h-25);ctx.moveTo(20+px,h-31);ctx.lineTo(20+px,h-25);ctx.stroke();ctx.fillStyle='#aac4cc';ctx.font='9px -apple-system,sans-serif';ctx.fillText(fmt(length)+' m',20,h-37);
}

function renderResults(){
 const data=S.results,scenario=$('scenario').value;
 const descriptions={S1:'Một packet tại tick dịch vụ; không nhìn trước/sau.',S2:'Wire trong 540–600 s; target là trung bình bốn GPS cuối. Đây chưa phải semantic stop inference.',S3:'Toàn wire 0–600 s; tái dựng 31 GPS theo timestamp, tính cả fake events.',S9:'GPS đầu phiên từ toàn transcript; attacker biết public start clock.',S10:'GPS cuối phiên từ toàn transcript; attacker biết phiên kết thúc ở 600 s.'};
 $('scenario-description').textContent=descriptions[scenario];
 const attacks=data.privacy.filter(a=>a.scenario===scenario);
 $('privacy-chart').innerHTML=data.methods.map(m=>{const a=attacks.find(a=>a.method===m.id);return `<div class="bar-row${m.id==='geoi_bq'?' proposed':''}"><span>${escape(m.label)}</span><div class="bar-track"><i class="bar-fill" style="width:${a.Hit100*100}%"></i></div><b>${fmt(a.Hit100*100,2)}%</b></div>`;}).join('');
 $('privacy-table').innerHTML=data.methods.map(m=>{const a=attacks.find(a=>a.method===m.id);return `<tr${rowClass(m.id)}><td>${escape(m.label)}<small title="${escape(JSON.stringify(a.selected_attacker))}">Hit: ${escape(a.selected_attacker.hit100)}<br>MAE: ${escape(a.selected_attacker.mae)}</small></td><td>${metricCell(a.Hit100,a.Hit100_CI95,3)}</td><td>${metricCell(a.MAE_m,a.MAE_m_CI95,1,1)}</td></tr>`;}).join('');
 const utility=data.utility.filter(u=>u.cache==='session'),primary=utility.find(u=>u.method==='geoi_bq').metrics.macro;
 $('macro-result').textContent=percent(primary.mean,3);$('macro-ci').textContent=`CI95% [${primary.absolute95.map(v=>fmt(v*100,3)).join('; ')}]%`;
 $('utility-table').innerHTML=data.methods.map(m=>{const u=utility.find(u=>u.method===m.id),delta=data.comparisons.find(c=>c.control===m.id)?.metrics.macro;return `<tr${rowClass(m.id)}><td>${escape(m.label)}</td>${purposeKeys.map(p=>`<td>${fmt(u.metrics[p].mean*100,3)}%</td>`).join('')}<td>${delta?`${fmt(delta.delta*100,3)}<small>CI [${delta.paired95.map(v=>fmt(v*100,3)).join('; ')}]</small>`:'—'}</td></tr>`;}).join('');
 $('cost-table').innerHTML=data.methods.map(m=>{const c=data.cost[m.id];return `<tr${rowClass(m.id)}><td>${escape(m.label)}</td><td>${fmt(c.packets,2)}</td><td>${fmt(c.requests,2)}</td><td>${fmt(c.request_bytes/1000,3)}</td><td>${fmt(c.reply_bytes/1000,3)}</td></tr>`;}).join('');
 $('native-metrics').innerHTML='<p class="muted">EIE S3 của point estimator chính là MAE. Eq.13 là local-target travel-cost proxy, chỉ áp cho output singleton. ASR/DER/path count còn N/A vì thiếu contract/đầu vào nguồn tương ứng.</p><div class="table-scroll"><table><thead><tr><th>Phương pháp</th><th>Eq.13 Δc (m)</th><th>ASR / DER / path count</th></tr></thead><tbody>'+data.methods.map(m=>`<tr${rowClass(m.id)}><td>${escape(m.label)}</td><td>${fmt(data.native_metrics[m.id].Eq13.mean,2)}</td><td>N/A</td></tr>`).join('')+'</tbody></table></div>';
 renderExtra();
}
function renderExtra(){
 const chosen=$('extra-scenario').value,data=S.results;
 const scopes={S4:'S4 đo route-family session association, chưa phải person/vehicle identity. CI theo 10 two-family blocks. Mốc BA cân bằng: 50%.',S5:'S5 chọn cạnh tương lai ở pha cửa sổ rẽ. Raw chỉ đạt 50%: chưa có positive control đủ mạnh, chưa kết luận contribution của defense.',S6:'S6 chọn một trong hai đích đã biết, sau tín hiệu rẽ. Raw đạt 100%; chưa là unrestricted destination prediction.',S7:'S7 là conditional payload audit với common cover wrapper ở cả sáu arm. Wire equality 100%; không phải natural-intent attacker hay ưu thế riêng trước paper gốc.',S8:'S8 V2 thêm kênh protected partner, tương quan tổng hợp sau căn clock. Không phải native co-travel; bank fusion chốt sau readout V1. Raw partner cho phép định vị target (available direct control 100%).'};
 $('extra-scope').textContent=scopes[chosen];
 const headers={S4:['BA % · CI95%','AUC theo block · CI95%'],S5:['Accuracy % · CI95%','Brier'],S6:['Accuracy % · CI95%','MAE đích m · CI95%'],S7:['Wire equality','Joint vulnerability'],S8:['Hit100 % · CI95%','MAE m · CI95%']};
 $('extra-head').innerHTML='<tr><th>Phương pháp</th>'+headers[chosen].map(h=>`<th>${h}</th>`).join('')+'</tr>';
 $('extra-table').innerHTML=data.methods.map(m=>{const id=chosen==='S4'?`S4--${m.id}`:chosen==='S5'?`S5--turn_window--${m.id}`:chosen==='S6'?`S6--post_turn--${m.id}`:chosen==='S7'?`S7--${m.id}`:`S8--protected_partner--${m.id}`;
 const task=(chosen==='S8'?data.fusion:data.remaining).find(t=>t.task===id);let a,b;
 if(chosen==='S4'){a=metricCell(task.metrics.balanced_accuracy.mean,task.metrics.balanced_accuracy.ci95);b=metricCell(task.metrics.group_mean_roc_auc.mean,task.metrics.group_mean_roc_auc.ci95,3,1);}
 if(chosen==='S5'){a=metricCell(task.metrics.accuracy.mean,task.metrics.accuracy.ci95);b=fmt(task.metrics.brier.mean,3);}
 if(chosen==='S6'){a=metricCell(task.metrics.accuracy.mean,task.metrics.accuracy.ci95);b=metricCell(task.destination_MAE_m.mean,task.destination_MAE_m.ci95,1,1);}
 if(chosen==='S7'){a=percent(task.equal_wire_rate);b=percent(task.metrics.joint_vulnerability.mean);}
 if(chosen==='S8'){a=metricCell(task.metrics.Hit100.mean,task.metrics.Hit100.ci95);b=metricCell(task.metrics.MAE_m.mean,task.metrics.MAE_m.ci95,1,1);}
 return `<tr${rowClass(m.id)}><td>${escape(m.label)}</td><td>${a}</td><td>${b}</td></tr>`;}).join('');
}

const tourSteps=[
 {t:0,stage:0,title:'1. Đầu phiên: GPS → Z',copy:'Đọc GPS cho lớp bảo vệ. REM lấy một Z trên mạng đường; tốn 1 unit Geo-I. Z là nội bộ, không phải tọa độ mặc định gửi lên server.'},
 {t:0,stage:2,title:'2. Từ protected history tới đoạn Q',copy:'Belief chấm điểm 24 ứng viên; kernel chỉ điều chỉnh probability quanh public base. Chọn chung đoạn ba khung, rồi phát năm Q đầu tiên.'},
 {t:20,stage:3,title:'3. Sau 20 giây: không đọc bảo vệ GPS mới',copy:'Client phát khung đã commit. Không tạo Z mới và không lấy mẫu lại một đoạn Q; đây là điểm cần nhìn rõ khi trình bày composition.'},
 {t:60,stage:0,title:'4. Đầu đoạn tiếp theo',copy:'Pacing cho phép private read khi dự toán cap còn đủ. Noisy test quyết định giữ hay tạo Z; cả hai nhánh đều phải được hạch toán.'},
 {t:180,stage:4,title:'5. Kiểm tra chất lượng dịch vụ',copy:'Đổi purpose, loại POI hoặc cache. Danh sách và Recall tại local thay đổi, nhưng bản tin Q không đổi. Nhấn xem thêm để thấy toàn danh sách đã sắp xếp.'},
 {t:600,stage:3,title:'6. Điểm cuối phiên và kết quả tổng hợp',copy:'Khung cuối vẫn phát theo public clock, không cần giấu thời điểm đóng. Qua tab Kết quả để xem S9/S10, sáu phương pháp và chi phí; một sample không thay cohort benchmark.'}
];
function tourAt(i){S.tour=Math.max(0,Math.min(tourSteps.length-1,i));const t=tourSteps[S.tour];$('tour').hidden=false;S.method='geoi_bq';S.perspective='analysis';$('method').value=S.method;$('perspective').value=S.perspective;S.stage=t.stage;stepTo(t.t/20);$('tour-count').textContent=`${S.tour+1} / ${tourSteps.length}`;$('tour-title').textContent=t.title;$('tour-copy').textContent=t.copy;$('tour-prev').disabled=S.tour===0;$('tour-next').disabled=S.tour===tourSteps.length-1;}

function bind(){
 document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>setTab(b.dataset.tab)));
 addEventListener('hashchange',()=>setTab(location.hash.slice(1)||'replay'));
 $('session').addEventListener('change',()=>{const s=S.index.sessions.find(s=>s.id===$('session').value);selectRun(s.draws[0]);});$('draw').addEventListener('change',()=>selectRun($('draw').value));
 $('method').addEventListener('change',()=>{S.method=$('method').value;S.more=20;render();});$('perspective').addEventListener('change',()=>{S.perspective=$('perspective').value;if(S.perspective==='server')fitAll();render();});
 for(const k of ['purpose','category','cache'])$(k).addEventListener('change',()=>{S[k]=$(k).value;S.more=20;render();});
 $('more-pois').addEventListener('click',()=>{S.more+=30;renderLocal();});
 $('timeline').addEventListener('input',()=>stepTo($('timeline').value));$('previous').addEventListener('click',()=>stepTo(S.step-1));$('next').addEventListener('click',()=>stepTo(S.step+1));$('reset').addEventListener('click',()=>stepTo(0));$('play').addEventListener('click',play);
 $('speed').addEventListener('change',()=>{const was=S.playing;stop();S.speed=Number($('speed').value);if(was)play();});
 $('fit-all').addEventListener('click',()=>{fitAll();drawMap();});$('fit-gps').addEventListener('click',()=>{fitGPS();drawMap();});
 for(const id of ['show-truth','show-anchor','show-attacker','show-pois'])$(id).addEventListener('change',drawMap);
 for(const [id,factor]of [['zoom-in',.7],['zoom-out',1.4]])$(id).addEventListener('click',()=>{S.camera.span=Math.min(50000,Math.max(100,S.camera.span*factor));S.zoomMode='manual';drawMap();});
 const canvas=$('map');canvas.addEventListener('wheel',e=>{e.preventDefault();S.camera.span=Math.min(50000,Math.max(100,S.camera.span*Math.exp(e.deltaY*.001)));S.zoomMode='manual';drawMap();},{passive:false});
 canvas.addEventListener('pointerdown',e=>{S.drag={x:e.clientX,y:e.clientY,cx:S.camera.x,cy:S.camera.y};canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(S.drag){const scale=projection().scale;S.camera.x=S.drag.cx-(e.clientX-S.drag.x)/scale;S.camera.y=S.drag.cy+(e.clientY-S.drag.y)/scale;S.zoomMode='manual';drawMap();return;}const point=S.points.filter(p=>Math.hypot(p.x-x,p.y-y)<12).at(-1),tip=$('map-tooltip');tip.hidden=!point;if(point){tip.textContent=point.label||'POI đã nhận';tip.style.left=Math.min(x+12,r.width-170)+'px';tip.style.top=Math.max(10,y-30)+'px';}});
 const end=()=>S.drag=null;canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);canvas.addEventListener('pointerleave',()=>{$('map-tooltip').hidden=true;});
 new ResizeObserver(drawMap).observe(canvas);
 document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>{S.stage=Number(b.dataset.stage);renderStage();}));
 $('scenario').addEventListener('change',renderResults);$('extra-scenario').addEventListener('change',renderExtra);
 $('prior').addEventListener('input',()=>{const p=Number($('prior').value)/100;$('prior-value').textContent=percent(p,0);$('posterior-value').textContent=percent(posteriorUpper(p),2);});
 $('tour-toggle').addEventListener('click',()=>tourAt(0));$('tour-close').addEventListener('click',()=>$('tour').hidden=true);$('tour-prev').addEventListener('click',()=>tourAt(S.tour-1));$('tour-next').addEventListener('click',()=>tourAt(S.tour+1));
 addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName)||$('replay').hidden||!S.run)return;if(e.code==='Space'){e.preventDefault();play();}if(e.key==='ArrowRight')stepTo(S.step+1);if(e.key==='ArrowLeft')stepTo(S.step-1);});
}

async function init(){try{
 [S.index,S.map,S.results]=await Promise.all([get('data/index.json'),get('data/map.json'),get('data/results.json')]);
 $('session').replaceChildren(...S.index.sessions.map(s=>option(s.id,`${s.family} · ${s.id.endsWith('day7')?'Ngày 7':'Ngày 8'}`)));
 $('method').replaceChildren(...S.results.methods.map(m=>option(m.id,m.label)));$('method').value='geoi_bq';
 $('category').replaceChildren(...S.map.categories.map(c=>option(c,categoryLabels[c]||c)));S.category=S.map.categories[0];$('category').value=S.category;
 const params=new URLSearchParams(location.search),id=params.get('sample')||S.index.default;
 if(S.results.methods.some(m=>m.id===params.get('method'))){S.method=params.get('method');$('method').value=S.method;}
 S.step=Math.max(0,Math.min(30,parseInt(params.get('step'),10)||0));bind();renderResults();
 $('source-status').textContent='Dữ liệu đóng băng · 108 pins';setTab(location.hash.slice(1)||'replay');
 await selectRun(S.index.sessions.some(s=>s.draws.includes(id))?id:S.index.default,false);
 }catch(e){fail(e);}}
init();
