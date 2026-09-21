"""Source-backed static report figures (matplotlib). No defender is run here."""
from pathlib import Path
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
teal='#087f7a';ink='#243b45';gray='#7b8589'
def save(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=170)
    plt.close(fig)
def load(p):return json.loads((ROOT/p).read_text())
def local(points,origin):
    scale=math.pi/180*6371000
    return ([(p[1]-origin[1])*scale*math.cos(math.radians(origin[0])) for p in points],[(p[0]-origin[0])*scale for p in points])

fig,ax=plt.subplots(figsize=(11,6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.add_patch(FancyBboxPatch((.015,.07),.72,.86,boxstyle='round,pad=.008',facecolor='#f6f9f8',edgecolor=gray,linestyle='--'))
ax.text(.035,.95,'THIẾT BỊ: giữ vị trí thật và trạng thái riêng',color=ink,weight='bold')
ax.text(.78,.95,'MÁY CHỦ LBS',color=ink,weight='bold')
def box(x,y,w,h,title,body,dashed=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor='white',edgecolor=teal,linestyle='--' if dashed else '-',linewidth=1.2))
 ax.text(x+w/2,y+h*.74,title,ha='center',va='center',weight='bold',fontsize=10)
 ax.text(x+w/2,y+h*.32,body,ha='center',va='center',fontsize=9,linespacing=1.5)
def arr(a,b,label=None):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':ink,'lw':1.2})
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.02,label,ha='center',fontsize=8,color=ink)
box(.04,.72,.18,.16,'1. Đầu vào','GPS xₜ, thời gian\nnhu cầu POI thật')
box(.29,.72,.18,.16,'2. Neo riêng tư','Test nhiễu / tái dùng\nsổ ngân sách B, H')
box(.54,.72,.17,.16,'3. Bộ lọc','Dừng / di chuyển\nbelief từ neo đã bảo vệ')
box(.54,.43,.17,.17,'4. Chọn K điểm','Miền tới được trên làn\nphủ POI + thay điểm')
box(.29,.43,.18,.17,'5. Cổng biên','S9/S10: đề xuất\nchưa tích hợp model',True)
box(.79,.43,.18,.17,'6. Truy hồi','K vị trí công bố\ncác loại truy vấn')
box(.29,.14,.42,.16,'7. Gộp / loại trùng / xếp hạng tại máy','Dùng vị trí thật để chọn top-5; không gửi bước lọc')
box(.04,.43,.18,.17,'Ngữ cảnh công khai','Mạng đường, POI\nlịch / tham số cố định')
arr((.22,.8),(.28,.8));arr((.47,.8),(.53,.8));arr((.625,.71),(.625,.61));arr((.53,.515),(.48,.515))
ax.plot([.47,.50,.76],[.46,.35,.35],color=ink,lw=1.2);arr((.76,.35),(.79,.46));ax.text(.61,.36,'transcript được phép',ha='center',fontsize=8,color=ink);arr((.875,.42),(.70,.27));ax.text(.87,.405,'phản hồi POI',ha='center',fontsize=8,color=ink,bbox={'facecolor':'white','edgecolor':'none','pad':1})
arr((.22,.52),(.28,.52));arr((.215,.61),(.54,.70))
ax.plot([.12,.025,.025,.28],[.71,.71,.21,.21],color=gray,ls=':',lw=1.1);arr((.28,.21),(.29,.21))
ax.text(.14,.25,'GPS thật chỉ dùng nội bộ',ha='center',fontsize=8,color=gray)
ax.text(.80,.095,'Đối thủ phân tích\ntranscript phía LBS.\nKhông nhận neo,\nGPS hoặc nhãn thật.',ha='left',fontsize=8.5,linespacing=1.5)
ax.text(.04,.015,'Nét đứt ở bước 5: mô-đun bảo vệ biên đang đề xuất; các bước lõi đã có trong mã nguồn.',fontsize=9,color=gray)
save(fig,'architecture')

fresh=load('artifacts/benchmarks/fresh_switching/confirmation.json')
r=next(r for r in fresh['rows'] if r['case_id']=='S3.A' and r['method']=='switching_exchange' and r['replicate']==1)
record=next(z for z in fresh['records'] if z['record_id']==r['record_id'])
truth=[(p['lat'],p['lon']) for p in record['points']];origin=truth[0]
allpts=truth+r['evaluator_anchors']+[(c['lat'],c['lon']) for e in r['public']['events'] for c in e['candidates']]
xs,ys=local(allpts,origin)
fig,axs=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
axs[0].plot(*local(truth,origin),'o-',color=ink,label='Vị trí thật',ms=4)
axs[0].plot(*local(r['evaluator_anchors'],origin),'s--',color=teal,label='Neo riêng tư',ms=4)
axs[0].set_title('Thiết bị / bộ đánh giá');axs[0].legend(fontsize=8)
for j in range(5):
 pts=[(e['candidates'][j]['lat'],e['candidates'][j]['lon']) for e in r['public']['events']]
 axs[1].plot(*local(pts,origin),marker='.',lw=1,alpha=.85,label=f'Dummy {j+1}')
axs[1].set_title('Máy chủ: chỉ thấy 5 track công bố');axs[1].legend(fontsize=8,ncol=2)
for a in axs:
 a.set_xlabel('Đông–Tây (m)');a.set_ylabel('Bắc–Nam (m)');a.set_aspect('equal',adjustable='box');a.set_xlim(min(xs)-70,max(xs)+70);a.set_ylim(min(ys)-70,max(ys)+70);a.grid(alpha=.16)
fig.suptitle(f'Mẫu thực nghiệm {r["record_id"]} · S3.A · lần lặp 1',fontsize=11)
save(fig,'protected_sample')

D=load('artifacts/datasets/urban_fresh_v2/dataset.json')
rr=next(r for r in D['records'] if r['case_id']=='S5.C')
sids=rr['session_ids'];origin=(D['traces'][sids[0]][0]['lat'],D['traces'][sids[0]][0]['lon'])
fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
for i,sid in enumerate(sids):
 t=D['traces'][sid];pts=[(p['lat'],p['lon']) for p in t];col=[ink,teal][i]
 ax.plot(*local(pts,origin),ls='--',color=col,lw=1,label=f'{sid}: toàn chuyến (đánh giá)')
 obs=[(t[j]['lat'],t[j]['lon']) for j in rr['observed_indices'][i]]
 ax.plot(*local(obs,origin),color=col,lw=3,label=f'{sid}: tiền tố được phép')
 ax.scatter(*local([pts[-1]],origin),marker='X',s=65,color=col)
 ax.annotate(f'Đích {i+1}',tuple(v[0] for v in local([pts[-1]],origin)),xytext=(6,6),textcoords='offset points',fontsize=9)
ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18);ax.set_xlabel('Đông–Tây (m)');ax.set_ylabel('Bắc–Nam (m)');ax.legend(fontsize=8,loc='best');ax.set_title('Cặp thật trong dataset: chung tiền tố, khác nhánh và khác đích',fontsize=11)
save(fig,'fork_sample')

ss=[s for s in fresh['summaries'] if s['method']=='switching_exchange'];labels=[s['case_id'] for s in ss];x=list(range(len(ss)))
fig,axs=plt.subplots(2,1,figsize=(9,5.6),sharex=True,layout='constrained')
axs[0].bar([i-.16 for i in x],[100*s['selected_hit100'] for s in ss],width=.3,color=teal,label='Đối thủ đã chọn')
axs[0].bar([i+.16 for i in x],[100*s['envelope_hit100'] for s in ss],width=.3,color=gray,label='Envelope đã thử')
axs[0].set_ylabel('Hit100 (%) ↓');axs[0].legend(fontsize=8,ncol=2);axs[0].grid(axis='y',alpha=.15)
for depth,col,marker in [('5',gray,'o'),('10',teal,'s')]:axs[1].plot(x,[100*s['utility'][depth]['poi_recall_at_5'] for s in ss],color=col,marker=marker,label='L='+depth)
axs[1].axhline(90,color=ink,ls='--',lw=1,label='Mốc 90%');axs[1].set_ylim(80,101);axs[1].set_ylabel('Recall@5 (%) ↑');axs[1].set_xticks(x,labels);axs[1].legend(fontsize=8,ncol=3);axs[1].grid(axis='y',alpha=.15)
fig.suptitle('Switching + exchange · kết quả xác nhận đã lưu · không chọn lại trên test',fontsize=11)
save(fig,'fresh_cases')

b=load('artifacts/benchmarks/report_boundary_audit/results.json')
fig,axs=plt.subplots(2,2,figsize=(9,5.8),sharex=True,layout='constrained')
for row,sc in enumerate(['S9','S10']):
 for method,col,lab in [('unprotected',gray,'Chỉ mặt nạ nền'),('br_private',teal,'BR-private')]:
  group=sorted([s for s in b['summaries'] if s['scenario']==sc and s['method']==method],key=lambda s:s['extra_cut_s'])
  xx=[s['extra_cut_s'] for s in group]
  axs[row,0].plot(xx,[100*s['envelope_hit100'] for s in group],color=col,marker='o',label=lab)
  axs[row,1].plot(xx,[100*s['utility']['recall_all_original_queries'] for s in group],color=col,marker='o',label=lab)
 axs[row,0].set_title(sc+' · Hit100 envelope thăm dò');axs[row,0].set_ylabel('Hit100 (%) ↓');axs[row,0].set_ylim(-2,50)
 axs[row,1].set_title(sc+' · tính cả truy vấn bị cắt');axs[row,1].set_ylabel('Recall cửa sổ gốc (%) ↑');axs[row,1].set_ylim(50,102)
 for a in axs[row]:a.set_xticks([0,40,80]);a.grid(alpha=.18);a.legend(fontsize=8)
for a in axs[-1]:a.set_xlabel('Phần cắt thêm (giây)')
save(fig,'boundary_tradeoff')
print('Rendered 5 source-backed figures as PDF/SVG/PNG.')
