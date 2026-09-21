"""Render exact FCD samples over archived SUMO geometry; no new simulation."""
from pathlib import Path
import json, math, hashlib, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
D=json.loads((ROOT/'artifacts/datasets/urban_fresh_v2/dataset.json').read_text())
P=json.loads((ROOT/'artifacts/benchmarks/paper_benchmark/results.json').read_text())
S={x['case_id']:x for x in json.loads((OUT/'printed_samples.json').read_text())}
FIRST={}
for r in D['records']:FIRST.setdefault(r['case_id'],r)
assert D['network']['osm_sha256']==P['manifests'][0]['provenance']['sha256']['osm']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#087f7a','#c17422','#72559b','#326aa2','#a94c64','#667933','#2f4149']
scale=math.pi/180*6371000; lat0=39.99

def xy(pts):
 a=np.array(pts);return np.column_stack(((a[:,0]-116.29)*scale*math.cos(math.radians(lat0)),(a[:,1]-39.96)*scale))
roads=[xy(r) for r in P['roads']]
notes={
'S1':['Một vị trí ở vùng có nhiều nhánh.','Một vị trí ít lựa chọn đường đi.','Vị trí gần POI hiếm: pharmacy.'],
'S2':['Dừng ngắn: 29 giây.','Dừng dài: 179 giây.','Hai lần dừng; không quan sát liên tục.'],
'S3':['Chuỗi lấy mẫu mỗi 20 giây.','Chuỗi trên đường ít nhánh.','Chuỗi thưa: mỗi 60 giây.'],
'S4':['Cùng người, cùng thiết bị.','Cùng người, khác thiết bị.','Khác người, cùng thiết bị.'],
'S5':['Đoán cạnh đường kế tiếp.','Đoán tiếp khi ít lựa chọn.','Chung tiền tố, khác nhánh tiếp theo.'],
'S6':['Chung tiền tố, đích cách 4,47 km.','Hai đích gần nhau: khoảng 46 m.','Sáu chuyến lịch sử + tiền tố chuyến mới.'],
'S7':['Query thật: clinic.','Clinic ẩn trong bó 6 loại query.','Chuỗi pharmacy → clinic → hospital.'],
'S8':['Đồng hành; gần nhau một phần.','Đồng hành; gần nhau toàn cửa sổ.','Gần nhau nhưng nhãn không đồng hành.'],
'S9':['Che 60 giây đầu; suy lại điểm xuất phát.','Hai điểm xuất phát khác nhau.','Hai phiên lặp cùng điểm xuất phát.'],
'S10':['Che đoạn cuối; suy điểm kết thúc.','Cùng tiền tố nhưng hai điểm kết thúc.','Hai phiên lặp cùng điểm kết thúc.']}

payload={'roads':P['roads'],'cases':[]}
for case,r in FIRST.items():
 tracks=[]
 for slot,sid in enumerate(r['session_ids']):
  t=D['traces'][sid]; lab=r['labels']; target=[]
  if 'future_indices' in lab:target=[lab['future_indices'][slot]]
  elif 'future_index' in lab:target=[lab['future_index']]
  elif 'target_indices' in lab:
   target=lab['target_indices'] if len(r['session_ids'])==1 else [lab['target_indices'][slot]]
  elif 'target_index' in lab and slot==lab.get('target_slot',0):target=[lab['target_index']]
  tracks.append({'id':sid,'points':[[p['lon'],p['lat']] for p in t],'observed':r['observed_indices'][slot],'target':target})
 payload['cases'].append({'case':case,'record':r['record_id'],'tracks':tracks,'label':S[case]['label_text'],'note':notes[r['scenario']]['ABC'.index(case[-1])]})
provenance={'background':'archived SUMO road polylines, paper_benchmark seed 81','osm_sha256':D['network']['osm_sha256'],'background_network_sha256':P['manifests'][0]['provenance']['sha256']['network'],'dataset_network_sha256':D['network']['sha256'],'exact_network_identity_verified':False,'limitation':'Same OSM hash and netconvert options; different network file hashes. Original dataset .net.xml unavailable, so exact geometry equivalence is not established.','dataset_sha256':hashlib.sha256((ROOT/'artifacts/datasets/urban_fresh_v2/dataset.json').read_bytes()).hexdigest(),'attribution':'© OpenStreetMap contributors; https://www.openstreetmap.org/copyright','counts':{'roads':len(roads),'cases':len(payload['cases'])}}
(OUT/'map_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
payload['provenance']=provenance

def plot(ax,c):
 ax.add_collection(LineCollection(roads,colors='#c4c9cc',linewidths=.6,zorder=0))
 focus=[]
 for k,tr in enumerate(c['tracks']):
  a=xy(tr['points']);obs=tr['observed'];col=colors[k%len(colors)]
  ax.plot(a[:,0],a[:,1],color=col,alpha=.35,ls='--',lw=.9)
  # Individual observed points only: never join sparse samples across hidden intervals.
  ax.scatter(a[obs,0],a[obs,1],s=15,color=col,zorder=3,label=f'{k+1}: {tr["id"]}')
  focus.extend(a[obs]);target=tr['target']
  if target:
   ax.scatter(a[target,0],a[target,1],s=58,marker='*',facecolor='#b72e35',edgecolor='white',linewidth=.35,zorder=5);focus.extend(a[target])
  start=a[obs[0]];ax.annotate(str(k+1),start,xytext=(5,5),textcoords='offset points',fontsize=8,color=col,weight='bold')
 f=np.array(focus);lo=f.min(0);hi=f.max(0);mid=(lo+hi)/2;span=max(float(max(hi-lo))*1.15,380)
 ax.set_xlim(mid[0]-span/2,mid[0]+span/2);ax.set_ylim(mid[1]-span/2,mid[1]+span/2);ax.set_aspect('equal')
 ax.set_xticks([]);ax.set_yticks([])
 bar=50 if span<600 else 200 if span<2000 else 1000
 x,y=mid[0]-span*.42,mid[1]-span*.43;ax.plot([x,x+bar],[y,y],color='#243b45',lw=2);ax.text(x,y+span*.025,f'{bar} m',fontsize=8)
 ax.text(.96,.94,'Bắc ↑',transform=ax.transAxes,ha='right',fontsize=8)
 ax.set_title(c['case']+' · '+c['record'],fontsize=9)
 return ax

if __name__ == '__main__':
    with PdfPages(OUT/'sample_maps.pdf') as pdf:
     for scenario in ['S1','S2','S3','S9','S10','S5','S6','S8','S4','S7']:
      cases=[next(c for c in payload['cases'] if c['case']==scenario+'.'+s) for s in 'ABC']
      fig,axs=plt.subplots(1,3,figsize=(12,4.6))
      for ax,c in zip(axs,cases):plot(ax,c);ax.set_xlabel(c['note'],fontsize=9,wrap=True)
      fig.subplots_adjust(left=.015,right=.99,top=.91,bottom=.22,wspace=.08)
      fig.text(.02,.10,'Chấm màu: mẫu được phép trước bảo vệ · Nét đứt: toàn chuyến (chỉ để đánh giá) · Sao đỏ: nhãn vị trí cần suy luận',fontsize=9)
      fig.text(.02,.045,'Nền SUMO lưu từ benchmark, cùng nguồn OSM; chưa xác minh trùng hình học mạng dataset. © OpenStreetMap contributors',fontsize=8,color='#626b70')
      for ext in ['pdf','svg','png']:fig.savefig(OUT/'figures'/f'map_{scenario}.{ext}',dpi=170,bbox_inches='tight')
      pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    # Offline explorer embeds all geometry, so opening from file:// works.
    template=(OUT/'sample_map_template.html').read_text()
    (OUT/'sample_maps.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))))
    print('Rendered 30 cases, 10 triptychs, atlas PDF and offline map explorer.')
