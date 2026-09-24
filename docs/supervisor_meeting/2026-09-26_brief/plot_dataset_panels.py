"""Source-backed maps with sample timing: spatial overlap is not one event."""
import runpy
from pathlib import Path
from dataset_content import REPRESENTATIVES
ns=runpy.run_path(str(Path(__file__).with_name('plot_sample_maps.py')))
plt=ns['plt'];out=ns['OUT'];colors=ns['colors']
plt.rcParams['svg.hashsalt']='scenario-risk-v1'
for name,suffix in REPRESENTATIVES.items():
    c=next(c for c in ns['payload']['cases'] if c['case']==name+'.'+suffix)
    fig=plt.figure(figsize=(5.4,5.7))
    many=len(c['tracks'])>3
    ax=fig.add_axes([.04,.35 if many else .29,.92,.60 if many else .66]);ns['plot'](ax,c)
    tx=fig.add_axes([.19,.10,.75,.19 if many else .12])
    common=min(tr['times'][tr['observed'][0]] for tr in c['tracks'])
    ticks=[];labels=[]
    for k,tr in enumerate(c['tracks']):
        obs=tr['observed']; origin=common if name=='S8' else tr['times'][obs[0]]
        times=[tr['times'][i]-origin for i in obs];row=len(c['tracks'])-1-k
        tx.scatter(times,[row]*len(times),s=14,color=colors[k%len(colors)],marker='|')
        ticks.append(row);labels.append(f'{k+1}: {len(obs)} mẫu')
        if len(c['tracks'])==1:
            unique=len(set(tuple(tr['points'][i]) for i in obs))
            ax.text(.02,.98,f'{len(obs)} mẫu / {unique} tọa độ',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'#cccccc','alpha':.95,'pad':3})
    tx.set_yticks(ticks,labels,fontsize=8);tx.tick_params(axis='x',labelsize=8);tx.set_ylim(-.7,len(ticks)-.3)
    if all(len(t['observed'])==1 for t in c['tracks']):tx.set_xlim(-1,1);tx.set_xticks([0])
    for edge in ['top','right','left']:tx.spines[edge].set_visible(False)
    tx.grid(axis='x',alpha=.18);tx.set_xlabel('Giây từ mẫu đầu chung (cặp xe)' if name=='S8' else 'Giây từ mẫu đầu mỗi phiên; mỗi vạch = 1 mẫu',fontsize=8)
    if name=='S2':fig.text(.5,.245,'Đợt 1: 9 mẫu  |  rời điểm dừng  |  đợt 2: 9 mẫu',ha='center',fontsize=8)
    if name=='S7':fig.text(.5,.245,'0 s: pharmacy → 20 s: clinic → 40 s: hospital',ha='center',fontsize=8)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/'figures'/f'sample_{name}.{ext}',bbox_inches='tight',dpi=170)
    plt.close(fig)
ns['write_explorer']()
