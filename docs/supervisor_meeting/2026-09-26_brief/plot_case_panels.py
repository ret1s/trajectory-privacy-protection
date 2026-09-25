"""Active map + time panels: S10 A/C, three cases for each other scenario."""
import runpy
from pathlib import Path
ns=runpy.run_path(str(Path(__file__).with_name('plot_sample_maps.py')))
plt=ns['plt'];out=ns['OUT'];colors=ns['colors']
plt.rcParams['svg.hashsalt']='all-abc-v1'
for c in ns['payload']['cases']:
    name=c['case'].split('.')[0]
    fig=plt.figure(figsize=(6.1,4.25))
    ax=fig.add_axes([.01,.04,.72,.9]);ns['plot'](ax,c)
    tx=fig.add_axes([.81,.14,.17,.73])
    common=min(t['times'][t['observed'][0]] for t in c['tracks'])
    labels=[]
    for k,tr in enumerate(c['tracks']):
        obs=tr['observed'];origin=common if name=='S8' else tr['times'][obs[0]]
        times=[tr['times'][i]-origin for i in obs]
        tx.scatter([k]*len(times),times,s=17,color=colors[k%len(colors)],marker='_')
        labels.append(str(k+1))
    tx.set_xticks(range(len(labels)),labels,fontsize=7);tx.set_xlim(-.6,len(labels)-.4)
    tx.set_ylabel('Giây từ mẫu đầu'+(' chung' if name=='S8' else ' mỗi phiên'),fontsize=7,labelpad=2)
    tx.tick_params(axis='y',labelsize=7);tx.invert_yaxis()
    if all(len(t['observed'])==1 for t in c['tracks']):tx.set_ylim(1,-1);tx.set_yticks([0])
    for edge in ['top','right','bottom']:tx.spines[edge].set_visible(False)
    tx.grid(axis='y',alpha=.18);tx.set_xlabel('Phiên',fontsize=7)
    n=sum(len(t['observed']) for t in c['tracks']);caption=f'{n} mẫu; mỗi vạch = 1 mẫu'
    if len(c['tracks'])==1:
        tr=c['tracks'][0];unique=len(set(tuple(tr['points'][i]) for i in tr['observed']))
        caption=f'{n} mẫu / {unique} tọa độ'
    ax.text(.02,.98,caption,transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'#ccc','alpha':.95,'pad':2})
    for ext in ['pdf','svg','png']:
        fig.savefig(out/'figures'/('case_'+c['case'].replace('.','_')+'.'+ext),bbox_inches='tight',dpi=160)
    plt.close(fig)
print('Rendered',len(ns['payload']['cases']),'active individual case panels.')
