"""Thesis-native fixed-population privacy/utility scatter with all grid points."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.export_coverage_frontier import OUTPUT,LABELS
from experiments.run_service_cover import read


def plot():
    data=read(OUTPUT/'readout.json')
    palette={'geometric':'#5b6066','mean_greedy':'#2676b8',
             'mean_exchange':'#174a73','capped_exchange':'#cf7b28'}
    markers={'geometric':'s','mean_greedy':'o','mean_exchange':'^','capped_exchange':'D'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(9,5.5),sharex=True,sharey=True)
    offsets={'geometric':(5,5),'mean_greedy':(-7,-14),
             'mean_exchange':(7,2),'capped_exchange':(-6,8)}
    for ax,depth in zip(axes,(5,10)):
        annotations=[]
        for method in LABELS:
            rows=[r for r in data['rows'] if (r['method'],r['depth'])==(method,depth)]
            assert len(rows)==3
            ax.scatter([100*r['recall'] for r in rows],[100*r['envelope_hit100'] for r in rows],
                label=LABELS[method],c=palette[method],marker=markers[method],s=55,
                edgecolors='white',linewidths=.5)
            for r in rows:
                offset=offsets[method]
                if method=='mean_greedy' and r['budget']==.12:offset=(-7,8)
                if method=='mean_greedy' and r['budget']==.48:offset=(7,-12)
                if method=='geometric' and r['budget']==.24:offset=(-6,5)
                annotations.append(ax.annotate(f'{r["budget"]:.2f}',(100*r['recall'],100*r['envelope_hit100']),
                            xytext=offset,textcoords='offset points',fontsize=9,
                            ha='right' if offset[0]<0 else 'left',color=palette[method]))
                annotations[-1].set_gid(r['id'])
        ax.set_title(f'LSP trả top-{depth}\nThiết bị giữ top-5',fontsize=11)
        ax.set_xlabel('Recall@5 trung bình (%)\ncao hơn tốt hơn')
        ax.set_xlim(79,100)
        ax.set_ylim(4.5,28)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15);ax.set_axisbelow(True)
        ax._budget_annotations=annotations
    axes[0].set_ylabel('Hit trong 100 m, cực trị bộ thử (%)\nthấp hơn tốt hơn')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.0),frameon=False,fontsize=10)
    fig.suptitle('Đánh đổi riêng tư và chất lượng dịch vụ',fontsize=13)
    fig.text(.5,.92,'9 ca · 4 nhóm phát triển dùng lại · K=5 · nhãn điểm: B (1/m)',ha='center',fontsize=10)
    fig.text(.5,.14,'Cực trị trong bộ thử; chi phí và thời gian được báo cáo riêng.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.76,bottom=.32,wspace=.15)
    fig.canvas.draw()
    for ax in axes:
        boxes=[a.get_window_extent(fig.canvas.get_renderer()) for a in ax._budget_annotations]
        overlaps=[(ax._budget_annotations[i].get_gid(),ax._budget_annotations[j].get_gid())
                  for i,a in enumerate(boxes) for j,b in enumerate(boxes) if i<j and a.overlaps(b)]
        assert not overlaps,overlaps
    fig.savefig(OUTPUT/'frontier.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__=='__main__':plot()
