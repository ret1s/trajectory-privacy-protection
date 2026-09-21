"""Reuse source-backed sample rendering for larger main-report panels."""
import runpy
from pathlib import Path
from dataset_content import REPRESENTATIVES
ns=runpy.run_path(str(Path(__file__).with_name('plot_sample_maps.py')))
plt=ns['plt'];out=ns['OUT']
for name,suffix in REPRESENTATIVES.items():
    c=next(c for c in ns['payload']['cases'] if c['case']==name+'.'+suffix)
    fig,ax=plt.subplots(figsize=(5.4,4.5))
    ns['plot'](ax,c)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.92)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/'figures'/f'sample_{name}.{ext}',bbox_inches='tight',dpi=170)
    plt.close(fig)
