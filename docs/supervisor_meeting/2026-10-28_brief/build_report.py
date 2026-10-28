"""Build paper-style tables/figure from the verified matched comparison only."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import gzip
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from benchmark.finite_session_bounds import all_duration_ratio,posterior_mass_bound
from core.research_release import load_release

BENCH=ROOT/'artifacts/benchmarks/paper_comparison_20261028_v1'
METHODS=('raw','geo_i_Z_k1','transprotect_markov','semantic_poi','fake_queries','geoi_bq')
LABELS=('Raw GPS','Geo-I/REM Z (K1)','TransProtect*','Semantic correlation*','Fake-query insertion*',r'\textbf{GeoI-BQ v1}')
PURPOSES=('macro','nearest_distance','fastest_travel','within_radius','minimum_detour')
PLABELS=('Trung bình','Gần nhất','Nhanh nhất','Bán kính','Ít vòng đường')
SCENARIOS=('S1','S2','S3','S9','S10')

def read(p):return json.loads(Path(p).read_text())
def read_json_gz(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def n(x,d=3):return '--' if x is None else f'{float(x):.{d}f}'.replace('.','{,}')
def pct(x,d=3):return n(100*float(x),d)
def ci(x,scale=100,d=3):return '['+'; '.join(n(scale*v,d) for v in x)+']'
def esc(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s.replace('−','-').replace('–','--').replace('ε',r'$\varepsilon$').replace('δ',r'$\delta$').replace('Δ',r'$\Delta$').replace('ρ',r'$\rho$').replace('ψ',r'$\psi$')
def row(a):return ' & '.join(map(str,a))+r' \\'+'\n'
def table(spec,head,rows):
    return (r'\par\smallskip\noindent\begin{tabularx}{\linewidth}{@{}'+spec+r'@{}}\toprule'+'\n'
        +row([r'\textbf{'+x+'}' for x in head])+r'\midrule'+'\n'
        +''.join(row(x) for x in rows)+r'\bottomrule\end{tabularx}\par\smallskip'+'\n')


def figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    attacks={(x['method'],x['scenario']):x for x in result['attackers']}
    utility={x['method']:x for x in result['utility'] if x['cache']=='session'}
    hits=[100*attacks[m,'S3']['Hit100'] for m in METHODS]
    costs=[result['cost'][m]['reply_bytes']/1000 for m in METHODS]
    recalls=[100*utility[m]['metrics']['macro']['mean'] for m in METHODS]
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.75),sharey=True,gridspec_kw={'width_ratios':[1,1.25]})
    y=np.arange(len(METHODS));colors=['#9aa6ad']*5+['#087f7a']
    axes[0].barh(y,hits,color=colors,height=.55)
    axes[0].set_yticks(y,['Raw','Geo-I Z','TransProtect*','Semantic*','Fake queries*','GeoI-BQ']);axes[0].invert_yaxis()
    axes[0].set_xlim(0,117);axes[0].set_xlabel('S3 Hit100 (%) - lower is better')
    for i,value in enumerate(hits):axes[0].text(value+1.2,i,f'{value:.2f}',va='center',fontsize=7)
    axes[1].barh(y,costs,color=colors,height=.55);axes[1].set_xlim(0,146)
    axes[1].set_xlabel('Reply-index JSON (KB / session)')
    for i,(cost,recall) in enumerate(zip(costs,recalls)):
        axes[1].text(cost+2,i,f'{recall:.3f}%',va='center',fontsize=7)
    axes[0].set_title('(a) Empirical localization privacy',fontsize=9)
    axes[1].set_title('(b) Data cost; labels = Recall@5',fontsize=9)
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout(pad=.7);(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/tradeoff.pdf',metadata={'CreationDate':None,'ModDate':None,'Creator':'matched paper-comparison report'})
    plt.close(fig)


def generated():
    spec=load_release();r=read(BENCH/'results.json');audit=read(BENCH/'validation.json')
    assert audit['status']=='pass' and audit['results_sha256']==sha(BENCH/'results.json')
    assert audit['protocol_sha256']==sha(BENCH/'protocol.json')==r['protocol_sha256']
    a={(x['method'],x['scenario']):x for x in r['attackers']}
    u={(x['method'],x['cache']):x for x in r['utility']}
    out={}
    out['privacy.tex']=table('P{3.1cm}XXXXX',('Phương pháp',)+SCENARIOS,
        [[label]+[pct(a[m,s]['Hit100'])+' / '+n(a[m,s]['MAE_m'],1) for s in SCENARIOS] for m,label in zip(METHODS,LABELS)])
    out['utility.tex']=table('P{3.9cm}XXXXX',('Phương pháp',)+PLABELS,
        [[label]+[pct(u[m,'session']['metrics'][p]['mean']) for p in PURPOSES] for m,label in zip(METHODS,LABELS)])
    out['cost.tex']=table('P{3.6cm}XXXXX',('Phương pháp','Packet','Query điểm','Request KB','Reply KB','Sinh / tick ms'),
        [[label,n(r['cost'][m]['packets'],2),n(r['cost'][m]['requests'],2),n(r['cost'][m]['request_bytes']/1000),
          n(r['cost'][m]['reply_bytes']/1000),('<0{,}001' if r['cost'][m]['generation_ms']/31<.001 else n(r['cost'][m]['generation_ms']/31,3)) if r['cost'][m]['generation_ms'] is not None else '--'] for m,label in zip(METHODS,LABELS)])
    comp={x['control']:x['metrics'] for x in r['comparisons']}
    out['comparison.tex']=table('P{4.5cm}XXX',('Đối chứng',r'Recall (\%)','BQ trừ đối chứng (pp)',r'CI95\% (pp)'),
        [[label,pct(u[m,'session']['metrics']['macro']['mean']),pct(comp[m]['macro']['delta']),ci(comp[m]['macro']['paired95'])] for m,label in zip(METHODS[:-1],LABELS[:-1])])
    out['utility_ci.tex']=table('P{3.2cm}XXX',('Purpose',r'BQ Recall (\%)',r'CI95\% tuyệt đối',r'Δ so TransProtect* (pp), CI95\%'),
        [[label,pct(u['geoi_bq','session']['metrics'][p]['mean']),ci(u['geoi_bq','session']['metrics'][p]['absolute95']),
          pct(comp['transprotect_markov'][p]['delta'])+'; '+ci(comp['transprotect_markov'][p]['paired95'])] for p,label in zip(PURPOSES,PLABELS)])
    out['cache.tex']=table('P{4cm}XXX',('Phương pháp',r'Current (\%)',r'Epoch60 (\%)',r'Session (\%)'),
        [[label]+[pct(u[m,c]['metrics']['macro']['mean']) for c in ('current','known','session')] for m,label in zip(METHODS,LABELS)])
    out['native.tex']=table('P{3.7cm}XXXXX',('Phương pháp','EIE S3 (m)',r'$\Delta c$ (m)','ASR','DER','Path count'),
        [[label,n(a[m,'S3']['MAE_m'],1),n(r['native_metrics'][m]['Eq13']['mean'],2) if r['native_metrics'][m]['Eq13']['mean'] is not None else 'N/A','N/A','N/A','N/A'] for m,label in zip(METHODS,LABELS)])
    out['coverage.tex']=table('P{4cm}XXX',('Phương pháp','TEST thử','Hoàn tất','Thất bại'),
        [[label,r['coverage'][m]['attempted'],r['coverage'][m]['completed'],r['coverage'][m]['failed']] for m,label in zip(METHODS,LABELS)])
    out['privacy_ci.tex']=table('P{3.4cm}P{.7cm}XXXX',('Phương pháp','Case',r'Hit100\%',r'CI95\% Hit','MAE m',r'CI95\% MAE'),
        [[label,s,pct(a[m,s]['Hit100']),ci(a[m,s]['Hit100_CI95']),n(a[m,s]['MAE_m'],1),ci(a[m,s]['MAE_m_CI95'],1,1)] for m,label in zip(METHODS,LABELS) for s in SCENARIOS])
    out['purpose_deltas.tex']=table('P{3.5cm}P{2.4cm}XX',('Đối chứng','Purpose','BQ trừ control (pp)',r'CI95\% (pp)'),
        [[label,plabel,pct(comp[m][p]['delta']),ci(comp[m][p]['paired95'])] for m,label in zip(METHODS[:-1],LABELS[:-1]) for p,plabel in zip(PURPOSES,PLABELS)])
    lit=read(HERE/'literature_evidence.json');related=[];bib=[]
    for p in lit['papers']:
        coverage=[]
        if p['direct']:coverage.append('F: '+', '.join(p['direct']))
        if p['partial']:coverage.append('P: '+', '.join(p['partial']))
        if p.get('uncertain'):coverage.append('CX: '+', '.join(p['uncertain']))
        related.append([r'\textbf{'+esc(p['name'])+'} '+str(p['year'])+r'\cite{'+p['key']+'}'+r'\newline '+esc(p['mechanism']),
            esc(p['output']),r'\newline '.join(coverage) or 'Không áp dụng',esc(p['privacy']),esc(p['utility']),esc(p['implementation'])])
        bib.append(r'\bibitem{'+p['key']+'} '+esc(p['authors'])+'. '+esc(p['title'])+'. '+esc(p['venue'])
                   +r'. \href{'+p['url']+'}{DOI: '+p['doi']+'}.\n')
    out['related.tex']=table('P{2.75cm}P{2.15cm}P{1.6cm}P{2.4cm}P{2.5cm}Y',('Paper / cơ chế','Output','Scenario','Privacy','Utility / cost','Tái lập trong bản này'),related)
    for p in lit['foundations']:
        bib.append(r'\bibitem{'+p['key']+'} '+esc(p['authors'])+'. '+esc(p['title'])+'. '+esc(p['venue'])+r'. \href{'+p['url']+'}{Nguồn tác giả}.\n')
    old=read(ROOT/'artifacts/benchmarks/main_depth_20261014_v1/results.json')
    ou={(x['method'],x['cache']):x for x in old['utility']}
    oldlabels={'baseline':'Service-cover K5','finite_static_g1':'Initial-only','cache_explore_g1':'Local explore','cache_target_g0':r'Public-base $\Gamma=0$','cache_target_g1':'GeoI-BQ v1'}
    out['ablation.tex']=table('P{4cm}XXXXX',('Ablation14/10',)+PLABELS,
        [[label]+[pct(ou[m,'session']['metrics'][p]['mean']) for p in PURPOSES] for m,label in oldlabels.items()])
    ro=read(ROOT/'artifacts/benchmarks/main_depth_20261014_v1/release_readout.json')
    out['depth_sensitivity.tex']=table('P{3cm}XXX',('Purpose',r'L30 cùng Q (\%)','L60 trừ L30 (pp)',r'CI95\% (pp)'),
        [[label,pct(ro['same_Q_L30_counterfactual'][p]['L30_mean']),pct(ro['same_Q_L30_counterfactual'][p]['L60_minus_L30']),ci(ro['same_Q_L30_counterfactual'][p]['paired95'])] for p,label in zip(PURPOSES,PLABELS)])
    # Fixed first TEST family/first trip, chosen by identifier, never by a score.
    sample=read_json_gz(BENCH/'families/cache-029.json.gz')['trips'][0]
    original=read_json_gz(ROOT/'artifacts/benchmarks/main_depth_20261014_v1/families/cache-029.json.gz')['trips'][0]
    x0=sample['truth'][0]['xy'];sample_rows=[]
    for i in (0,1,2,3,6,9):
        point=sample['truth'][i];ledger=original['GeoI_ledger'][i]
        delta='('+n(point['xy'][0]-x0[0],0)+'; '+n(point['xy'][1]-x0[1],0)+')'
        pair=lambda xy:'('+n(xy[0],4)+'; '+n(xy[1],4)+')'
        branch={'fresh':'Tạo Z','reuse':'Giữ Z','public_clock_skip':'Không đọc'}[ledger['branch']]
        sample_rows.append([point['t'],delta,branch+' / '+str(ledger['spent_units']),pair(original['GeoI_anchors'][i]),
            pair(sample['events']['geoi_bq'][i]['coordinates'][0]),int(sample['masks']['geoi_bq'][i]['session'],16).bit_count()])
    out['sample_trace.tex']=table('P{.6cm}P{2.3cm}P{2.7cm}P{3.3cm}P{3.3cm}X',
        ('s',r'$\Delta$GPS (m)','Nhánh / units tích lũy',r'Z (lat; lon)','Q1 (lat; lon)','POI cache'),sample_rows)
    upper=all_duration_ratio(1,cutoff=11);assert upper<F(2437372,1000000)
    primary=u['geoi_bq','session']['metrics']['macro']
    constants=(r'\newcommand{\PrimaryRecall}{'+pct(primary['mean'])+'}\n'
        +r'\newcommand{\PrimaryCI}{'+ci(primary['absolute95'])+'}\n'
        +r'\newcommand{\RatioUpper}{2{,}437372}'+'\n'
        +r'\newcommand{\PosteriorUpper}{70{,}91}'+'\n')
    assert 100*posterior_mass_bound(F(1,2),upper)<F(7091,100)
    sources=[BENCH/'protocol.json',BENCH/'results.json',BENCH/'validation.json',BENCH/'resources.json',
        HERE/'literature_evidence.json',ROOT/'configs/geo_i_bq_v1.json',
        ROOT/'docs/research/2026-10-06_report_evidence.md',
        ROOT/'docs/research/2026-10-14_geoi_bq_mathematical_foundation.md',
        ROOT/'docs/reviews/2026-10-14_mathematical_adversarial_audit.md',
        ROOT/'artifacts/benchmarks/main_depth_20261014_v1/results.json',
        ROOT/'artifacts/benchmarks/main_depth_20261014_v1/release_readout.json',
        BENCH/'families/cache-029.json.gz',
        ROOT/'artifacts/benchmarks/main_depth_20261014_v1/families/cache-029.json.gz']
    provenance=dict(schema='paper-style-report-sources-v1',date='2026-10-28',main_release_pins_verified=len(spec['files_sha256']),
        main_methods=list(METHODS),main_privacy_cells=30,benchmark_sha256=sha(BENCH/'results.json'),
        own_model_regenerated=False,benchmark_fitted_by_report_builder=False,old_ablation_separate=True,
        literature_window=lit['window'],sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        figure_sha256=sha(HERE/'figures/tradeoff.pdf') if (HERE/'figures/tradeoff.pdf').exists() else None)
    return out,constants,''.join(bib),provenance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    if not a.check:figure(read(BENCH/'results.json'))
    tables,constants,bib,sources=generated()
    files={Path('tables')/n:s for n,s in tables.items()}
    files[Path('report_constants.tex')]=constants;files[Path('references.tex')]=bib
    files[Path('report_sources.json')]=json.dumps(sources,ensure_ascii=False,indent=2)+'\n'
    for name,content in files.items():
        path=HERE/name
        if a.check:assert path.exists() and path.read_text()==content,str(name)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    print('PASS: verified paper comparison;'+str(len(tables))+' tables;108 main pins; no model/attacker fitting')


if __name__=='__main__':main()
