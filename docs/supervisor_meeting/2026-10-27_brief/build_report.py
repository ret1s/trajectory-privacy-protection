"""Export saved evidence for a review report; never fit, sample or rescore."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.finite_session_bounds import all_duration_ratio, posterior_mass_bound

BASE = 'artifacts/benchmarks/'
MAIN = BASE + 'main_depth_20261014_v1/'
BARE = BASE + 'geo_i_anchor_comparator_20261014_v1/'
PILOT = BASE + 'continual_geoi_20261017_v1/'
LABELS = {
    'geo_i_Z_k1': r'Geo-I Z (K1)', 'baseline': r'Service-cover K5',
    'finite_static_g1': r'Initial-only K5', 'cache_explore_g1': r'Local explore',
    'cache_target_g0': r'Public-base $\Gamma=0$',
    'cache_target_g1': r'\textbf{GeoI-BQ $\Gamma=1$}',
}
METHODS = tuple(LABELS)
PURPOSES = ('macro', 'nearest_distance', 'fastest_travel', 'within_radius', 'minimum_detour')
PURPOSE_LABELS = ('Trung bình', 'Gần nhất', 'Nhanh nhất', 'Bán kính', 'Ít vòng đường')
SCENARIOS = ('S1', 'S2', 'S3', 'S9', 'S10')
SOURCES = [
    'configs/geo_i_bq_v1.json', 'thesis/WRITING_BRIEF.md',
    'docs/research/2026-10-14_geoi_bq_mathematical_foundation.md',
    'docs/reviews/2026-10-14_mathematical_adversarial_audit.md',
    'docs/reviews/2026-10-14_mathematical_completion.json',
    MAIN+'results.json', MAIN+'protocol.json', MAIN+'release_readout.json',
    MAIN+'independent_validation.json', BARE+'results.json',
    BASE+'public_depth_design_20261014_v1/certificate.json',
    PILOT+'results.json', PILOT+'protocol.json', PILOT+'validation.json',
    BASE+'paper_benchmark/results.json', BASE+'native_metrics_20261005/readout.json',
    BASE+'native_metrics_20261005/verification.json',
    'benchmark/finite_query_kernel.py', 'benchmark/anchor_belief.py',
]


def read(name): return json.loads((ROOT/name).read_text())
def sha(name): return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def num(x, digits=3): return f'{float(x):.{digits}f}'.replace('.', '{,}')
def pct(x, digits=3): return num(100*float(x), digits)
def ci(v): return '[' + '; '.join(pct(x) for x in v) + ']'
def row(values): return ' & '.join(map(str, values)) + r' \\' + '\n'
def table(spec, headers, rows):
    return (r'\par\smallskip\noindent\begin{tabularx}{\linewidth}{@{}'+spec+r'@{}}\toprule'+'\n'
            +row([r'\textbf{'+s+'}' for s in headers])+r'\midrule'+'\n'
            +''.join(row(r) for r in rows)+r'\bottomrule\end{tabularx}\par\smallskip'+'\n')


def generated():
    manifest=read('configs/geo_i_bq_v1.json')
    for name,digest in manifest['files_sha256'].items():
        if sha(name)!=digest: raise ValueError('Frozen reference changed: '+name)
    r, v, bare=read(MAIN+'results.json'),read(MAIN+'release_readout.json'),read(BARE+'results.json')
    audit=read(MAIN+'independent_validation.json')
    assert audit['status']=='pass' and audit['results_sha256']==sha(MAIN+'results.json')
    assert audit['protocol_sha256']==sha(MAIN+'protocol.json')==r['protocol_sha256']
    assert v['results_sha256']==sha(MAIN+'results.json') and r['promotion']['utility_gate']
    pilot,pa=read(PILOT+'results.json'),read(PILOT+'validation.json')
    assert pa['status']=='pass' and pa['results_sha256']==sha(PILOT+'results.json')
    assert pa['protocol_sha256']==sha(PILOT+'protocol.json')==pilot['protocol_sha256']
    utility={(x['method'],x['cache']):x for x in r['utility']}
    attacks={(x['method'],x['scenario']):x for x in r['attackers']+bare['attackers']}
    def mean(m,p,cache='session'):
        return bare['metrics'][p]['bare_mean'] if m=='geo_i_Z_k1' else utility[m,cache]['metrics'][p]['mean']
    out={}
    out['utility.tex']=table('P{4.2cm}XXXXX',('Phương án',)+PURPOSE_LABELS,
        [[LABELS[m]]+[pct(mean(m,p)) for p in PURPOSES] for m in METHODS])
    out['privacy.tex']=table('P{3.6cm}XXXXX',('Phương án',)+SCENARIOS,
        [[LABELS[m]]+[pct(attacks[m,s]['Hit100'])+' / '+num(attacks[m,s]['MAE_m'],1) for s in SCENARIOS] for m in METHODS])
    costs={**r['cost'],'geo_i_Z_k1':bare['cost']}
    out['cost.tex']=table('P{4.4cm}XXX',('Phương án','Số query điểm','Request (KB)','Reply-index (KB)'),
        [[LABELS[m],num(costs[m]['requests'],0),num(costs[m]['request_bytes']/1000),num(costs[m]['reply_bytes']/1000)] for m in METHODS])
    comparisons={x['control']:x['metrics'] for x in v['comparisons']}
    comparison_rows=[]
    for m in METHODS[:-1]:
        b=bare['metrics']['macro'] if m=='geo_i_Z_k1' else comparisons[m]['macro']
        comparison_rows.append([LABELS[m],pct(b['primary_minus_bare'] if m=='geo_i_Z_k1' else b['delta']),ci(b['paired95'])])
    out['comparisons.tex']=table('P{6cm}XX',('Đối chứng','BQ trừ đối chứng (pp)',r'CI ghép 95\% (pp)'),comparison_rows)
    primary=utility['cache_target_g1','session']['metrics']
    out['gate.tex']=table('P{3.1cm}XXXX',('Purpose',r'BQ (\%)',r'CI tuyệt đối 95\%','Δ so K5 (pp)','CI Δ (pp)'),
        [[label,pct(primary[p]['mean']),ci(v['primary_absolute95'][p]),pct(primary[p]['paired_delta']),ci(primary[p]['paired95'])] for label,p in zip(PURPOSE_LABELS,PURPOSES)])
    out['cache.tex']=table('P{4.3cm}XXX',('Phương án',r'Current (\%)',r'Epoch 60 s (\%)',r'Session (\%)'),
        [[LABELS[m]]+[pct(mean(m,'macro',c)) for c in ('current','known','session')] for m in METHODS[1:]])
    out['coverage.tex']=table('P{4.3cm}XXX',('Phương án','Cache đầu','Cache cuối',r'Recall 200-600 s (\%)'),
        [[LABELS[m],num(v['POI_coverage'][m]['initial'],2),num(v['POI_coverage'][m]['final'],2),pct(v['periods'][2]['means'][m])] for m in METHODS[1:]])
    depth=read(BASE+'public_depth_design_20261014_v1/certificate.json')
    out['depth_design.tex']=table('XXX',('L / loại / Q',r'Mean miền công khai (\%)',r'Goal đạt floor 99\%'),
        [[x['L'],pct(F(x['exact_best_public_mean_recall'])),len(x['qualified_initial_goals'])] for x in depth['rows']])
    out['depth_sensitivity.tex']=table('P{3.3cm}XXX',('Purpose',r'L30 cùng Q (\%)','L60 trừ L30 (pp)','CI Δ (pp)'),
        [[label,pct(v['same_Q_L30_counterfactual'][p]['L30_mean']),pct(v['same_Q_L30_counterfactual'][p]['L60_minus_L30']),ci(v['same_Q_L30_counterfactual'][p]['paired95'])] for label,p in zip(PURPOSE_LABELS,PURPOSES)])
    out['local_gps.tex']=table('P{4.3cm}XXXXX',('Phương án',)+PURPOSE_LABELS,
        [[LABELS[m]]+[pct(v['stale_local_GPS_60s'][m][p]['mean']) for p in PURPOSES] for m in ('baseline','cache_target_g0','cache_target_g1')])
    def attacker(a):
        known={'centroid':'Tâm Q','median':'Trung vị Q','shadow_tree':'ExtraTrees64','shadow_knn1':'Shadow kNN1',
            'shadow_prior_median':'Prior median','geometry_nearest_prior_median':'Geometry + prior',
            'geometry_nearest_slot0_ols3_boundary':'Geometry + OLS3 (slot 0)'}
        return known.get(a,a.replace('_',r'\_'))
    out['attack_selection.tex']=table('P{3.7cm}P{1cm}YY',('Phương án','Case','Chọn theo Hit100','Chọn theo MAE'),
        [[LABELS[m],s,attacker(attacks[m,s]['selected_attacker']['hit100']),attacker(attacks[m,s]['selected_attacker']['mae'])]
         for m in ('geo_i_Z_k1','cache_target_g1') for s in SCENARIOS])
    sample=v['training_sample_walkthrough']
    out['sample.tex']=table('P{1.1cm}P{2.5cm}XXXX',('t (s)','GPS bảo vệ / nhánh','Chọn đoạn','ε Q mới','Q đã dành','POI cache'),
        [[x['t'],{'fresh':'Đọc / tạo mới','reuse':'Đọc / giữ Z','public_clock_skip':'Không đọc'}[x['GeoI_branch']],
          'Có' if x['selection'] else 'Không',num(x['epsilon_Q'],4) if x['selection'] else '--',num(x['budget_reserved'],4),x['cached_POIs']]
         for x in sample if x['t'] in (0,20,40,60,120,300,600)])
    history=read(BASE+'paper_benchmark/results.json')
    hi={(x['method'],x['scenario']):x for x in history['summary'] if x['k']==5}
    hm=[('unprotected','GPS thật'),('dls_graph_adaptation','DLS*'),('transprotect_adaptation','TransProtect*'),
        ('semantic_correlation_local_adaptation','Semantic*'),('br_private','BR--Geo-I lịch sử')]
    out['historical.tex']=table('P{4cm}XXX',('Phương pháp','S1','S2','S3'),
        [[label]+[pct(hi[m,s]['location_hit_100m'],2)+' / '+num(hi[m,s]['location_mae_m'],0)+' / '+pct(hi[m,s]['poi_recall_at_k'],2) for s in SCENARIOS[:3]] for m,label in hm])
    native=read(BASE+'native_metrics_20261005/readout.json')
    assert native['source_sha256'][BASE+'paper_benchmark/results.json']==sha(BASE+'paper_benchmark/results.json')
    ni={x['method']:x for x in native['summaries'] if x['cohort']=='paper_v2_test' and x['scenario']=='S3' and x['k']==5}
    def native_value(m,key):
        metric=ni[m]['metrics'][key]
        return '--' if metric['value'] is None else num(metric['value'],2)
    hm.append(('anotherme_adaptation','AnotherMe* (offline)'))
    out['native_metrics.tex']=table('P{4.4cm}XXXX',('Phương pháp','Hoàn tất / thử','EIE (m)','Dịch chuyển (m)','Sinh / event (ms)'),
        [[label,f"{ni[m]['completed_rows']} / {ni[m]['attempted_rows']}",native_value(m,'eie_point_estimate_m'),native_value(m,'single_release_displacement_m'),native_value(m,'generation_ms_per_event')] for m,label in hm])
    pu={(x['method'],x['cache'],x['window']):x for x in pilot['utility']}
    pl={'capped_g1':r'Giữ cap, $\Gamma=1$','continual_g1':r'Continual, $\Gamma=1$','continual_g0':r'Continual, $\Gamma=0$'}
    out['pilot_utility.tex']=table('P{4.2cm}XXX',('Pilot',r'0-600 s (\%)',r'0-1800 s (\%)',r'720-1800 s (\%)'),
        [[label]+[pct(pu[m,'session',w]['metrics']['macro']['mean']) for w in ('native_10min','full_30min','synthetic_after_12min')] for m,label in pl.items()])
    out['pilot_purposes.tex']=table('P{4.2cm}XXXXX',('Pilot, cache phiên',)+PURPOSE_LABELS,
        [[label]+[pct(pu[m,'session','full_30min']['metrics'][p]['mean']) for p in PURPOSES] for m,label in pl.items()])
    pi={(x['method'],x['scenario']):x for x in pilot['attackers']}
    out['pilot_privacy.tex']=table('P{3.6cm}XXXXX',('Pilot',)+SCENARIOS,
        [[label]+[pct(pi[m,s]['Hit100'])+' / '+num(pi[m,s]['MAE_m'],1) for s in SCENARIOS] for m,label in pl.items()])
    out['pilot_gate.tex']=table('P{4cm}XX',('Pilot, toàn 30 phút','Δ so cap (pp)',r'CI 95\% (pp)'),
        [[label,pct(pu[m,'session','full_30min']['metrics']['macro']['paired_delta']),ci(pu[m,'session','full_30min']['metrics']['macro']['paired95'])] for m,label in pl.items()])
    out['pilot_cost.tex']=table('P{4cm}XXX',('Pilot','Query điểm','Request KB','Reply-index KB'),
        [[label,num(pilot['cost'][m]['requests'],0),num(pilot['cost'][m]['request_bytes']/1000,2),num(pilot['cost'][m]['reply_bytes']/1000,2)] for m,label in pl.items()])
    upper=all_duration_ratio(1,cutoff=11)
    assert upper<F(2437372,1000000)
    constants=(r'\newcommand{\RatioUpper}{2{,}437372}'+'\n'
        +r'\newcommand{\PosteriorUpper}{70{,}91}'+'\n'
        +r'\newcommand{\PrimaryRecall}{'+pct(mean('cache_target_g1','macro'))+'}\n')
    assert float(100*posterior_mass_bound(F(1,2),upper))<70.91
    # Numerical illustration only; no private trajectory or sampler is used.
    pts=(0,100,300); prior=(.5,.3,.2)
    likelihood=[math.exp(-.005*abs(s-100))/sum(math.exp(-.005*abs(s-vv)) for vv in pts) for s in pts]
    product=[a*b for a,b in zip(prior,likelihood)]; belief=[x/sum(product) for x in product]
    out['belief_example.tex']=table('XXXX',('Vùng / đại diện','Prior','Likelihood Z=100','Belief sau'),
        [[f'{label} / {s} m',num(p,2),num(l,4),pct(b,2)+r'\%'] for label,s,p,l,b in zip('ABC',pts,prior,likelihood,belief)])
    provenance=dict(schema='geoi-bq-review-report-v1',prepared_date='2026-10-27',
        main_release_pins_verified=len(manifest['files_sha256']),main_release_changed=False,
        benchmark_rescored=False,models_fitted=False,new_private_samples=False,
        sources_sha256={s:sha(s) for s in SOURCES},
        main_methods=list(METHODS),main_privacy_cells=len(METHODS)*len(SCENARIOS),
        main_utility_purposes=list(PURPOSES),main_cache_views=['current','known','session'],
        historical_results_separate=True,continual_pilot_separate=True,
        all_duration_ratio_upper_exact=str(upper),illustrative_belief=belief,
        derived_tables=sorted(out))
    return out,constants,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();tables,constants,provenance=generated()
    files={Path('tables')/n:t for n,t in tables.items()}
    files[Path('report_constants.tex')]=constants
    files[Path('report_sources.json')]=json.dumps(provenance,ensure_ascii=False,indent=2)+'\n'
    for name,value in files.items():
        path=HERE/name
        if args.check:
            assert path.exists() and path.read_text()==value,'Regenerate '+str(name)
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    print(f'PASS: {len(tables)} saved-evidence tables; {provenance["main_release_pins_verified"]} frozen pins; no rescoring')


if __name__=='__main__':main()
