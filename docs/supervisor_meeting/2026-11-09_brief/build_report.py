"""Export the verified S4--S8 supplement; never fit or generate protection."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from core.research_release import load_release

B1=ROOT/'artifacts/benchmarks/remaining_scenarios_20261109_v1'
B2=ROOT/'artifacts/benchmarks/s8_fusion_20261109_v2'
OLD=ROOT/'artifacts/benchmarks/paper_comparison_20261028_v1'
METHODS=('raw','geo_i_Z_k1','transprotect_markov','semantic_poi','fake_queries','geoi_bq')
LABELS=('Raw GPS','REM-Z K1','TransProtect*','Semantic correlation*','Fake-query insertion*',r'\textbf{GeoI-BQ v1}')
VIEWS=('target_only','protected_partner','raw_partner','unrelated_partner')
VLABELS=('Target','+ protected','+ raw','+ unrelated')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x,d=2):return f'{float(x):.{d}f}'.replace('.','{,}')
def pct(x,d=2):return num(100*x,d)
def metric(s,scale=100,d=2):return num(scale*s['mean'],d)+' ['+'; '.join(num(scale*v,d) for v in s['ci95'])+']'
def multiline(s,scale=100,d=2):return num(scale*s['mean'],d)+r'\newline ['+'; '.join(num(scale*v,d) for v in s['ci95'])+']'
def row(values):return ' & '.join(map(str,values))+r' \\'+'\n'
def table(columns,head,rows):return (r'\par\smallskip\noindent\begin{tabularx}{\linewidth}{@{}'+columns+r'@{}}\toprule'+'\n'
    +row([r'\textbf{'+v+'}' for v in head])+r'\midrule'+'\n'+''.join(row(v) for v in rows)+r'\bottomrule\end{tabularx}\par\smallskip'+'\n')


def generated():
    spec=load_release();r=read(B1/'results.json');v=read(B1/'validation.json')
    q=read(B2/'results.json');w=read(B2/'validation.json')
    assert v['status']==w['status']=='pass' and v['result_sha256']==sha(B1/'results.json')
    assert w['results_sha256']==sha(B2/'results.json') and r['full_coverage'] and q['full_coverage']
    assert r['old_paper_comparison_sha256']==sha(OLD/'results.json')
    a={t['task']:t for t in r['tasks']};b={t['task']:t for t in q['tasks']};out={}
    out['s4.tex']=table('P{3cm}XXX',('Phương pháp',r'BA\% [CI95\%]',r'AUC theo block [CI95\%]','Macro-F1'),
        [[label,metric(a['S4--'+m]['metrics']['balanced_accuracy']),
          metric(a['S4--'+m]['metrics']['group_mean_roc_auc'],1,3),num(a['S4--'+m]['metrics']['macro_f1'],3)] for m,label in zip(METHODS,LABELS)])
    out['s5.tex']=table('P{3.2cm}XX',('Phương pháp',r'Trước fork: Acc\% [CI95\%]',r'Cửa sổ rẽ: Acc\% [CI95\%]'),
        [[label]+[metric(a['S5--'+phase+'--'+m]['metrics']['accuracy']) for phase in ('before_fork','turn_window')] for m,label in zip(METHODS,LABELS)])
    out['s6.tex']=table('P{3.1cm}XXX',('Phương pháp',r'Acc\% [CI95\%]',r'Hit100\% [CI95\%]',r'MAE m [CI95\%]'),
        [[label,metric(a['S6--post_turn--'+m]['metrics']['accuracy']),metric(a['S6--post_turn--'+m]['destination_Hit100']),
          metric(a['S6--post_turn--'+m]['destination_MAE_m'],1,1)] for m,label in zip(METHODS,LABELS)])
    out['s7.tex']=table('P{3.2cm}XXXX',('Phương pháp','Wire bằng nhau',r'Purpose\%',r'Category\%',r'Joint\%'),
        [[label,pct(a['S7--'+m]['equal_wire_rate'])]+[pct(a['S7--'+m]['metrics'][key]['mean'])
            for key in ('purpose_vulnerability','category_vulnerability','joint_vulnerability')] for m,label in zip(METHODS,LABELS)])
    out['s8.tex']=table('P{3.2cm}XXXX',('Phương pháp',)+VLABELS,
        [[label]+[pct(b['S8--'+view+'--'+m]['metrics']['Hit100']['mean'])+' / '+num(b['S8--'+view+'--'+m]['metrics']['MAE_m']['mean'],1)
          for view in VIEWS] for m,label in zip(METHODS,LABELS)])
    out['fusion.tex']=table('P{3.1cm}XXXX',('Phương pháp',r'V1 Hit\%',r'V2 Hit\%','V1 MAE m','V2 MAE m'),
        [[label,pct(a['S8--protected_partner--'+m]['metrics']['Hit100']['mean']),pct(b['S8--protected_partner--'+m]['metrics']['Hit100']['mean']),
          num(a['S8--protected_partner--'+m]['metrics']['MAE_m']['mean'],1),num(b['S8--protected_partner--'+m]['metrics']['MAE_m']['mean'],1)] for m,label in zip(METHODS,LABELS)])
    attack_rows=[]
    def escaped(text):return text.replace('_',r'\_').replace('@',r'\, @\, ')
    for key,label in [('S4--geoi_bq','S4 association'),('S5--turn_window--geoi_bq','S5 turn'),('S6--post_turn--geoi_bq','S6 post-turn')]:
        attack_rows.append([label,escaped(a[key]['selected'])])
    for view,label in zip(VIEWS,VLABELS):
        t=b['S8--'+view+'--geoi_bq'];attack_rows.append(['S8 '+label,'Hit: '+escaped(t['selected']['hit100'])+'; MAE: '+escaped(t['selected']['mae'])])
    out['selected.tex']=table('P{3cm}X',('Tác vụ GeoI-BQ','Attacker đã chọn trên SELECTION'),attack_rows)
    for scenario in ('S5','S6'):
        phases=('before_fork','turn_window') if scenario=='S5' else ('before_fork','turn_window','post_turn')
        plabels={'before_fork':'Trước fork','turn_window':'Cửa sổ rẽ','post_turn':'Sau rẽ'}
        rows=[]
        for m,label in zip(METHODS,LABELS):
            for phase in phases:
                t=a[scenario+'--'+phase+'--'+m]
                values=[label,plabels[phase],metric(t['metrics']['accuracy']),num(t['metrics']['brier']['mean'],3),num(t['metrics']['log_loss']['mean'],3)]
                if scenario=='S6':values.append(metric(t['destination_MAE_m'],1,1))
                rows.append(values)
        out[scenario.lower()+'_full.tex']=table('P{3cm}P{2.1cm}X P{1cm}P{1cm}'+('X' if scenario=='S6' else ''),
            ('Phương pháp','Pha',r'Acc\% [CI95\%]','Brier','Log loss')+((r'MAE m [CI95\%]',) if scenario=='S6' else ()),rows)
    rows=[]
    for m,label in zip(METHODS,LABELS):
        for view,vlabel in zip(VIEWS,VLABELS):
            t=b['S8--'+view+'--'+m]
            rows.append([label,vlabel,multiline(t['metrics']['Hit100']),multiline(t['metrics']['MAE_m'],1,1),
                multiline(t['paired_vs_target_only']['Hit100']),multiline(t['paired_vs_target_only']['MAE_m'],1,1)])
    out['s8_full.tex']=table('P{2.8cm}P{1.5cm}XXXX',('Phương pháp','View',r'Hit\% [CI]',r'MAE m [CI]',r'ΔHit pp [CI]',r'ΔMAE m [CI]'),rows)
    out['s8_cost.tex']=table('P{3.2cm}XXXX',('Phương pháp',)+VLABELS,
        [[label]+[num(b['S8--'+view+'--'+m]['mean_observed_coordinates'],1) for view in VIEWS] for m,label in zip(METHODS,LABELS)])
    constants=r'\newcommand{\PrimaryBA}{'+pct(a['S4--geoi_bq']['metrics']['balanced_accuracy']['mean'])+'}\n'
    constants+=r'\newcommand{\PrimaryFuture}{'+pct(a['S6--post_turn--geoi_bq']['metrics']['accuracy']['mean'])+'}\n'
    sources=[B1/'results.json',B1/'validation.json',B1/'protocol.json',B2/'results.json',B2/'validation.json',B2/'protocol.json',OLD/'results.json',ROOT/'configs/geo_i_bq_v1.json']
    receipt=dict(schema='remaining-scenarios-report-sources-v1',date='2026-11-09',
        main_pins_verified=len(spec['files_sha256']),tasks_v1=66,tasks_s8_v2=24,table_count=len(out),
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        source_outputs_regenerated=False,post_v1_attacker_diagnostic_disclosed=True)
    return out,constants,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    tables,constants,sources=generated()
    files={Path('tables')/k:v for k,v in tables.items()}
    files[Path('report_constants.tex')]=constants;files[Path('report_sources.json')]=json.dumps(sources,ensure_ascii=False,indent=2)+'\n'
    for name,value in files.items():
        path=HERE/name
        if args.check:assert path.read_text()==value,name
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    print('PASS: derived',len(tables),'tables;108 main pins; no fitting or defense sampling')


if __name__=='__main__':main()
