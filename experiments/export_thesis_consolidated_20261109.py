"""Export a canonical thesis from sealed evidence, without fitting or sampling."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'thesis';GENERATED=HERE/'generated'
sys.path.insert(0,str(ROOT))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m


PAPER=module('paper_report',ROOT/'docs/supervisor_meeting/2026-10-28_brief/build_report.py')
SUPPLEMENT=module('scenario_report',ROOT/'docs/supervisor_meeting/2026-11-09_brief/build_report.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def row(values):return ' & '.join(map(str,values))+r' \\'+'\n'


def longtable(text,caption,label,size='small'):
    start=text.index(r'\begin{tabularx}{\linewidth}{')+len(r'\begin{tabularx}{\linewidth}{')
    stop=text.index(r'\toprule',start)
    spec=text[start:stop].rstrip('}')
    # Restore the final @{} brace removed with the environment-spec delimiter.
    if spec.endswith('@{'):spec+='}'
    fixed=sum(map(float,re.findall(r'P\{([0-9.]+)cm\}',spec)))
    flex=spec.count('X')+spec.count('Y')
    plain=re.sub(r'P\{[^}]+\}','',spec)
    count=len(re.findall(r'P\{',spec))+flex+sum(plain.count(c) for c in 'lcr')
    if flex:
        width=(16.-fixed-8*(count-1)*.03514598)/flex
        assert width>0,(label,spec)
        spec=spec.replace('X',f'P{{{width:.3f}cm}}').replace('Y',f'P{{{width:.3f}cm}}')
    head_start=stop+len(r'\toprule');head_stop=text.index(r'\midrule',head_start)
    head=text[head_start:head_stop].replace(r'\textbf{Case}', r'\textbf{S}')
    body=text[head_stop+len(r'\midrule'):text.rindex(r'\bottomrule')]
    keep = r'\Needspace{210pt}'+'\n' if body.count(r'\\')<=7 else ''
    return (keep+r'\par\begingroup'+'\\'+size+r'\setstretch{1.04}\setlength{\parskip}{0pt}'+'\n'
        +r'\begin{longtable}{'+spec+'}\n'+r'\caption{'+caption+r'}\label{tab:'+label+r'}\\'+'\n'
        +r'\toprule'+head+r'\midrule\endfirsthead'+'\n'
        +r'\multicolumn{'+str(count)+r'}{l}{\small\textit{Bảng \thetable\ (tiếp)}}\\\toprule'
        +head+r'\midrule\endhead'+'\n'+r'\midrule\multicolumn{'+str(count)+r'}{r}{\small\textit{Còn tiếp ở trang sau}}\\\endfoot'+'\n'
        +r'\bottomrule\endlastfoot'+body+r'\end{longtable}\endgroup\par'+'\n')


TABLES={
 'privacy':(r'Privacy: Hit100 (\%) / MAE (m), cùng cohort và cửa sổ quan sát','main-privacy'),
 'utility':(r'Session Recall@5 (\%) theo bốn mục đích truy vấn','main-utility'),
 'cost':('Chi phí trung bình mỗi phiên 600 giây; KB = 1.000 byte','main-cost'),
 'comparison':('Chênh lệch macro Recall giữa GeoI-BQ và từng đối chứng','utility-differences'),
 'utility_ci':('Recall GeoI-BQ theo purpose và chênh lệch so với TransProtect*','purpose-ci'),
 'cache':('Macro Recall@5 theo phạm vi cache trên cùng output','cache'),
 'native':('Metrics gốc áp dụng được và các trường hợp N/A','native'),
 'coverage':('Coverage TEST và số lần thực thi thất bại của trial chính','coverage'),
 'privacy_ci':(r'CI95\% đầy đủ cho năm tác vụ định vị/tái dựng','privacy-ci'),
 'purpose_deltas':('Chênh lệch Recall theo purpose: GeoI-BQ trừ đối chứng','purpose-deltas'),
 'related':('Tổng hợp nghiên cứu: cơ chế, output, scenario, metrics và fidelity','related'),
 'ablation':('Ablation nội bộ ngày 14/10, tách khỏi đối chiếu paper','ablation'),
 'depth_sensitivity':('Truncation L30 so với L60 trên cùng Q/time','depth'),
 'sample_trace':('Một prefix nhiều bước cố định từ family cache-029','sample'),
}
EXTRA={
 's4':('Liên kết phiên S4 ở mức route-family proxy','s4'),
 's5':('S5: accuracy chọn cạnh tương lai trong hai ứng viên','s5'),
 's6':('S6 sau rẽ: chọn đích tương lai trong hai ứng viên','s6'),
 's7':('S7: độ phân biệt payload có điều kiện trong common wrapper','s7'),
 's8':(r'S8 với attacker fusion: Hit100 (\%) / MAE (m)','s8'),
 'fusion':('S8 protected-partner: V1 và vòng attacker fusion V2','fusion'),
 'selected':('Attacker được chọn cho GeoI-BQ trong đánh giá bổ sung','extra-selected'),
 's5_full':('Tất cả pha S5, xác suất dự đoán và CI','s5-full'),
 's6_full':('Tất cả pha S6, xác suất dự đoán và sai số đích','s6-full'),
 's8_full':('S8 V2: CI và chênh lệch ghép theo block','s8-full'),
 's8_cost':('Số tọa độ attacker quan sát tại cutoff S8, tính cả hai kênh','s8-observations'),
}


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    shutil.copyfile(PAPER.HERE/'figures/tradeoff.pdf',folder/'privacy_utility_cost.pdf')
    r=read(SUPPLEMENT.B1/'results.json');v=read(SUPPLEMENT.B2/'results.json')
    a={t['task']:t for t in r['tasks']};b={t['task']:t for t in v['tasks']}
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    labels=['Raw','REM-Z','TransProtect*','Semantic*','Fake queries*','GeoI-BQ']
    colors=['#9aa6ad']*5+['#087f7a'];y=np.arange(6)
    fig,ax=plt.subplots(figsize=(6.2,2.9))
    stats=[a['S6--post_turn--'+m]['metrics']['accuracy'] for m in PAPER.METHODS]
    means=np.array([100*s['mean'] for s in stats]);lo=means-np.array([100*s['ci95'][0] for s in stats]);hi=np.array([100*s['ci95'][1] for s in stats])-means
    ax.barh(y,means,color=colors,height=.6,xerr=[lo,hi],capsize=3,error_kw={'lw':.9})
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,105);ax.axvline(50,color='#6d777a',ls='--',lw=.9)
    ax.set_xlabel('S6 two-candidate accuracy (%) - lower means less inference')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout()
    fig.savefig(folder/'future_choice.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,2.9));width=.32
    before=[100*a['S8--protected_partner--'+m]['metrics']['Hit100']['mean'] for m in PAPER.METHODS]
    after=[100*b['S8--protected_partner--'+m]['metrics']['Hit100']['mean'] for m in PAPER.METHODS]
    ax.barh(y-width/2,before,height=width,label='V1 bank',color='#9aa6ad')
    ax.barh(y+width/2,after,height=width,label='V2: added fusion',color='#087f7a')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('S8 protected-partner Hit100 (%)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(folder/'joint_attack_fusion.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def generated():
    one,c1,bib,p1=PAPER.generated();two,c2,p2=SUPPLEMENT.generated();out={}
    for name,text in one.items():
        cap,label=TABLES[name[:-4]]
        out[name]=longtable(text,cap,label,'footnotesize' if name in ('related.tex','privacy_ci.tex','sample_trace.tex') else 'small')
    for name,text in two.items():
        cap,label=EXTRA[name[:-4]]
        out[name]=longtable(text,cap,label,'footnotesize' if name in ('s6_full.tex','s8_full.tex') else 'small')
    primary=read(PAPER.BENCH/'results.json')
    selected=[]
    for task in primary['attackers']:
        def wrap(s):return r'\path{'+s+'}'
        label=PAPER.LABELS[PAPER.METHODS.index(task['method'])]
        selected.append([label,task['scenario'],wrap(task['selected_attacker']['hit100']),wrap(task['selected_attacker']['mae'])])
    raw=PAPER.table('P{3.1cm}P{.7cm}XX',('Phương pháp','S','Attacker Hit100','Attacker MAE'),selected)
    out['main_selected.tex']=longtable(raw,'Toàn bộ attacker thực sự được chọn trong trial chính','main-selected','footnotesize')
    years={'semantic':2026,'fake':2026,'ev':2024,'cpcrok':2025,'dpfetc':2025,'ptppm':2025}
    lines=[]
    for line in bib.splitlines():
        key=re.search(r'\\bibitem\{([^}]+)\}',line)[1]
        if key in years: line=line.replace(r' \href{',f' {years[key]}. '+r'\href{',1)
        lines.append(line)
    bib='\n'.join(lines)+'\n'
    foundations=read(HERE/'references_foundations.json')
    for p in foundations:
        bib+=r'\bibitem{'+p['key']+'} '+p['entry']+'\n'
    files=[PAPER.HERE/'literature_evidence.json',HERE/'references_foundations.json',
        ROOT/'docs/research/2026-10-14_geoi_bq_mathematical_foundation.md',
        ROOT/'docs/reviews/2026-10-14_mathematical_adversarial_audit.md',
        ROOT/'docs/reviews/2026-10-14_mathematical_completion.json',
        ROOT/'docs/research/2026-11-09_remaining_scenarios_protocol.md',
        ROOT/'docs/research/2026-11-09_s8_fusion_protocol.md']
    source_hashes={**p1['sources_sha256'],**p2['sources_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in files}}
    provenance=dict(schema='consolidated-masters-thesis-sources-v1',date='2026-11-09',
        canonical_source='thesis/main.tex',main_chapters=5,main_release_pins_verified=108,
        main_privacy_cells=30,supplement_tasks_v1=66,joint_fusion_tasks_v2=24,
        generated_tables=len(out),models_fitted=False,defense_outputs_regenerated=False,
        main_comparison='three audited paper adaptations plus Raw/REM-Z; not official reproductions',
        sources_sha256=source_hashes,
        figures_sha256={p.name:sha(p) for p in sorted((HERE/'figures').glob('*.pdf'))})
    return out,c1+c2,bib,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if not args.check:figures()
    tables,constants,bib,sources=generated();files={GENERATED/n:t for n,t in tables.items()}
    files[GENERATED/'numbers.tex']=constants;files[GENERATED/'references.tex']=bib
    files[GENERATED/'sources.json']=json.dumps(sources,ensure_ascii=False,indent=2)+'\n'
    for path,content in files.items():
        if args.check:assert path.read_text()==content,str(path.relative_to(ROOT))
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    print('PASS:',len(tables),'tables;108 frozen main pins; no model fitting or protection sampling')


if __name__=='__main__':main()
