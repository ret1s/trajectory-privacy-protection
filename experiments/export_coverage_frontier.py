"""Verified exact tables and resource-aware development frontier for the thesis."""
from collections import defaultdict

import numpy as np

from evaluation.retrieval_frontier import pareto_ids
from experiments.run_coverage_frontier import OUTPUT,METHODS,BUDGETS,DEPTHS
from experiments.run_service_cover import read,write,sha,ROOT

LABELS={'geometric':'BR-làn','mean_greedy':'Phủ trung bình',
        'mean_exchange':'Phủ + thay điểm','capped_exchange':'Phủ cân bằng'}


def build_readout():
    rows=[];deltas=[];cases=[];selection={};pins={}
    timing=read(OUTPUT/'timing.json')
    assert len(timing['rows'])==36 and all(r['public_matches'] for r in timing['rows'])
    assert timing['source_sha256']==sha(ROOT/'experiments/profile_coverage_frontier.py')
    assert timing['amendment_sha256']==sha(ROOT/'thesis/notes/coverage_frontier_timing_amendment.md')
    for budget in BUDGETS:
        directory=OUTPUT/f'b{budget:.2f}'
        receipt=read(directory/'verification.json');assert receipt['verified']
        assert receipt['verifier_sha256']==sha(ROOT/'experiments/verify_coverage_frontier.py')
        for phase,h in receipt['phase_sha256'].items(): assert sha(directory/f'{phase}.json')==h
        assert sha(directory/'selection.json')==receipt['selection_sha256']
        p=read(directory/'development.json');val=read(directory/'validation.json')
        chosen=read(directory/'selection.json')
        selection[str(budget)]=chosen['method_selection_by_depth']
        pins[str(budget)]={'development_sha256':sha(directory/'development.json'),
                          'verification_sha256':sha(directory/'verification.json')}
        for method in METHODS:
            group=[s for s in p['summaries'] if s['method']==method];assert len(group)==9
            vgroup=[s for s in val['summaries'] if s['method']==method]
            profile=next(s for s in timing['summaries'] if (s['budget'],s['method'])==(budget,method))
            for depth in map(str,DEPTHS):
                utility=[s['utility'][depth] for s in group]
                row=dict(id=f'{budget:.2f}/{method}/{depth}',budget=budget,method=method,
                    label=LABELS[method],depth=int(depth),k=5,cases=9,families=4,
                    recall=float(np.mean([u['poi_recall_at_5'] for u in utility])),
                    min_case_recall=min(u['poi_recall_at_5'] for u in utility),
                    complete=float(np.mean([u['poi_complete_rate'] for u in utility])),
                    selected_hit100=float(np.mean([s['selected_hit100'] for s in group])),
                    selected_mae_m=float(np.mean([s['selected_mae_m'] for s in group])),
                    envelope_hit100=float(np.mean([s['envelope_hit100'] for s in group])),
                    envelope_mae_m=float(np.mean([s['envelope_mae_m'] for s in group])),
                    requests_per_event=float(np.mean([u['requests_per_event'] for u in utility])),
                    reply_items=float(np.mean([u['reply_items_per_event'] for u in utility])),
                    response_id_bytes=float(np.mean([u['response_id_bytes_per_event'] for u in utility])),
                    profile_step_mean_ms=profile['step_mean_ms'],profile_step_p95_ms=profile['step_p95_ms'],
                    validation_min_recall=min(s['utility'][depth]['poi_recall_at_5'] for s in vgroup),
                    validation_feasible=round(min(s['utility'][depth]['poi_recall_at_5'] for s in vgroup),12)>=.9,
                    development_feasible=round(min(u['poi_recall_at_5'] for u in utility),12)>=.9,
                    categories={c:float(np.mean([u['by_category'][c] for u in utility if c in u['by_category']]))
                                for c in sorted({c for u in utility for c in u['by_category']})})
                rows.append(row)
                for s in group:
                    cases.append(dict(budget=budget,method=method,depth=int(depth),case_id=s['case_id'],
                        recall=s['utility'][depth]['poi_recall_at_5'],categories=s['utility'][depth]['by_category'],
                        selected_hit100=s['selected_hit100'],envelope_hit100=s['envelope_hit100']))
        for family in sorted({r['family_id'] for r in p['rows']}):
            for depth in map(str,DEPTHS):
                means={}
                for method in METHODS:
                    g=[r for r in p['rows'] if (r['family_id'],r['method'])==(family,method)]
                    assert len(g)==27
                    # Equal case/replicate weights, not independent events.
                    means[method]={'recall':float(np.mean([r['utility'][depth]['poi_recall_at_5'] for r in g])),
                        'hit':float(np.mean([np.mean(np.asarray(r['errors_by_attack'][chosen['attackers'][method+'/'+r['case_id']]['hit']])<=100) for r in g]))}
                for method in METHODS:
                    if method=='mean_greedy':continue
                    deltas.append(dict(budget=budget,depth=int(depth),family=family,method=method,
                        recall_delta=means[method]['recall']-means['mean_greedy']['recall'],
                        selected_hit_delta=means[method]['hit']-means['mean_greedy']['hit']))
    # Do not let a weaker formal budget dominate a stronger one just because
    # this finite attack bank happens to fail: compare within each B only.
    frontier=[];timed=[]
    for budget in BUDGETS:
        group=[r for r in rows if r['budget']==budget]
        frontier+=pareto_ids(group,[('recall',-1),('selected_hit100',1),('response_id_bytes',1)])
        timed+=pareto_ids(group,[('recall',-1),('selected_hit100',1),('response_id_bytes',1),('profile_step_mean_ms',1)])
    return dict(schema='coverage-frontier-readout-v1',scope='reused development; no SOTA or confirmation claim',
        definitions={'recall':'equal-case mean of per-run eligible category-event Recall@5',
            'gate':'minimum case Recall>=.90; researcher-defined, not conference standard',
            'selected':'per-case attackers chosen separately for Hit and MAE on validation only',
            'envelope':'per-case best attack within tested bank, then case mean; exploratory only',
            'bytes':'JSON ID lists, all six categories per event; not actual network bytes',
            'timing':'separate 36-run serial local-host profile, three validation records',
            'frontier':'within each B only; exploratory finite-grid non-dominance; no statistical superiority'},
        sources=pins,timing_sha256=sha(OUTPUT/'timing.json'),rows=rows,cases=cases,
        family_deltas=deltas,selection=selection,frontier_ids=frontier,timing_frontier_ids=timed)


def latex_table(caption,columns,header,rows,label):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
        '\\caption{'+caption+'}\\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',
        header+r'\\\midrule',*[r' & '.join(row)+r'\\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])


def export():
    readout=build_readout();write(OUTPUT/'readout.json',readout)
    tables=[]
    for depth in DEPTHS:
        table=[]
        for r in readout['rows']:
            if r['depth']!=depth:continue
            table.append([f'{r["budget"]:.2f}',r['label'],f'{100*r["recall"]:.2f}',
                f'{100*r["min_case_recall"]:.2f}',f'{100*r["selected_hit100"]:.2f}',
                f'{100*r["envelope_hit100"]:.2f}',f'{r["response_id_bytes"]:.0f}'])
        tables.append(latex_table(f'Biên thực nghiệm với phản hồi top-{depth}; đáp án tham chiếu luôn top-5. '
            'Recall và Hit tính theo phần trăm; byte là danh sách ID mỗi sự kiện. '
            'Hit chọn trên validation khác cực trị thăm dò.',
            'rlrrrrr',r'$B$ & Bộ chọn & Recall & Ca thấp & Hit chọn & Hit cực trị & Byte',table,
            f'tab:coverage-frontier-{depth}'))
    timing=[]
    for r in readout['rows']:
        if r['depth']==5:
            timing.append([f'{r["budget"]:.2f}',r['label'],f'{r["selected_mae_m"]:.1f}',
                f'{r["envelope_mae_m"]:.1f}',f'{r["profile_step_mean_ms"]:.1f}',f'{r["profile_step_p95_ms"]:.1f}'])
    tables.append(latex_table('Sai số suy luận (m) và thời gian sinh (ms). Thời gian đo tuần tự riêng trên '
        'ba bản ghi validation, không phải tốc độ thiết bị di động; MAE không đổi khi chỉ tăng độ sâu phản hồi.',
        'rlrrrr',r'$B$ & Bộ chọn & MAE chọn & MAE cực trị & Bước TB & Bước p95',timing,'tab:coverage-frontier-timing'))
    choices=[]
    for budget in BUDGETS:
        for depth in DEPTHS:
            decision=readout['selection'][str(budget)][str(depth)]['chosen']
            if decision is None:
                choices.append([f'{budget:.2f}',str(depth),'Không có','---','---'])
            else:
                row=next(r for r in readout['rows'] if
                    (r['budget'],r['method'],r['depth'])==(budget,decision['method'],depth))
                choices.append([f'{budget:.2f}',str(depth),row['label'],
                    f'{100*row["validation_min_recall"]:.2f}',f'{100*row["min_case_recall"]:.2f}'])
    tables.append(latex_table('Cấu hình chọn trước khi chấm phát triển. Điều kiện là Recall thấp nhất '
        'theo ca đạt 90\\% trên validation; không có ứng viên thì không chọn phương án thay thế. '
        'Hai cột cuối là Recall của ca thấp nhất (\\%).',
        'rrlrr',r'$B$ & $L$ & Chọn trên validation & Validation & Phát triển',choices,
        'tab:coverage-frontier-selection'))
    path=OUTPUT/'tables.tex'
    with path.open('x') as stream:stream.write('\n\n'.join(tables)+'\n')
    print('Verified readout/tables exported',flush=True)


if __name__=='__main__':export()
