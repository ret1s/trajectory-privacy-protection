"""Scientific figures from completed depth readouts, without fitting/selection.

All development depths remain visible. Optional fresh figures require the
recorded before-score depth freeze and exactly one saved TEST primary. No Q,
private truth, key, model, bootstrap or new response-depth score is computed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEPTHS = (20,30,40,60)
PURPOSES = ('nearest_distance','fastest_travel','within_radius','minimum_detour')
LABELS = dict(nearest_distance='Nearest',fastest_travel='Fastest',within_radius='Within radius',
    minimum_detour='Minimum detour',equal_purpose_macro='Equal-purpose macro',family_lower_tail='Family lower-tail mean')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rate(value, *, difference=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
        raise ValueError('Finite saved numeric rate required')
    low=-1. if difference else 0.
    if not low-1e-12 <= value <= 1.+1e-12: raise ValueError('Saved rate outside its contract')
    return float(value)


def estimate(row, *, tail=False):
    value=row['lower_tail_mean_difference' if tail else 'mean_difference']
    ci=row['percentile95_lower_tail_difference' if tail else 'percentile95_family_bootstrap']
    if value is None:
        if ci is not None: raise ValueError('Undefined score must keep its interval N/A')
        return None
    value=rate(value,difference=True)
    if not isinstance(ci,list) or len(ci)!=2: raise ValueError('Saved95% family interval required')
    low,high=(rate(v,difference=True) for v in ci)
    if low>high: raise ValueError('Ordered interval required')
    return value,(low,high)


def coverage(value):
    names=('defined_windows','total_windows','defined_categories','total_categories')
    if any(isinstance(value.get(k),bool) or not isinstance(value.get(k),int) or value[k]<0 for k in names):
        raise ValueError('Nonnegative integer reference counts required')
    if value['defined_windows']>value['total_windows'] or value['defined_categories']>value['total_categories']:
        raise ValueError('Defined reference counts exceed total')
    return {k:value[k] for k in names}


def contrast_key(depth,split,purpose):
    return f'service_l{depth}--minus--service_l20--{split}--current--all--{purpose}'


def prepare_development(saved,paired,selection):
    if saved.get('schema')!='qplanner-response-depth-readout-v1': raise ValueError('Completed depth readout required')
    if (paired.get('schema')!='qplanner-response-depth-paired-readout-v1'
            or paired.get('defense_selected_by_this_readout') is not False
            or paired.get('no_private_generation') is not True):
        raise ValueError('Completed paired depth readout required')
    selected=selection.get('selected_depth')
    if selected not in DEPTHS[1:]: raise ValueError('Already-selected development depth required')
    if [c['depth'] for c in selection['candidates']] != list(DEPTHS[1:]):
        raise ValueError('All predeclared candidates must remain visible')
    if any(row.get('primary_contrast') for row in paired['contrasts'].values()):
        raise ValueError('Development cannot contain a fresh primary')
    rows=[]; baseline=saved['summary']['20']['selection']
    ref={p:coverage(baseline['current']['all'][p]) for p in PURPOSES}
    baseline_bytes=baseline['cost']['reply_bytes']
    if isinstance(baseline_bytes,bool) or not isinstance(baseline_bytes,int) or baseline_bytes<=0:
        raise ValueError('Positive actual baseline reply bytes required')
    for depth in DEPTHS:
        s=saved['summary'][str(depth)]['selection']; purposes=s['current']['all']
        recalls={p:None if purposes[p]['family_mean'] is None else rate(purposes[p]['family_mean'])
            for p in (*PURPOSES,'equal_purpose_macro')}
        if recalls['equal_purpose_macro'] is None: raise ValueError('Defined saved development macro required')
        for purpose in PURPOSES:
            if coverage(purposes[purpose])!=ref[purpose]: raise ValueError('Depth reference N/A coverage changed')
        costs=s['cost']
        if any(isinstance(costs[k],bool) or not isinstance(costs[k],int) or costs[k]<0
                for k in ('requests','request_bytes','reply_bytes')):
            raise ValueError('Nonnegative actual JSON cost counts required')
        if costs['requests']!=baseline['cost']['requests']:
            raise ValueError('Response depth changed request count')
        delta=recalls['equal_purpose_macro']-rate(baseline['current']['all']['equal_purpose_macro']['family_mean'])
        ci=None
        if depth!=20:
            row=paired['contrasts'][contrast_key(depth,'selection','equal_purpose_macro')]
            value,ci=estimate(row)
            if not np.isclose(delta,value,atol=1e-12,rtol=0): raise ValueError('Paired development mean differs from saved curve')
        rows.append(dict(depth=depth,recalls=recalls,macro_delta=delta,macro_ci=ci,
            reply_bytes=costs['reply_bytes'],reply_ratio=costs['reply_bytes']/baseline_bytes,
            request_bytes=costs['request_bytes'],requests=costs['requests']))
    return dict(selected_depth=selected,rows=rows,coverage=ref,
        independent_family_clusters=len(baseline['current']['all']['equal_purpose_macro']['family_values']))


def prepare_fresh(paired,freeze,*,selected_depth):
    if (paired.get('schema')!='qplanner-response-depth-paired-readout-v1'
            or paired.get('defense_selected_by_this_readout') is not False
            or paired.get('no_private_generation') is not True
            or paired.get('exact_Q_clock_ledger_certificate_checked') is not True):
        raise ValueError('Completed certified paired depth readout required')
    if (freeze.get('schema')!='qplanner-selected-response-depth-freeze-v1'
            or freeze.get('fresh_depth_scores_viewed') is not False
            or freeze.get('selected_depth')!=selected_depth or freeze.get('baseline_depth')!=20):
        raise ValueError('Recorded before-score selected depth freeze required')
    primary=[k for k,r in paired['contrasts'].items() if r.get('primary_contrast')]
    key=contrast_key(selected_depth,'test','equal_purpose_macro')
    if primary!=[key]: raise ValueError('Exactly the frozen TEST/current/all macro primary required')
    rows={}; counts={}
    for purpose in (*PURPOSES,'equal_purpose_macro'):
        row=paired['contrasts'][contrast_key(selected_depth,'test',purpose)]
        estimate(row); estimate(row,tail=True)
        ids=row['family_ids']; differences=row.get('family_differences',{})
        if len(ids)!=len(set(ids)) or len(ids)!=row['independent_family_clusters'] or set(ids)!=set(differences):
            raise ValueError('Paired family IDs and saved differences differ')
        for value in differences.values(): rate(value,difference=True)
        if purpose!='equal_purpose_macro':
            left,right=coverage(row['left_reference_coverage']),coverage(row['right_reference_coverage'])
            if left!=right: raise ValueError('Fresh response depth changed reference coverage')
            counts[purpose]=left
        rows[purpose]=row
    primary_row=rows['equal_purpose_macro']; draws=primary_row['within_draw']['draws']
    if set(draws)!= {'1','2','3'}: raise ValueError('All three nested fresh draws required')
    draw_values={d:None if draws[d]['paired_mean_difference'] is None else rate(draws[d]['paired_mean_difference'],difference=True)
        for d in ('1','2','3')}
    costs={f'service_l{d}':dict(requests=0,request_bytes=0,reply_bytes=0) for d in (20,selected_depth)}
    seen=set()
    for row in paired['family_draw_costs']:
        if row['split']!='test': continue
        identity=(row['method'],row['family_id'],row['draw'])
        if row['method'] not in costs or identity in seen: raise ValueError('Unexpected or duplicate fresh cost cluster')
        seen.add(identity)
        for name in costs[row['method']]:
            value=row[name]
            if isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError('Actual integer fresh bytes/counts required')
            costs[row['method']][name]+=value
    left,right=(costs[f'service_l{d}'] for d in (selected_depth,20))
    if right['reply_bytes']<=0 or left['requests']!=right['requests']:
        raise ValueError('Positive matching fresh cost inventory required')
    cost_ids={m:{(f,d) for method,f,d in seen if method==m} for m in costs}
    if len({frozenset(v) for v in cost_ids.values()})!=1: raise ValueError('Fresh cost family/draw pairs differ')
    for method in costs:
        cell=paired['conditional_family_cells'][f'{method}--test--current--all']['equal_purpose_macro']
        expected=set()
        for family,values in cell['family_draw_values'].items():
            if set(values)!={'1','2','3'}: raise ValueError('Fresh family cost requires every declared draw')
            expected.update((family,int(d)) for d in values)
        if not expected or cost_ids[method]!=expected:
            raise ValueError('Fresh cost inventory omitted or added declared family/draw clusters')
    return dict(primary_key=key,rows=rows,coverage=counts,draw_values=draw_values,
        cost=costs,reply_ratio=left['reply_bytes']/right['reply_bytes'],
        criterion_result=paired['primary_criterion_result'])


def load_inputs(development,fresh=None):
    development=Path(development); inputs={}
    def load(label,path):
        inputs[label]={'path':str(Path(path).resolve()),'sha256':sha(path)}
        return read(path)
    saved=load('development_readout',development/'readout.json')
    paired=load('development_paired_readout',development/'paired_readout.json')
    selection=load('development_selection',development/'depth_selection.json')
    protocol=load('development_protocol',development/'protocol.json')
    pp=load('development_paired_protocol',development/'paired_protocol.json')
    validation=load('development_validation',development/'validation.json')
    if (selection['readout_sha256']!=inputs['development_readout']['sha256']
            or paired['source_readout_sha256']!=inputs['development_readout']['sha256']
            or paired['paired_protocol_sha256']!=inputs['development_paired_protocol']['sha256']
            or saved['protocol_sha256']!=inputs['development_protocol']['sha256']
            or validation['status']!='pass' or validation['readout_sha256']!=inputs['development_readout']['sha256']
            or validation['protocol_sha256']!=inputs['development_protocol']['sha256']
            or validation['selected_depth']!=selection['selected_depth']
            or selection['protocol_sha256']!=inputs['development_protocol']['sha256']
            or pp['source_readout_sha256']!=inputs['development_readout']['sha256']
            or pp['scope']!='development' or pp['source_protocol_sha256']!=inputs['development_protocol']['sha256']):
        raise ValueError('Completed development input hash chain differs')
    prepared=prepare_development(saved,paired,selection); prepared['fresh']=None
    if fresh is not None:
        fresh=Path(fresh)
        fs=load('fresh_readout',fresh/'readout.json'); fp=load('fresh_paired_readout',fresh/'paired_readout.json')
        fprotocol=load('fresh_protocol',fresh/'protocol.json'); fpp=load('fresh_paired_protocol',fresh/'paired_protocol.json')
        freeze=load('fresh_depth_freeze',fresh/'depth_freeze.json')
        if (fp['source_readout_sha256']!=inputs['fresh_readout']['sha256']
                or fp['paired_protocol_sha256']!=inputs['fresh_paired_protocol']['sha256']
                or fs['protocol_sha256']!=inputs['fresh_protocol']['sha256']
                or fpp['source_protocol_sha256']!=inputs['fresh_protocol']['sha256']
                or freeze['depth_protocol_sha256']!=inputs['fresh_protocol']['sha256']
                or freeze['development_selection_sha256']!=inputs['development_selection']['sha256']
                or fpp['scope']!='independent-synthetic-generalization'
                or freeze['criterion']!=fprotocol['criterion'] or fpp['criterion']!=freeze['criterion']):
            raise ValueError('Fresh readout does not bind the saved pre-score freeze/development selection')
        prepared['fresh']=prepare_fresh(fp,freeze,selected_depth=prepared['selected_depth'])
    return prepared,inputs


def coverage_text(value):
    c,t=value['defined_categories'],value['total_categories']
    percent=f'{100*c/t:.1f}%' if t else 'N/A'
    return f'{c:,}/{t:,} ({percent})\nwindows {value["defined_windows"]:,}/{value["total_windows"]:,}'


def plot(prepared):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
    fresh=prepared['fresh']; nrows=3 if fresh else 2
    fig=plt.figure(figsize=(12.,10.2 if fresh else 7.))
    grid=fig.add_gridspec(nrows,2,height_ratios=[1.1,1.05,.8] if fresh else [1.1,.8],
        left=.095,right=.97,top=.90,bottom=.115,wspace=.40,hspace=.57)
    utility=fig.add_subplot(grid[0,0]); cost=fig.add_subplot(grid[0,1])
    rows=prepared['rows']; selected=prepared['selected_depth']
    x=[r['depth'] for r in rows]; y=[100*r['macro_delta'] for r in rows]
    utility.plot(x,y,color='#0072B2',marker='o',label='Equal-purpose macro')
    for r in rows[1:]:
        low,high=r['macro_ci']; utility.vlines(r['depth'],100*low,100*high,color='#0072B2',linewidth=1.4)
    base=rows[0]['recalls']
    for purpose,color in zip(PURPOSES,('#009E73','#E69F00','#CC79A7','#D55E00')):
        values=[None if r['recalls'][purpose] is None or base[purpose] is None
            else 100*(r['recalls'][purpose]-base[purpose]) for r in rows]
        utility.plot(x,values,color=color,linestyle=':',linewidth=1.,label=LABELS[purpose])
    utility.axhline(2.,color='.5',linestyle='--',linewidth=.7,label='Fixed development gate: +2 pp')
    chosen=next(r for r in rows if r['depth']==selected)
    utility.scatter([selected],[100*chosen['macro_delta']],s=95,facecolors='none',edgecolors='#0072B2',linewidths=1.8,zorder=5)
    utility.annotate(f'Chosen L{selected}: +{100*chosen["macro_delta"]:.3f} pp',
        xy=(selected,100*chosen['macro_delta']),xytext=(5,14),textcoords='offset points',fontsize=8,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1.))
    utility.set(xticks=x,xlabel='Public service depth L per category',ylabel='ΔRecall@5 vs L20 (percentage points)')
    utility.set_title(f'A  DEVELOPMENT utility: {prepared["independent_family_clusters"]} families\nSaved paired-family95% macro CI; exploratory',loc='left')
    utility.legend(fontsize=6.8,loc='upper left',frameon=False)
    utility.grid(alpha=.18);utility.spines[['top','right']].set_visible(False)
    positions=np.arange(len(rows)); volumes=[r['reply_bytes']/2**20 for r in rows]
    bars=cost.bar(positions,volumes,color=['#777777' if r['depth']==20 else '#0072B2' if r['depth']==selected else '#b7d7e8' for r in rows],width=.65)
    for bar,r in zip(bars,rows):
        cost.text(bar.get_x()+bar.get_width()/2,bar.get_height(),
            f'{r["reply_ratio"]:.3f}×\n(+{100*(r["reply_ratio"]-1):.1f}%)',ha='center',va='bottom',fontsize=7)
    cost.set(xticks=positions,xticklabels=[f'L{r["depth"]}' for r in rows],ylabel='Reply JSON payload (MiB)',ylim=(0,max(volumes)*1.25))
    cost.set_title('B  DEVELOPMENT response cost\nSame Q/clock/request count; additional bytes paid',loc='left')
    cost.grid(axis='y',alpha=.18);cost.spines[['top','right']].set_visible(False)
    if fresh:
        forest=fig.add_subplot(grid[1,0]);drawax=fig.add_subplot(grid[1,1])
        names=['equal_purpose_macro',*PURPOSES,'family_lower_tail']
        for j,purpose in enumerate(names):
            row=fresh['rows'].get(purpose,fresh['rows']['equal_purpose_macro'])
            saved=estimate(row,tail=purpose=='family_lower_tail')
            if saved is None:forest.text(0.,j,' N/A',va='center',fontsize=8);continue
            mean,(low,high)=saved;color='#0072B2' if j==0 else '.45'
            forest.hlines(j,100*low,100*high,color=color,linewidth=1.4)
            forest.scatter(100*mean,j,color=color,s=32 if j==0 else 23,zorder=3)
        forest.axvline(0.,color='.5',linewidth=.7,linestyle='--')
        forest.set_yticks(range(len(names)),[LABELS[n]+(' (primary)' if n=='equal_purpose_macro' else '') for n in names],fontsize=7)
        forest.set_ylim(len(names)-.5,-.5);forest.set_xlabel('ΔRecall@5 (pp); saved95% whole-family CI')
        forest.set_title(f'C  Fresh L{selected} − L20: current replies\nSame-map synthetic; one predeclared primary',loc='left')
        forest.grid(axis='x',alpha=.18);forest.spines[['top','right']].set_visible(False)
        primary=estimate(fresh['rows']['equal_purpose_macro'])
        if primary is not None:
            mean,(low,high)=primary
            drawax.axhspan(100*low,100*high,color='#0072B2',alpha=.12,label='Overall saved95% family CI')
            drawax.axhline(100*mean,color='#0072B2',linewidth=1.,label='Overall family mean')
        for i,(draw,value) in enumerate(fresh['draw_values'].items(),start=1):
            if value is None:drawax.text(i,0.,'N/A',ha='center',fontsize=8)
            else:drawax.scatter(i,100*value,color='#D55E00',s=36,zorder=3)
        drawax.axhline(0.,color='.5',linewidth=.7,linestyle='--')
        drawax.set(xticks=[1,2,3],xticklabels=['Draw1','Draw2','Draw3'],xlim=(.5,3.5),ylabel='Macro ΔRecall@5 (pp)')
        ratio=fresh['reply_ratio']
        drawax.set_title(f'D  Nested private-draw diagnostics\nFresh reply JSON: +{100*(ratio-1):.1f}% vs L20',loc='left')
        drawax.legend(fontsize=7,frameon=False,loc='best');drawax.grid(axis='y',alpha=.18);drawax.spines[['top','right']].set_visible(False)
    tableax=fig.add_subplot(grid[-1,:]);tableax.axis('off')
    tableax.set_title(('E' if fresh else 'C')+'  Purpose utility and reference coverage: empty reference stays N/A',loc='left',pad=10)
    body=[]
    for purpose in PURPOSES:
        baseline,adopted=base[purpose],chosen['recalls'][purpose]
        cells=[LABELS[purpose],'N/A' if baseline is None else f'{100*baseline:.2f}%',
            'N/A' if adopted is None else f'{100*adopted:.2f}%',coverage_text(prepared['coverage'][purpose])]
        if fresh:cells.append(coverage_text(fresh['coverage'][purpose]))
        body.append(cells)
    headers=['Purpose','DEV L20 Recall',f'DEV L{selected} Recall','DEV reference coverage']+(['Fresh reference coverage'] if fresh else [])
    table=tableax.table(cellText=body,colLabels=headers,cellLoc='center',bbox=[0.,0.,1.,.90])
    table.auto_set_font_size(False);table.set_fontsize(7)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('.8');cell.set_linewidth(.5)
        if r==0:cell.set_facecolor('#eeeeee')
    title='Geo-I: fixed Q, public response-depth utility / cost'
    if not fresh:title+=' — fresh confirmation pending'
    fig.suptitle(title,x=.095,ha='left',fontsize=13)
    caption='Current-only conditional static-service utility. L is per category, for each of5 Q; application JSON excludes HTTP/TLS.\n'
    caption+=('Draws remain nested in family; purpose/tail intervals are exploratory. ' if fresh else 'Development tuning and intervals are exploratory. ')
    caption+='Paid response depth is not a novel Q algorithm, a privacy improvement or a same-cost SOTA comparison.'
    fig.text(.095,.025,caption,fontsize=7,color='.25')
    return fig


def render(development,output_dir,*,fresh=None):
    prepared,inputs=load_inputs(development,fresh)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    fig=plot(prepared);paths=[]
    for suffix in ('pdf','png'):
        path=output_dir/f'response_depth_utility_cost.{suffix}'
        fig.savefig(path,dpi=300,facecolor='white');paths.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    metadata=dict(schema='qplanner-response-depth-scientific-figure-v1',inputs=inputs,
        figure_source_sha256=sha(Path(__file__)),selected_depth=prepared['selected_depth'],
        development_depths=list(DEPTHS),fresh_included=prepared['fresh'] is not None,
        units=dict(utility='percentage points =100 times saved0..1 rate differences',reply_cost='MiB = JSON bytes /2^20'),
        statistical_unit='whole family; nested draws/sessions retained',
        intervals='saved paired-family95% intervals; no new bootstrap, fit, scoring or selection',
        raw_coordinates_or_rng_keys_read=False,output_sha256={p.name:sha(p) for p in paths})
    with (output_dir/'figure_provenance.json').open('x') as stream:
        json.dump(metadata,stream,indent=2,allow_nan=False);stream.write('\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--fresh',type=Path,help='Completed frozen fresh depth output; omit while pending')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();render(args.development,args.output_dir,fresh=args.fresh)
    print('Saved source-backed response-depth PDF/PNG:',args.output_dir)


if __name__=='__main__':main()
