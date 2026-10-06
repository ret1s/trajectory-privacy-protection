"""Plot completed paired utility evidence without fitting or selecting models.

The sole data input is paired_family_readout.json. Absolute per-method family
Recall is absent from that contract, so the figure displays paired differences
in percentage points, its saved family bootstrap intervals, and N/A coverage.
No file is produced until actual completed evidence passes the input contract.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PURPOSES = ('nearest_distance','fastest_travel','within_radius','minimum_detour')
LABELS = dict(nearest_distance='Nearest distance',fastest_travel='Fastest travel',
              within_radius='Within radius',minimum_detour='Minimum detour',
              equal_purpose_macro='Four-purpose family mean',family_lower_tail='Family lower25% mean')
SUFFIX = '--test--current--all--'
METHOD_LABELS = dict(legacy_l10='Legacy signature L10',aligned_nearest='Nearest signature L20',
    multi_mean='Joint-profile mean',tail25='Joint-profile tail λ=.25',tail50='Joint-profile tail λ=.50',
    normalized_mean='Normalized mean',normalized_tight='Normalized mean, slack0',
    normalized_tail='Normalized tail λ=.25')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _estimate(row, *, tail=False):
    value = row['lower_tail_mean_difference' if tail else 'mean_difference']
    interval = row['percentile95_lower_tail_difference' if tail else 'percentile95_family_bootstrap']
    if value is None:
        if interval is not None: raise ValueError('Undefined estimate must keep its interval N/A')
        return None
    if (not isinstance(interval,list) or len(interval)!=2 or not np.isfinite([value,*interval]).all()
            or interval[0]>interval[1] or max(abs(v) for v in [value,*interval])>1.+1e-12):
        raise ValueError('Finite saved rate-difference estimate and ordered95% interval required')
    return float(value),tuple(map(float,interval))


def _coverage(value):
    names=('defined_windows','total_windows','defined_categories','total_categories')
    if any(isinstance(value.get(k),bool) or not isinstance(value.get(k),int) or value[k]<0 for k in names):
        raise ValueError('Nonnegative integer coverage counts required')
    if value['defined_windows']>value['total_windows'] or value['defined_categories']>value['total_categories']:
        raise ValueError('Defined reference coverage cannot exceed its denominator')
    return {k:value[k] for k in names}


def prepare(readout, *, alignment_control='aligned_nearest'):
    """Select only the recorded primary and fixed alignment secondary, no tuning."""
    if readout.get('schema')!='qplanner-paired-family-readout-v1' or readout.get('test_independent_unit')!='family':
        raise ValueError('Completed paired-family readout contract required')
    if readout.get('defense_selected_by_this_readout') is not False:
        raise ValueError('Figure cannot silently select a defense')
    contrasts=readout['contrasts']
    primary=[(key,row) for key,row in contrasts.items() if row.get('primary_contrast')]
    if len(primary)!=1:raise ValueError('Exactly one already-recorded fresh primary contrast required')
    key,record=primary[0]
    suffix=SUFFIX+'equal_purpose_macro'
    if not key.endswith(suffix) or '--minus--' not in key:
        raise ValueError('Primary must be TEST/current/all/equal-purpose family Recall')
    left,right=key.removesuffix(suffix).split('--minus--')
    if left==right:raise ValueError('Primary methods must be distinct')
    rights=[right]+([alignment_control] if alignment_control not in (left,right) else [])
    groups=[];coverage={};families=set()
    for control in rights:
        rows={}
        for purpose in (*PURPOSES,'equal_purpose_macro'):
            name=f'{left}--minus--{control}{SUFFIX}{purpose}'
            if name not in contrasts:raise ValueError('Missing declared purpose or alignment contrast: '+name)
            row=contrasts[name];_estimate(row);_estimate(row,tail=True)
            ids=row['family_ids'];differences=row.get('family_differences',{})
            if len(ids)!=len(set(ids)) or len(ids)!=row['independent_family_clusters'] or set(ids)!=set(differences):
                raise ValueError('Paired family identities/denominators do not match saved differences')
            if not np.isfinite(list(differences.values())).all() or any(abs(v)>1.+1e-12 for v in differences.values()):
                raise ValueError('Finite rate differences required')
            rows[purpose]=row
            if purpose!='equal_purpose_macro':
                for method,field in ((left,'left_reference_coverage'),(control,'right_reference_coverage')):
                    counts=_coverage(row[field]);slot=(method,purpose)
                    if slot in coverage and coverage[slot]!=counts:
                        raise ValueError('Same method has inconsistent coverage across contrasts')
                    coverage[slot]=counts
        families.update(rows['equal_purpose_macro']['family_ids'])
        groups.append(dict(left=left,right=control,primary=control==right,rows=rows))
    return dict(primary_key=key,left=left,right=right,alignment_control=alignment_control,
        groups=groups,family_ids=sorted(families),coverage=coverage,
        methods=[left,*rights],scope='TEST/current/all; conditional static service; paired family differences',
        absolute_family_recall_available=False)


def coverage_text(counts):
    total=counts['total_categories'];defined=counts['defined_categories']
    percentage=f'{100*defined/total:.1f}%' if total else 'N/A'
    return f'{defined}/{total} ({percentage})\nwindows {counts["defined_windows"]}/{counts["total_windows"]}'


def plot(prepared):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                        'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
    height=max(8.,2.+.23*len(prepared['family_ids']))
    fig=plt.figure(figsize=(13.,height))
    grid=fig.add_gridspec(2,2,width_ratios=[1.,1.7],height_ratios=[1.5,1.],
                         left=.12,right=.98,bottom=.15,top=.88,wspace=.40,hspace=.38)
    family_ax=fig.add_subplot(grid[:,0]);forest_ax=fig.add_subplot(grid[0,1]);coverage_ax=fig.add_subplot(grid[1,1])
    colors=['#0072B2','#D55E00'];markers=['o','^'];legend=[]
    ids=prepared['family_ids'];ys=np.arange(len(ids))
    for i,group in enumerate(prepared['groups']):
        label=f'{METHOD_LABELS.get(group["left"],group["left"])} − {METHOD_LABELS.get(group["right"],group["right"])}'+(' (primary)' if group['primary'] else ' (alignment secondary)')
        legend.append(Line2D([],[],color=colors[i],marker=markers[i],linestyle='',label=label))
        values=group['rows']['equal_purpose_macro'].get('family_differences',{})
        offset=(i-(len(prepared['groups'])-1)/2)*.24
        for j,family in enumerate(ids):
            if family in values:
                family_ax.scatter(100*values[family],j+offset,color=colors[i],marker=markers[i],s=23,zorder=3)
    family_ax.axvline(0.,color='.5',linewidth=.8,linestyle='--')
    family_ax.set_yticks(ys,ids,fontsize=7);family_ax.invert_yaxis()
    family_ax.set_xlabel('Paired ΔRecall@5 (percentage points)')
    family_ax.set_title('A  Each independent family\n(all sessions and draws retained)',loc='left')
    family_ax.grid(axis='x',alpha=.2);family_ax.spines[['top','right']].set_visible(False)
    if not ids:family_ax.text(.5,.5,'N/A: no defined family pairs',ha='center',transform=family_ax.transAxes)

    names=[*PURPOSES,'equal_purpose_macro','family_lower_tail']
    for i,group in enumerate(prepared['groups']):
        offset=(i-(len(prepared['groups'])-1)/2)*.23
        for j,purpose in enumerate(names):
            row=group['rows'].get(purpose,group['rows']['equal_purpose_macro'])
            estimate=_estimate(row,tail=purpose=='family_lower_tail')
            y=j+offset
            if estimate is None:
                forest_ax.text(0.,y,' N/A',color=colors[i],fontsize=7,va='center')
                continue
            mean,(low,high)=estimate
            forest_ax.hlines(y,100*low,100*high,color=colors[i],linewidth=1.4)
            forest_ax.scatter(100*mean,y,color=colors[i],marker=markers[i],s=25,zorder=3)
    forest_ax.axvline(0.,color='.5',linewidth=.8,linestyle='--')
    forest_ax.set_yticks(np.arange(len(names)),[LABELS[n] for n in names],fontsize=8)
    forest_ax.set_ylim(len(names)-.5,-.5);forest_ax.set_xlabel('ΔRecall@5 (percentage points); saved95% family-bootstrap CI')
    forest_ax.set_title('B  Mean and lower-tail contrasts\nLCVaR difference = difference of method tails',loc='left')
    forest_ax.grid(axis='x',alpha=.2);forest_ax.spines[['top','right']].set_visible(False)

    coverage_ax.axis('off');coverage_ax.set_title('C  Reference coverage\nDefined categories / all categories; empty reference remains N/A',loc='left',pad=14)
    cells=[[coverage_text(prepared['coverage'][(m,p)]) for m in prepared['methods']] for p in PURPOSES]
    table=coverage_ax.table(cellText=cells,rowLabels=[LABELS[p] for p in PURPOSES],
        colLabels=[METHOD_LABELS.get(m,m) for m in prepared['methods']],cellLoc='center',rowLoc='left',bbox=[0.,0.,1.,.92])
    table.auto_set_font_size(False);table.set_fontsize(7)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('.8');cell.set_linewidth(.5)
        if r==0:cell.set_facecolor('#eeeeee')
    fig.suptitle('Geo-I public-service planner: paired family utility',fontsize=13,x=.12,ha='left')
    fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.12,.055),frameon=False,fontsize=8)
    fig.text(.12,.026,'Current responses only. Conditional static-service utility; same-map synthetic generalization.\n'
             'Exploratory secondary intervals are unadjusted. Figure establishes neither empirical privacy equivalence nor publication readiness.',fontsize=7,color='.25')
    return fig


def render(input_path,output_dir,*,alignment_control='aligned_nearest'):
    input_path,output_dir=Path(input_path),Path(output_dir)
    readout=json.loads(input_path.read_text())
    prepared=prepare(readout,alignment_control=alignment_control)
    output_dir.mkdir(parents=True,exist_ok=False)
    fig=plot(prepared)
    paths=[]
    for suffix in ('pdf','png'):
        path=output_dir/('paired_family_current_recall.'+suffix)
        fig.savefig(path,dpi=300,facecolor='white');paths.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    metadata=dict(schema='qplanner-paired-figure-v1',input_path=str(input_path.resolve()),input_sha256=sha(input_path),
        figure_source_sha256=sha(Path(__file__)),saved_analysis_protocol_sha256=readout.get('protocol_sha256'),
        primary_key=prepared['primary_key'],alignment_control=alignment_control,
        displayed_method_labels={m:METHOD_LABELS.get(m,m) for m in prepared['methods']},
        displayed_units='percentage-point rate differences (100 times saved0..1 differences)',
        confidence_intervals='saved95% paired whole-family percentile bootstrap; no refit or interval recomputation',
        absolute_family_recall_available=False,output_sha256={p.name:sha(p) for p in paths},
        scope=prepared['scope'],test_independent_unit='family',private_data_or_rng_keys_read=False)
    with (output_dir/'figure_provenance.json').open('x') as stream:
        json.dump(metadata,stream,indent=2,allow_nan=False);stream.write('\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--alignment-control',default='aligned_nearest')
    args=parser.parse_args()
    render(args.input,args.output_dir,alignment_control=args.alignment_control)
    print('Saved paired-family scientific figure with source/input hashes:',args.output_dir)


if __name__=='__main__':main()
