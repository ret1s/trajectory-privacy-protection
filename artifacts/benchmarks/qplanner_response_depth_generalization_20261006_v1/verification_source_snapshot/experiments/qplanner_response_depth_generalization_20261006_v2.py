"""Frozen-depth synthetic utility replay and independent paired-family readout.

Constant response depth pays bandwidth for more static POIs. The Geo-I Q,
clocks, K, anchors, reads and attacker inputs stay frozen. This is neither a
new privacy mechanism nor a comparison with competing algorithms.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import qplanner_response_depth_20261006 as kernel
from experiments.qplanner_paired_readout_20261006 import (
    CACHES, REPLICATES, SEED, build_readout, declared_clock_inventory,
    primary_decision, summarize_bundles, validate_study_sources,
)

ROOT = kernel.ROOT
BASE_VERIFIER = 'experiments/verify_qplanner_depth_base_q_20261006_v2.py'
DEV_VERIFIER = 'experiments/verify_qplanner_response_depth_20261006.py'
VERIFIER_HELPERS = ('experiments/verify_qplanner_study_20261006.py',
                    'experiments/verify_qplanner_study_20261006_v2.py')
THIS = 'experiments/qplanner_response_depth_generalization_20261006_v2.py'
PREVIOUS_THIS = 'experiments/qplanner_response_depth_generalization_20261006.py'
CRITERION = {'minimum_absolute_mean_gain': .02, 'paired95_lower_bound_gt': 0.,
    'every_private_draw_gain_gt': 0., 'family_bootstrap_replicates': 10000,
    'public_analysis_seed': 2026100617}
DRAW_SCHEDULE = {'train': 1, 'selection': 1, 'test': 3}
PRIMARY = 'selected constant public depth minus20; current/all/equal-purpose/equal-family conditional Recall@5'
SERVICE_KEYS = ('full_reply60_sha256', 'frozen_reply20_sha256',
                'depth_context_sha256', 'exact_l20_prefix_asserted')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return kernel.common.read(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kernel.save_new(path, value)


def local(root, name):
    root = Path(root).resolve(); name = Path(name)
    if name.is_absolute() or not (root/name).resolve().is_relative_to(root):
        raise ValueError('Root-relative pinned path required')
    return root/name


def artifact(root, name):
    path = Path(name)
    if not path.is_absolute(): return local(root, path)
    if path.resolve().is_relative_to(Path(root).resolve()): return path
    if 'artifacts' not in path.parts: raise ValueError('Published artifact path required')
    return local(root, Path(*path.parts[path.parts.index('artifacts'):]))


def relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def check_pins(root, output, pins, snapshot='source_snapshot'):
    for name, pin in pins.items():
        if sha(local(root, name)) != pin or sha(local(Path(output)/snapshot, name)) != pin:
            raise ValueError('Pinned source/snapshot differs: '+name)


def equivalent_public_inputs(fresh, development, *, root=ROOT):
    """Permit ONLY an identical native archive copied beside the new dataset.

    The old independently verified dataset may deliberately reuse an archive
    from its earlier version. Authenticate that old dataset before reading its
    network metadata; the fresh archive must be beside the declared dataset.
    The full base-Q verifier later independently checks the fresh dataset's
    actual network.compressed_path and both compressed/native digests.
    No fresh dataset bytes or labels are read to establish this equivalence.
    """
    left,right=fresh['public_inputs_sha256'],development['public_inputs_sha256']
    if left==right:
        for name,digest in left.items():
            if sha(local(root,name))!=digest:
                raise ValueError('Unrelocated public input bytes differ from pinned digest')
        return {'native_archive_relocated':False,'all_other_public_inputs_identical':True}
    a=str(Path(fresh['dataset_path']).parent/'public_native.net.xml.gz')
    old_dataset=local(root,development['dataset_path'])
    if sha(old_dataset)!=development['dataset_sha256']:
        raise ValueError('Development dataset bytes differ before reading network metadata')
    network=read(old_dataset)['network']
    b=network['compressed_path']
    local(root,b)  # Reject absolute/out-of-root metadata paths.
    if (a==b or a not in left or b not in right
            or set(left)-set(right)!={a} or set(right)-set(left)!={b}
            or {k:v for k,v in left.items() if k!=a}!={k:v for k,v in right.items() if k!=b}
            or left[a]!=right[b] or network['compressed_sha256']!=right[b]):
        raise ValueError('Public inputs differ beyond an identical native-archive path relocation')
    if sha(local(root,a))!=left[a] or sha(local(root,b))!=right[b]:
        raise ValueError('Relocated native archive bytes differ from pinned identical digests')
    native_sha=hashlib.sha256(gzip.decompress(local(root,b).read_bytes())).hexdigest()
    if native_sha!=network['native_sha256']:
        raise ValueError('Development native archive digest differs from network metadata')
    return {'native_archive_relocated':True,'fresh_native_archive_path':a,
        'development_native_archive_path':b,'identical_archive_sha256':left[a],
        'development_dataset_sha256':development['dataset_sha256'],
        'identical_native_sha256':native_sha,
        'all_other_public_inputs_identical':True,
        'full_base_Q_validation_checks_dataset_network_path_after_freeze':True}


def development_contract(development, *, root=ROOT):
    """Bind the durable fixed development selection; do not read fresh data."""
    development = Path(development); p = read(development/'protocol.json')
    if sha(development/'protocol.json') != (development/'protocol.sha256').read_text().strip():
        raise ValueError('Development depth protocol changed')
    if p['selection_criterion'] != kernel.CRITERION or p['depths'] != list(kernel.DEPTHS):
        raise ValueError('Depth selection gates changed')
    if 'test' in p['splits']: raise ValueError('Development must exclude TEST')
    check_pins(root, development, p['source_sha256'])
    for name, pin in p['source_files_sha256'].items():
        if sha(artifact(root, p['source_output'])/name) != pin:
            raise ValueError('Original development Q source changed')
    selection = read(development/'depth_selection.json')
    if (selection['schema'] != 'qplanner-response-depth-selection-v1'
            or selection['criterion'] != kernel.CRITERION
            or selection['protocol_sha256'] != sha(development/'protocol.json')
            or selection['readout_sha256'] != sha(development/'readout.json')):
        raise ValueError('Depth selection provenance/gate differs')
    selected = selection['selected_depth']
    candidates = selection['candidates']
    if [c['depth'] for c in candidates] != [30, 40, 60]:
        raise ValueError('All predeclared depth candidates must be retained')
    for candidate in candidates:
        differences=candidate['differences'];ratio=candidate['reply_json_byte_ratio']
        gates={'macro_gain_at_least_2pp':differences['equal_purpose_macro']>=.02-1e-12,
            'nearest_no_loss':differences['nearest_distance']>=-1e-12,
            'reply_byte_ratio_at_most_2_5':ratio<=2.5+1e-12}
        if (not all(np.isfinite([*differences.values(),ratio])) or ratio<=0
                or candidate['gates']!=gates or candidate['eligible']!=all(gates.values())):
            raise ValueError('Claimed development eligibility differs from fixed numeric gates')
    eligible = [c['depth'] for c in candidates if c['eligible'] is True and all(c['gates'].values())]
    if selected not in kernel.DEPTHS[1:] or not eligible or selected != min(eligible):
        raise ValueError('No eligible smallest fixed depth selected; do not force a winner')
    resource = read(development/'resources.json')
    readout = read(development/'readout.json')
    if (readout['protocol_sha256'] != sha(development/'protocol.json')
            or readout['resources_sha256'] != sha(development/'resources.json')
            or set(readout['family_files_sha256']) != set(p['family_files_sha256'])
            or resource['exact_l20_prefix_asserted'] is not True):
        raise ValueError('Development replay/resource inventory changed')
    certificate = read(development/'validation.json')
    if (certificate['status'] != 'pass'
            or certificate['protocol_sha256'] != sha(development/'protocol.json')
            or certificate['readout_sha256'] != sha(development/'readout.json')
            or certificate['verifier_sha256'] != sha(local(root,DEV_VERIFIER))
            or certificate.get('selected_depth') != selected
            or certificate.get('fixed_Q_clock_anchors_ledger_inputs_unchanged') is not True):
        raise ValueError('Full independent development depth validation required')
    original = read(artifact(root, p['source_output'])/'protocol.json')
    return p, selection, resource, original


def declare_freeze(development, base_q, output, *, root=ROOT):
    """Seal a selected depth before base Q generation or fresh utility scoring."""
    root = Path(root); development, base_q, output = map(Path, (development, base_q, output))
    if output.exists() and any(output.iterdir()): raise FileExistsError('New empty depth output required')
    if (base_q/'generation_started.json').exists() or (base_q/'generation.json').exists():
        raise ValueError('Freeze must precede the first fresh Q generation')
    if (base_q/'freeze.json').exists(): raise FileExistsError('Existing base-Q freeze is immutable')
    dp, selection, resources, original = development_contract(development, root=root)
    base = read(base_q/'protocol.json')
    if sha(base_q/'protocol.json') != (base_q/'protocol.sha256').read_text().strip():
        raise ValueError('Predeclared base-Q protocol changed')
    if (list(base['configuration']['methods']) != ['legacy_l10']
            or base['splits'] != ['train', 'selection', 'test']
            or base['draws_by_split'] != DRAW_SCHEDULE):
        raise ValueError('Sole legacy Q with predeclared1/1/3 draws required')
    if (base['dataset_path'] == original['dataset_path'] or base['dataset_sha256'] == original['dataset_sha256']):
        raise ValueError('An independent synthetic cohort is required')
    if base['source_sha256']!=original['source_sha256']:
        raise ValueError('Frozen base-Q source differs')
    public_equivalence=equivalent_public_inputs(base,original,root=root)
    nonmethods = lambda c: {k:v for k,v in c.items() if k != 'methods'}
    if (nonmethods(base['configuration']) != nonmethods(dp['fixed_configuration'])
            or base['configuration']['methods']['legacy_l10'] != dp['fixed_configuration']['methods']['legacy_l10']):
        raise ValueError('All privacy, planner, clock and service controls must remain unchanged')
    check_pins(root, base_q, base['source_sha256'])
    for name, pin in base['public_inputs_sha256'].items():
        if sha(local(root, name)) != pin: raise ValueError('Frozen public input changed')
    names = sorted(set(dp['source_sha256']) | {THIS, PREVIOUS_THIS, BASE_VERIFIER, DEV_VERIFIER, *VERIFIER_HELPERS})
    source_pins = {name: sha(local(root, name)) for name in names}
    value = {
        'schema': 'qplanner-response-depth-generalization-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'development_depth_output': relative(root, development),
        'development_files_sha256': {n:sha(development/n) for n in
            ('protocol.json','protocol.sha256','depth_selection.json','readout.json','resources.json','validation.json',
             'paired_protocol.json','paired_readout.json')},
        'base_q_output': relative(root, base_q), 'base_q_protocol_sha256': sha(base_q/'protocol.json'),
        'base_q_configuration': base['configuration'], 'base_q_source_sha256': base['source_sha256'],
        'public_inputs_sha256': base['public_inputs_sha256'],
        'public_input_equivalence': public_equivalence,
        'dataset_path': base['dataset_path'], 'dataset_sha256': base['dataset_sha256'],
        'splits': base['splits'], 'draws_by_split': DRAW_SCHEDULE,
        'selected_depth': selection['selected_depth'], 'baseline_depth': 20,
        'depths': [20,selection['selected_depth']], 'criterion': CRITERION, 'primary': PRIMARY,
        'service_kernel': {k:resources[k] for k in SERVICE_KEYS},
        'source_sha256': source_pins, 'base_q_verifier': BASE_VERIFIER,
        'minimum_fresh_test_families': 24,
        'claim_scope': 'independent same-map synthetic static utility/cost; paid bandwidth, no new Q mechanism/Geo-I proof/SOTA superiority; offline/bulk workload gate unresolved',
    }
    output.mkdir(parents=True, exist_ok=True); save(output/'protocol.json', value)
    (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    for name in names:
        path = local(output/'source_snapshot', name); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(local(root,name).read_bytes())
    depth_freeze = {'schema':'qplanner-selected-response-depth-freeze-v1',
        'depth_protocol_sha256':sha(output/'protocol.json'), 'selected_depth':value['selected_depth'],
        'baseline_depth':20, 'criterion':CRITERION, 'primary':PRIMARY,
        'base_q_protocol_sha256':value['base_q_protocol_sha256'],
        'development_selection_sha256':sha(development/'depth_selection.json'),
        'freeze_source_sha256':sha(local(root,THIS)), 'fresh_depth_scores_viewed':False}
    save(output/'depth_freeze.json', depth_freeze)
    save(base_q/'freeze.json', {'schema':'qplanner-base-Q-for-selected-depth-freeze-v1',
        'fresh_protocol_sha256':value['base_q_protocol_sha256'], 'configuration':base['configuration'],
        'fresh_test_evaluated':False, 'depth_output':relative(root,output),
        'depth_protocol_sha256':sha(output/'protocol.json'),
        'depth_freeze_sha256':sha(output/'depth_freeze.json')})
    return value


def depth_contract(output, *, root=ROOT):
    """Authenticate adoption/source before opening any fresh dataset or score."""
    output = Path(output); root = Path(root); p = read(output/'protocol.json')
    digest = sha(output/'protocol.json')
    if (digest != (output/'protocol.sha256').read_text().strip()
            or p['schema'] != 'qplanner-response-depth-generalization-v1'
            or p['criterion'] != CRITERION or p['primary'] != PRIMARY
            or p['baseline_depth'] != 20 or p['depths'] != [20,p['selected_depth']]
            or p['draws_by_split'] != DRAW_SCHEDULE or p['minimum_fresh_test_families'] != 24):
        raise ValueError('Fresh depth protocol/gates differ')
    check_pins(root, output, p['source_sha256'])
    development = local(root,p['development_depth_output'])
    for name,pin in p['development_files_sha256'].items():
        if sha(development/name) != pin: raise ValueError('Frozen depth development evidence changed')
    dp, selection, resource, original = development_contract(development,root=root)
    if (p['selected_depth'] != selection['selected_depth']
            or p['service_kernel'] != {k:resource[k] for k in SERVICE_KEYS}):
        raise ValueError('Depth/kernel differs from the selected development policy')
    base_q = local(root,p['base_q_output']); base = read(base_q/'protocol.json')
    if sha(base_q/'protocol.json') != p['base_q_protocol_sha256'] or sha(base_q/'protocol.json') != (base_q/'protocol.sha256').read_text().strip():
        raise ValueError('Fresh base-Q protocol changed')
    if (list(base['configuration']['methods']) != ['legacy_l10'] or base['splits'] != ['train','selection','test']
            or base['draws_by_split'] != DRAW_SCHEDULE or base['configuration'] != p['base_q_configuration']):
        raise ValueError('Sole unchanged legacy Q and1/1/3 schedule required')
    if (base['dataset_path'],base['dataset_sha256']) != (p['dataset_path'],p['dataset_sha256']):
        raise ValueError('Frozen fresh dataset binding differs')
    if base['dataset_path']==original['dataset_path'] or base['dataset_sha256']==original['dataset_sha256']:
        raise ValueError('Old development dataset is not a fresh cohort')
    if base['source_sha256']!=p['base_q_source_sha256'] or base['source_sha256']!=original['source_sha256']:
        raise ValueError('Fresh source differs from fixed development')
    if base['public_inputs_sha256']!=p['public_inputs_sha256'] or equivalent_public_inputs(base,original,root=root)!=p['public_input_equivalence']:
        raise ValueError('Fresh public catalogue or native-archive equivalence differs')
    nonmethods=lambda c:{k:v for k,v in c.items() if k!='methods'}
    if (nonmethods(base['configuration']) != nonmethods(dp['fixed_configuration'])
            or base['configuration']['methods']['legacy_l10'] != dp['fixed_configuration']['methods']['legacy_l10']):
        raise ValueError('Fresh base-Q mechanism/public controls differ')
    check_pins(root,base_q,base['source_sha256'])
    for name,pin in base['public_inputs_sha256'].items():
        if sha(local(root,name)) != pin: raise ValueError('Public input changed')
    freeze=read(output/'depth_freeze.json'); base_freeze=read(base_q/'freeze.json')
    expected={'schema':'qplanner-selected-response-depth-freeze-v1','depth_protocol_sha256':digest,
        'selected_depth':p['selected_depth'],'baseline_depth':20,'criterion':CRITERION,'primary':PRIMARY,
        'base_q_protocol_sha256':p['base_q_protocol_sha256'],
        'development_selection_sha256':sha(development/'depth_selection.json'),
        'freeze_source_sha256':sha(local(root,THIS)),'fresh_depth_scores_viewed':False}
    if freeze != expected: raise ValueError('Pre-test depth freeze differs')
    expected_base={'schema':'qplanner-base-Q-for-selected-depth-freeze-v1',
        'fresh_protocol_sha256':p['base_q_protocol_sha256'],'configuration':base['configuration'],
        'fresh_test_evaluated':False,'depth_output':relative(root,output),
        'depth_protocol_sha256':digest,'depth_freeze_sha256':sha(output/'depth_freeze.json')}
    if base_freeze != expected_base: raise ValueError('Base-Q pre-generation depth freeze differs')
    if (base_q/'execution_protocol.json').exists():
        e=read(base_q/'execution_protocol.json')
        if (sha(base_q/'execution_protocol.json')!=(base_q/'execution_protocol.sha256').read_text().strip()
                or e['common_protocol_sha256']!=p['base_q_protocol_sha256']
                or e.get('paired_development') is not None
                or e['predeclared_files_sha256'].get('freeze.json')!=sha(base_q/'freeze.json')):
            raise ValueError('Fresh execution does not bind predeclared freeze/no old pairing')
        for name,pin in e['source_sha256'].items():
            if sha(local(root,name))!=pin or sha(local(base_q/'execution_source_snapshot',name))!=pin:
                raise ValueError('Fresh execution source changed')
            if name in base['source_sha256'] and pin!=base['source_sha256'][name]:
                raise ValueError('Execution common source differs')
    return {'status':'pass','depth_protocol_sha256':digest,'depth_freeze_sha256':sha(output/'depth_freeze.json'),
        'base_q_protocol_sha256':p['base_q_protocol_sha256'],'selected_depth':p['selected_depth'],
        'fresh_dataset_opened_for_contract':False,'fresh_metrics_opened_for_contract':False}


def base_q_receipt(output, *, root=ROOT):
    depth_contract(output,root=root); p=read(Path(output)/'protocol.json')
    base=local(root,p['base_q_output']); generation=read(base/'generation.json')
    receipt=read(base/'validation.json')
    if (receipt['status']!='pass' or receipt['protocol_sha256']!=p['base_q_protocol_sha256']
            or receipt['generation_sha256']!=sha(base/'generation.json')
            or receipt['verifier_sha256']!=p['source_sha256'][BASE_VERIFIER]
            or receipt.get('protected_tapes_verified') is not True
            or receipt.get('no_depth_scores_opened') is not True):
        raise ValueError('Full independent frozen base-Q verification required before depth scoring')
    validate_study_sources(base,read(base/'protocol.json'),generation,root=root)
    return base,generation,receipt


def replay_fresh(output,workdir,public_cache=None):
    """Only two frozen response variants, after full base-Q validation."""
    output=Path(output); base,generation,receipt=base_q_receipt(output)
    p=read(output/'protocol.json')
    save(output/'replay_started.json',{'depth_protocol_sha256':sha(output/'protocol.json'),
        'base_generation_sha256':sha(base/'generation.json'),'base_validation_sha256':sha(base/'validation.json'),
        'started_utc':datetime.now(timezone.utc).isoformat()})
    rn,evaluators,metadata=kernel.build_depth_resources(local(ROOT,p['dataset_path']),workdir,public_cache)
    if {k:metadata[k] for k in SERVICE_KEYS}!=p['service_kernel']:
        raise ValueError('Fresh public reply kernel/catalogue/prefix differs')
    evaluators={depth:evaluators[depth] for depth in p['depths']}
    save(output/'resources.json',metadata)
    data=read(local(ROOT,p['dataset_path']));families={f['family_id']:f for f in data['families']}
    if len([f for f in families.values() if f['split']=='test'])<p['minimum_fresh_test_families']:
        raise ValueError('Fresh family test cohort smaller than predeclared24')
    pins={};source_pins={};rows=[];wires=[]
    for name,pin in generation['family_files_sha256'].items():
        source=base/'families'/name
        if sha(source)!=pin:raise ValueError('Frozen base-Q bundle changed')
        bundle=read(source);value=kernel.replay_bundle(bundle,families[bundle['evaluator_only']['family_id']],rn,evaluators)
        value['source_bundle_sha256']=pin
        target=output/'families'/name;target.parent.mkdir(parents=True,exist_ok=True)
        kernel.common.compressed_save(target,value);pins[name]=sha(target);source_pins[name]=pin
        rows.extend(value['utility']);wires.extend(value['wire'])
    save(output/'readout.json',{'schema':'qplanner-response-depth-fresh-replay-v1',
        'protocol_sha256':sha(output/'protocol.json'),'resources_sha256':sha(output/'resources.json'),
        'base_generation_sha256':sha(base/'generation.json'),'base_validation_sha256':sha(base/'validation.json'),
        'family_files_sha256':pins,'source_family_files_sha256':source_pins,
        'L20_source_rows_exactly_reproduced':True,'static_event_recall_monotonicity_asserted':True,
        'no_private_generation':True,'only_selected_and_baseline_scored':True})
    return read(output/'readout.json')


def augmented_variant(block,source,depths):
    """Link exact variant clock inventory and private tapes to original Q."""
    if block['evaluator_only']!={k:source['evaluator_only'][k] for k in ('family_id','split','draw')}:
        raise ValueError('Depth family/draw identity differs')
    ledger=[{k:v for k,v in s['ledger']['legacy_l10'].items() if k!='step_ms'}
            for s in source['evaluator_only']['sessions']]
    expected={'Q_stream_sha256':kernel.canonical_sha(source['public']['streams']['legacy_l10']),
        'ledger_and_anchors_sha256':kernel.canonical_sha(ledger),
        'Q_not_regenerated':True,'private_reads_not_performed':True}
    if block['frozen_controls']!=expected:raise ValueError('Q/private-tape certificate differs')
    result=dict(block);public=dict(source['public'])
    public['streams']={f'service_l{depth}':source['public']['streams']['legacy_l10'] for depth in depths}
    result['public']=public
    inventory=declared_clock_inventory(result,tuple(public['streams']))
    event_ids={(slot,e['timestamp_s']):e['event_id'] for slot,session in enumerate(public['streams'][f'service_l{depths[0]}'])
               for e in session['events']}
    for key in ('utility','wire'):
        for row in block[key]:
            if (row['slot'],row['t']) not in inventory or row['event_id']!=event_ids[row['slot'],row['t']]:
                raise ValueError('Depth row clock/event does not match frozen Q')
    return result


def paired_depth_readout(output,*,scope='development'):
    """Independent row aggregation and whole-family bootstrap; no winner rotation."""
    output=Path(output);p=read(output/'protocol.json')
    if scope=='development':
        if p['schema']!='qplanner-response-depth-development-v1':
            raise ValueError('Development analysis may not open fresh metrics')
        kernel.validate_depth_study(output);base=artifact(ROOT,p['source_output'])
        depths=p['depths'];draw_counts=p['draws_by_split'];primary=None;criterion=None
        source_pins=p['family_files_sha256']
    elif scope=='independent-synthetic-generalization':
        base,generation,_=base_q_receipt(output);depths=p['depths'];draw_counts=p['draws_by_split']
        primary=f'service_l{p["selected_depth"]}';criterion=p['criterion'];source_pins=generation['family_files_sha256']
    else:raise ValueError('Explicit development or frozen independent-synthetic scope required')
    saved=read(output/'readout.json')
    if scope=='independent-synthetic-generalization' and (
            saved['base_generation_sha256']!=sha(base/'generation.json')
            or saved['base_validation_sha256']!=sha(base/'validation.json')):
        raise ValueError('Fresh replay receipt differs')
    if (saved['protocol_sha256']!=sha(output/'protocol.json')
            or saved['resources_sha256']!=sha(output/'resources.json')
            or set(saved['family_files_sha256'])!=set(source_pins)):
        raise ValueError('Depth readout source/inventory differs')
    declaration={'schema':'qplanner-response-depth-paired-protocol-v1','scope':scope,
        'source_protocol_sha256':sha(output/'protocol.json'),'source_readout_sha256':sha(output/'readout.json'),
        'analysis_source_sha256':sha(Path(__file__)),'family_bootstrap_replicates':REPLICATES,
        'public_analysis_seed':SEED,'depths':depths,'primary':PRIMARY if primary else None,
        'criterion':criterion,'unit':'whole family; retain all8sessions and all nested draws',
        'purposes':'four declared workloads, nearest/fastest correlated; not independent replications',
        'development_intervals':'all arms retained, exploratory/unadjusted; no defense selection by readout'}
    save(output/'paired_protocol.json',declaration)
    def bundles():
        for name,pin in saved['family_files_sha256'].items():
            path=output/'families'/name;source=base/'families'/name
            if sha(path)!=pin or sha(source)!=source_pins[name]:raise ValueError('Depth/source bundle changed')
            block=read(path)
            if block['source_bundle_sha256']!=source_pins[name]:raise ValueError('Depth source bundle binding differs')
            yield augmented_variant(block,read(source),depths)
    draws={s:list(range(1,n+1)) for s,n in draw_counts.items()};methods=[f'service_l{d}' for d in depths]
    cells,costs=summarize_bundles(bundles(),kernel.common.PURPOSES,draws,
        expected_methods=methods,expected_caches=CACHES,require_public_clock=True)
    contrasts,_=build_readout(cells,draws,methods,'service_l20',primary,'service_l20' if primary else None)
    values={f'{m}--{s}--{c}--{ph}':entries for (m,s,c,ph),entries in cells.items()}
    if scope=='development':
        for depth,split_cells in saved['summary'].items():
            for split,cache_cells in split_cells.items():
                for cache in CACHES:
                    for phase in ('all','cold','temporal_tail_400_600'):
                        for purpose,expected in cache_cells[cache][phase].items():
                            actual=cells[f'service_l{depth}',split,cache,phase][purpose]['family_values']
                            comparable={f:v for f,v in actual.items() if v is not None} if purpose=='equal_purpose_macro' else actual
                            kernel._assert_baseline(expected['family_values'],comparable)
    result={'schema':'qplanner-response-depth-paired-readout-v1','paired_protocol_sha256':sha(output/'paired_protocol.json'),
        'source_readout_sha256':sha(output/'readout.json'),'contrasts':contrasts,'conditional_family_cells':values,
        'family_draw_costs':[dict(method=m,split=s,family_id=f,draw=d,**cost) for (m,s,f,d),cost in sorted(costs.items())],
        'primary_criterion_result':primary_decision(contrasts,primary,'service_l20',criterion) if primary else None,
        'exact_Q_clock_ledger_certificate_checked':True,'no_private_generation':True,
        'privacy_scope':'sameQ/clock/K/anchors/reads/budget; deterministic static public replies givenQ and publicL; no newprivacy theorem or attackerexhaustiveness proof',
        'cost_scope':'publicL differs; full POImetadata reply and request compactJSON estimates; extra replybytes paid explicitly',
        'defense_selected_by_this_readout':False}
    save(output/'paired_readout.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('declare','contract','replay','readout','development-readout'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--development',type=Path);parser.add_argument('--base-q',type=Path)
    parser.add_argument('--workdir',type=Path);parser.add_argument('--public-cache',type=Path)
    args=parser.parse_args()
    if args.stage=='declare':
        if args.development is None or args.base_q is None:parser.error('--development and --base-q required')
        result=declare_freeze(args.development,args.base_q,args.output)
    elif args.stage=='contract':result=depth_contract(args.output)
    elif args.stage=='replay':
        if args.workdir is None:parser.error('--workdir required')
        result=replay_fresh(args.output,args.workdir,args.public_cache)
    else:result=paired_depth_readout(args.output,scope='development' if args.stage=='development-readout' else 'independent-synthetic-generalization')
    print(json.dumps({k:result[k] for k in ('schema','status','selected_depth','primary_criterion_result') if k in result},indent=2))


if __name__=='__main__':main()
