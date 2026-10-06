"""Independent arithmetic and public/private separation checks for S7 evidence."""
from pathlib import Path
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    raw=Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_mean(rows,L,purpose=None,split='test'):
    values=defaultdict(list)
    for r in rows:
        if r['split']==split and r['L']==L and r['recall'] is not None and (purpose is None or r['purpose']==purpose):
            values[r['family_id']].append(r['recall'])
    means=[sum(v)/len(v) for v in values.values()]
    return sum(means)/len(means) if means else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,
        default=ROOT/'artifacts/benchmarks/query_purpose_20261005')
    args=parser.parse_args();checked=0
    for folder in ('snapshot','sequence'):
        base=args.out/folder;protocol=read(base/'protocol.json');result=read(base/'readout.json')
        for name,h in protocol['source_sha256'].items():assert sha(ROOT/name)==h,name
        rows=read(base/'utility_rows.json.gz');attack=read(base/'attack_rows.json.gz')
        splitsets={s:{f for f,role in protocol['split_by_family'].items() if role==s}
                   for s in ('fit','selection','test')}
        assert all(not splitsets[a]&splitsets[b] for a,b in [('fit','selection'),('fit','test'),('selection','test')])
        for row in rows:
            assert len(row['reference'])==len(set(row['reference']))<=5
            assert len(row['returned'])==len(set(row['returned']))<=5
            expected=len(set(row['reference'])&set(row['returned']))/len(row['reference']) if row['reference'] else None
            assert row['recall']==expected
            assert row['family_id'] in splitsets[row['split']]
            checked+=1
        selection=result['selection'] if folder=='snapshot' else {'selected_L':result['selected_L']}
        L=selection['selected_L']
        actual=result['test_recall_all']['value'] if folder=='snapshot' else result['test_recall']['value']
        assert math.isclose(family_mean(rows,L),actual,abs_tol=1e-12)
        streams=read(base/'public_streams.json.gz')
        for stream in streams:
            assert 0<=stream['spent_bound']<=.23+1e-12
            for event in stream['events']:
                assert set(event)=={'timestamp_s','coordinates'}
                assert len(event['coordinates'])==5
        if folder=='snapshot':
            depths=selection['all_development_depths']
            for item in depths:
                assert math.isclose(family_mean(rows,item['L'],split='selection'),item['selection_recall']['value'])
                for purpose,v in item['by_purpose'].items():
                    assert math.isclose(family_mean(rows,item['L'],purpose,split='selection'),v)
            expected=next((d['L'] for d in depths if d['passed']),protocol['response_depth_grid'][-1])
            assert expected==L
            groups=defaultdict(list)
            for r in attack:
                if r['method']=='full_cover':groups[(r['record_id'],r['time_s'])].append(r)
            assert all(len({tuple(r['features']) for r in g})==1 for g in groups.values())
        else:
            depths=result['selection_depths']
            for item in depths:assert math.isclose(family_mean(rows,item['L'],split='selection'),item['value'])
            assert L==next((d['L'] for d in depths if d['value']>=.90-1e-12),protocol['depths'][-1])
            # Intent-counterfactual streams have identical coordinates, including all prefixes.
            groups=defaultdict(list)
            for stream in streams:groups[stream['session_id_evaluator_only']].append(stream['events'])
            assert all(all(g==group[0] for g in group) for group in groups.values())
            groups=defaultdict(list)
            for r in attack:
                if r['method']=='full_cover':groups[(r['family_id'],r['prefix_time_s'],r['L'])].append(r)
            assert all(len({tuple(r['features']) for r in group})==1 for group in groups.values())
            assert all(a['L']==L for a in result['intent_attacks_by_public_prefix'])
    comparison=read(args.out/'purpose_comparison.json')
    for name,h in comparison['source_sha256'].items():assert sha(ROOT/name)==h,name
    snapshot=read(args.out/'snapshot/utility_rows.json.gz')
    near={(r['record_id'],r['time_s'],r['L'],r['category']):r['returned']
          for r in snapshot if r['purpose']=='nearest_distance'}
    detour=defaultdict(list)
    for r in snapshot:
        if r['split']=='test' and r['L']==comparison['L'] and r['purpose']=='minimum_detour' and r['reference']:
            got=near[(r['record_id'],r['time_s'],r['L'],r['category'])]
            detour[r['family_id']].append((len(set(got)&set(r['reference']))/len(r['reference']),r['recall']))
    means=[(sum(v[0] for v in vs)/len(vs),sum(v[1] for v in vs)/len(vs)) for vs in detour.values()]
    metric=comparison['purpose_results']['minimum_detour']
    assert math.isclose(sum(v[0] for v in means)/len(means),metric['distance_only_recall'])
    assert math.isclose(sum(v[1] for v in means)/len(means),metric['purpose_reranking_recall'])
    files=[p for p in args.out.rglob('*') if p.is_file() and p.name!='validation.json']
    validation={'status':'passed','utility_rows_independently_recomputed':checked,
        'source_hashes_unchanged':True,'family_splits_disjoint':True,
        'selection_recomputed_without_test':True,'null_empty_references_preserved':True,
        'public_stream_fields_whitelisted':True,'budget_cap_checked':True,
        'selected_depth_attacker_alignment_checked':True,
        'paired_private_intents_share_public_prefixes':True,
        'paired_distance_vs_purpose_ranking_recomputed':True,
        'outputs':{str(p.relative_to(args.out)):sha(p) for p in files}}
    (args.out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='outputs'}))


if __name__=='__main__':main()
