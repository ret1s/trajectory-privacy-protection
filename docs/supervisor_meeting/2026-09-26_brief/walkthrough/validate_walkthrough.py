"""Independent accounting checks and noninterference checks for the teaching run."""
from pathlib import Path
import copy
import hashlib
import json
import math
import subprocess
import sys
import numpy as np
import fitz

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
import build_walkthrough as build
from data.lane_states import build_lane_states
from evaluation.lane_travel import SparseTravel


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=json.loads((OUT/'walkthrough.json').read_text())
    for name,h in d['source_hashes'].items():assert sha(ROOT/name)==h,name
    data=json.loads(build.DATA.read_text());sid=d['session_id'];trace=data['traces'][sid]
    assert sid==sorted(data['traces'])[0]
    assert d['true_trace']==[{'t':p['time_s']-trace[0]['time_s'],'lat':p['lat'],
                 'lon':p['lon'],'speed_m_s':p['speed_m_s']} for p in trace]
    rn=build_lane_states(OUT/'demo.net.xml')
    assert rn.catalogue_sha256==d['graph']['catalogue_sha256']
    assert sha(OUT/'demo.net.xml')==d['graph']['net_sha256']
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')}
          for p in json.loads(build.POIS.read_text())['pois_used']]
    work=Path('/private/tmp/trajectory-walkthrough')/rn.catalogue_sha256[:12]
    work.mkdir(parents=True,exist_ok=True)
    service=build.LanePoiService(rn,pois,k=5)
    context=build.PublicPoiContext(service,work/'demo-poi5.npz')
    reply=build.PublicPoiContext(build.LanePoiService(rn,pois,k=10),work/'demo-poi10.npz')
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,
                          return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    base=build.PublicAnchorModel(rn,context,prior,cache_path=work/'demo-belief.npz')
    belief=build.ResponseAwareAnchorModel(base,reply)
    ranking=build.RankedRoadPois(service,work/'demo-ranking.npy')
    travel=SparseTravel(rn)
    assert list(ranking.pois)==d['pois']
    checks=0
    for name,run in d['runs'].items():
        spending=0.;sent=[]
        for row in run['rows']:
            expected_cost=0. if row['head_skipped'] or not row.get('private_read') else (
                .01 if row['old_anchor'] is None or row['branch']=='reuse' else .02)
            spending+=expected_cost
            assert math.isclose(row.get('cost',0.),expected_cost,abs_tol=1e-12)
            assert math.isclose(row['spent'],spending,abs_tol=1e-12)
            assert math.isclose(row['remaining'],.23-spending,abs_tol=1e-12)
            if row.get('distance_m') is not None:
                assert math.isclose(row['noisy_distance_m'],row['distance_m']+row['test_noise_m'])
                assert (row['noisy_distance_m']<=200)==(row['branch']=='reuse')
            if not row['head_skipped']:
                assert len(row['queries'])==5
                assert abs(sum(row['region_weights'])-1)<1e-9
                if 'objective_loss' in row['objective']:
                    o=row['objective'];assert math.isclose(o['objective_loss'],
                        o['objective_before_slack']-o['objective_after_slack'],abs_tol=1e-12)
                    assert o['objective_loss']<=.03+1e-12
                for j,s in enumerate(row['states']):
                    if row['previous_states']:
                        assert s in travel.reachable(row['previous_states'][j],row['t']-row['previous_query_t'])
            score=[]
            for ref,got in zip(row['service']['reference'],row['service']['returned']):
                assert len(got)<=5
                if ref:score.append(len(set(ref)&set(got))/len(ref))
            expected=sum(score)/len(score) if score else None
            assert row['service']['recall']==expected
            assert all(len(cat)<=10 for reply in row['service']['replies'] for cat in reply)
            sent+=row['released_source_times'];checks+=1
        assert math.isclose(run['budget_spent'],spending,abs_tol=1e-12)
        assert len(sent)==run['accounting']['released_events']
        mean=sum(r['service']['recall'] for r in run['rows'])/len(run['rows'])
        assert math.isclose(mean,run['recall_all_input_times'],abs_tol=1e-12)
        for p in run['publications']:
            src=next(r for r in run['rows'] if r['t']==p['source_t'])
            assert p['coordinates']==src['queries']
            assert p['publication_t']-p['source_t']>=run['policy']['delay_s']
        for e in run['public_transcript']['events']:
            assert set(e)=={'event_id','timestamp_s','candidates'}
            assert all(set(c)=={'candidate_id','lat','lon'} for c in e['candidates'])
        for row in run['rows']:
            if row.get('tail_cancelled'):assert row['t']+run['policy']['delay_s']>d['duration_s']
        assert run['accounting']['protected_events']==len(sent)+run['accounting']['tail_cancelled']
    # Temporary copies only. Alter the head and a skipped private-read input;
    # the source dataset remains byte-identical and no changed sample is published.
    changed=copy.deepcopy(trace)
    for point in changed:
        t=point['time_s']-changed[0]['time_s']
        if t<60 or t==80:
            point['lat'],point['lon']=39.995,116.305
    other=build.run(rn,belief,ranking,changed,60.,60.)
    original=d['runs']['boundary']
    for left,right in zip(original['rows'],other['rows']):
        for key in ('anchor','queries','states','region_weights','spent','cost','branch'):
            assert json.dumps(left.get(key))==json.dumps(right.get(key)),key
    assert other['public_transcript']==original['public_transcript']
    for name,h in d['source_hashes'].items():assert sha(ROOT/name)==h,name
    pdf=fitz.open(OUT/'walkthrough.pdf');assert len(pdf)==4
    for page in pdf:
        for word in page.get_text('words'):
            assert word[0]>=0 and word[1]>=0 and word[2]<=page.rect.width+.1 and word[3]<=page.rect.height+.1,word
    fonts={f[0] for p in pdf for f in p.get_fonts()};assert all(pdf.extract_font(f)[3] for f in fonts)
    assert not any('S1.A' in p.get_text() or 'A/B/C' in p.get_text() for p in pdf)
    import sumo
    result=subprocess.run([str(Path(sumo.SUMO_HOME)/'bin/netconvert'),'--sumo-net-file',
        str(OUT/'demo.net.xml'),'--output-file','/private/tmp/trajectory-walkthrough/validated.net.xml',
        '--no-warnings','true'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    public={name:run['public_transcript'] for name,run in d['runs'].items()}
    (OUT/'public_transcript.json').write_text(json.dumps(public,indent=2)+'\n')
    subprocess.run([sys.executable,str(OUT/'validate_explorer.py')],check=True)
    validation={'status':'passed','rows_checked':checks,'source_hashes_unchanged':True,
        'selection_fixed_before_scores':True,'original_gps_unchanged':True,
        'budget_recomputed':True,'belief_normalization_checked':True,
        'slack_loss_recomputed':True,'directed_reachability_checked_in_demo_graph':True,
        'service_recall_recomputed_from_ids':True,'boundary_queue_accounting_checked':True,
        'head_and_skipped_gps_noninterference_passed':True,
        'public_transcript_has_no_truth_or_internal_states':True,
        'sumo_network_import_checked':True,'original_benchmark_map_equivalence':False,
        'privacy_attacks_run':False,'pdf_pages':4,'embedded_fonts':len(fonts),
        'pdf_visual_review':'All four slides inspected; map legends and budget annotation repositioned.',
        'browser_visual_review':'Unavailable: no browser enabled in computer-use tool.',
        'explorer_logic':json.loads((OUT/'explorer_validation.json').read_text()),
        'outputs':{n:sha(OUT/n) for n in ['walkthrough.json','walkthrough.pdf','walkthrough.html','demo.net.xml','public_transcript.json']}}
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','rows':checks,'noninterference':True,'sumo_import':True,'pdf_pages':4}))


if __name__=='__main__':main()
