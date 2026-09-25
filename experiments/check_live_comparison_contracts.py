"""Prefix/future invariance and coordinate-snap audit before interpretation."""
import json
import numpy as np
from core.demo_protocol import TrajectoryPoint
from benchmark.paper_comparators import generate
from experiments.run_live_paper_comparison import protocol,resources,data_path,OUT,write_json
from experiments.research_loop_category_confirmation import sources
from experiments.research_loop_resources import sha


def main():
    protocol();rn,_,history,shared,_=resources()
    data=json.loads(data_path('auxiliary').read_text());ss,_=sources(data,rn)
    audits=[]
    for source in ss[:3]:
        p=tuple(TrajectoryPoint(t,*rn.latlon(s)) for t,s in zip(source['timestamps_s'],source['reference_states']))
        n=max(2,len(p)//2);seed=8021
        changed=p[:n]+tuple(TrajectoryPoint(v.timestamp_s,*rn.latlon((i*151)%len(rn))) for i,v in enumerate(p[n:],1))
        for method in ('dls','rdg','transprotect_markov','semantic_poi','fake_queries'):
            whole=generate(method,p,rn,history,shared,seed)
            prefix=generate(method,p[:n],rn,history,shared,seed)
            altered=generate(method,changed,rn,history,shared,seed)
            cutoff=p[n-1].timestamp_s
            a=[e for e in whole['events'] if e['timestamp_s']<=cutoff]
            b=[e for e in altered['events'] if e['timestamp_s']<=cutoff]
            assert a==prefix['events']==b,(method,source['session_id'])
            audits.append({'method':method,'session':source['session_id'],'prefix_equal':True,'future_perturbation_equal':True})
    snap={}
    for split in ('development','new_groups'):
        data=json.loads(data_path(split).read_text());ss,_=sources(data,rn)
        before=[s for x in ss for s in x['reference_states']]
        after=[rn.nearest(*rn.latlon(s))[0] for s in before]
        snap[split]={'service_events':len(before),'state_changed_by_coordinate_roundtrip':sum(a!=b for a,b in zip(before,after)),
                     'scope':'Representation ambiguity is shared by all methods; raw control uses the same snapped input. Service target retains original nearest-FCD state.'}
    write_json(OUT/'contract_checks.json',{'status':'passed','protocol_sha256':sha(OUT/'protocol.json'),'prefix_checks':audits,
        'anotherme':'Offline full-trip reference; no online prefix-invariance claim','coordinate_snap_audit':snap})
    print('Passed',len(audits),'causal contracts; snap audit',snap,flush=True)

if __name__=='__main__':main()
