"""Fixed public depth design; no private trajectory is consulted."""
import json
from benchmark.public_poi_context import PublicPoiContext
from benchmark.exact_public_reference import certified_goals
from evaluation.lane_travel import LanePoiService
from experiments import query_bundle_audit_20261010 as original

ROOT,NETWORK,RESOURCES,CONFIG=original.ROOT,original.NETWORK,original.RESOURCES,original.CONFIG
library=original.library
DESIGN='artifacts/benchmarks/public_depth_design_20261014_v1/certificate.json'


def resources(work):
    rn,reference,reply30,ids=original.resources(work)
    design=json.loads((ROOT/DESIGN).read_text())
    if design['selected_L']!=60 or design['exact_public_mean_floor']!='99/100':
        raise ValueError('Pinned PUBLIC depth design required')
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')} for p in json.loads((ROOT/RESOURCES).read_text())['pois_used']]
    reply=PublicPoiContext(LanePoiService(rn,pois,k=60),work/'reply60.npz')
    return rn,reference,reply,ids
