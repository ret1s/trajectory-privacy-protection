"""BR-Dummy extension: fixed public lane states and unsnapped private inputs.

Inherits the existing noisy-reuse primitive and ledger, not a new DP theorem.
With raw planar x, the ideal full-support anchor kernel has the same triangle
inequality proof on continuous input space. Float sampling is approximate.
"""
from dataclasses import replace
import math

from benchmark.engines.budgeted import BudgetedReachableDummy
from evaluation.lane_travel import SparseTravel


class LaneBudgetedDummy(BudgetedReachableDummy):
    name='lane_budgeted_dummy'

    def __init__(self,rn,**kwargs):
        if rn.graph.graph.get('schema')!='sumo-lane-progress-v1':
            raise ValueError('A fixed public SUMO lane-state catalogue is required')
        super().__init__(rn,**kwargs)
        self.travel=SparseTravel(rn)

    def reset(self):
        super().reset()
        self.evaluator_states=[]

    def protect_step(self,lat,lon,timestamp_s):
        # Validate only when actually reading a private coordinate. Beyond H
        # the coordinate is deliberately ignored, not inspected for diagnostics.
        if self.n<self.horizon and (not math.isfinite(lat) or not math.isfinite(lon) or
                                   not -90<=lat<=90 or not -180<=lon<=180):
            raise ValueError('Finite valid private coordinate required within horizon')
        result=super().protect_step(lat,lon,timestamp_s)
        self.evaluator_states.append(list(self.previous))
        return result

    def protect_run(self,points):
        run=super().protect_run(points)
        params={**dict(run.transcript.public_parameters),'input_representation':'raw_GPS_in_fixed_local_projection',
                'motion_model':'directed_lane_progress_no_lane_changes_free_flow',
                'catalogue_sha256':self.rn.catalogue_sha256,
                'state_spacing_m':self.rn.graph.graph['spacing_m']}
        return replace(run,transcript=replace(run.transcript,public_parameters=params))
