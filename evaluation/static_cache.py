"""Static-service controls: no remote request is needed for a known catalogue.

This assumption audit does not apply to live inventory, private server records,
or changing traffic. The cache is built from PUBLIC context, never from a target's
truth or evaluation labels. Private coordinates are used only on the device.
"""
import numpy as np

class StaticPublicPoiCache:
    def __init__(self,context):
        self.context=context
        self.index={category:i for i,category in enumerate(context.categories)}

    def query(self,lat,lon,category):
        state,_=self.context.rn.nearest(lat,lon)
        # Match the service's nearest-state contract once, not a second snap.
        ids=self.context.signatures[state,self.index[category]]
        return [self.context.pois[int(i)]['id'] for i in ids if i>=0]

    @property
    def index_bytes(self):
        return self.context.signatures.nbytes+self.context.access.nbytes

    @property
    def remote_events(self):return ()
