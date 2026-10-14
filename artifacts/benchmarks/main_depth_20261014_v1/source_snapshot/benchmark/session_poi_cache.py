"""Static-metadata session cache; wire stays purpose-independent and unchanged.

Coordinates/category/location records must remain valid for the public session.
This does NOT extend validity of volatile opening/status/availability replies.
Restart/retry uses a trusted checkpoint; rollback invalidates the invariant.
"""
import copy
import math
import numpy as np
from benchmark.query_purpose import PurposeIndependentCoverClient


class SessionCoverClient(PurposeIndependentCoverClient):
    def __init__(self,*args,session_token,static_catalogue_id,**kwargs):
        super().__init__(*args,**kwargs)
        if not session_token or not static_catalogue_id:raise ValueError('Public session/catalogue context required')
        self.session_token,self.static_catalogue_id=session_token,static_catalogue_id
        self.session_known=np.zeros_like(self.cache.known)
        self.last_wire=None

    def step(self,timestamp_s,protected_coordinates,server):
        coordinates=tuple(tuple(c) for c in protected_coordinates)
        if self.last_t is not None and timestamp_s==self.last_t:
            if coordinates!=tuple(tuple(r['coordinate']) for r in self.last_wire['requests']):
                raise ValueError('Retry cannot change the published Q')
            return copy.deepcopy(self.last_wire)
        wire=super().step(timestamp_s,coordinates,server)
        self.session_known|=wire['current']
        wire['session']=self.session_known.copy()
        self.last_wire=copy.deepcopy(wire)
        return wire

    def checkpoint(self):
        wire=None if self.last_wire is None else {k:(v.tolist() if isinstance(v,np.ndarray) else copy.deepcopy(v)) for k,v in self.last_wire.items()}
        return dict(schema='static-session-cache-v1',session_token=self.session_token,
            static_catalogue_id=self.static_catalogue_id,categories=list(self.categories),
            k=self.k,response_l=self.response_l,epoch_seconds=self.epoch_seconds,
            size=len(self.session_known),last_t=self.last_t,epoch=self.cache.epoch,
            session=self.session_known.tolist(),known=self.cache.known.tolist(),last_wire=wire)

    @classmethod
    def restore(cls,payload,*,session_token,static_catalogue_id):
        p=copy.deepcopy(payload)
        if p['schema']!='static-session-cache-v1' or p['session_token']!=session_token or p['static_catalogue_id']!=static_catalogue_id:
            raise ValueError('Immutable public context mismatch')
        client=cls(p['categories'],p['size'],k=p['k'],response_l=p['response_l'],epoch_seconds=p['epoch_seconds'],session_token=session_token,static_catalogue_id=static_catalogue_id)
        for name in ('session','known'):
            if len(p[name])!=p['size'] or any(type(v) is not bool for v in p[name]):raise ValueError('Boolean POI masks required')
        client.session_known=np.array(p['session'],dtype=bool);client.cache.known=np.array(p['known'],dtype=bool)
        if np.any(client.cache.known&~client.session_known):raise ValueError('Epoch cache must be contained in session cache')
        client.last_t=p['last_t'];client.cache.epoch=p['epoch'];client.last_wire=p['last_wire']
        if client.last_t is None:
            if client.cache.epoch is not None or client.last_wire is not None or client.session_known.any():raise ValueError('Invalid unstarted checkpoint')
        else:
            if not math.isfinite(client.last_t) or client.last_t<0 or client.cache.epoch!=int(client.last_t//client.epoch_seconds) or client.last_wire is None:raise ValueError('Invalid public clock checkpoint')
            for name in ('current','known','session'):
                a=client.last_wire[name]
                if len(a)!=p['size'] or any(type(v) is not bool for v in a):raise ValueError('Boolean saved wire mask required')
                client.last_wire[name]=np.array(a,dtype=bool)
            if not np.array_equal(client.last_wire['session'],client.session_known) or not np.array_equal(client.last_wire['known'],client.cache.known):raise ValueError('Checkpoint masks disagree')
        return client
