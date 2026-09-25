"""Public calendar refresh, independent of trip boundaries and local POI use.

The interval must be subscribed before private activity and remain fixed even
when no trip/query occurs. This is not an on-demand padding promise: activation,
early cancellation, region changes, clicks and failures are outside the claim.
"""
from copy import deepcopy
import math

from evaluation.category_cover import reply_mask


class ScheduledCategoryClient:
    def __init__(self, plan, poi_count, start_s=0, end_s=3600, epoch_s=60):
        if (not all(math.isfinite(x) for x in (start_s,end_s,epoch_s)) or
                start_s < 0 or epoch_s <= 0 or end_s <= start_s or
                start_s % epoch_s or end_s % epoch_s):
            raise ValueError('Public interval must contain whole server epochs')
        if not plan['queries'] or poi_count < 1:
            raise ValueError('Frozen nonempty public plan and catalogue required')
        self.plan=deepcopy(plan);self.poi_count=poi_count
        self.start_s,self.end_s,self.epoch_s=start_s,end_s,epoch_s
        self.next_tick=start_s;self.cached_epoch=None;self.known=None

    def tick(self, timestamp_s, query):
        """Scheduler-only API: cannot receive GPS, activity or endpoint labels.

        A missed tick is an error, not a fictitious backdated refresh. The query
        callback receives only public category/coordinate/epoch information.
        """
        if timestamp_s != self.next_tick or timestamp_s >= self.end_s:
            raise ValueError('Execute the next public tick at its actual deadline')
        epoch=int(timestamp_s//self.epoch_s);requests=[];replies=[]
        for q in self.plan['queries']:
            request={'category_index':q['category_index'],'coordinate':tuple(q['coordinate']),
                     'epoch':epoch,'time_s':timestamp_s}
            requests.append(request);replies.append(list(query(deepcopy(request))))
        known=reply_mask(replies,self.poi_count)
        self.known=known;self.cached_epoch=epoch;self.next_tick+=self.epoch_s
        return {'timestamp_s':timestamp_s,'requests':requests,'replies':replies,'epoch':epoch}

    def local_snapshot(self, timestamp_s):
        """Read-only local service. A read never emits or reschedules a request."""
        if (not math.isfinite(timestamp_s) or timestamp_s < self.start_s or
                timestamp_s >= self.end_s or self.cached_epoch != int(timestamp_s//self.epoch_s)):
            raise ValueError('No valid response in the public service interval')
        return self.known.copy()
