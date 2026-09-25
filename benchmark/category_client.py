"""Online client for a frozen public category plan and versioned responses."""
from copy import deepcopy
import math

from evaluation.category_cover import reply_mask


class PublicCategoryClient:
    """No GPS argument: local ranking is outside the network-facing policy.

    Every session uses the same plan. Refreshing once per epoch is optional and
    depends only on the public clock and a server validity guarantee. This does
    not hide that clock, account identity, plan-region choice, or app lifetime.
    """
    def __init__(self, plan, poi_count, epoch_seconds=60., refresh_once_per_epoch=False):
        if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise ValueError('Positive finite epoch duration required')
        if not plan['queries'] or poi_count < 1:
            raise ValueError('Nonempty frozen plan and catalogue required')
        self.plan = deepcopy(plan)
        self.poi_count = poi_count
        self.epoch_seconds = epoch_seconds
        self.refresh_once_per_epoch = bool(refresh_once_per_epoch)
        self.last_timestamp = None
        self.epoch = None
        self.known = None

    def step(self, timestamp_s, query):
        if (not math.isfinite(timestamp_s) or timestamp_s < 0 or
                (self.last_timestamp is not None and timestamp_s <= self.last_timestamp)):
            raise ValueError('Strictly increasing finite nonnegative public time required')
        epoch = int(timestamp_s // self.epoch_seconds)
        requests, replies = [], []
        if not self.refresh_once_per_epoch or self.epoch != epoch:
            for q in self.plan['queries']:
                request = {'category_index': q['category_index'], 'category': q['category'],
                           'coordinate': tuple(q['coordinate']), 'epoch': epoch}
                replies.append(list(query(request)))
                requests.append(request)
            self.known = reply_mask(replies, self.poi_count)
        self.epoch = epoch
        self.last_timestamp = timestamp_s
        return {'requests': requests, 'replies': replies, 'known': self.known.copy(), 'epoch': epoch}
