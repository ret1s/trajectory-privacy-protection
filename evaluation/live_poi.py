"""Versioned synthetic availability and a causal response cache.

Simulation seeds belong to the evaluator/server. A client receives only replies;
it never receives the full status mask through the response-cache interface.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

from evaluation.lane_travel import matrix


class RankedRoadPois:
    """Exact public road-distance order, including every reachable POI.

    No top-L truncation before applying availability. Stable lexical POI ties
    match PoiService. Coordinates must first pass through the server snap rule.
    """
    def __init__(self, service, cache_path=None):
        self.rn = service.rn
        self.pois = tuple(sorted(service.pois, key=lambda p: p['id']))
        self.categories = tuple(service.categories)
        self.n = len(self.pois)
        self.slices = []
        offset = 0
        for category in self.categories:
            count = sum(p['category'] == category for p in self.pois)
            self.slices.append(slice(offset, offset + count))
            offset += count
        metadata = {'schema': 'full-road-poi-ranking-v1',
                    'catalogue': self.rn.catalogue_sha256, 'pois': self.pois,
                    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        metadata = json.loads(json.dumps(metadata))
        path = Path(cache_path) if cache_path else None
        meta_path = path.with_suffix('.json') if path else None
        if path and path.exists():
            saved = json.loads(meta_path.read_text())
            assert saved['metadata'] == metadata, 'Stale ranking cache'
            assert saved['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            self.rank = np.load(path, mmap_mode='r', allow_pickle=False)
        else:
            self.rank = np.full((len(self.rn), self.n), -1, dtype='<i4')
            reverse = matrix(self.rn).transpose().tocsr()
            for category, section in zip(self.categories, self.slices):
                ids = np.array([i for i, p in enumerate(self.pois) if p['category'] == category])
                distances = np.column_stack([
                    dijkstra(reverse, directed=True, indices=self.pois[int(i)]['vertex']) for i in ids])
                order = np.argsort(distances, axis=1, kind='stable')
                ranked = ids[order].astype('<i4')
                ranked[~np.isfinite(np.take_along_axis(distances, order, axis=1))] = -1
                self.rank[:, section] = ranked
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.rank, allow_pickle=False)
                meta_path.write_text(json.dumps({'metadata': metadata,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2)+'\n')
        assert self.rank.shape == (len(self.rn), self.n)

    def top(self, state, available, k=5):
        if k < 1 or np.shape(available) != (self.n,):
            raise ValueError('Positive k and a complete status mask required')
        result = []
        for section in self.slices:
            ids = self.rank[int(state), section]
            ids = ids[ids >= 0]
            result.append(ids[np.asarray(available, dtype=bool)[ids]][:k].tolist())
        return result


class AvailabilityWorld:
    def __init__(self, size, seed, probability=.8, epoch_seconds=60):
        if not 0 <= probability <= 1 or epoch_seconds <= 0 or size < 1:
            raise ValueError('Invalid availability world')
        self.size, self.seed = int(size), int(seed)
        self.probability, self.epoch_seconds = float(probability), float(epoch_seconds)
        self.snapshots = {}

    def epoch(self, timestamp):
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Nonnegative finite timestamp required')
        return int(timestamp // self.epoch_seconds)

    def at_epoch(self, epoch):
        if epoch < 0 or int(epoch) != epoch:
            raise ValueError('Nonnegative integer epoch required')
        if epoch not in self.snapshots:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, int(epoch)])))
            mask = rng.random(self.size) < self.probability
            mask.setflags(write=False)
            self.snapshots[epoch] = mask
        return self.snapshots[epoch]


class LivePointService:
    """Server-only world, bounded response cache; client sees returned IDs only."""
    def __init__(self, ranking, world, response_l=10, cache_limit=16384):
        self.ranking, self.world, self.response_l = ranking, world, int(response_l)
        self.cache_limit, self.cache = cache_limit, OrderedDict()

    def query(self, state, epoch):
        key = int(state), int(epoch)
        if key not in self.cache:
            self.cache[key] = self.ranking.top(state, self.world.at_epoch(epoch), self.response_l)
            if len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


class EpochResponseCache:
    """Local postprocessing only: no true position and no server status input.

    Validity is a server contract: responses are current for one fixed epoch.
    No request suppression, prefetch, or request-coordinate choice is made here.
    """
    def __init__(self, size):
        self.epoch = None
        self.known = np.zeros(size, dtype=bool)

    def receive(self, epoch, replies):
        if self.epoch is not None and epoch < self.epoch:
            raise ValueError('Responses cannot travel backwards in time')
        if epoch != self.epoch:
            self.known[:] = False
            self.epoch = epoch
        current = np.zeros_like(self.known)
        for reply in replies:
            for category in reply:
                ids = np.asarray(category, dtype=int)
                if np.any(ids < 0) or np.any(ids >= len(current)):
                    raise ValueError('Unknown POI ID')
                current[ids] = True
        self.known |= current
        return current, self.known.copy()


def score_returned(reference, returned, availability):
    category = [len(set(a) & set(b))/len(a) if a else None for a, b in zip(reference, returned)]
    eligible = [v for v in category if v is not None]
    ids = [i for row in returned for i in row]
    return {'recall': float(np.mean(eligible)) if eligible else None,
            'category_recall': category, 'empty_reference_categories': len(category)-len(eligible),
            'returned_items': len(ids), 'unavailable_returned_items': sum(not availability[i] for i in ids)}
