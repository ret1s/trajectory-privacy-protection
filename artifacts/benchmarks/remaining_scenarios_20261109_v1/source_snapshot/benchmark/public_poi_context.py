"""Offline public POI context. No trajectory, scenario label or private query.

Reverse shortest paths build a top-k POI signature at every lane state. Query
access follows the existing service's coordinate-to-state rule, including its
known lane-direction ambiguity. This is a public proxy, not access to LSP state.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

from evaluation.lane_travel import matrix


class PublicPoiContext:
    def __init__(self, service, cache_path=None):
        self.rn = service.rn
        self.categories = tuple(service.categories)
        self.pois = tuple(sorted(service.pois, key=lambda p: p['id']))
        self.k = service.k
        self.metadata = {
            'schema': 'public-poi-context-v1', 'catalogue_sha256': self.rn.catalogue_sha256,
            'pois': self.pois, 'categories': self.categories, 'k': self.k,
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.metadata_json = json.dumps(self.metadata, sort_keys=True, separators=(',', ':'))
        cache = Path(cache_path) if cache_path else None
        if cache and cache.exists():
            with np.load(cache, allow_pickle=False) as arrays:
                if str(arrays['metadata']) != self.metadata_json:
                    raise ValueError('Stale public context: use a fresh cache path')
                self.signatures = arrays['signatures'].copy()
                self.access = arrays['access'].copy()
        else:
            self.signatures = self._build(service)
            self.access = self.rn.tree.query(self.rn.xy)[1].astype('<i4')
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, metadata=self.metadata_json,
                                    signatures=self.signatures, access=self.access)
        if self.signatures.shape != (len(self.rn), len(self.categories), self.k):
            raise ValueError('Invalid public POI signature shape')
        if self.access.shape != (len(self.rn),) or np.any(self.access < 0) or np.any(self.access >= len(self.rn)):
            raise ValueError('Invalid public query access mapping')
        if np.any(self.signatures < -1) or np.any(self.signatures >= len(self.pois)):
            raise ValueError('Invalid POI index')
        digest = hashlib.sha256(self.metadata_json.encode())
        digest.update(self.signatures.astype('<i4').tobytes())
        digest.update(self.access.astype('<i4').tobytes())
        self.sha256 = digest.hexdigest()

    def _build(self, service):
        reverse = matrix(self.rn).transpose().tocsr()
        n, c, k = len(self.rn), len(self.categories), self.k
        signatures = np.full((n, c, k), -1, dtype='<i4')
        best = np.full((n, c, k), np.inf)
        for index, poi in enumerate(self.pois):
            category = self.categories.index(poi['category'])
            dist = dijkstra(reverse, directed=True, indices=poi['vertex'])
            # POIs arrive in lexical ID order; stable sort preserves that tie rule.
            values = np.c_[best[:, category], dist]
            ids = np.c_[signatures[:, category], np.full(n, index, dtype='<i4')]
            order = np.argsort(values, axis=1, kind='stable')[:, :k]
            best[:, category] = np.take_along_axis(values, order, axis=1)
            signatures[:, category] = np.take_along_axis(ids, order, axis=1)
            signatures[:, category][~np.isfinite(best[:, category])] = -1
        return signatures

    def query_indices(self, state):
        return self.signatures[self.access[int(state)]]

    def reference_weights(self, anchor):
        """Macro category recall weights at a protected anchor, NOT real GPS."""
        state, _ = self.rn.nearest(*anchor)
        references = self.signatures[state]
        nonempty = sum(np.any(row >= 0) for row in references)
        weights = np.zeros(len(self.pois) + 1)
        for row in references:
            valid = row[row >= 0]
            if len(valid):
                weights[valid] = 1. / (nonempty * len(valid))
        return weights

    def marginal_gain(self, states, weights, selected):
        remaining = weights.copy()
        for state in selected:
            ids = self.query_indices(state)
            remaining[ids[ids >= 0]] = 0.
        ids = self.signatures[self.access[np.asarray(states, dtype=int)]]
        # The extra weight is zero and represents missing POI slots.
        return remaining[np.where(ids >= 0, ids, len(self.pois))].sum(axis=(1, 2))
