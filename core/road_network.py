"""
Road-network wrapper used by the road-constrained privacy mechanisms.

Wraps an OSMnx graph as flat numpy arrays of candidate output locations (the
graph vertices) in a local planar coordinate frame (meters), plus a KD-tree
for nearest-candidate queries. All mechanisms and attacks operate on this
candidate set.
"""
import pickle

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_000.0


class LocalProjection:
    """Equirectangular projection around a reference latitude — accurate to
    well under 1m over a ~10km city box, and trivially invertible."""

    def __init__(self, lat0):
        self.lat0 = lat0
        self.m_per_deg_lat = np.pi / 180.0 * EARTH_RADIUS_M
        self.m_per_deg_lon = self.m_per_deg_lat * np.cos(np.radians(lat0))

    def to_xy(self, lat, lon):
        return (
            np.asarray(lon) * self.m_per_deg_lon,
            np.asarray(lat) * self.m_per_deg_lat,
        )

    def to_latlon(self, x, y):
        return (
            np.asarray(y) / self.m_per_deg_lat,
            np.asarray(x) / self.m_per_deg_lon,
        )


class RoadNetwork:
    """Candidate output set = vertices of an OSM road graph."""

    def __init__(self, graph):
        self.graph = graph
        node_ids, lats, lons = [], [], []
        for nid, data in graph.nodes(data=True):
            node_ids.append(nid)
            lats.append(data["y"])
            lons.append(data["x"])
        self.node_ids = np.array(node_ids)
        self.lats = np.array(lats)
        self.lons = np.array(lons)
        self.proj = LocalProjection(float(np.mean(self.lats)))
        self.xs, self.ys = self.proj.to_xy(self.lats, self.lons)
        self.xy = np.column_stack([self.xs, self.ys])
        self.tree = cKDTree(self.xy)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def __len__(self):
        return len(self.node_ids)

    def point_xy(self, lat, lon):
        x, y = self.proj.to_xy(lat, lon)
        return float(x), float(y)

    def nearest(self, lat, lon):
        """Index of the nearest candidate vertex to (lat, lon)."""
        d, idx = self.tree.query([*self.point_xy(lat, lon)])
        return int(idx), float(d)

    def within(self, lat, lon, radius_m):
        """Indices of candidate vertices within radius_m of (lat, lon)."""
        return np.array(
            self.tree.query_ball_point([*self.point_xy(lat, lon)], radius_m),
            dtype=int,
        )

    def latlon(self, idx):
        return float(self.lats[idx]), float(self.lons[idx])

    def dist_to_road(self, lat, lon):
        """Distance (m) from a point to the nearest candidate vertex — used
        as the on-road realism proxy."""
        return self.nearest(lat, lon)[1]
