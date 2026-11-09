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
        """Distance (m) from a point to the nearest candidate VERTEX. This is a
        vertex proximity, NOT true on-road distance — a point mid-segment on a
        long edge is far from any vertex yet on the road (verifier R4-003). Use
        `dist_to_edge` for the on-road metric; kept only for candidate lookups."""
        return self.nearest(lat, lon)[1]

    def _build_edge_index(self):
        """Lazily build a projected point-to-edge index (verifier R4-003): every
        graph edge as a projected LineString (using its OSM `geometry` polyline
        when present, else the straight segment between endpoints) in an STRtree,
        so `dist_to_edge` measures true distance to the road, not to a vertex."""
        from shapely.geometry import LineString
        from shapely.strtree import STRtree

        geoms = []
        for u, v, data in self.graph.edges(data=True):
            geom = data.get("geometry")
            if geom is not None:
                lonlat = list(geom.coords)  # [(lon, lat), ...]
                lons = np.array([c[0] for c in lonlat])
                lats = np.array([c[1] for c in lonlat])
            else:
                lons = np.array([self.graph.nodes[u]["x"], self.graph.nodes[v]["x"]])
                lats = np.array([self.graph.nodes[u]["y"], self.graph.nodes[v]["y"]])
            xs, ys = self.proj.to_xy(lats, lons)
            coords = list(zip(np.atleast_1d(xs).tolist(), np.atleast_1d(ys).tolist()))
            if len(coords) >= 2:
                geoms.append(LineString(coords))
        self._edge_geoms = geoms
        self._edge_tree = STRtree(geoms)

    def dist_to_edge(self, lat, lon):
        """True projected distance (m) from a point to the nearest road EDGE."""
        from shapely.geometry import Point

        if not hasattr(self, "_edge_tree"):
            self._build_edge_index()
        x, y = self.point_xy(lat, lon)
        p = Point(x, y)
        idx = int(self._edge_tree.nearest(p))
        return float(self._edge_geoms[idx].distance(p))
