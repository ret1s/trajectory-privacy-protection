"""Deprecated import compatibility for the early thesis demo class."""

from benchmark.engines.proposed import AnchoredDummyBatch, GeoIAnchoredDummyEngine


class GeoIAnchoredDummyTrajectoriesLite(GeoIAnchoredDummyEngine):
    name = "geo_i_anchored_dummy_lite"
    demo_only = True

__all__ = ["AnchoredDummyBatch", "GeoIAnchoredDummyTrajectoriesLite"]
