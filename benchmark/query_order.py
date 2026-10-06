"""Remove explicit Q slot-order labels without changing the protected Q set.

Coordinates can still be associated geometrically across time. This is public
postprocessing of the Geo-I result, not identity anonymity or a new primitive.
"""
import numpy as np

from benchmark.query_purpose import PurposeIndependentCoverClient


class PrivateOrderCoverClient(PurposeIndependentCoverClient):
    """Network requests expose no persistent Q ID or stable internal slot order.

    A private fresh RNG is used by default. A supplied Generator is intended
    for controlled experiments or a private session stream, never public seeds
    in production. Internal Q motion state stays in its original order.
    """
    def __init__(self, categories, poi_count, *, rng=None, **kwargs):
        super().__init__(categories,poi_count,**kwargs)
        if rng is not None and not isinstance(rng,np.random.Generator):
            raise ValueError('Private NumPy Generator required when supplied')
        self._order_rng = np.random.default_rng() if rng is None else rng

    def step(self,timestamp_s,protected_coordinates,server):
        coordinates = tuple(tuple(c) for c in protected_coordinates)
        order = self._order_rng.permutation(len(coordinates))
        published = tuple(coordinates[int(i)] for i in order)
        return super().step(timestamp_s,published,server)
