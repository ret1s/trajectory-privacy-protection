"""
Geo-indistinguishability implementation module.
Provides mechanisms to apply differential privacy to geographical coordinates.
"""
import numpy as np
from math import cos, pi


class GeoIndistinguishability:
    """
    Implements geo-indistinguishability by adding calibrated noise to
    geographical coordinates to provide differential privacy guarantees.
    """

    def __init__(self, epsilon=0.1, delta=200):
        """
        Initialize the geo-indistinguishability mechanism.

        Args:
            epsilon: Privacy parameter (lower values provide stronger privacy)
            delta: Maximum distance in meters for QoS guarantee
        """
        self.epsilon = epsilon
        self.delta = delta  # Maximum allowed distance distortion in meters

    def _sample_from_polar(self):
        """
        Sample a point from the polar Laplacian distribution.

        Returns:
            A tuple (r, theta) where r is the radius and theta is the angle
        """
        # Sample from exponential distribution for radius
        r = np.random.exponential(scale=1.0 / self.epsilon)
        # Sample uniform angle
        theta = np.random.uniform(0, 2 * pi)

        return r, theta

    def add_noise(self, lat, lon):
        """
        Add calibrated noise to a geographical point.

        Args:
            lat: Latitude of the original point
            lon: Longitude of the original point

        Returns:
            A tuple (obfuscated_lat, obfuscated_lon) with noise added
        """
        # Sample from polar Laplacian
        r, theta = self._sample_from_polar()

        # Scale radius to meters - cap to ensure QoS
        r = min(r, self.delta)

        # Convert radius in meters to degrees (approximation)
        lat_degree_per_meter = 1 / 111111  # 1 degree ≈ 111111 meters for latitude
        # Longitude degrees per meter varies with latitude
        lon_degree_per_meter = 1 / (111111 * cos(lat * pi / 180))

        # Convert polar coordinates to Cartesian shift
        lat_shift = r * lat_degree_per_meter * np.cos(theta)
        lon_shift = r * lon_degree_per_meter * np.sin(theta)

        # Add noise
        obfuscated_lat = lat + lat_shift
        obfuscated_lon = lon + lon_shift

        return obfuscated_lat, obfuscated_lon
