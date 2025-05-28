"""
Trajectory privacy module for generating realistic fake trajectories.
"""
import numpy as np
import osmnx as ox
import folium
import pandas as pd
from shapely.geometry import Point, LineString
import geopandas as gpd
from sklearn.metrics import pairwise_distances
import haversine
from math import radians
import networkx as nx
from geo_indistinguishability import GeoIndistinguishability


class TrajectoryPrivacy:
    """
    Implementation of trajectory privacy mechanism that generates
    realistic privacy-preserving fake trajectories.
    """

    def __init__(self, epsilon=0.1, qos_radius=200):
        """
        Initialize the trajectory privacy mechanism.

        Args:
            epsilon: Privacy parameter (lower = stronger privacy)
            qos_radius: Maximum allowed distance in meters between real and fake points
        """
        self.geo_indist = GeoIndistinguishability(epsilon, qos_radius)
        self.qos_radius = qos_radius
        self.real_trajectory = []
        self.fake_trajectory = []
        self.graph = None
        self.spatial_constraints = None

    def load_spatial_constraints(self, bounds, excluded_types=None):
        """
        Load spatial constraints from OpenStreetMap.

        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            excluded_types: Types of areas to exclude (buildings, water, etc.)
        """
        if excluded_types is None:
            excluded_types = ['building', 'water', 'river']

        # Get the street network graph for the area
        self.graph = ox.graph_from_bbox(
            bounds[2], bounds[0], bounds[3], bounds[1],
            network_type='drive'
        )

        # Get buildings, water bodies, and other excluded areas
        # Using the updated OSMnx API for version 1.1.2+
        gdf_list = []
        for feature_type in excluded_types:
            try:
                # Create a feature query for each type
                tags = {feature_type: True}
                gdf = ox.features.features_from_bbox(
                    bounds[2], bounds[0], bounds[3], bounds[1],
                    tags=tags
                )
                if not gdf.empty:
                    gdf_list.append(gdf)
            except Exception as e:
                print(f"Warning: Could not fetch {feature_type} features: {e}")

        # Combine all GeoDataFrames if any were returned
        if gdf_list:
            self.spatial_constraints = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        else:
            self.spatial_constraints = None
            print("Warning: No spatial constraints were loaded. All locations will be considered valid.")

    def is_valid_location(self, lat, lon):
        """
        Check if a location is valid (not in buildings, water, etc.)

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Boolean indicating if location is valid
        """
        if self.spatial_constraints is None:
            return True

        point = Point(lon, lat)  # Note: GeoJSON uses (lon, lat) order

        # Check if point is in any excluded area
        for _, area in self.spatial_constraints.iterrows():
            if area.geometry.contains(point):
                return False

        # Check proximity to road network
        nearest_edge = ox.distance.nearest_edges(
            self.graph, lon, lat, return_dist=True
        )

        # If too far from road network, consider invalid
        if nearest_edge[2] > 30:  # 30 meters threshold
            return False

        return True

    def add_point(self, lat, lon):
        """
        Add a real point to the trajectory and generate corresponding fake point.

        Args:
            lat: Latitude of real point
            lon: Longitude of real point

        Returns:
            Tuple of (fake_lat, fake_lon)
        """
        # Store real point
        self.real_trajectory.append((lat, lon))

        # Apply geo-indistinguishability mechanism to get initial fake point
        fake_lat, fake_lon = self.geo_indist.add_noise(lat, lon)

        # Ensure fake point is valid (not in building, water, etc.)
        attempts = 0
        max_attempts = 10

        while (not self.is_valid_location(fake_lat, fake_lon)) and attempts < max_attempts:
            fake_lat, fake_lon = self.geo_indist.add_noise(lat, lon)
            attempts += 1

        # Ensure continuity with previous fake points
        if len(self.fake_trajectory) > 0:
            fake_lat, fake_lon = self._ensure_trajectory_continuity(fake_lat, fake_lon)

        # Store fake point
        self.fake_trajectory.append((fake_lat, fake_lon))

        return fake_lat, fake_lon

    def _ensure_trajectory_continuity(self, lat, lon):
        """
        Adjust a fake point to ensure trajectory continuity.

        Args:
            lat: Latitude of proposed fake point
            lon: Longitude of proposed fake point

        Returns:
            Adjusted (lat, lon)
        """
        if len(self.fake_trajectory) < 2:
            return lat, lon

        # Get last two points in fake trajectory
        prev_lat, prev_lon = self.fake_trajectory[-1]
        prev2_lat, prev2_lon = self.fake_trajectory[-2]

        # Calculate expected direction
        expected_dlat = prev_lat - prev2_lat
        expected_dlon = prev_lon - prev2_lon

        # Mix original and expected direction (smoothing)
        mix_ratio = 0.7  # 70% influence from expected direction

        adjusted_lat = prev_lat + mix_ratio * expected_dlat + (1 - mix_ratio) * (lat - prev_lat)
        adjusted_lon = prev_lon + mix_ratio * expected_dlon + (1 - mix_ratio) * (lon - prev_lon)

        # Ensure adjusted point respects QoS
        real_lat, real_lon = self.real_trajectory[-1]
        if not self._check_qos(real_lat, real_lon, adjusted_lat, adjusted_lon):
            # If QoS violated, use the original fake point
            return lat, lon

        return adjusted_lat, adjusted_lon

    def _check_qos(self, real_lat, real_lon, fake_lat, fake_lon):
        """
        Check if QoS constraint is satisfied for a given fake point.

        Args:
            real_lat, real_lon: Real location
            fake_lat, fake_lon: Fake location

        Returns:
            Boolean indicating if QoS is satisfied
        """
        # Calculate distance between real and fake point
        distance = haversine.haversine(
            (real_lat, real_lon),
            (fake_lat, fake_lon),
            unit='m'  # meters
        )

        # Check if distance is within QoS radius
        return distance <= self.qos_radius

    def snap_to_road_network(self):
        """
        Snap fake trajectory points to nearest roads for realism.
        """
        if self.graph is None or len(self.fake_trajectory) == 0:
            return

        snapped_trajectory = []

        for i, (lat, lon) in enumerate(self.fake_trajectory):
            # Find nearest node in graph
            nearest_node = ox.distance.nearest_nodes(self.graph, lon, lat)

            # Get node coordinates
            node = self.graph.nodes[nearest_node]
            snapped_lat, snapped_lon = node['y'], node['x']

            # Ensure QoS constraint is still met
            if self._check_qos(self.real_trajectory[i][0], self.real_trajectory[i][1],
                              snapped_lat, snapped_lon):
                snapped_trajectory.append((snapped_lat, snapped_lon))
            else:
                # Keep original fake point if QoS would be violated
                snapped_trajectory.append((lat, lon))

        self.fake_trajectory = snapped_trajectory

    def visualize_trajectories(self, center=None, zoom_start=15):
        """
        Visualize real and fake trajectories on an interactive map.

        Args:
            center: (lat, lon) for map center, defaults to first point
            zoom_start: Initial zoom level

        Returns:
            Folium map object
        """
        if not self.real_trajectory:
            raise ValueError("No trajectory data to visualize")

        if center is None:
            center = self.real_trajectory[0]

        # Create map
        m = folium.Map(location=center, zoom_start=zoom_start, tiles='OpenStreetMap')

        # Add real trajectory
        points = [(lat, lon) for lat, lon in self.real_trajectory]
        folium.PolyLine(
            points,
            color='blue',
            weight=4,
            opacity=0.7,
            tooltip="Real Trajectory"
        ).add_to(m)

        # Add fake trajectory
        fake_points = [(lat, lon) for lat, lon in self.fake_trajectory]
        folium.PolyLine(
            fake_points,
            color='red',
            weight=4,
            opacity=0.7,
            tooltip="Obfuscated Trajectory"
        ).add_to(m)

        # Add markers for real points
        for i, (lat, lon) in enumerate(self.real_trajectory):
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup=f"Real Point {i+1}: {lat:.6f}, {lon:.6f}"
            ).add_to(m)

        # Add markers for fake points
        for i, (lat, lon) in enumerate(self.fake_trajectory):
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"Fake Point {i+1}: {lat:.6f}, {lon:.6f}"
            ).add_to(m)

        # Add lines connecting real and fake points
        for i in range(min(len(self.real_trajectory), len(self.fake_trajectory))):
            folium.PolyLine(
                [self.real_trajectory[i], self.fake_trajectory[i]],
                color='gray',
                weight=1,
                opacity=0.5,
                dash_array='5'
            ).add_to(m)

        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px;
                    opacity: 0.8;">
        <b>Legend</b><br>
        <i class="fa fa-circle" style="color:blue"></i> Real Trajectory<br>
        <i class="fa fa-circle" style="color:red"></i> Obfuscated Trajectory<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def evaluate_privacy(self):
        """
        Evaluate privacy metrics for the trajectories.

        Returns:
            Dict of privacy metrics
        """
        if not self.real_trajectory or not self.fake_trajectory:
            raise ValueError("No trajectory data to evaluate")

        metrics = {}

        # Average distance between real and fake points
        total_distance = 0
        for i in range(min(len(self.real_trajectory), len(self.fake_trajectory))):
            real_lat, real_lon = self.real_trajectory[i]
            fake_lat, fake_lon = self.fake_trajectory[i]
            distance = haversine.haversine(
                (real_lat, real_lon),
                (fake_lat, fake_lon),
                unit='m'  # meters
            )
            total_distance += distance

        metrics['avg_distance'] = total_distance / len(self.real_trajectory)

        # Maximum distance between real and fake points
        max_distance = 0
        for i in range(min(len(self.real_trajectory), len(self.fake_trajectory))):
            real_lat, real_lon = self.real_trajectory[i]
            fake_lat, fake_lon = self.fake_trajectory[i]
            distance = haversine.haversine(
                (real_lat, real_lon),
                (fake_lat, fake_lon),
                unit='m'  # meters
            )
            max_distance = max(max_distance, distance)

        metrics['max_distance'] = max_distance

        # QoS satisfaction rate
        qos_satisfied = sum(1 for i in range(min(len(self.real_trajectory), len(self.fake_trajectory)))
                            if self._check_qos(self.real_trajectory[i][0], self.real_trajectory[i][1],
                                              self.fake_trajectory[i][0], self.fake_trajectory[i][1]))
        metrics['qos_satisfaction'] = qos_satisfied / len(self.real_trajectory)

        return metrics

