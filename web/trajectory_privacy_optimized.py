"""
Optimized TrajectoryPrivacy class with better error handling.
"""
import numpy as np
import osmnx as ox
from shapely.geometry import Point
import warnings
from haversine import haversine, Unit
from core.geo_indistinguishability import GeoIndistinguishability

warnings.filterwarnings("ignore")

class TrajectoryPrivacyOptimized:
    def __init__(self, epsilon=0.1, qos_radius=200):
        """
        Initialize trajectory privacy protection with optimizations.
        
        Args:
            epsilon: Privacy parameter (lower = more privacy)
            qos_radius: Quality of service radius in meters
        """
        self.epsilon = epsilon
        self.qos_radius = qos_radius
        self.real_trajectory = []
        self.fake_trajectory = []
        self.graph = None
        self.spatial_constraints = None
        self.has_road_network = False
        self.geo_ind = GeoIndistinguishability(epsilon=epsilon, delta=qos_radius)
        
    def load_spatial_constraints(self, bounds, excluded_types=None):
        """
        Load spatial constraints for the given area with better error handling.
        
        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            excluded_types: List of OSM feature types to exclude
        """
        if excluded_types is None:
            excluded_types = ['building']
        
        # Try to get road network
        try:
            # Expand bounds slightly to ensure we catch roads
            expanded_bounds = (
                bounds[0] - 0.005,  # min_lat
                bounds[1] - 0.005,  # min_lon
                bounds[2] + 0.005,  # max_lat
                bounds[3] + 0.005   # max_lon
            )
            
            # Configure OSMnx
            ox.settings.use_cache = True
            ox.settings.log_console = False
            
            # Try to get road network
            self.graph = ox.graph_from_bbox(
                bbox=(expanded_bounds[1], expanded_bounds[0], expanded_bounds[3], expanded_bounds[2]),
                network_type='all',  # Try to get all types of paths
                simplify=True,
                retain_all=False
            )
            
            if self.graph and len(self.graph.nodes) > 0:
                self.has_road_network = True
                print(f"Loaded road network with {len(self.graph.nodes)} nodes")
            else:
                self.has_road_network = False
                print("Warning: No road network found in area")
                
        except Exception as e:
            print(f"Warning: Could not load road network: {e}")
            self.has_road_network = False
            self.graph = None
        
        # Try to load spatial constraints (buildings, etc.)
        try:
            gdf_list = []
            for feature_type in excluded_types:
                try:
                    tags = {feature_type: True}
                    gdf = ox.features.features_from_bbox(
                        bbox=(bounds[1], bounds[0], bounds[3], bounds[2]),
                        tags=tags
                    )
                    if not gdf.empty:
                        gdf_list.append(gdf)
                except:
                    pass
            
            if gdf_list:
                import pandas as pd
                self.spatial_constraints = pd.concat(gdf_list, ignore_index=True)
            else:
                self.spatial_constraints = None
                
        except Exception as e:
            print(f"Warning: Could not load spatial constraints: {e}")
            self.spatial_constraints = None
    
    def is_valid_location(self, lat, lon):
        """
        Check if a location is valid (not inside buildings, etc.).
        Always returns True if no constraints are loaded.
        """
        if self.spatial_constraints is None:
            return True
        
        point = Point(lon, lat)
        
        # Check if point is inside any constraint
        for _, constraint in self.spatial_constraints.iterrows():
            if hasattr(constraint, 'geometry') and constraint.geometry:
                if constraint.geometry.contains(point):
                    return False
        
        return True
    
    def add_point(self, lat, lon):
        """
        Add a real point and generate a corresponding fake point.
        
        Args:
            lat: Latitude of real point
            lon: Longitude of real point
            
        Returns:
            tuple: (fake_lat, fake_lon)
        """
        # Add real point
        self.real_trajectory.append((lat, lon))
        
        # Generate fake point with noise
        max_attempts = 50
        for attempt in range(max_attempts):
            # Add noise
            noise_lat, noise_lon = self.geo_ind.add_noise(lat, lon)
            
            # Check QoS constraint
            distance = haversine((lat, lon), (noise_lat, noise_lon), unit=Unit.METERS)
            
            if distance <= self.qos_radius:
                # Check validity if we have constraints
                if self.is_valid_location(noise_lat, noise_lon):
                    self.fake_trajectory.append((noise_lat, noise_lon))
                    return noise_lat, noise_lon
        
        # If no valid point found, use a point at QoS radius boundary
        angle = np.random.uniform(0, 2 * np.pi)
        # Approximate meters to degrees (rough conversion)
        radius_deg = self.qos_radius / 111000.0
        fake_lat = lat + radius_deg * np.sin(angle)
        fake_lon = lon + radius_deg * np.cos(angle) / np.cos(np.radians(lat))
        
        self.fake_trajectory.append((fake_lat, fake_lon))
        return fake_lat, fake_lon
    
    def snap_to_road_network(self):
        """
        Snap fake trajectory to road network if available.
        If no road network, trajectory remains unchanged.
        """
        if not self.has_road_network or not self.graph or not self.fake_trajectory:
            print("No road network available for snapping")
            return
        
        snapped_trajectory = []
        
        for lat, lon in self.fake_trajectory:
            try:
                # Find nearest node
                nearest_node = ox.distance.nearest_nodes(self.graph, lon, lat)
                
                # Get node coordinates
                node_lat = self.graph.nodes[nearest_node]['y']
                node_lon = self.graph.nodes[nearest_node]['x']
                
                # Check if snapped point is within reasonable distance
                distance = haversine((lat, lon), (node_lat, node_lon), unit=Unit.METERS)
                
                if distance < 500:  # Only snap if within 500m
                    snapped_trajectory.append((node_lat, node_lon))
                else:
                    snapped_trajectory.append((lat, lon))
                    
            except Exception as e:
                # If snapping fails, keep original point
                snapped_trajectory.append((lat, lon))
        
        self.fake_trajectory = snapped_trajectory
    
    def snap_real_trajectory_to_road_network(self):
        """
        Snap real trajectory to road network if available.
        """
        if not self.has_road_network or not self.graph or not self.real_trajectory:
            return
        
        snapped_trajectory = []
        
        for lat, lon in self.real_trajectory:
            try:
                nearest_node = ox.distance.nearest_nodes(self.graph, lon, lat)
                node_lat = self.graph.nodes[nearest_node]['y']
                node_lon = self.graph.nodes[nearest_node]['x']
                
                distance = haversine((lat, lon), (node_lat, node_lon), unit=Unit.METERS)
                
                if distance < 500:
                    snapped_trajectory.append((node_lat, node_lon))
                else:
                    snapped_trajectory.append((lat, lon))
                    
            except:
                snapped_trajectory.append((lat, lon))
        
        self.real_trajectory = snapped_trajectory
    
    def evaluate_privacy(self):
        """
        Evaluate privacy metrics between real and fake trajectories.
        """
        if not self.real_trajectory or not self.fake_trajectory:
            return {
                'avg_distance': 0,
                'max_distance': 0,
                'qos_satisfaction': 1.0
            }
        
        distances = []
        qos_satisfied = 0
        
        for (real_lat, real_lon), (fake_lat, fake_lon) in zip(self.real_trajectory, self.fake_trajectory):
            distance = haversine((real_lat, real_lon), (fake_lat, fake_lon), unit=Unit.METERS)
            distances.append(distance)
            
            if distance <= self.qos_radius:
                qos_satisfied += 1
        
        return {
            'avg_distance': np.mean(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'qos_satisfaction': qos_satisfied / len(self.real_trajectory) if self.real_trajectory else 1.0
        }
    
    def visualize_trajectories(self, zoom_start=13):
        """
        Create a Folium map visualization of real and fake trajectories.
        """
        import folium
        
        if not self.real_trajectory:
            raise ValueError("No trajectory data to visualize")
        
        # Center map on middle of real trajectory
        center_lat = np.mean([p[0] for p in self.real_trajectory])
        center_lon = np.mean([p[1] for p in self.real_trajectory])
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        # Add real trajectory
        if self.real_trajectory:
            folium.PolyLine(
                locations=self.real_trajectory,
                color='green',
                weight=4,
                opacity=0.8,
                popup='Real Trajectory'
            ).add_to(m)
            
            # Add markers for real points
            for i, (lat, lon) in enumerate(self.real_trajectory):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=f'Real Point {i+1}',
                    color='green',
                    fill=True,
                    fillColor='green'
                ).add_to(m)
        
        # Add fake trajectory
        if self.fake_trajectory:
            folium.PolyLine(
                locations=self.fake_trajectory,
                color='red',
                weight=4,
                opacity=0.8,
                popup='Fake Trajectory',
                dash_array='10'
            ).add_to(m)
            
            # Add markers for fake points
            for i, (lat, lon) in enumerate(self.fake_trajectory):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=f'Fake Point {i+1}',
                    color='red',
                    fill=True,
                    fillColor='red'
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin: 0;"><strong>Legend</strong></p>
        <p style="margin: 5px 0;"><span style="color: green;">━━━</span> Real Trajectory</p>
        <p style="margin: 5px 0;"><span style="color: red;">┅┅┅</span> Fake Trajectory</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m