"""
Flask web application for interactive trajectory privacy demonstration.
"""
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import numpy as np
from core.trajectory_privacy import TrajectoryPrivacy
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
import warnings
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import threading
import time

warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Custom print function that sends to console
def console_log(message, level='info'):
    """Send log message to web console."""
    socketio.emit('console_output', {
        'message': str(message),
        'level': level,
        'timestamp': time.strftime('%H:%M:%S')
    })
    print(message)  # Also print to server console

def interpolate_route_points(route_points, target_count=20):
    """
    Interpolate points along a route to get a desired number of points.
    """
    if len(route_points) < 2:
        return route_points
    
    # Create a LineString from the route
    line = LineString([(p['lng'], p['lat']) for p in route_points])
    
    # Calculate distances for interpolation
    distances = np.linspace(0, line.length, target_count)
    
    # Interpolate points
    interpolated_points = []
    for distance in distances:
        point = line.interpolate(distance, normalized=True)
        interpolated_points.append({
            'lat': point.y,
            'lng': point.x
        })
    
    return interpolated_points

def snap_to_road_network_points(points, G):
    """
    Snap a list of points to the nearest road network nodes.
    """
    snapped_points = []
    
    for point in points:
        # Find nearest node
        nearest_node = ox.nearest_nodes(G, point['lng'], point['lat'])
        
        # Get node coordinates
        node_data = G.nodes[nearest_node]
        snapped_points.append({
            'lat': node_data['y'],
            'lng': node_data['x']
        })
    
    return snapped_points

# Optimized TrajectoryPrivacy class that inherits from the original
class OptimizedTrajectoryPrivacy(TrajectoryPrivacy):
    """Optimized version with progress reporting and faster loading."""
    
    def __init__(self, epsilon=0.1, qos_radius=200):
        super().__init__(epsilon, qos_radius)
        self.console_log = console_log
        self.has_road_network = False
    
    def load_spatial_constraints(self, bounds, excluded_types=None):
        """
        Optimized version that loads only essential features and provides progress updates.
        """
        if excluded_types is None:
            # Reduce the number of excluded types for faster loading
            excluded_types = ['building']  # Only load buildings, skip water/river for speed
        
        self.console_log("📍 Starting spatial constraints loading...")
        self.console_log(f"📐 Bounds: {bounds[0]:.6f},{bounds[1]:.6f} to {bounds[2]:.6f},{bounds[3]:.6f}")
        
        # Configure OSMnx for better performance
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        try:
            # Get the street network graph for the area
            self.console_log("🛣️ Loading road network...")
            start_time = time.time()
            
            try:
                # Try with expanded bounds to catch nearby roads
                expanded_bounds = (
                    bounds[0] - 0.005,
                    bounds[1] - 0.005,
                    bounds[2] + 0.005,
                    bounds[3] + 0.005
                )
                
                # Use a simplified network type for faster loading
                self.graph = ox.graph_from_bbox(
                    bbox=(expanded_bounds[2], expanded_bounds[0], expanded_bounds[3], expanded_bounds[1]),
                    network_type='all',  # Try all types of paths
                    simplify=True,
                    retain_all=False
                )
                
                if self.graph and len(self.graph.nodes) > 0:
                    road_time = time.time() - start_time
                    self.console_log(f"✅ Road network loaded in {road_time:.2f}s ({len(self.graph.nodes)} nodes)")
                    self.has_road_network = True
                else:
                    raise ValueError("No roads found")
                    
            except Exception as e:
                self.console_log(f"⚠️ No road network found in area: {str(e)}", level='warning')
                self.console_log("🚶 Proceeding without road snapping", level='info')
                self.graph = None
                self.has_road_network = False
            
            # Load spatial constraints with progress updates
            gdf_list = []
            for feature_type in excluded_types:
                try:
                    self.console_log(f"🏢 Loading {feature_type} features...")
                    start_time = time.time()
                    
                    tags = {feature_type: True}
                    gdf = ox.features.features_from_bbox(
                        bbox=(bounds[2], bounds[0], bounds[3], bounds[1]),
                        tags=tags
                    )
                    
                    if not gdf.empty:
                        gdf_list.append(gdf)
                        feature_time = time.time() - start_time
                        self.console_log(f"✅ Loaded {len(gdf)} {feature_type}s in {feature_time:.2f}s")
                    else:
                        self.console_log(f"ℹ️ No {feature_type} features found in area")
                        
                except Exception as e:
                    self.console_log(f"⚠️ Could not fetch {feature_type}: {str(e)}", level='warning')
            
            # Combine all spatial constraints
            if gdf_list:
                import pandas as pd
                self.spatial_constraints = pd.concat(gdf_list, ignore_index=True)
                self.console_log(f"✅ Total spatial constraints loaded: {len(self.spatial_constraints)} features")
            else:
                self.spatial_constraints = None
                self.console_log("ℹ️ No spatial constraints found")
                
            self.console_log("🎉 Spatial constraints loading complete!", level='success')
            
        except Exception as e:
            self.console_log(f"❌ Error loading spatial constraints: {str(e)}", level='error')
            raise

@app.route('/')
def index():
    """Render the main page with interactive map."""
    return render_template('index.html')

@app.route('/process_route', methods=['POST'])
def process_route():
    """Process a user-drawn route and return fake trajectories."""
    try:
        console_log("🚀 Starting route processing...", level='info')
        
        data = request.json
        route_points = data['route']
        epsilon = float(data.get('epsilon', 0.1))
        qos_radius = float(data.get('qos_radius', 175))
        
        console_log(f"📊 Parameters: ε={epsilon}, QoS radius={qos_radius}m")
        
        if len(route_points) < 2:
            return jsonify({'error': 'Please draw a route with at least 2 points'}), 400
        
        # Use the points as drawn by the user
        trajectory_points = route_points
        console_log(f"✅ Using {len(trajectory_points)} user-drawn points")
        
        # Convert to list of tuples
        trajectory = [(p['lat'], p['lng']) for p in trajectory_points]
        
        # Create optimized trajectory privacy object
        tp = OptimizedTrajectoryPrivacy(epsilon=epsilon, qos_radius=qos_radius)
        
        # Load spatial constraints for the area
        min_lat = min(p['lat'] for p in trajectory_points) - 0.01
        max_lat = max(p['lat'] for p in trajectory_points) + 0.01
        min_lng = min(p['lng'] for p in trajectory_points) - 0.01
        max_lng = max(p['lng'] for p in trajectory_points) + 0.01
        
        tp.load_spatial_constraints((min_lat, min_lng, max_lat, max_lng))
        
        # Process trajectory
        console_log("🔐 Generating privacy-protected trajectory...")
        fake_trajectory = []
        for i, (lat, lng) in enumerate(trajectory):
            fake_lat, fake_lng = tp.add_point(lat, lng)
            fake_trajectory.append({'lat': fake_lat, 'lng': fake_lng})
            console_log(f"📍 Point {i+1}/{len(trajectory)}: ({lat:.6f},{lng:.6f}) → ({fake_lat:.6f},{fake_lng:.6f})")
        
        # Snap to road network
        console_log("🛣️ Snapping fake trajectory to road network...")
        tp.snap_to_road_network()
        
        # Get the snapped fake trajectory
        snapped_fake_trajectory = []
        for lat, lng in tp.fake_trajectory:
            snapped_fake_trajectory.append({'lat': lat, 'lng': lng})
        
        # Also snap the real trajectory for visualization
        console_log("🛣️ Snapping real trajectory to road network...")
        tp.snap_real_trajectory_to_road_network()
        snapped_real_trajectory = []
        for lat, lng in tp.real_trajectory:
            snapped_real_trajectory.append({'lat': lat, 'lng': lng})
        
        # Calculate privacy metrics
        console_log("📈 Calculating privacy metrics...")
        metrics = tp.evaluate_privacy()
        
        console_log("✅ Route processing complete!", level='success')
        console_log(f"📊 Metrics: Avg distance={metrics['avg_distance']:.2f}m, Max={metrics['max_distance']:.2f}m, QoS={metrics['qos_satisfaction']*100:.1f}%")
        
        return jsonify({
            'success': True,
            'real_trajectory': snapped_real_trajectory,
            'fake_trajectory': snapped_fake_trajectory,
            'original_fake_trajectory': fake_trajectory,
            'metrics': {
                'avg_distance': round(metrics['avg_distance'], 2),
                'max_distance': round(metrics['max_distance'], 2),
                'qos_satisfaction': round(metrics['qos_satisfaction'] * 100, 1)
            }
        })
        
    except Exception as e:
        error_msg = str(e)
        console_log(f"❌ Error processing route: {error_msg}", level='error')
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    console_log("👤 Client connected", level='info')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    console_log("👤 Client disconnected", level='info')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5002, allow_unsafe_werkzeug=True)