"""
Optimized Flask web application with caching for trajectory privacy demonstration.
"""
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import numpy as np
from web.trajectory_privacy_optimized import TrajectoryPrivacyOptimized
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
import warnings
import time
import pickle
import os
from datetime import datetime, timedelta
import hashlib

warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Cache directory for road networks
CACHE_DIR = 'road_network_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache for frequently used road networks
road_network_cache = {}

def console_log(message, level='info'):
    """Send log message to web console."""
    socketio.emit('console_output', {
        'message': str(message),
        'level': level,
        'timestamp': time.strftime('%H:%M:%S')
    })
    print(message)

def get_cache_key(bounds):
    """Generate a cache key for the given bounds."""
    return hashlib.md5(f"{bounds[0]:.6f},{bounds[1]:.6f},{bounds[2]:.6f},{bounds[3]:.6f}".encode()).hexdigest()

def load_cached_road_network(bounds):
    """Load road network from cache if available."""
    cache_key = get_cache_key(bounds)
    
    # Check in-memory cache first
    if cache_key in road_network_cache:
        console_log("📦 Using in-memory cached road network", level='success')
        return road_network_cache[cache_key]
    
    # Check disk cache
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        # Check if cache is less than 7 days old
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_age < timedelta(days=7):
            try:
                console_log("💾 Loading road network from disk cache...", level='info')
                with open(cache_file, 'rb') as f:
                    graph = pickle.load(f)
                road_network_cache[cache_key] = graph
                console_log(f"✅ Loaded cached road network ({len(graph.nodes)} nodes)", level='success')
                return graph
            except Exception as e:
                console_log(f"⚠️ Cache load failed: {e}", level='warning')
    
    return None

def save_road_network_to_cache(bounds, graph):
    """Save road network to cache."""
    cache_key = get_cache_key(bounds)
    
    # Save to memory cache
    road_network_cache[cache_key] = graph
    
    # Save to disk cache
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(graph, f)
        console_log("💾 Saved road network to cache", level='success')
    except Exception as e:
        console_log(f"⚠️ Failed to save cache: {e}", level='warning')

def interpolate_route_points(route_points, target_count=None):
    """
    Interpolate points along a route.
    If target_count is None, use the number of points the user drew.
    """
    if len(route_points) < 2:
        return route_points
    
    # If no target count specified, use the original number of points
    if target_count is None:
        target_count = len(route_points)
    
    # If user drew more points than target, just use their points
    if len(route_points) >= target_count:
        return route_points[:target_count]
    
    # Only interpolate if we need more points
    line = LineString([(p['lng'], p['lat']) for p in route_points])
    distances = np.linspace(0, line.length, target_count)
    
    interpolated_points = []
    for distance in distances:
        point = line.interpolate(distance, normalized=True)
        interpolated_points.append({
            'lat': point.y,
            'lng': point.x
        })
    
    return interpolated_points

class OptimizedTrajectoryPrivacy(TrajectoryPrivacyOptimized):
    """Optimized version with caching and progress reporting."""
    
    def __init__(self, epsilon=0.1, qos_radius=200):
        super().__init__(epsilon, qos_radius)
        self.console_log = console_log
    
    def load_spatial_constraints(self, bounds, excluded_types=None):
        """
        Optimized version that uses cached road networks and handles areas without roads.
        """
        if excluded_types is None:
            excluded_types = ['building']
        
        self.console_log("📍 Starting spatial constraints loading...")
        self.console_log(f"📐 Bounds: {bounds[0]:.6f},{bounds[1]:.6f} to {bounds[2]:.6f},{bounds[3]:.6f}")
        
        # Configure OSMnx for better performance
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        # Try to load road network
        try:
            # Try to load cached road network first
            self.graph = load_cached_road_network(bounds)
            
            if self.graph is None:
                # Download road network
                self.console_log("🌐 Downloading road network from OpenStreetMap...")
                start_time = time.time()
                
                try:
                    # First try with expanded bounds
                    expanded_bounds = (
                        bounds[0] - 0.005,
                        bounds[1] - 0.005,
                        bounds[2] + 0.005,
                        bounds[3] + 0.005
                    )
                    
                    self.graph = ox.graph_from_bbox(
                        bbox=(expanded_bounds[2], expanded_bounds[0], expanded_bounds[3], expanded_bounds[1]),
                        network_type='all',  # Try all types of paths
                        simplify=True,
                        retain_all=False
                    )
                    
                    if self.graph and len(self.graph.nodes) > 0:
                        road_time = time.time() - start_time
                        self.console_log(f"✅ Road network downloaded in {road_time:.2f}s ({len(self.graph.nodes)} nodes)")
                        self.has_road_network = True
                        # Save to cache
                        save_road_network_to_cache(bounds, self.graph)
                    else:
                        raise ValueError("No roads found")
                        
                except Exception as e:
                    self.console_log(f"⚠️ No road network found in area: {str(e)}", level='warning')
                    self.console_log("🚶 Proceeding without road snapping", level='info')
                    self.graph = None
                    self.has_road_network = False
            else:
                # Using cached network
                self.has_road_network = True
            
            # Load spatial constraints (buildings, etc.) - continue even if no roads
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
                
            if self.has_road_network:
                self.console_log("🎉 Spatial constraints loading complete with road network!", level='success')
            else:
                self.console_log("🎉 Spatial constraints loading complete (no roads available)!", level='success')
            
        except Exception as e:
            self.console_log(f"❌ Critical error: {str(e)}", level='error')
            # Don't raise - allow processing to continue without constraints
            self.graph = None
            self.has_road_network = False
            self.spatial_constraints = None

@app.route('/')
def index():
    """Render the main page with interactive map."""
    return render_template('index_optimized.html')

@app.route('/preload_area', methods=['POST'])
def preload_area():
    """Preload road network for a specific area."""
    try:
        data = request.json
        bounds = data['bounds']  # [min_lat, min_lng, max_lat, max_lng]
        
        console_log("🗺️ Preloading area map data...", level='info')
        
        # Check if already cached
        existing = load_cached_road_network(bounds)
        if existing:
            return jsonify({'success': True, 'message': 'Area already cached'})
        
        # Download and cache
        console_log("🌐 Downloading area from OpenStreetMap...")
        try:
            # Try with expanded bounds
            expanded_bounds = [
                bounds[0] - 0.005,
                bounds[1] - 0.005,
                bounds[2] + 0.005,
                bounds[3] + 0.005
            ]
            
            graph = ox.graph_from_bbox(
                bbox=(expanded_bounds[2], expanded_bounds[0], expanded_bounds[3], expanded_bounds[1]),
                network_type='all',
                simplify=True,
                retain_all=False
            )
            
            if graph and len(graph.nodes) > 0:
                save_road_network_to_cache(bounds, graph)
                console_log(f"✅ Area preloaded successfully ({len(graph.nodes)} nodes)", level='success')
                return jsonify({'success': True, 'nodes': len(graph.nodes)})
            else:
                console_log("⚠️ No road network found in this area", level='warning')
                return jsonify({'success': True, 'message': 'No roads in area', 'nodes': 0})
                
        except Exception as e:
            console_log(f"⚠️ Could not download roads: {str(e)}", level='warning')
            return jsonify({'success': True, 'message': 'Area has no road data', 'nodes': 0})
        
    except Exception as e:
        console_log(f"❌ Error preloading area: {str(e)}", level='error')
        return jsonify({'error': str(e)}), 500

@app.route('/process_route', methods=['POST'])
def process_route():
    """Process a user-drawn route and return fake trajectories."""
    try:
        console_log("🚀 Starting route processing...", level='info')
        
        data = request.json
        route_points = data['route']
        epsilon = float(data.get('epsilon', 0.1))
        qos_radius = float(data.get('qos_radius', 175))
        use_interpolation = data.get('use_interpolation', False)
        
        console_log(f"📊 Parameters: ε={epsilon}, QoS radius={qos_radius}m")
        console_log(f"📍 User drew {len(route_points)} points")
        
        if len(route_points) < 2:
            return jsonify({'error': 'Please draw a route with at least 2 points'}), 400
        
        # Only interpolate if requested
        if use_interpolation and len(route_points) < 20:
            console_log("🔄 Interpolating to 20 points (as requested)...")
            trajectory_points = interpolate_route_points(route_points, target_count=20)
        else:
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
            if i % 5 == 0:  # Log every 5th point to reduce console spam
                console_log(f"📍 Processing point {i+1}/{len(trajectory)}...")
        
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

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the road network cache."""
    try:
        # Clear in-memory cache
        road_network_cache.clear()
        
        # Clear disk cache
        for file in os.listdir(CACHE_DIR):
            if file.endswith('.pkl'):
                os.remove(os.path.join(CACHE_DIR, file))
        
        console_log("🗑️ Cache cleared successfully", level='success')
        return jsonify({'success': True})
        
    except Exception as e:
        console_log(f"❌ Error clearing cache: {str(e)}", level='error')
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    console_log("👤 Client connected", level='info')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    console_log("👤 Client disconnected", level='info')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)