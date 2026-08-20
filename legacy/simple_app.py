"""
Simple Flask web application for trajectory privacy demonstration.
"""
from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from core.trajectory_privacy import TrajectoryPrivacy
import osmnx as ox
from shapely.geometry import LineString
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

def interpolate_route_points(route_points, target_count=20):
    """Interpolate points along a route."""
    if len(route_points) < 2:
        return route_points
    
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

@app.route('/')
def index():
    """Render the main page."""
    return render_template('simple_index.html')

@app.route('/process_route', methods=['POST'])
def process_route():
    """Process a user-drawn route."""
    try:
        data = request.json
        route_points = data['route']
        epsilon = float(data.get('epsilon', 0.1))
        qos_radius = float(data.get('qos_radius', 175))
        
        if len(route_points) < 2:
            return jsonify({'error': 'Please draw a route with at least 2 points'}), 400
        
        # Interpolate route
        trajectory_points = interpolate_route_points(route_points, target_count=20)
        trajectory = [(p['lat'], p['lng']) for p in trajectory_points]
        
        # Create trajectory privacy object
        tp = TrajectoryPrivacy(epsilon=epsilon, qos_radius=qos_radius)
        
        # Load spatial constraints
        min_lat = min(p['lat'] for p in trajectory_points) - 0.01
        max_lat = max(p['lat'] for p in trajectory_points) + 0.01
        min_lng = min(p['lng'] for p in trajectory_points) - 0.01
        max_lng = max(p['lng'] for p in trajectory_points) + 0.01
        
        tp.load_spatial_constraints((min_lat, min_lng, max_lat, max_lng))
        
        # Process trajectory
        fake_trajectory = []
        for lat, lng in trajectory:
            fake_lat, fake_lng = tp.add_point(lat, lng)
            fake_trajectory.append({'lat': fake_lat, 'lng': fake_lng})
        
        # Snap to road network
        tp.snap_to_road_network()
        
        snapped_fake_trajectory = [{'lat': lat, 'lng': lng} for lat, lng in tp.fake_trajectory]
        
        tp.snap_real_trajectory_to_road_network()
        snapped_real_trajectory = [{'lat': lat, 'lng': lng} for lat, lng in tp.real_trajectory]
        
        # Calculate metrics
        metrics = tp.evaluate_privacy()
        
        return jsonify({
            'success': True,
            'real_trajectory': snapped_real_trajectory,
            'fake_trajectory': snapped_fake_trajectory,
            'metrics': {
                'avg_distance': round(metrics['avg_distance'], 2),
                'max_distance': round(metrics['max_distance'], 2),
                'qos_satisfaction': round(metrics['qos_satisfaction'] * 100, 1)
            }
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)