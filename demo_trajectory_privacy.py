"""
Demo script for trajectory privacy protection.
"""
import folium
import numpy as np
from trajectory_privacy import TrajectoryPrivacy
import os
import datetime
import random
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString, Point
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

def generate_realistic_trajectory(center_lat, center_lon, target_length=20, max_distance=1500):
    """
    Generate a realistic trajectory that follows road networks.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        target_length: Target number of trajectory points
        max_distance: Maximum distance in meters for the route

    Returns:
        List of (lat, lon) tuples representing a trajectory that follows roads
    """
    print("Fetching road network...")
    # Get a street network graph around the center point
    try:
        G = ox.graph_from_point((center_lat, center_lon), dist=max_distance, network_type='drive')
    except Exception as e:
        print(f"Error fetching road network: {e}")
        print("Falling back to simplified network generation...")
        # Fallback to a simpler approach with a smaller area
        G = ox.graph_from_point((center_lat, center_lon), dist=800, network_type='drive_service')

    # Get nodes from the graph
    nodes = list(G.nodes)

    if not nodes:
        print("No nodes found in graph. Using fallback method.")
        return generate_fallback_trajectory(center_lat, center_lon, target_length)

    # Select a random starting node
    start_node = random.choice(nodes)

    # Find a destination node that's reasonably far away
    potential_end_nodes = []
    for node in nodes:
        try:
            # Calculate path using shortest path
            path = nx.shortest_path(G, start_node, node, weight='length')
            path_length = sum(ox.utils_graph.get_route_edge_attributes(G, path, 'length'))

            # Consider nodes that create paths of suitable length
            if 500 < path_length < max_distance:
                potential_end_nodes.append((node, path_length))
        except nx.NetworkXNoPath:
            continue

    if not potential_end_nodes:
        print("No suitable end nodes found. Using fallback method.")
        return generate_fallback_trajectory(center_lat, center_lon, target_length)

    # Sort by path length and select a random node from the top candidates
    potential_end_nodes.sort(key=lambda x: x[1], reverse=True)
    end_node, path_length = random.choice(potential_end_nodes[:min(5, len(potential_end_nodes))])

    print(f"Found a route with path length of {path_length:.1f} meters")

    try:
        # Calculate the shortest path between start and end nodes
        route = nx.shortest_path(G, start_node, end_node, weight='length')

        # Extract coordinates for each node in the route
        route_coords = []
        for node in route:
            lat = G.nodes[node]['y']
            lon = G.nodes[node]['x']
            route_coords.append((lat, lon))

        # For longer routes, we need to interpolate points between nodes
        if len(route_coords) < target_length:
            # Convert to a more detailed trajectory by interpolating
            detailed_coords = []

            for i in range(len(route_coords) - 1):
                # Get the two adjacent points
                start = route_coords[i]
                end = route_coords[i + 1]

                # Calculate number of points to add between these nodes
                # More points for longer segments
                start_point = Point(start[1], start[0])  # lon, lat order for Point
                end_point = Point(end[1], end[0])
                seg_distance = start_point.distance(end_point) * 111000  # rough conversion to meters

                # Number of points proportional to segment length
                num_points = max(1, int(seg_distance / 30))  # One point roughly every 30 meters

                # Add the starting point
                detailed_coords.append(start)

                # Add interpolated points
                for j in range(1, num_points):
                    ratio = j / num_points
                    lat = start[0] + (end[0] - start[0]) * ratio
                    lon = start[1] + (end[1] - start[1]) * ratio
                    detailed_coords.append((lat, lon))

            # Add the final point
            detailed_coords.append(route_coords[-1])

            route_coords = detailed_coords

        # If we have too many points, sample them
        if len(route_coords) > target_length:
            # Keep first and last points, sample the rest
            indices = [0] + sorted(random.sample(range(1, len(route_coords) - 1), target_length - 2)) + [len(route_coords) - 1]
            route_coords = [route_coords[i] for i in indices]

        print(f"Generated trajectory with {len(route_coords)} points following road network")
        return route_coords

    except Exception as e:
        print(f"Error generating path: {e}")
        return generate_fallback_trajectory(center_lat, center_lon, target_length)

def generate_fallback_trajectory(center_lat, center_lon, num_points=20):
    """
    Fallback method to generate a more realistic-looking trajectory when OSMnx fails.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        num_points: Number of points to generate

    Returns:
        List of (lat, lon) tuples
    """
    print("Using fallback trajectory generation method")
    trajectory = []

    # Parameters for realistic movement
    step_sizes = [0.0001, 0.00015, 0.0002, 0.00025]  # Different step sizes in degrees
    direction_change_probability = 0.3  # Probability of changing direction

    # Generate starting point
    current_lat = center_lat + random.uniform(-0.002, 0.002)
    current_lon = center_lon + random.uniform(-0.002, 0.002)
    trajectory.append((current_lat, current_lon))

    # Initialize direction (angle in radians)
    current_direction = random.uniform(0, 2 * np.pi)

    # Generate subsequent points with a more natural movement pattern
    for _ in range(num_points - 1):
        # Occasionally change direction (to simulate turns)
        if random.random() < direction_change_probability:
            # Make a significant turn (30-90 degrees)
            turn_angle = random.uniform(np.pi/6, np.pi/2) * random.choice([-1, 1])
            current_direction += turn_angle

        # Some small random variation in direction (slight curves)
        current_direction += random.uniform(-0.1, 0.1)

        # Select a step size (distance to move)
        step_size = random.choice(step_sizes)

        # Calculate new position based on direction and step size
        current_lat += step_size * np.sin(current_direction)
        current_lon += step_size * np.cos(current_direction)

        trajectory.append((current_lat, current_lon))

    return trajectory

def main():
    # Parameters
    epsilon = 0.1  # Privacy parameter (lower = stronger privacy)
    qos_radius = 175  # Quality of service radius in meters (between 150-200m)

    # Create trajectory privacy object
    trajectory_privacy = TrajectoryPrivacy(epsilon=epsilon, qos_radius=qos_radius)

    # Generate a realistic trajectory in San Francisco area
    # Center point of San Francisco
    sf_center_lat = 37.7749
    sf_center_lon = -122.4194

    # Generate realistic trajectory following road network
    random_trajectory = generate_realistic_trajectory(sf_center_lat, sf_center_lon)

    print(f"Generated random trajectory with {len(random_trajectory)} points")

    # Load spatial constraints for the area around the trajectory
    min_lat = min(point[0] for point in random_trajectory) - 0.01
    min_lon = min(point[1] for point in random_trajectory) - 0.01
    max_lat = max(point[0] for point in random_trajectory) + 0.01
    max_lon = max(point[1] for point in random_trajectory) + 0.01

    print("Loading spatial constraints from OpenStreetMap...")
    trajectory_privacy.load_spatial_constraints((min_lat, min_lon, max_lat, max_lon))
    print("Spatial constraints loaded.")

    # Process trajectory
    print("Generating privacy-preserving trajectory...")
    for lat, lon in random_trajectory:
        fake_lat, fake_lon = trajectory_privacy.add_point(lat, lon)
        print(f"Real point: {lat:.6f}, {lon:.6f} -> Fake point: {fake_lat:.6f}, {fake_lon:.6f}")

    # Snap to road network for increased realism
    print("Snapping fake trajectory to road network...")
    trajectory_privacy.snap_to_road_network()

    # Also snap real trajectory to road network for consistent visualization
    print("Snapping real trajectory to road network...")
    trajectory_privacy.snap_real_trajectory_to_road_network()

    # Visualize both trajectories
    print("Creating visualization...")
    map_viz = trajectory_privacy.visualize_trajectories(zoom_start=14)

    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"trajectory_privacy_map_{timestamp}.html"

    # Save the map
    map_viz.save(output_file)
    print(f"Map visualization saved to {os.path.abspath(output_file)}")

    # Evaluate privacy metrics
    metrics = trajectory_privacy.evaluate_privacy()
    print("\nPrivacy Metrics:")
    print(f"Average distance between real and fake points: {metrics['avg_distance']:.2f} meters")
    print(f"Maximum distance between real and fake points: {metrics['max_distance']:.2f} meters")
    print(f"QoS satisfaction rate: {metrics['qos_satisfaction'] * 100:.1f}%")

if __name__ == "__main__":
    main()

