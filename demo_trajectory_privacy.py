"""
Demo script for trajectory privacy protection.
"""
import folium
import numpy as np
from trajectory_privacy import TrajectoryPrivacy
import os
import datetime

def main():
    # Parameters
    epsilon = 0.1  # Privacy parameter (lower = stronger privacy)
    qos_radius = 175  # Quality of service radius in meters (between 150-200m)

    # Create trajectory privacy object
    trajectory_privacy = TrajectoryPrivacy(epsilon=epsilon, qos_radius=qos_radius)

    # Sample trajectory (San Francisco area)
    # This would typically come from real user movement
    sf_trajectory = [
        (37.7749, -122.4194),  # Starting point
        (37.7746, -122.4183),
        (37.7742, -122.4167),
        (37.7737, -122.4155),
        (37.7732, -122.4141),
        (37.7725, -122.4125),
        (37.7718, -122.4110),
        (37.7711, -122.4097),
        (37.7705, -122.4085),
        (37.7698, -122.4071),
        (37.7691, -122.4059),
        (37.7685, -122.4045)   # End point
    ]

    # Load spatial constraints for the area around the trajectory
    min_lat = min(point[0] for point in sf_trajectory) - 0.01
    min_lon = min(point[1] for point in sf_trajectory) - 0.01
    max_lat = max(point[0] for point in sf_trajectory) + 0.01
    max_lon = max(point[1] for point in sf_trajectory) + 0.01

    print("Loading spatial constraints from OpenStreetMap...")
    trajectory_privacy.load_spatial_constraints((min_lat, min_lon, max_lat, max_lon))
    print("Spatial constraints loaded.")

    # Process trajectory
    print("Generating privacy-preserving trajectory...")
    for lat, lon in sf_trajectory:
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
