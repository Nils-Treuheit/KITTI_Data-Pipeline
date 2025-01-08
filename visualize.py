import os
import numpy as np
import plotly.io as pio
pio.renderers.default = 'notebook'
import plotly.graph_objects as go
from ipywidgets import interact,IntSlider
from data_utils import get_files,load_point_cloud,load_labels,load_calibration,\
    apply_calibration,create_x_rotation_matrix,get_3d_bbox

BBOX_COLORS = {
    'Car': 'lime', 
    'Van': 'greenyellow', 
    'Truck': 'darkgreen',
    'Pedestrian': 'red', 
    'Person_sitting': 'orange', 
    'Cyclist': 'deeppink', 
    'Tram': 'aquamarine',
    'Misc': 'cornflowerblue',
    'DontCare': 'silver' 
}


def generate_plane_two_corners(corner1, corner2, z_fixed=None, color='blue'):
    """
    Plot a 3D scatter plot with a rectangular plane defined by two corners.
    Args:
        corner1: First corner of the rectangle (x1, y1, z1).
        corner2: Opposite corner of the rectangle (x2, y2, z2).
        z_fixed: Fix Z-coordinate for the plane (optional, overrides corner Z).
    """
    # Extract coordinates for corners
    x1, y1, z1 = corner1
    x2, y2, z2 = corner2

    # Optionally fix the Z-coordinate (e.g., if the plane should be flat)
    if z_fixed is not None:
        z1 = z2 = z_fixed
        corner1[2] = z_fixed
        corner2[2] = z_fixed

    # Calculate the other two corners of the rectangle
    corner3 = (x2, y1, z1)  # Same X as corner2, same Y as corner1
    corner4 = (x1, y2, z2)  # Same X as corner1, same Y as corner2

    # Plane corners in 3D space
    plane_corners = [corner1, corner3, corner2, corner4]

    # Extract plane coordinates for plotting
    x_plane = [corner[0] for corner in plane_corners]
    y_plane = [corner[1] for corner in plane_corners]
    z_plane = [corner[2] for corner in plane_corners]

    # Create the plane
    plane = go.Mesh3d(
        x=x_plane,
        y=y_plane,
        z=z_plane,
        color=color,
        opacity=0.25,
        i=[0, 1, 2],  # Triangle 1
        j=[1, 2, 3],  # Triangle 2
        k=[2, 3, 0],  # Triangle 3
    )
    return plane


def plot_point_cloud_with_labels(point_cloud, labels, include_cls=None, planes=None):
    if include_cls == None:
        include_cls = ['Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 
                       'Cyclist', 'Tram', 'Misc'] # do not include 'DontCare'
    scatter = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1, color=point_cloud[:, 3], colorscale='Viridis', opacity=0.8)
    )

    bbox_traces = []
    for lbl in labels:
        if lbl['label'] in include_cls:
            corners = get_3d_bbox(lbl['dimensions'], lbl['location'], lbl['rotation_y'])
            x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            for start, end in edges:
                bbox_traces.append(go.Scatter3d(
                    x=[x[start], x[end]],
                    y=[y[start], y[end]],
                    z=[z[start], z[end]],
                    mode='lines',
                    line=dict(color=BBOX_COLORS[lbl['label']], width=2)
                ))

    data = [scatter] if planes == None else [scatter, *planes]
    fig = go.Figure(data=data + bbox_traces)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'))
    return fig


def visualize_point_cloud(pointcloud_folder, label_folder, calib_folder, minimal_area,maximal_area):
    """Interactive visualization of point clouds and labels."""
    pointcloud_files,label_files,calib_files = get_files(pointcloud_folder,label_folder,calib_folder)
    
    min_plane = generate_plane_two_corners(minimal_area[0], minimal_area[1], z_fixed=0., color='green')
    max_plane = generate_plane_two_corners(maximal_area[0], maximal_area[1], z_fixed=0., color='red')
    def update(index):
        bin_path = os.path.join(pointcloud_folder, pointcloud_files[index])
        label_path = os.path.join(label_folder, label_files[index])
        calib_path = os.path.join(calib_folder, calib_files[index])

        point_cloud = load_point_cloud(bin_path)
        labels = load_labels(label_path)
        calib = load_calibration(calib_path)

        point_cloud = apply_calibration(point_cloud,calib)
        rotation_matrix = create_x_rotation_matrix()
        point_cloud = np.hstack((np.dot(point_cloud[:, :3], rotation_matrix.T), point_cloud[:, 3:]))
        
        fig = plot_point_cloud_with_labels(point_cloud, labels, planes=[min_plane,max_plane])
        fig.show()

    interact(update, index=IntSlider(min=0, max=len(pointcloud_files) - 1, step=1, description='File Index'))
