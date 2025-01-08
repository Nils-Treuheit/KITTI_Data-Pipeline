import numpy as np
from os import listdir

def get_files(pointcloud_folder,label_folder,calib_folder):
    pointcloud_files = sorted([f for f in listdir(pointcloud_folder) if f.endswith('.bin')])
    label_files = sorted([f for f in listdir(label_folder) if f.endswith('.txt')])
    calib_files = sorted([f for f in listdir(calib_folder) if f.endswith('.txt')])
    return pointcloud_files, label_files, calib_files


def load_point_cloud(bin_path):
    """Load the point cloud from a KITTI .bin file."""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud


def load_labels(label_path):
    """Load labels from a KITTI label file."""
    boxes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            data = line.split()
            box = dict()
            box['label'] = data[0]
            box['truncated'] = float(data[1])
            box['occluded'] = int(data[2])
            box['alpha'] = float(data[3]) # observation angle -pi to pi
            box['bbox'] = np.array(data[4:8], dtype=np.float32) # left,top,right,bottom
            box['dimensions'] = np.array(data[8:11], dtype=np.float32) # height, width, length
            box['location'] = np.array(data[11:14], dtype=np.float32) # x, y, z
            box['rotation_y'] = float(data[14]) # y-axis angle -pi to pi
            boxes.append(box)
    return boxes


def load_calibration(calib_path):
    """
    Load KITTI calibration file and parse transformation matrices.
    Args:
        calib_path: Path to the calibration file.
    Returns:
        A dictionary with calibration matrices.
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if not(':' in line): continue
            key, value = line.split(':', 1)
            # Flatten the values into a numpy array
            calib[key] = np.array([float(x) for x in value.split()])
    
    # Reshape matrices where necessary
    calib['P2'] = calib['P2'].reshape(3, 4)  # Projection matrix
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)  # Rectification matrix
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)  # Lidar-to-camera transformation
    return calib


def transform_point_cloud_to_camera(point_cloud, Tr_velo_to_cam, R0_rect):
    """
    Transform point cloud from lidar to camera coordinates.
    Args:
        point_cloud: Nx4 array (x, y, z, intensity).
        Tr_velo_to_cam: Transformation matrix (3x4).
        R0_rect: Rectification matrix (4x4).
    """
    point_cloud_homogeneous = np.hstack((point_cloud[:, :3],np.ones((point_cloud.shape[0], 1))))

    # Apply velo-to-cam transformation
    points_cam = np.dot(point_cloud_homogeneous, Tr_velo_to_cam.T)
    
    # Apply rectification
    points_rect = np.dot(points_cam,R0_rect.T)

    # Append intensity (if available) to the transformed points
    if point_cloud.shape[1] > 3:
        return np.hstack((points_rect[:, :3], point_cloud[:, 3:]))  # Nx4
    else:
        return points_rect[:, :3]  # Nx3


def apply_calibration(point_cloud, calib):
    """
    Transform to align point cloud with labels using calibration files.
    """
    # Extract transformation matrices
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    R0_rect = calib['R0_rect']

    # Transform point cloud
    point_cloud_transformed = transform_point_cloud_to_camera(point_cloud, Tr_velo_to_cam, R0_rect)
    return point_cloud_transformed


def create_x_rotation_matrix():
    """
    Create a rotation matrix that aligns the axes as required:
    - X -> Length
    - Y -> Width
    - Z -> Height (upwards)
    This will rotate the point cloud and labels.
    """
    rotation_matrix = np.array([
        [1, 0, 0],  # X -> X
        [0, 0, 1],  # Y -> Z
        [0, -1, 0]   # Z -> Y
    ])  # This is a 90 degree rotation around X-axis
    return rotation_matrix


def get_3d_bbox(dimensions, location, rotation_y):
    """Generate a 3D bounding box from dimensions, location, and rotation."""
    h, w, l = dimensions
    x, y, z = location

    # Create a bounding box around the origin
    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners])

    # Rotation around Y-axis
    rotation_matrix = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    # Apply rotation and translation
    rotated_corners = np.dot(rotation_matrix, corners)
    rotated_corners[0, :] += x
    rotated_corners[1, :] += y
    rotated_corners[2, :] += z
    rotation_matrix = create_x_rotation_matrix()
    rotated_corners = np.dot(rotated_corners.T, rotation_matrix.T)
    return rotated_corners
