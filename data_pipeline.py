import numpy as np
from os.path import join, basename
from visualize import get_3d_bbox
from data_utils import get_files,load_point_cloud, load_labels, load_calibration, \
    apply_calibration, create_x_rotation_matrix


def get_analysis(pointcloud_folder,label_folder,calib_folder):
    pointcloud_files,label_files,calibration_files = get_files(pointcloud_folder,label_folder,calib_folder)
    print('Found',len(pointcloud_files),'PoCl |',len(label_files),'Lbl |',len(calibration_files),'Cal Files!')

    min_obj_corner,max_obj_corner = np.array([1e8,1e8,1e8]),np.array([1e-8,1e-8,1e-8])
    min_point, max_point = np.array([1e8,1e8,1e8,1e8]),np.array([1e-8,1e-8,1e-8,1e-8])
    min_dim, max_dim = np.array([1e8,1e8,1e8]),np.array([1e-8,1e-8,1e-8])
    min_loc, max_loc = np.array([1e8,1e8,1e8,1e8]),np.array([1e-8,1e-8,1e-8,1e-8])
    pocl_files,calib_files = list(),list()
    for bin_path,label_path,calib_path in zip(pointcloud_files,label_files,calibration_files):
        pocl_file = join(pointcloud_folder,bin_path)
        calib_file = join(calib_folder,calib_path)
        pocl_files.append(pocl_file)
        calib_files.append(calib_file)
        point_cloud = load_point_cloud(pocl_file)
        labels = load_labels(join(label_folder,label_path))
        calib = load_calibration(calib_file)

        point_cloud = apply_calibration(point_cloud,calib)
        rotation_matrix = create_x_rotation_matrix()
        point_cloud = np.hstack((np.dot(point_cloud[:, :3], rotation_matrix.T), point_cloud[:, 3:]))
        min_point = np.array([np.minimum(point_cloud[:,i].min(),min_point[i]) for i in range(point_cloud.shape[1])])
        max_point = np.array([np.maximum(point_cloud[:,i].max(),max_point[i]) for i in range(point_cloud.shape[1])])

        for label in labels:
            rotated_corners = get_3d_bbox(label['dimensions'],label['location'],label['rotation_y'])
            if label['label']!='DontCare':
                min_dim  = [np.minimum(label['dimensions'][i], min_dim[i]) for i in range(len(label['dimensions']))]
                max_dim = [np.maximum(label['dimensions'][i], max_dim[i]) for i in range(len(label['dimensions']))]
                min_loc  = [np.minimum(label['location'][i], min_loc[i]) for i in range(len(label['location']))]
                max_loc = [np.maximum(label['location'][i], max_loc[i]) for i in range(len(label['location']))]
                min_obj_corner = np.array([np.minimum(rotated_corners[:,i].min(),min_obj_corner[i]) for i in range(rotated_corners.shape[1])])
                max_obj_corner = np.array([np.maximum(rotated_corners[:,i].max(),max_obj_corner[i]) for i in range(rotated_corners.shape[1])])
    
    min_dim[1],min_dim[2] = min_dim[2],min_dim[1]
    max_dim[1],max_dim[2] = max_dim[2],max_dim[1]
    min_loc[1],min_loc[2] = min_loc[2],min_loc[1]
    max_loc[1],max_loc[2] = max_loc[2],max_loc[1]
    print("Min. Lbl. Location:",[round(float(min_loc[i]),2) for i in range(len(min_loc))])
    print("Max. Lbl. Location:",[round(float(max_loc[i]),2) for i in range(len(max_loc))])
    print("Min. Lbl. Dimension:",[round(float(min_dim[i]),2) for i in range(len(min_dim))])
    print("Max. Lbl. Dimension:",[round(float(max_dim[i]),2) for i in range(len(max_dim))])
    print("Objects in area spanning from",[round(float(coord),2) for coord in min_obj_corner],"to",[round(float(coord),2) for coord in max_obj_corner])
    print("Points in area spanning from",[round(float(coord),2) for coord in min_point],"to",[round(float(coord),2) for coord in max_point])
    minimal_area = [[round(float(np.maximum(min_loc[i],np.maximum(min_obj_corner[i],min_point[i]))),2) \
          for i in range(len(min_loc))],[round(float(np.minimum(max_loc[i],np.minimum(max_obj_corner[i],max_point[i]))),2) \
          for i in range(len(max_loc))]]
    maximal_area = [round(float(np.maximum(min_point[i],np.minimum(min_obj_corner[i],min_loc[i]))),2) \
          for i in range(len(min_loc))],[round(float(np.minimum(max_point[i],np.maximum(max_obj_corner[i],max_loc[i]))),2) \
          for i in range(len(max_loc))]
    print("Minimal Area of Objects:",minimal_area[0],"to",minimal_area[1])
    print("Maximal Area of Objects:",maximal_area[0],"to",maximal_area[1])
    return minimal_area, maximal_area, pocl_files, calib_files


def gen_cut_out_dataset(point_cloud_files, calib_files, area_borders, path="./"):
    folder = join(path,'LABELED_Area_PoCls')
    point_numbers = list()
    for pocl_file,calib_file in zip(point_cloud_files, calib_files):
        pocl = load_point_cloud(pocl_file)
        calib = load_calibration(calib_file)

        pocl = apply_calibration(pocl,calib)
        rotation_matrix = create_x_rotation_matrix()
        pocl = np.hstack((np.dot(pocl[:, :3], rotation_matrix.T), pocl[:, 3:]))
        mask = (pocl[:,0]>area_borders[0][0]) & (pocl[:,1]>area_borders[0][1]) & (pocl[:,2]>area_borders[0][2]) \
               & (pocl[:,0]<area_borders[1][0]) & (pocl[:,1]<area_borders[1][1]) & (pocl[:,2]<area_borders[1][2])
        pocl = pocl[mask]
        point_numbers.append(pocl.shape[0])
        pocl.tofile(join(folder,basename(pocl_file)))
    print('Min point numbers:', min(point_numbers))
    print('AvG point numbers:', round(sum(point_numbers)/len(point_numbers)))
    print('Max point numbers:', max(point_numbers))


if __name__=="__main__":
    pointcloud_folder = '/path/to/KITTI_Dataset/training/velodyne'
    label_folder = '/path/to/KITTI_Dataset/training/label_2'
    calib_folder = '/path/to/KITTI_Dataset/training/calib'

    minimal_area, maximal_area, pocl_files, calib_files = get_analysis(pointcloud_folder,label_folder,calib_folder)
    gen_cut_out_dataset(pocl_files, calib_files, maximal_area)
    # TODO: Patching - Visualize them too
    #  -> Dense_1 1/6 close to center(0), Dense2 1/6 from 1/7 to 13/42
    #  -> Medium Dense_1 1/5 from 1/4 to 9/20, Medium Dense_2 1/5 from 2/5 to 3/5
    #  -> Least Dense_1 1/4 from 1/2 to 3/4, Least Dense_2 1/3 from 2/3 to 1
