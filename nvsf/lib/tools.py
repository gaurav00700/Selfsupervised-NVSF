import os
import copy
import time
from datetime import datetime
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import yaml
import json
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, Delaunay
from pyquaternion import Quaternion
from typing import List, Union

def load_yaml(file_path:str):
    '''
    Load yaml file using file path
    :param file_path:str, path of yaml file
    :return:dict, serialised data in the form of dict
    '''
    # Load box24 yaml file
    with open(file_path, "r") as stream:
        try:
            data_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data_yaml

def load_json(file_path:str):
    '''
    Load Json file using path
    :param file_path:str, file path
    :return:dict, json file (dict)
    '''
    with open(file_path) as f:
        f_out = json.load(f)
    return f_out

def save_json(data:dict, save_path:str ='', file_name:str='data'):
    '''
    Save dict to json file
    :param data:dict, input_data
    :param save_path:str, path to save json file
    :param file_name:str, file name
    :return: save .json file at provided path
    '''
    def NumpyEncoder(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))
    file_name = file_name + '_' + str(int(time.time())) + '.json'
    with open(os.path.join(save_path, file_name), 'w') as fp:
        json.dump(data, fp, default=NumpyEncoder, indent=4)

def load_csv(csv_path:str):
    '''
    Load csv file from path
    :param csv_path:str, path to csv file
    :return:
    '''

    # Read csv file
    df_input = pd.read_csv(csv_path, index_col=False)

    # Drop empty rows of df
    # df_input.drop(df_input[(df_input.Bbox_3d_id == '[nan]') & (df_input.Bbox_2d == '[nan]') & (df_input.pcd_3d == '[nan]') & (df_input.cam == '[nan]')].index, inplace=True)
    df_input.drop(df_input[df_input.pcd_3d == '[]'].index, inplace=True)
    # df_input.dropna(inplace=True)

    # Convert df dtype from string to list
    import ast
    # Change the df dtype from string to list
    input_cols = ['Bbox_3d_id', 'Bbox_3d', 'Bbox_2d', 'pcd_3d', 'cam', 'gnss', 'box_calib']
    for col in input_cols:
        df_input[col] = df_input[col].apply(str)  # convert to string to prevent eval error
        df_input[col] = df_input[col].apply(ast.literal_eval)  # convert string items to true dtype

    return df_input

def filter_data_outliers(data: list, std_limit:tuple=(-3,3)):
    '''
    Filter data for outliers using std (sigma) limit
    +-1 std from the Mean: 68%
    +-2 std from the Mean: 95%
    +-3 std from the Mean: 99.7%
    :param data:list, For data to be filtered
    :param std_limit:tuple, For std limit
    :return:
    list, filtered data
    list, outliers data
    '''
    # calculate summary statistics
    data_mean, data_std = np.nanmean(data), np.nanstd(data)
    # identify outliers
    #cut_off = data_std * 2
    l_limit = data_mean + data_std * std_limit[0]
    u_limit = data_mean + data_std * std_limit[1]

    # identify outliers
    # data_outliers = [x for x in data if x < l_limit or x > u_limit]

    # remove outliers
    # data_wo_outliers = [x for x in data if x >= l_limit and x <= u_limit]
    data_wo_outliers = []
    index = []
    for i, item in enumerate(data):
        if item >= l_limit and item <= u_limit:
            data_wo_outliers.append(item)
            index.append(i)

    return data_wo_outliers, index

def euler_to_quat(rot_xyz):
    '''
    Convert euler angles (xyz) to quaternions angles
    :param rot_xyz:list, angles in radians
    :return:list, quaternions angles
    '''

    r = Rotation.from_euler('xyz', rot_xyz, degrees=False)
    return r.as_quat()

def quat_to_euler(rot_xyzw):
    '''
    Convert quaternions to euler angles
    :param rot_xyzw:list, quaternions angles
    :return:list, euler angles (xyz) in radians
    '''

    r = Rotation.from_quat(rot_xyzw)
    return r.as_euler('xyz', degrees=False)


def check_in_hull(point, hull):
    """Check if pcd points (`p`) in convex hull or not
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation will be computed.

    Args:
        point (array): pcd points
        hull (array | hull): object or np.array, hull object or 'MxK' array

    Returns:
        list : pcd belongs to hull
        list : hull mask
    """
    
    # Get hull object
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    
    # compute hull mask
    inhull_mask = hull.find_simplex(point[:, :3]) >= 0

    return point[inhull_mask], inhull_mask

class dict_to_cls:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def get_bbox3d_corner_points(
    position:Union[List[float], np.ndarray], 
    size:Union[List[float], np.ndarray], 
    rotation_quaternion: Union[List[float], np.ndarray]
    )-> np.ndarray:
    """Create 3d bounding boxes corner points using position, size and orientation

    args:
        - position(list): center coordinate of 3d bbox in format (x, y, z)
        - size(list): size of 3d bbox in format (l, w, h)
        - rotation_quaternion(list): orientation of 3dbbox, quaternion in format (w, x, y, z)
    return:
        - corner_points(array): eight corner points of 3d bbox (8, 3)
    """
    length, width, height = size
    #1: Convert quaternion to rotation matrix
    rotation_matrix = Quaternion(rotation_quaternion).rotation_matrix # (w, x, y, z) format
    # rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix() #  (x, y, z, w) format

    #2: Calculate half dimensions
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    #3: Calculate corner offsets
    corner_offsets = [
        np.array([half_length, half_width, half_height]),
        np.array([-half_length, half_width, half_height]),
        np.array([-half_length, -half_width, half_height]),
        np.array([half_length, -half_width, half_height]),
        np.array([half_length, half_width, -half_height]),
        np.array([-half_length, half_width, -half_height]),
        np.array([-half_length, -half_width, -half_height]),
        np.array([half_length, -half_width, -half_height])
    ]
    
    #4: rotate offset values using rotation matrix
    rotated_corner_offsets = [np.dot(rotation_matrix, offset) for offset in corner_offsets]
    
    #5: Calculate coordinates of corner points using position and offsets 
    corner_points = np.asarray([position + offset for offset in rotated_corner_offsets])

    return corner_points

def poses_to_T(pose:dict, heading:dict) -> np.ndarray:
    """Convert postions and heading dict to Transformation matrix
    Args:
        - pose (dict): x, y, z
        - heading (dict): x, y, z, w
    Return:
        - T_mat (np.ndarray): 4x4 Transformation matrix
    """
    # Ego vehicle position and heading
    pose = np.array(list(pose.values())) # (x, y, z)
    r = Rotation.from_quat(list(heading.values())) #(x, y, z, w) scalar-last
    heading = r.as_euler('xyz', degrees=False)
    
    #create transformation matrix from gnss to world frame
    T_mat = np.eye(4) # 4x4
    T_mat[:3, :3] = r.as_matrix()
    T_mat[:3, 3] = pose
    return T_mat

def convert_to_o3dpcd(np_pcd:np.ndarray, save_path:str=None) -> o3d.geometry.PointCloud:
    """Convert numpy point cloud to open3d point cloud
    Args:
       np_pcd(np.ndarray): pcd with 3 or 4 channels
    """
    pcd = o3d.t.geometry.PointCloud() #create pcd object using geometry tensor class
    pcd.point['positions'] = np_pcd[:, :3] #set 'positions' attribute to  pcd.point
    
    # set intensity attribute
    if np_pcd.shape[1] == 4:
        pcd.point['intensity'] = np_pcd[:, 3:] #set 'intensity' attribute to pcd.point 

    #save pcd
    if save_path:
        o3d.t.io.write_point_cloud( 
                            save_path, 
                            pcd, 
                            write_ascii=True
                            ) 
    return pcd

