from typing import Literal, Union
from networkx import out_degree_centrality
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
import glob
import json
from scipy.spatial.transform import Rotation
from collections import defaultdict
import matplotlib.pyplot as plt
from nvsf.lib import convert
from nvsf.lib.convert import (
    lidar_to_pano_with_intensities,
    lidar_to_pano_with_intensities_with_bbox_mask
    )

def all_points_to_world(pcd_path_list, lidar2world_list):
    pc_w_list = []
    for i, pcd_path in enumerate(pcd_path_list):
        point_cloud = np.load(pcd_path)
        point_cloud[:, -1] = 1
        points_world = (point_cloud @ (lidar2world_list[i].reshape(4, 4)).T)[:, :3]
        pc_w_list.append(points_world)
    return pc_w_list


def oriented_bounding_box(data):
    data_norm = data - data.mean(axis=0)
    C = np.cov(data_norm, rowvar=False)
    vals, vecs = np.linalg.eig(C)
    vecs = vecs[:, np.argsort(-vals)]
    Y = np.matmul(data_norm, vecs)
    offset = 0.03
    xmin = min(Y[:, 0]) - offset
    xmax = max(Y[:, 0]) + offset
    ymin = min(Y[:, 1]) - offset
    ymax = max(Y[:, 1]) + offset

    temp = list()
    temp.append([xmin, ymin])
    temp.append([xmax, ymin])
    temp.append([xmax, ymax])
    temp.append([xmin, ymax])

    pointInNewCor = np.asarray(temp)
    OBB = np.matmul(pointInNewCor, vecs.T) + data.mean(0)
    return OBB


def get_dataset_bbox(all_class, dataset_root, out_dir):
    object_bbox = {}
    for class_name in all_class:
        lidar_path = os.path.join(dataset_root, class_name)
        rt_path = os.path.join(lidar_path, "lidar2world.txt")
        filenames = os.listdir(lidar_path)
        filenames.remove("lidar2world.txt")
        filenames.sort(key=lambda x: int(x.split(".")[0]))
        show_interval = 1
        pcd_path_list = [os.path.join(lidar_path, filename) for filename in filenames][
            ::show_interval
        ]
        print(f"{lidar_path}: {len(pcd_path_list)} frames")
        lidar2world_list = list(np.loadtxt(rt_path))[::show_interval]
        all_points = all_points_to_world(pcd_path_list, lidar2world_list)
        pcd = np.concatenate(all_points).reshape((-1, 3))

        OBB_xy = oriented_bounding_box(pcd[:, :2])
        z_min, z_max = min(pcd[:, 2]), max(pcd[:, 2])
        OBB_buttum = np.concatenate([OBB_xy, np.tile(z_min, 4).reshape(4, 1)], axis=1)
        OBB_top = np.concatenate([OBB_xy, np.tile(z_max, 4).reshape(4, 1)], axis=1)
        OBB = np.concatenate([OBB_top, OBB_buttum])
        object_bbox[class_name] = OBB
    np.save(os.path.join(out_dir, "dataset_bbox_7k.npy"), object_bbox)


def LiDAR_2_Pano_NeRF_MVL(
    local_points_with_intensities,
    lidar_H,
    lidar_W,
    intrinsics,
    OBB_local,
    max_depth=80.0,
):
    pano, intensities = lidar_to_pano_with_intensities_with_bbox_mask(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        bbox_local=OBB_local,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


def generate_nerf_mvl_train_data(
    H,
    W,
    intrinsics,
    all_class,
    dataset_bbox,
    nerf_mvl_parent_dir,
    out_dir,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    for class_name in all_class:
        OBB = dataset_bbox[class_name]
        lidar_path = os.path.join(nerf_mvl_parent_dir, "nerf_mvl_7k", class_name)
        filenames = os.listdir(lidar_path)
        filenames.remove("lidar2world.txt")
        filenames.sort(key=lambda x: int(x.split(".")[0]))
        save_path = os.path.join(out_dir, class_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy(
            os.path.join(lidar_path, "lidar2world.txt"),
            os.path.join(save_path, "lidar2world.txt"),
        )
        lidar2world = np.loadtxt(os.path.join(lidar_path, "lidar2world.txt"))
        avaliable_frames = [i for i in range(0, len(filenames))]
        print(class_name, " frames num ", len(avaliable_frames))
        for idx in tqdm(avaliable_frames):
            pcd = np.load(os.path.join(lidar_path, filenames[idx]))
            OBB_local = (
                np.concatenate([OBB, np.ones((8, 1))], axis=1)
                @ np.linalg.inv(lidar2world[idx].reshape(4, 4)).T
            )
            pano = LiDAR_2_Pano_NeRF_MVL(pcd, H, W, intrinsics, OBB_local)
            np.savez_compressed(
                os.path.join(save_path, "{:010d}.npz").format(idx), data=pano
            )


def create_nerf_mvl_rangeview(H_lidar:int= 256, W_lidar:int=1800, fov_up:float=15, fov:float=40):
    project_root = Path(__file__).parent.parent
    nerf_mvl_root = project_root / "data" / "nerf_mvl" / "nerf_mvl_7k"
    nerf_mvl_parent_dir = nerf_mvl_root.parent
    out_dir = nerf_mvl_parent_dir / "nerf_mvl_7k_pano"

    all_class = [
        "water_safety_barrier",
        "tire",
        "pier",
        "plant",
        "warning_sign",
        "traffic_cone",
        "bollard",
        "pedestrian",
        "car",
    ]

    # get_dataset_bbox
    if not os.path.exists(os.path.join(nerf_mvl_parent_dir, "dataset_bbox_7k.npy")):
        get_dataset_bbox(all_class, nerf_mvl_root, nerf_mvl_parent_dir)
    dataset_bbox = np.load(
        os.path.join(nerf_mvl_parent_dir, "dataset_bbox_7k.npy"), allow_pickle=True
    ).item()

    # generate train rangeview images
    intrinsics = (fov_up, fov)
    generate_nerf_mvl_train_data(
        H=H_lidar,
        W=W_lidar,
        intrinsics=intrinsics,
        all_class=all_class,
        dataset_bbox=dataset_bbox,
        nerf_mvl_parent_dir=nerf_mvl_parent_dir,
        out_dir=out_dir,
    )


def LiDAR_2_Pano(
        local_points_with_intensities:np.ndarray, 
        lidar_H:int, 
        lidar_W:int, 
        intrinsics:list[float], 
        intrinsics_hoz:list[float],
        max_depth:float
    ):
    """
    Convert lidar pcd to Range images
    Args:
        - local_points_with_intensities(list): List of point cloud data (x, y, z, i)
        - lidar_H(int): Height of Range image
        - lidar_W(int): Width of Range image
        - intrinsics(list): list of lidar intrinsics parameters in deg (fov_up, fov)
        - intrinsics_hoz(list, optional): list of horizontal lidar intrinsics parameters in deg (fov_up, fov)
        - num_returns(int, optional): number of returns of lidar beam, default is 1
        - max_depth(int): Range of lidar depth
    """
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        lidar_K_hoz=intrinsics_hoz,
        max_depth=max_depth,
    )

    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano

    return range_view

def generate_train_data(
    H:int,
    W:int,
    intrinsics:list[float],
    intrinsics_hoz:list[float],
    max_depth:float,
    points_dim:int,
    lidar_paths:list[str],
    out_dir:str,
):
    """
    Args:
        - H: Heights of the range view.
        - W: Width of the range view.
        - intrinsics: lidar vertical fov in degree (fov_up, fov) of the range view.
        - intrinsics_hoz: lidar horizontal fov in degree (fov_up, fov) of the range view.
        - max_depth: Maximum range of lidar in m.
        - poins_dim: Dimension of point cloud data
        - lidar_paths: list of lidar point clouds paths
        - out_dir: Output directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lidar_path in tqdm(lidar_paths, desc="Converting and saving lidar to pano (.npy)"):
        #Get pcd data
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        
        #Convert to pano/range image
        pano = LiDAR_2_Pano(point_cloud, H, W, intrinsics, intrinsics_hoz, max_depth)

        #get frame name
        frame_name = lidar_path.split("/")[-1]
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy") 

        #save pano/range image in .npy file
        np.save(out_dir / frame_name, pano) 

def create_kitti_rangeview(
        H_lidar:int= 66,
        W_lidar:int= 1030,
        fov_up:float= 2.0,          #vertical   
        fov:float= 26.9,            #vertical
        fov_hoz_up:float= 180.0,    #horizontal   
        fov_hoz:float= 360.0,       #horizontal   
        lidar_range:float=80.0,
        points_dim:int=4.0,
        recording_name:str = "2013_05_28_drive_0000",
        sequence_name:str= "1908",
        out_dir:str=None
    ):
    
    #Path information
    project_root = Path(__file__).parent.parent
    kitti_360_root = project_root / "data" / "kitti360" / "source_data"
    kitti_360_parent_dir = kitti_360_root.parent
    
    #output directory
    if not out_dir:
        out_dir = kitti_360_parent_dir / "train" / sequence_name
    else:
        out_dir = Path(out_dir)

    # Get target frame ids.
    frame_ids = list(range(int(sequence_name), int(sequence_name) + 64))
    
    # intrinsics
    intrinsics = fov_up, fov
    intrinsics_hoz = fov_hoz_up, fov_hoz
    
    # Get all lidar .bin files directories
    lidar_dir = (
        kitti_360_root
        / "data_3d_raw"
        / f"{recording_name}_sync"
        / "velodyne_points"
        / "data"
    )
    lidar_paths = [
        os.path.join(lidar_dir, "%010d.bin" % frame_id) for frame_id in frame_ids
    ]  

    # Generate train rangeview images and save to output directory.
    generate_train_data(
        H=H_lidar,
        W=W_lidar,
        intrinsics=intrinsics,
        intrinsics_hoz=intrinsics_hoz,
        max_depth=lidar_range,
        points_dim=points_dim,
        lidar_paths=lidar_paths,
        out_dir=out_dir,
    )

def create_daas_rangeview(
            H_lidar:int,
            W_lidar:int,
            fov_up:float,           
            fov:float,
            fov_hoz_up:float,   
            fov_hoz:float,     
            lidar_range:float,
            points_dim:int,
            sequence_name:str,    
            out_dir=None,
            save_pcd=False,
            loader=None
    ):
    """Process raw pcd files to generate range/pano view images for AVL DaaS dataset.

    Args:
        - H_lidar (int, optional): Horizontal size of pano image. Defaults to 256.
        - W_lidar (int, optional): Vertical size of pano image. Defaults to 1800.
        - fov_up (float, optional): Upper (+ve) vertical field of view of lidar. Defaults to 15.
        - fov (float, optional): Full vertical field of view of lidar. Defaults to 40.
        - fov_hoz_up (float, optional): Upper (+ve) horizontal field of view of lidar. Defaults to 180.0.
        - fov_hoz (float, optional): Full horizontal field of view of lidar. Defaults to 360.0.
        - lidar_range (float, optional): Range of lidar sensor. Defaults to 245.
        - points_dim (float, optional): Number of dimensions of lidar pcd. Defaults to 4.
        - sequence_name (str, optional): Name of sequence. Defaults to "CityStreet_dgt_2021-07-15-13-31-59_0_s0".
        - out_dir (_type_, optional): Path to save processed pano images. Defaults to None.
        - save_pcd (bool, optional): Flag to save pcd to txt file. Defaults to False.
        - loader (AVL_loader, optional): Data loader object. Defaults to None.
    """

    #Path information
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "daas" / "source_data"
    root_dir_parent = data_root.parent
    
    #output directory
    if out_dir is None:
        out_dir = root_dir_parent / "train" / sequence_name
    
    # Lidar vertical and horizontal intrinsics
    intrinsics = fov_up, fov
    intrinsics_hoz = fov_hoz_up, fov_hoz

    # Get all lidar files directories
    raw_data_dir = data_root / sequence_name
    lidar_file_paths = glob.glob(os.path.join(raw_data_dir / "dataset/*.json"))
    cam_file_paths = glob.glob(os.path.join(raw_data_dir / "dataset/*.png"))
    
    # make directories
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Daas data parser
    if not loader: loader = AVL_loader.DaaS_loader(raw_data_dir)
    pcd_world_frames = loader.pcd_world_frames
    T_lidar2world_frames = loader.T_lidar2world_frames

    # convert pcd from world frame to lidar frame
    pcd_lidar_frames = defaultdict(lambda: defaultdict(list))
    lidar_ids = ['RSFord_SHC_LF', 'RSFord_SHC_LN', 'RSFord_SHR_LN', 'RSFord_SHL_LN']
    for (frame, pcd_world), (_, Tr_lidar2world) in zip(pcd_world_frames.items(), T_lidar2world_frames.items()):
        for lidar_id in lidar_ids:
            Tr = np.linalg.inv(T_lidar2world_frames[frame][lidar_id]) #convert world to lidar frame
            pcd_world = pcd_world_frames[frame][lidar_id] 
            
            #transfrom points from world to lidar frame
            points_pcd_lidar = (Tr @ np.column_stack((pcd_world[:,:3], np.ones(len(pcd_world)))).T).T[:, :3]
            pcd_lidar = np.column_stack((points_pcd_lidar, pcd_world[:,3])) #put back intensity
            pcd_lidar_frames[frame][lidar_id] = pcd_lidar #append data
            
            #save pcd
            if save_pcd:
                np.savetxt(out_dir / str(f'org_pcd_lidar_{lidar_id}_{frame}.txt'), pcd_lidar)
                np.savetxt(out_dir / str(f'org_pcd_world_{lidar_id}_{frame}.txt'), pcd_world)

    # [debug] 1. Visualize the ego poses
    # gnss_poses = np.stack([i[:3,3] for i in loader.T_gnss2world_frames.values()])
    # plt.scatter(gnss_poses[:,0], gnss_poses[:,1], s=1)
    
    # [debug] 2. Visualize camera image 
    # img_path = loader.camera_images['000001']['RSFord_SHC_CF']
    # img = plt.imread(img_path) 
    # plt.imshow(img)
    
    # [debug] 3. Visualize point cloud on image frame
    # pcd_world = loader.pcd_world_frames['000001']['RSFord_SHC_LF'] #(N,4)
    # T_world2camera = np.linalg.inv(loader.T_cam2world_frames['000001']['RSFord_SHC_CF']) #(4,4)
    # K_camera = loader.K_mats()['000001']['RSFord_SHC_CF'] #(3,3)
    # points_camera = (T_world2camera @ np.column_stack((pcd_world[:,:3], np.ones(len(pcd_world)))).T).T #(N,4)
    # points_img = (K_camera @ points_camera[:,:3].T).T # transform to image frame
    # front_points_mask = points_img[:,2] > 0 #mask for 3D points in front of camera
    # points_img = points_img / points_img[:,2:3] #normalize by z values
    # plt.scatter(points_img[front_points_mask, 0], points_img[front_points_mask, 1], c=pcd_world[front_points_mask, 3], s=0.5, edgecolors='none', cmap='viridis')
    # plt.xlim(0,img.shape[1]);plt.ylim(img.shape[0],0) #canvas limit
    # plt.savefig(f"daas_camera_lidar_calibration_check_{sequence_name}.png", dpi=300, bbox_inches='tight')

    lidar_ids = ['RSFord_SHC_LF']
    for frame in tqdm(pcd_lidar_frames, desc="Converting lidar data to pano image"):
        for lidar_id in lidar_ids:
            #Get pcd data
            point_cloud = pcd_lidar_frames[frame][lidar_id].astype(np.float32)
            point_cloud = point_cloud.reshape((-1, points_dim))
            
            #Convert to pano/range image
            pano = LiDAR_2_Pano(point_cloud, H_lidar, W_lidar, intrinsics, intrinsics_hoz, lidar_range)

            #get frame name
            frame_name = frame + ".npy"

            #save pano/range image in .npy file
            np.save(out_dir / frame_name, pano) 

            #save pano image to pcd
            if save_pcd:
                # PCD in Lidar frame
                pcd_from_pano = convert.pano_to_lidar_with_intensities(pano[:,:,2],  pano[:,:,1], intrinsics, intrinsics_hoz)
                np.savetxt(out_dir / frame_name.replace(".npy", f"_{lidar_id}_lidar.txt"), pcd_from_pano)

                # PCD in World frame
                Tr_l2w = T_lidar2world_frames[frame][lidar_id] #Lidar frame to world frame
                points_pcd_world = (Tr_l2w @ np.column_stack((pcd_from_pano[:,:3], np.ones(len(pcd_from_pano)))).T).T[:, :3]
                pcd_pano_world = np.column_stack((points_pcd_world, pcd_from_pano[:,3])) #put back intensity
                np.savetxt(out_dir / frame_name.replace(".npy", f"_{lidar_id}_world.txt"), pcd_pano_world)
    
    print(f"Pano images are saved at: \n{out_dir}")

def create_dgt_rangeview(
            H_lidar:int,
            W_lidar:int,
            fov_up:float,           
            fov:float,
            fov_hoz_up:float,   
            fov_hoz:float,     
            lidar_range:float,
            points_dim:int,
            sequence_name:str,    
            out_dir=None,
            save_pcd=False,
            loader=None
    ):
    """Process raw pcd files to generate range/pano view images for AVL DGT dataset.

    Args:
        - H_lidar (int, optional): Horizontal size of pano image. Defaults to 256.
        - W_lidar (int, optional): Vertical size of pano image. Defaults to 1800.
        - fov_up (float, optional): Upper (+ve) vertical field of view of lidar. Defaults to 15.
        - fov (float, optional): Full vertical field of view of lidar. Defaults to 40.
        - fov_hoz_up (float, optional): Upper (+ve) horizontal field of view of lidar. Defaults to 180.0.
        - fov_hoz (float, optional): Full horizontal field of view of lidar. Defaults to 360.0.
        - lidar_range (float, optional): Range of lidar sensor. Defaults to 245.
        - points_dim (float, optional): Number of dimensions of lidar pcd. Defaults to 4.
        - sequence_name (str, optional): Name of sequence. Defaults to "CityStreet_dgt_2021-07-15-13-31-59_0_s0".
        - out_dir (_type_, optional): Path to save processed pano images. Defaults to None.
        - save_pcd (bool, optional): Flag to save pcd to txt file. Defaults to False.
        - loader (AVL_loader, optional): Data loader object. Defaults to None.
    """

    #Path information
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "dgt" / "source_data"
    root_dir_parent = data_root.parent
    
    #output directory
    if out_dir is None:
        out_dir = root_dir_parent / "train" / sequence_name
    
    # Lidar vertical and horizontal intrinsics
    intrinsics = fov_up, fov
    intrinsics_hoz = fov_hoz_up, fov_hoz

    # Get all lidar files directories
    raw_data_dir = data_root / sequence_name
    lidar_file_paths = glob.glob(os.path.join(raw_data_dir / "dataset/*.json"))
    cam_file_paths = glob.glob(os.path.join(raw_data_dir / "dataset/*.png"))
    
    # make directories
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Daas data parser
    if not loader: loader = AVL_loader.DGT_loader(raw_data_dir)
    pcd_world_frames = loader.pcd_world_frames
    T_lidar2world_frames = loader.T_lidar2world_frames

    # convert pcd from world frame to lidar frame
    pcd_lidar_frames = defaultdict(lambda: defaultdict(list))
    lidar_ids = ['lidar_front', 'lidar_left', 'lidar_right']
    for (frame, pcd_world), (_, Tr_lidar2world) in zip(pcd_world_frames.items(), T_lidar2world_frames.items()):
        for lidar_id in lidar_ids:
            Tr = np.linalg.inv(T_lidar2world_frames[frame][lidar_id]) #convert world to lidar frame
            pcd_world = pcd_world_frames[frame][lidar_id] 
            
            #transfrom points from world to lidar frame
            points_pcd_lidar = (Tr @ np.column_stack((pcd_world[:,:3], np.ones(len(pcd_world)))).T).T[:, :3]
            pcd_lidar = np.column_stack((points_pcd_lidar, pcd_world[:,3])) #put back intensity
            pcd_lidar_frames[frame][lidar_id] = pcd_lidar #append data
            
            #save pcd
            if save_pcd:
                np.savetxt(out_dir / str(f'org_pcd_lidar_{lidar_id}_{frame}.txt'), pcd_lidar)
                np.savetxt(out_dir / str(f'org_pcd_world_{lidar_id}_{frame}.txt'), pcd_world)

    # [debug] 1. Visualize the ego poses
    # gnss_poses = np.stack([i[:3,3] for i in loader.T_gnss2world_frames.values()])
    # plt.scatter(gnss_poses[:,0], gnss_poses[:,1], s=1)
    
    # [debug] 2. Visualize camera image 
    # img_path = loader.camera_images['1']['camera_front']
    # img = plt.imread(img_path) 
    # plt.imshow(img)
    
    # [debug] 3. Visualize point cloud on image frame
    # pcd_world = loader.pcd_world_frames['1']['lidar_front'] #(N,4)
    # T_world2camera = np.linalg.inv(loader.T_cam2world_frames['1']['camera_front']) #(4,4)
    # K_camera = loader.K_cameras['camera_front']['intrinsic_matrix'] #(3,3)
    # points_camera = (T_world2camera @ np.column_stack((pcd_world[:,:3], np.ones(len(pcd_world)))).T).T #(N,4)
    # points_img = (K_camera @ points_camera[:,:3].T).T # transform to image frame
    # front_points_mask = points_img[:,2] > 0 #mask for 3D points in front of camera
    # points_img = points_img / points_img[:,2:3] #normalize by z values
    # plt.scatter(points_img[front_points_mask, 0], points_img[front_points_mask, 1], c=pcd_world[front_points_mask, 3], s=1, edgecolors='none', cmap='viridis')
    # plt.xlim(0,img.shape[1]);plt.ylim(img.shape[0],0) #canvas limit
    # plt.savefig(f"dgt_camera_lidar_calibration_check_{sequence_name}.png", dpi=300, bbox_inches='tight')

    lidar_ids = ['lidar_front']
    for frame in tqdm(pcd_lidar_frames, desc="Converting lidar data to pano image"):
        for lidar_id in lidar_ids:
            #Get pcd data
            point_cloud = pcd_lidar_frames[frame][lidar_id].astype(np.float32)
            point_cloud = point_cloud.reshape((-1, points_dim))

            #Filter the PCD data
            # 1. DGT front lidar has points below the surface of the ground. That is noisy data.
            #TODO: Better approach is required to filter noisy 3D points.
            point_cloud = point_cloud[point_cloud[:,2] > -2.75]
            # 2. Remove points close to the sensor (1.5m)
            point_cloud = point_cloud[np.linalg.norm(point_cloud[:, :3], axis=1) > 1.5] 
            
            #Convert to pano/range image
            pano = LiDAR_2_Pano(point_cloud, H_lidar, W_lidar, intrinsics, intrinsics_hoz, lidar_range)

            #get frame name
            frame_name = frame + ".npy"

            #save pano/range image in .npy file
            np.save(out_dir / frame_name, pano) 

            #save pano image to pcd
            if save_pcd:
                # PCD in Lidar frame
                pcd_from_pano = convert.pano_to_lidar_with_intensities(pano[:,:,2],  pano[:,:,1], intrinsics, intrinsics_hoz)
                np.savetxt(out_dir / frame_name.replace(".npy", f"_{lidar_id}_lidar.txt"), pcd_from_pano)

                # PCD in World frame
                Tr_l2w = T_lidar2world_frames[frame][lidar_id] #Lidar frame to world frame
                points_pcd_world = (Tr_l2w @ np.column_stack((pcd_from_pano[:,:3], np.ones(len(pcd_from_pano)))).T).T[:, :3]
                pcd_pano_world = np.column_stack((points_pcd_world, pcd_from_pano[:,3])) #put back intensity
                np.savetxt(out_dir / frame_name.replace(".npy", f"_{lidar_id}_world.txt"), pcd_pano_world)
    
    print(f"Pano images are saved at: \n{out_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["kitti360", "dgt", "daas"], help="The dataset loader to use.",)
    parser.add_argument("--sequence_name", type=str, required=True, help="Name of the sequence")
    args = parser.parse_args()

    # Ganerate rage images as per dataset.
    if args.dataset == "kitti360":
        create_kitti_rangeview(
            H_lidar= 66, #default is 66
            W_lidar= 1030, #default is 1030
            fov_up= 2.0,    #vertical
            fov=26.9,       #vertical
            fov_hoz_up= 180.0,  #horizontal   
            fov_hoz= 360.0,     #horizontal
            max_depth= 80.0,
            sequence_name= args.sequence_name, #options: static- 1538, 1728, 1908, 3353, dynamic- 2350, 4950, 8120, 10200, 10750, 11400
            out_dir= None,
        )
    
    elif args.dataset == "daas":
        create_daas_rangeview(
            H_lidar= 128,       #128
            W_lidar= 940,       #1875     
            fov_up= 13.8,       #15.0
            fov= 24.6,          #40.0
            fov_hoz_up= 90.0,   #180
            fov_hoz= 180.0,     #360
            lidar_range= 245,
            sequence_name= args.sequence_name,
            out_dir= None,
        )
    elif args.dataset == "dgt":
        create_dgt_rangeview(
            H_lidar= 128,       #128
            W_lidar= 940,       #1875     
            fov_up= 13.8,       #15.0
            fov= 24.6,          #40.0
            fov_hoz_up= 90.0,   #180
            fov_hoz= 180.0,     #360
            lidar_range= 245,
            sequence_name= args.sequence_name,
            out_dir= None,
        )
    else:
        raise ValueError(f"Dataset not supported: {args.dataset}")
