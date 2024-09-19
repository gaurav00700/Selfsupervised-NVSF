import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import argparse
from nvsf.lib.convert import pano_to_lidar

np.set_printoptions(suppress=True)

def cal_centerpose_bound_scale(
    lidar_rangeview_paths, 
    lidar2worlds, 
    intrinsics, 
    intrinsics_hoz, 
    bound=1.0
):
    """Calculate offset, scale, far and near values of scene """

    near = 200
    far = 0
    points_world_list = []
    for i, lidar_rangeview_path in enumerate(lidar_rangeview_paths):
        pano = np.load(lidar_rangeview_path)
        point_cloud = pano_to_lidar(pano=pano[:, :, 2], lidar_K=intrinsics, lidar_K_hoz=intrinsics_hoz)
        point_cloud = np.concatenate(
            [point_cloud, np.ones(point_cloud.shape[0]).reshape(-1, 1)], -1
        )

        # distance of each point from the sensor origin
        dis = np.linalg.norm(point_cloud, axis=1)

        #update near and far bounds
        near = min(min(dis), near)
        far = max(far, max(dis))

        #transform to world frame
        points_world = (point_cloud @ lidar2worlds[i].T)[:, :3]
        points_world_list.append(points_world)
    
    # print("Near, Far:", near, far)

    # plt.figure(figsize=(16, 16))
    pc_all_w = np.concatenate(points_world_list)[:, :3]

    # plt.scatter(pc_all_w[:, 0], pc_all_w[:, 1], s=0.001)
    # lidar2world_scene = np.array(lidar2worlds)
    # plt.plot(lidar2world_scene[:, 0, -1], lidar2world_scene[:, 1, -1])
    # plt.savefig('vis/points-trajectory.png')
    
    # Get cental position of all frame (sequence)
    centerpose = [
        (np.max(pc_all_w[:, 0]) + np.min(pc_all_w[:, 0])) / 2.0,
        (np.max(pc_all_w[:, 1]) + np.min(pc_all_w[:, 1])) / 2.0,
        (np.max(pc_all_w[:, 2]) + np.min(pc_all_w[:, 2])) / 2.0,
    ]

    # print("Centerpose: ", centerpose)
    pc_all_w_centered = pc_all_w - centerpose

    # plt.figure(figsize=(16, 16))
    # plt.scatter(pc_all_w_centered[:, 0], pc_all_w_centered[:, 1], s=0.001)
    # plt.savefig('vis/points-centered.png')

    bound_ori = [
        np.max(pc_all_w_centered[:, 0]),
        np.max(pc_all_w_centered[:, 1]),
        np.max(pc_all_w_centered[:, 2]),
    ]
    scale = bound / np.max(bound_ori)
    # print("Scale: ", scale)

    # pc_all_w_centered_scaled = pc_all_w_centered * scale
    # plt.figure(figsize=(16, 16))
    # plt.scatter(pc_all_w_centered_scaled[:, 0],
    #             pc_all_w_centered_scaled[:, 1],
    #             s=0.001)
    # plt.savefig('vis/points-centered-scaled.png')
    return centerpose, scale, near, far


def get_path_pose_from_json(root_path, sequence_name):
    with open(
        os.path.join(root_path, "train", sequence_name, f"transforms_{sequence_name}_all.json"), "r"
    ) as f:
        transform = json.load(f)
    frames = transform["frames"]
    poses_lidar = []
    paths_lidar = []
    for f in tqdm.tqdm(frames, desc=f"Loading {type} data"):
        pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]
        f_lidar_path = os.path.join(root_path, f["lidar_file_path"])
        # f_lidar_path = f["lidar_file_path"]
        poses_lidar.append(pose_lidar)
        paths_lidar.append(f_lidar_path)
    return paths_lidar, poses_lidar

def main(
        dataset:str, 
        sequence_name:str,
        fov_up:float,           
        fov:float,
        fov_hoz_up:float,   
        fov_hoz:float,     
        lidar_range:float,
        **kwargs
        ):
    """Calculate centerpose, scale, near and far values of a scene. Save them to a txt file""" 

    # Load lidar pano image path and transformation matrices
    lidar_rangeview_paths, lidar2worlds = get_path_pose_from_json(
        root_path= f"nvsf/data/{dataset}", 
        sequence_name= sequence_name
        )

    # Lidar horizontal and vertical intrinsics
    intrinsics = [fov_up, fov]
    intrinsics_hoz = [fov_hoz_up, fov_hoz] 
    
    # Calculate centerpose and scale
    centerpose, scale, near, far = cal_centerpose_bound_scale(
        lidar_rangeview_paths, 
        lidar2worlds, 
        intrinsics=intrinsics,
        intrinsics_hoz=intrinsics_hoz,
        )

    print(f"=========================[SCENE INFO]================================")
    print("Near, Far:", near, far, sep='\t')
    print("Centerpose/Offset: ", centerpose, sep='\t')
    print("Scale: ", scale , sep='\t')
    print(f"============================[END]====================================")

    #save scene info in Config file
    config_path = f"nvsf/configs/{dataset}_{sequence_name}.txt"
    with open(config_path, "w") as f:
        f.write("dataloader = {}\n".format(dataset))
        f.write("path = {}\n".format(f"nvsf/data/{dataset}"))
        f.write("sequence_id = {}\n".format(sequence_name))
        f.write("num_frames = {}\n".format(len(lidar_rangeview_paths)))
        f.write("intrinsics_lidar = {}\n".format(intrinsics))
        f.write("intrinsics_hoz_lidar = {}\n".format(intrinsics_hoz))
        f.write("lidar_max_depth = {}\n".format(lidar_range))
        f.write("scale = {}\n".format(scale))
        f.write("offset = {}\n".format(centerpose))
    
    print(f"[INFO] Config file saved at: {config_path}", sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["kitti360", "dgt", "daas"], help="The dataset loader to use.")
    parser.add_argument("--sequence_name", type=str, required=True, help="Name of the sequence")
    parser.add_argument("--sequence_id", type=str, default=None, help="ID of the sequence for Kitti360")
    args = parser.parse_args()
    
    main(dataset=args.dataset, sequence_name=args.sequence_name, sequence_id=args.sequence_id)
