import numpy as np
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from nvsf.lib import tools
import open3d as o3d
import cv2

def lidar_to_pano_with_intensities_with_bbox_mask(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    bbox_local: np.ndarray,
    max_depth=80,
    max_intensity=255.0,
):
    """
    Convert lidar frame to pano frame with intensities with bbox_mask.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        bbox_local: (8x4), world bbox in local.
        max_depth: max depth in meters.
        max_intensity: max intensity.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """

    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))

    # bbox mask
    pano[:, :] = -1
    r_min, r_max, c_min, c_max = 1e5, -1, 1e5, -1
    for bbox_local_point in bbox_local:
        x, y, z, _ = bbox_local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi

        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue
        else:
            r_min, r_max, c_min, c_max = (
                min(r_min, r),
                max(r_max, r),
                min(c_min, c),
                max(c_max, c),
            )

    pano[r_min:r_max, c_min:c_max] = 0

    # Fill pano and intensities.
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity / max_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity / max_intensity

    return pano, intensities


def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: list[float],
    lidar_K_hoz: list[float],
    max_depth:float= 80.0,
):
    """Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        - local_points_with_intensities (np.ndarray): (N, 4), float32, in lidar frame, with intensities.
        - lidar_H (int): pano height.
        - lidar_W (int): pano width.
        - lidar_K (list): lidar intrinsics vertical in deg (fov_up, fov).
        - lidar_K_hoz (list): lidar intrinsics horizontal in deg (fov_up, fov).
        - max_depth (float, optional): lidar range in meters. Defaults to 80.0.
    Returns:
        - pano (np.ndarray): array for depth. [nr, H, W]
        - intensities (np.ndarray): array for intensity. [nr, H, W]
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]

    #Lidar intrinsics
    fov_up, fov = lidar_K
    fov_down = fov - fov_up
    fov_hoz_up, fov_hoz = lidar_K_hoz

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)
    depth_num = bound_num = clash_num = 0

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_point, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            depth_num += 1
            continue

        x, y, z = local_point  # point coordinates
        
        # azimuth angle
        beta = fov_hoz_up * np.pi / 180 - np.arctan2(y, x)
        
        # inclination angle
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi  

        # Calculate pixel coordinate (row, column) of lidar pano/range image
        c = int(round(beta / ((fov_hoz * np.pi / 180) / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            bound_num += 1
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        else:
            clash_num +=1 
            # pano_e[r,c] += 1
            
    return pano, intensities

def lidar_to_pano(
    local_points: np.ndarray, 
    lidar_H:int, 
    lidar_W:int, 
    lidar_K:list[float], 
    lidar_K_hoz:list[float], 
    max_depth:float= 80
) -> np.ndarray:
    """
    Convert lidar frame to pano frame. Lidar points are in local coordinates.

    Args:
        - local_points: (N, 3), float32, in lidar frame.
        - lidar_H: pano height.
        - lidar_W: pano width.
        - lidar_K: lidar intrinsics vertical.
        - lidar_K_hoz: lidar intrinsics horizontal.
        - max_depth: max depth in meters.

    Return:
        - pano: (H, W), float32.
    """

    # (N, 3) -> (N, 4), filled with zeros.
    local_points_with_intensities = np.concatenate(
        [local_points, np.zeros((local_points.shape[0], 1))], axis=1
    )
    pano, _ = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=lidar_K,
        lidar_K_hoz=lidar_K_hoz,
        max_depth=max_depth,
    )
    return pano


def pano_to_lidar_with_intensities(
        pano: np.ndarray, 
        intensities:np.ndarray, 
        lidar_K:list[float],
        lidar_K_hoz:list[float],
    ) -> np.ndarray :
    """Convert pano image to point cloud with intensity
    Args:
        - pano (np.ndarray): (H, W), float32. Depth/range of the point cloud
        - intensities (np.ndarray): (H, W), float32.Intensity of the point cloud
        - lidar_K (list): vertical lidar intrinsics (fov_up, fov)
        - lidar_K_hoz (list): horizontal lidar intrinsics in deg (fov_up, fov).

    Return:
        - local_points_with_intensities (np.ndarray): (N, 4), float32, in lidar frame.
    """
    fov_up, fov = lidar_K
    fov_hoz_up, fov_hoz = lidar_K_hoz

    H, W = pano.shape
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    # beta = -(i - W / 2) / W * 2 * np.pi
    beta = -(i - W / 2) / W * fov_hoz / 180 * np.pi
    alpha = (fov_up - j / H * fov) / 180 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2
    )

    # Filter empty points.
    idx = np.where(pano != 0.0)
    local_points_with_intensities = local_points_with_intensities[idx]

    return local_points_with_intensities


def pano_to_lidar(
    pano:np.ndarray, 
    lidar_K:list[float], 
    lidar_K_hoz:list[float]
    ) -> np.ndarray:
    """
    Args:
        - pano: (H, W), float32.
        - lidar_K (list): vertical lidar intrinsics (fov_up, fov)
        - lidar_K_hoz (list): horizontal lidar intrinsics in deg (fov_up, fov).

    Return:
        - local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
        lidar_K_hoz=lidar_K_hoz
    )
    return local_points_with_intensities[:, :3]


def lidar_to_pano_with_intensities_fpa(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
    z_buffer_len=10,
):
    """
    Convert lidar frame to pano frame with intensities with bbox_mask.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.
        z_buffer_len: length of the z_buffer.

    Return:
        rangeview image: (H, W, 3), float32.
    """

    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    range_view = np.zeros((lidar_H, lidar_W, 3, z_buffer_len + 1))

    for local_point, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        position = range_view[r, c, 2, 0] + 1
        if position > z_buffer_len:
            depth_z_buffer = list(range_view[r, c, 2][1:]) + [dist]
            intensity_z_buffer = list(range_view[r, c, 1][1:]) + [local_point_intensity]
            position = position - 1

            sort_index = np.argsort(depth_z_buffer)
            depth_z_buffer = np.insert(
                np.array(depth_z_buffer)[sort_index][:z_buffer_len], 0, position
            )
            intensity_z_buffer = np.insert(
                np.array(intensity_z_buffer)[sort_index][:z_buffer_len], 0, position
            )
            range_view[r, c, 2] = depth_z_buffer
            range_view[r, c, 1] = intensity_z_buffer

        else:
            range_view[r, c, 2, int(position)] = dist
            range_view[r, c, 1, int(position)] = local_point_intensity
        range_view[r, c, 2, 0] = position
    range_view = parse_z_buffer(range_view, lidar_H, lidar_W)
    return range_view[:, :, 2], range_view[:, :, 1]


def parse_z_buffer(novel_pano, lidar_H, lidar_W, threshold=0.2):
    range_view = np.zeros((lidar_H, lidar_W, 3))
    for i in range(lidar_H):
        for j in range(lidar_W):
            range_pixel = novel_pano[i, j, 2]
            intensity_pixel = novel_pano[i, j, 1]
            z_buffer_num = int(range_pixel[0])
            if z_buffer_num == 0:
                continue
            if z_buffer_num == 1:
                range_view[i][j][2] = range_pixel[1]
                range_view[i][j][1] = intensity_pixel[1]
                continue

            depth_z_buffer = range_pixel[1:z_buffer_num]
            closest_points = min(depth_z_buffer)
            index = depth_z_buffer <= (closest_points + threshold)

            final_depth_z_buffer = np.array(depth_z_buffer)[index]
            final_dis = np.average(
                final_depth_z_buffer, weights=1 / final_depth_z_buffer
            )
            range_view[i][j][2] = final_dis

            intensity_z_buffer = intensity_pixel[1:z_buffer_num]
            final_intensity_z_buffer = np.array(intensity_z_buffer)[index]
            final_intensity = np.average(
                final_intensity_z_buffer, weights=1 / final_depth_z_buffer
            )
            range_view[i][j][1] = final_intensity
    return range_view

def trans_pcd_lidar_to_world(pcd_lidar_dict: dict, 
                                 cam0_to_world_path:str ="data/kitti360/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt"):
    """Transform pcd data from lidar frame to world frame

    Args:
        pcd_lidar_dict (dict): dictionary of pcd data in lidar frame {frame_id : pcd_data}
        cam0_to_world_path (str, optional): _description_. Defaults to "data/kitti360/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt".

    Returns:
        dict: dictionary of pcd data in world frame {frame_id : pcd_data}
    """
    #read transformation file
    cam0_to_world_mats = np.loadtxt(cam0_to_world_path)

    # get camera to lidar/velodyne T matrix for required frames
    idx = np.searchsorted(cam0_to_world_mats[..., 0], 
                          list(pcd_lidar_dict.keys()), 
                          side="left")
    
    T_cam0_2_w = cam0_to_world_mats[idx][..., 1:].reshape(-1, 4, 4)

    # get camera to lidar/velodyne T matrix
    cam0_to_velo_path = os.path.join(
        cam0_to_world_path.split("data_poses")[0],
        "calibration",
        "calib_cam_to_velo.txt",
    )
    with open(cam0_to_velo_path, "r") as fid:
        line = fid.readline().split()
        line = [float(x) for x in line]
        cam0_to_velo = np.array(line).reshape(3, 4)

    # Transform pcd from lidar/velo frame to world frame and save pcd .txt file
    pcd_world_dict = dict()
    for frame_id, T_cam0_2_w_ in tqdm(zip(pcd_lidar_dict, T_cam0_2_w), 
                                      desc="Processing and saving pcd files"):
        pcd = np.asarray(pcd_lidar_dict[frame_id])
        pcd_ = np.column_stack((pcd[:, :3], np.ones(len(pcd)))).T
        # get T mat for lidar to world using velo -> camera -> world frame
        T_l2w = np.dot(
            T_cam0_2_w_,
            np.linalg.inv(np.row_stack((cam0_to_velo, [0.0, 0.0, 0.0, 1.0]))),
        )
        pcd_world = np.column_stack( (np.dot(T_l2w, pcd_).T[:, :3], 
                                      pcd[:, 3:]) ) #add attributes (intensity ...) of pcd
        pcd_world_dict[frame_id] = pcd_world # append to dict
    
    return pcd_world_dict

def convert_pcd_source2target(
    lidar_K:list[float],
    lidar_K_hoz:list[float],
    file_dirs:list, 
    start_frame:int, 
    source_pcd_dim:int, 
    target_pcd_format:str, 
    cord_frame:str='lidar', 
    save_dir: str= None
    ):
    """Convert pcd data from source to target file format. 
    Source data should be in xyzi format and lidar frame.

    Args:
        file_dirs (list): path list of pcd files 
        start_frame (int): start frame number
        source_pcd_dim (int): input pcd dimensions [4,5] -> xyzi or xyzii
        target_pcd_format (str): target pcd format -> ['bin', 'nus_bin', 'nus_woi_bin', 'txt', 'npy'] 
        cord_frame (str, optional): target pcd coordinate frame ->['lidar', 'world'] . Defaults to 'lidar'.
        save_dir (str, optional): path to save file in bin format. Defaults to None.

    Raises:
        ValueError: If file format is not from format ['bin', 'txt', 'npy'] 

    Returns:
        dict: Converted pcd data in dictionary format {frame_id: pcd_data}
    """
    assert source_pcd_dim in (4,5), "source_pcd_dim should 4 or 5"
    assert target_pcd_format in ['bin', 'nus_bin', 'nus_woi_bin', 'txt', 'npy'] , f"target_pcd_format:{target_pcd_format} is not supported"
    assert cord_frame in ['lidar', 'world'] , f"cord_frame: {cord_frame} is not supported "
    assert len(file_dirs) > 0 , "No pcd files found"

    pcds_dict = dict()
    for frame_id, file in tqdm(enumerate(file_dirs, start= start_frame), 
                               total= len(file_dirs), 
                               desc=f"Converting pcd files to {target_pcd_format} format"):
        # Reading pcd file
        if '.txt' in os.path.basename(file):
            pcd_data = np.loadtxt(file, dtype=np.float32) #read .txt file
        elif '.bin' in os.path.basename(file):
                pcd_data = np.fromfile(file, dtype=np.float32).reshape(-1, source_pcd_dim)  # hardcode read .bin file, xyzi
        elif '.npy' in os.path.basename(file):
            pcd_data = np.load(file).astype(np.float32) # read .npy file
            #check pcd data for range image dimensions
            if pcd_data.shape[-1] == 3: #TODO: hard coded
                pcd_data = pano_to_lidar_with_intensities(pano = pcd_data[:, :, 2], 
                                                          intensities = pcd_data[:, :, 1], 
                                                          lidar_K = lidar_K, #hard code
                                                          lidar_K_hoz = lidar_K_hoz) #hard code
        else:
            raise ValueError(f"Pcd file format is not supported, format should be from ['.txt', '.bin', '.npy']")

        # Convert data as per target pcd format
        #TODO: target pcd=range image to be implemented
        if target_pcd_format == 'nus_bin':
            # add column for intensity, xyzii
            pcd_data = np.column_stack((pcd_data[:, :4], 
                                        np.zeros((len(pcd_data), 1), dtype=np.float32) ))  #pcd_xyzii
        elif target_pcd_format == 'nus_woi_bin':
            # add zeros column
            pcd_data = np.column_stack((pcd_data[:, :3], 
                                        np.zeros((len(pcd_data), 2), dtype=np.float32) )) #pcd_xyzii
        else:
            pcd_data = pcd_data[:, :4]
        
        # frame_id = int(os.path.basename(file).split('.')[0])
        pcds_dict[frame_id] = pcd_data
    
    #transform pcd
    #TODO: world to lidar transformation 
    # if source_frame == 'lidar' and target_frame == 'world'
    if cord_frame == 'world':
        pcds_dict = trans_pcd_lidar_to_world(pcds_dict)
    
    #Save pcd
    if save_dir:
        #functions for saving files
        pcd_save_command = {
        'bin': lambda: pcds_dict[frame_id].tofile(file_name, sep="", format="%f"),
        'txt': lambda: np.savetxt(file_name, pcds_dict[frame_id], delimiter=" ", fmt="%f"),
        'npy': lambda: np.save(file_name, pcds_dict[frame_id])
        }

        if target_pcd_format in ['nus_bin', 'nus_woi_bin']:
            target_pcd_format = 'bin'

        for frame_id in tqdm(pcds_dict, desc= f"Saving pcd files in {target_pcd_format} format"):
            file_name = os.path.join(save_dir, 
                                    "%010d." %frame_id + target_pcd_format)
            pcd_save_command[target_pcd_format]() #save pcd
    
    if save_dir:
        print(f"Pcd files are saved at location {save_dir}")

    return pcds_dict

if __name__ == "__main__":

    do = False
    # Lidar fov_up, fov
    lidar_K = {
        'kitti360': [2.0, 26.9],    
        'dgt': [7.0, 20.0],        # TODO: add dgt lidar intrinsics
        'daas': [13.8, 24.6],       # [15.0, 40.0]
    }  
    lidar_K_hoz = {
        'kitti360': [180.0, 360.0], 
        'dgt': [180.0, 360.0],        # TODO: add dgt lidar intrinsics
        'daas': [90.0, 180.0],
    }  
    if do:
        # =================================DATA CONVERSIONS=============================================#
        data_dir = "nvsf/data/kitti360/train/1908"
        
        start_frame = 4950  # options: 1538, 1728, 1908, 3353, 4950, 8120
        nr_frame = 50 #options: 64, 50
        # file_dirs = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        file_dirs = [os.path.join(data_dir, "%010d.bin" % frame_id)
                      for frame_id in range(start_frame, start_frame + nr_frame+1)]

        pcds_dict = convert_pcd_source2target(
            lidar_K = lidar_K['kitti360'],
            lidar_K_hoz = lidar_K_hoz['kitti360'], #hard code #hard code
            file_dirs= file_dirs, 
            start_frame= start_frame, # options: 1538, 1728, 1908, 3353
            source_pcd_dim= 4,  # options: 4, 5
            target_pcd_format= 'txt', # options: 'bin', 'nus_bin', 'nus_woi_bin', 'txt', 'npy'
            cord_frame= 'lidar', # options: 'lidar', 'world'
            save_dir= "nvsf/data/kitti360/train_pcd/from_bin_files/lidar_frame"
            )

        # ======================================OTHER===============================================#

        # Rename pcd .bin files
        dir_path = "nvsf/data/kitti360/pcd_in_nuScene_format/1538/exp"
        files = sorted(glob.glob(os.path.join(dir_path, "*.bin")))
        start_frame = 1538  # #options: 1538, 1728, 1908, 3353
        for frame_id, file_ in enumerate(files, start=start_frame):
            dest = file_.replace(os.path.basename(file_), "%010d.bin" %frame_id )
            os.rename(file_, dest)

        # Multi channel
        # Save npy to txt file for multi channel
        for file in file_dirs:  
            pcd_data = np.load(file).astype(np.float32) # read .npy file
            pcd_data = pcd_data.transpose(2,0,1,3) # (66, 1030, 2, 3) -> (2, 66, 1030, 3)
            for c , pcd_data_ in enumerate(pcd_data, start=1):
                pcd_data_ = pano_to_lidar_with_intensities(pano = pcd_data_[:, :, 2], 
                                                            intensities = pcd_data_[:, :, 1], 
                                                            lidar_K = lidar_K['kitti360'],
                                                            lidar_K_hoz = lidar_K_hoz['kitti360']) #hard code #hard code
                file_path = os.path.join("nvsf/data/kitti360/train_pcd/from_range_images_66x1030_mr/lidar_frame",
                                         os.path.basename(file.replace('.npy', f"_{c}.txt")))
                np.savetxt(file_path, pcd_data_, delimiter=" ", fmt="%f")
        
        #Convert txt to pcd
        dir = "nvsf/log/daas/nvsf/results/CityStreet_dgt_2021-07-13-11-21-58_0_s0/1000eps,2048,1024@100m/dgt_nerf/dgt_lidar_fov(2.5, 9.5)"
        files = sorted(glob.glob(os.path.join(dir, "*lidar.txt")))
        pcd = o3d.t.geometry.PointCloud() #create pcd object using geometry tensor class
        for file in files:
            pcd_data = np.loadtxt(file).astype(np.float32)
            pcd.point['positions'] = pcd_data[:, :3] #set 'positions' attribute to  pcd.point 
            pcd.point['intensity'] = pcd_data[:, 3:] #set 'intensity' attribute to pcd.point 
            file_path= os.path.join(dir, os.path.basename(file).replace('lidar.txt', 'lidar.pcd'))
            o3d.t.io.write_point_cloud(file_path, pcd, write_ascii=True) #save pcd file

        

