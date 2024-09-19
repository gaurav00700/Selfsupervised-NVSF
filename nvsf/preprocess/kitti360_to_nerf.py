from pathlib import Path
from nvsf.preprocess.kitti360_loader import KITTI360Loader
import camtools as ct
import numpy as np
import json
from typing import Union

def normalize_Ts(Ts):
    # New Cs.
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])
    normalize_mat = ct.normalize.compute_normalize_mat(Cs)
    Cs_new = ct.project.homo_project(Cs.reshape((-1, 3)), normalize_mat)

    # New Ts.
    Ts_new = []
    for T, C_new in zip(Ts, Cs_new):
        pose = ct.convert.T_to_pose(T)
        pose[:3, 3] = C_new
        T_new = ct.convert.pose_to_T(pose)
        Ts_new.append(T_new)

    return Ts_new


def main(sequence_name:str, recording_name:str = "2013_05_28_drive_0000", range_view_dir:str= None):
    """Create transformation json files for training, validation, testing and all
    Sequence id and path for range/pano images are required.

    Args:
        - sequence_name (str): sequence id eg: 1538, 1728, 1908, 3353
        - range_view_dir (str, optional): Path for range images. Defaults to None.
    """

    # Get paths
    project_root = Path(__file__).parent.parent
    kitti_360_root = project_root / "data" / "kitti360" / "source_data"
    kitti_360_parent_dir = kitti_360_root.parent

    if not range_view_dir:
        range_view_dir = kitti_360_parent_dir / "train" / sequence_name
    else:
        range_view_dir = Path(range_view_dir)

    # if sequence_name !="2013_05_28_drive_0000":
    #     raise ValueError(f"Sequence '{sequence_name}' not supported")
    
    # Specify frames and splits.
    if sequence_name == "1538":
        s_frame_id = 1538
        e_frame_id = 1601  # Inclusive
        val_frame_ids = [1551, 1564, 1577, 1590]
    elif sequence_name == "1728":
        s_frame_id = 1728
        e_frame_id = 1791  # Inclusive
        val_frame_ids = [1741, 1754, 1767, 1780]
    elif sequence_name == "1908":
        s_frame_id = 1908
        e_frame_id = 1971  # Inclusive
        val_frame_ids = [1921, 1934, 1947, 1960]
    elif sequence_name == "3353":
        s_frame_id = 3353
        e_frame_id = 3416  # Inclusive
        val_frame_ids = [3366, 3379, 3392, 3405]
    elif sequence_name == "2350":
        s_frame_id = 2350
        e_frame_id = 2400  # Inclusive
        val_frame_ids = [2360, 2370, 2380, 2390]
    elif sequence_name == "4950":
        s_frame_id = 4950
        e_frame_id = 5000  # Inclusive
        val_frame_ids = [4960, 4970, 4980, 4990]
    elif sequence_name == "8120":
        s_frame_id = 8120
        e_frame_id = 8170  # Inclusive
        val_frame_ids = [8130, 8140, 8150, 8160]
    elif sequence_name == "10200":
        s_frame_id = 10200
        e_frame_id = 10250  # Inclusive
        val_frame_ids = [10210, 10220, 10230, 10240]
    elif sequence_name == "10750":
        s_frame_id = 10750
        e_frame_id = 10800  # Inclusive
        val_frame_ids = [10760, 10770, 10780, 10790]
    elif sequence_name == "11400":
        s_frame_id = 11400
        e_frame_id = 11450  # Inclusive
        val_frame_ids = [11410, 11420, 11430, 11440]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_name}.\
                         \nSelect sequence from [1538,1728,1908,3353,2350,4950,8120,10200,10750,11400]")
    
    # Get list of frame ids for training and testing
    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    num_frames = len(frame_ids)

    test_frame_ids = val_frame_ids
    train_frame_ids = [x for x in frame_ids if x not in val_frame_ids]

    # Load KITTI-360 dataset.
    k3 = KITTI360Loader(kitti_360_root)

    # Get image paths.
    cam_00_im_paths = k3.get_image_paths("cam_00", recording_name, frame_ids)
    cam_01_im_paths = k3.get_image_paths("cam_01", recording_name, frame_ids)
    im_paths = cam_00_im_paths + cam_01_im_paths

    # Get Ks, Ts.
    cam_00_Ks, cam_00_Ts = k3.load_cameras("cam_00", recording_name, frame_ids)
    cam_01_Ks, cam_01_Ts = k3.load_cameras("cam_01", recording_name, frame_ids)
    Ks = np.concatenate([cam_00_Ks, cam_01_Ks], axis=0)
    Ts = np.concatenate([cam_00_Ts, cam_01_Ts], axis=0)
    # Ts = normalize_Ts(Ts)

    # Get image dimensions, assume all images have the same dimensions.
    im_rgb = ct.io.imread(cam_00_im_paths[0])
    im_h, im_w, _ = im_rgb.shape

    # Get lidar paths (range view not raw data)    
    range_view_paths = [
        range_view_dir / "{:010d}.npy".format(int(frame_id)) for frame_id in frame_ids
    ]

    # Get lidar2world.
    lidar2world = k3.load_lidars(recording_name, frame_ids)

    # Get image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_paths[0])
    lidar_h, lidar_w, _ = lidar_range_image.shape

    # Split by train/test/val.
    all_indices = [i - s_frame_id for i in frame_ids]
    train_indices = [i - s_frame_id for i in train_frame_ids]
    val_indices = [i - s_frame_id for i in val_frame_ids]
    test_indices = [i - s_frame_id for i in test_frame_ids]

    # Get transformation json files for training, validation and testing
    split_to_all_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
        "all": all_indices,
    }
    for split, indices in split_to_all_indices.items():
        print(f"Split {split} has {len(indices)} frames.")
        id_split = [frame_ids[i] for i in indices]
        im_paths_split = [im_paths[i] for i in indices]
        lidar_paths_split = [range_view_paths[i] for i in indices]
        lidar2world_split = [lidar2world[i] for i in indices]
        Ks_split = [Ks[i] for i in indices]
        Ts_split = [Ts[i] for i in indices]

        json_dict = {
            "w": im_w,
            "h": im_h,
            "w_lidar": lidar_w,
            "h_lidar": lidar_h,
            "fl_x": Ks_split[0][0, 0],
            "fl_y": Ks_split[0][1, 1],
            "cx": Ks_split[0][0, 2],
            "cy": Ks_split[0][1, 2],
            "frame_start": s_frame_id,
            "frame_end": e_frame_id,
            "num_frames": num_frames,
            "num_frames_split": len(id_split),
            "aabb_scale": 2,
            "frames": [
                {
                    "frame_id": id,
                    "file_path": str(path.relative_to(kitti_360_parent_dir)),
                    "transform_matrix": ct.convert.T_to_pose(T).tolist(),
                    "lidar_file_path": str(lidar_path.relative_to(kitti_360_parent_dir)),
                    "lidar2world": lidar2world.tolist(),
                }
                for (
                    id,
                    path,
                    T,
                    lidar_path,
                    lidar2world,
                ) in zip(
                    id_split,
                    im_paths_split,
                    Ts_split,
                    lidar_paths_split,
                    lidar2world_split,
                )
            ],
        }
        #Save json
        json_path = range_view_dir / f"transforms_{sequence_name}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")

if __name__ == "__main__":
    main(
        sequence_name='1908',  #options: static- 1538, 1728, 1908, 3353, dynamic- 2350, 4950, 8120, 10200, 10750, 11400
        range_view_dir = Path('data/kitti360/train')
    ) 