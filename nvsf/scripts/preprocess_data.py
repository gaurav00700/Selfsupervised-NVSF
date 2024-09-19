import argparse
from pathlib import Path
from pyparsing import str_type
from nvsf.preprocess import (
    generate_rangeview, 
    cal_centerpose_bound, 
    kitti360_to_nerf, 
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["kitti360", "dgt", "daas"], help="The dataset loader to use.",)
parser.add_argument("--sequence_name", type=str, required=True, help="Name of the sequence")
parser.add_argument("--save_pcd", action="store_true", help="Convert pano image to pcd")
args = parser.parse_args()

#dataset directory
raw_data_dir = Path(__file__).resolve().parent.parent / "data" / args.dataset / "source_data" / args.sequence_name

if args.dataset == "kitti360":  
    
    # Lidar parameters 
    lidar_params = {
        "H_lidar": 66,
        "W_lidar": 1030,
        "fov_up": 2.0,
        "fov": 26.9,
        "fov_hoz_up": 180.0,
        "fov_hoz": 360.0,
        "lidar_range": 80.0,
        "points_dim": 4,
        } 
    # Generate rage images as per dataset.
    generate_rangeview.create_kitti_rangeview(
        **lidar_params,
        sequence_name= args.sequence_name, #options: 1538, 1728, 1908, 3353, 4950, 8120
        )

    # Generate configs json files
    kitti360_to_nerf.main(
        sequence_name=args.sequence_name, #options: 1538, 1728, 1908, 3353, 4950, 8120 ...
    )

else:
    raise ValueError(f"Dataset not supported: {args.dataset}")

# Calculate center pose and bounds
cal_centerpose_bound.main(
    **lidar_params,
    dataset= args.dataset, 
    sequence_name= args.sequence_name,
    )

print("[INFO] Finished Data Preprocessing")