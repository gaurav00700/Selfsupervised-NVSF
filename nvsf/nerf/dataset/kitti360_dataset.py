import json
import os
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Literal
from nvsf.lib import convert
from nvsf.nerf.dataset import base_dataset
from nvsf.lib.convert import pano_to_lidar
from nvsf.lib import tools
from nvsf.kitti360Scripts.kitti360scripts.helpers import annotation

class KITTI360Dataset(base_dataset.BaseDataset):
     
    def _load_renderings(self):
        """Load extras and post processing"""

        self.load_annotations()

        # [debug] 1. visualize range image
        # plt.imshow(self.images_lidar[0][:,:,2], cmap='gray')

        # [debug] 2. visualize camera image
        # plt.imshow(self.images[0])

        # [debug] 3. Visualize point cloud on image frame
        # pcd_lidar = convert.pano_to_lidar_with_intensities(
        #     pano=self.images_lidar[0][:,:,2] / self.scale, 
        #     intensities=self.images_lidar[0][:,:,1], 
        #     lidar_K=self.intrinsics_lidar,
        #     lidar_K_hoz=self.intrinsics_hoz_lidar,
        # )
        # points_img, idx = dataset_utils.project_points(pcd_lidar[:,:3], self.intrinsics, self.poses_lidar[0], self.poses[0], self.W, self.H)
        # plt.scatter(points_img[idx, 0], points_img[idx, 1], c=pcd_lidar[idx, 3], s=0.5, edgecolors='none', cmap='viridis')
        # plt.xlim(0, self.W); plt.ylim(self.H, 0) #canvas limit
        # plt.savefig(f"kitti360_camera_lidar_calibration_check_{self.sequence_id}.png", dpi=300, bbox_inches='tight')

        # [debug] 4. Visualize Camera and Lidar poses
        # plt.scatter(self.poses[:,0,-1], self.poses[:,1, -1])
        # plt.scatter(self.poses_lidar[:,0,-1], self.poses_lidar[:,1, -1])

        self.load_annotations()

    def load_annotations(self):
        """Load 3d annotations"""

        self.annotations = defaultdict(list)
        ann_dir = os.path.join(self.root_path, 'source_data', 'data_3d_bboxes')
        if os.path.exists(ann_dir):
            ann = annotation.Annotation3D(
                labelDir= ann_dir, 
                sequence= self.frames[0]['file_path'].split(os.sep)[-4]
            )
            #load annotation data (currently dynamic objects are loaded)
            for i, frame_id in enumerate(self.frame_ids):
                for global_id in ann.objects:
                    if frame_id in ann.objects[global_id]:
                        #object postion, orientation, 3dbbox vertices 
                        obj_cls = ann.objects[global_id][frame_id].name #object class
                        obj_pos =  ann.objects[global_id][frame_id].T # object position
                        obj_ori =  ann.objects[global_id][frame_id].R # object position 
                        vertices_3dbbox = ann.objects[global_id][frame_id].vertices # [8,3] in world frame
                        # vertices_3dbbox = np.column_stack([vertices_3dbbox, np.ones(vertices_3dbbox.shape[0])])
                        # vertices_3dbbox = np.matmul(np.linalg.inv(pose_lidar), vertices_3dbbox.T).T[:,:3] #lidar_frame
                        self.annotations[i].append({'frame_id': frame_id,
                                                    'class': obj_cls, 
                                                    'type' : 'dynamic' if frame_id > 0 else 'static',
                                                    'position': obj_pos, 
                                                    'orientation': obj_ori, 
                                                    'vertices': vertices_3dbbox})
        else:
         print(f"[WARN] No annotations found for {self.sequence_id}")