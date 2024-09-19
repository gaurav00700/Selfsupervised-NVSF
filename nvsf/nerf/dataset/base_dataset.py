import json
import os
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import tqdm
import copy
import abc
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Dataset
import trimesh
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Tuple
from collections import defaultdict
from nvsf.lib import convert
from nvsf.lib import tools
from nvsf.nerf.dataset import dataset_utils

@dataclass
class BaseDataset:
    device: str = "cpu"
    split: Literal['train', 'val', 'test'] = "train"  # train, val, test
    root_path: str = "nvsf/data/daas"
    sequence_id: str = "CityStreet_dgt_2021-07-13-11-21-58_0_s0"
    training: bool = True #training of nerf
    preload: bool = True  # preload data into GPU
    scale: float = (1) # camera radius scale to make sure camera are inside the bounding box.
    offset: list = field(default_factory=list)  # offset
    intrinsics_lidar: list = field(default_factory=lambda: [13.8, 24.6])  # (2.0, 26.9) => vertical fov_up, fov
    intrinsics_hoz_lidar: list = field(default_factory=lambda: [90.0, 180.0])  # (2.0, 26.9) => horizontal fov_up, fov
    fp16: bool = True  # if preload, load into fp16.
    patch_size: int = 1  # size of the image patch to be sample for camera.
    patch_size_lidar: int = 1  # size of the pano image patch to sample for lidar.
    enable_rgb: bool = True #camera modality
    enable_lidar: bool = True #lidar modality
    permute_lidar_camera_index: bool = False
    color_space: str = 'srgb'
    num_rays: int = 4096 #ray batch for camera image
    num_rays_lidar: int = 4096 #ray batch for lidar pano image 
    use_error_map:bool = True #use error map for pixel sampling
    delta_position: list = field(default_factory=lambda: [0., 0., 0.]) # delta position of lidar
    delta_orientation: list = field(default_factory=lambda: [0., 0., 0.]) # delta position of lidar
    H_lidar_new: int = 0 # desired height of pano image
    W_lidar_new: int = 0 # desired width of pano image
    intrinsics_lidar_new: list = field(default_factory=lambda: [0.0, 0.0])  # vertical (fov_up, fov)
    intrinsics_hoz_lidar_new: list = field(default_factory=lambda: [0.0, 0.0])  # horizontal (fov_up, fov)
    delta_pos_camera: list = field(default_factory=lambda: [0., 0., 0.]) # delta position of camera
    delta_orient_camera: list = field(default_factory=lambda: [0., 0., 0.]) # delta position of camera
    H_new: int = 0 # desired height of camera image
    W_new: int = 0 # desired width of camera image
    
    def __post_init__(self):
        
        # self.training = self.split in ["train", "trainval"]
        self.num_rays = self.num_rays if self.training else -1
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        
        # load nerf-compatible format data (Json file)
        with open(
            os.path.join(self.root_path, "train", self.sequence_id, f"transforms_{self.sequence_id}_{self.split}.json"),"r",) as f:
            transform = json.load(f)
        
        # load image size
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"])
            self.W = int(transform["w"])
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # Get lidar pano image dimesions (height->vertical resolution and width-> Horizontal resolution)
        if "h_lidar" in transform and "w_lidar" in transform:
            self.H_lidar = int(transform["h_lidar"])
            self.W_lidar = int(transform["w_lidar"])
        if "nr_returns" in transform:
            self.nr_returns = int(transform["nr_returns"])

        #total number of frames
        self.num_frames = transform["num_frames"]
        
        # read images
        frames = transform["frames"]
        self.frames = sorted(frames, key=lambda d: d['file_path'])

        # get start and end frame number
        self.frame_start = frame_start = transform["frame_start"]
        self.frame_end = frame_end = transform["frame_end"]

        # load camera intrinsics
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y'])
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x'])
        cx = (transform['cx']) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy']) if 'cy' in transform else (self.H / 2)
        self.intrinsics = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

        self.poses = []
        self.images = []
        self.image_paths = []
        self.poses_lidar = [] # lidar2world transformation matrices for all frames
        self.images_lidar = [] # lidar range images for all frame
        self.image_depths = []
        self.times = []
        self.frame_ids = []
        for i, f in tqdm.tqdm(enumerate(self.frames), total=len(self.frames) ,desc=f"Loading {self.split} data"):
            #images
            f_path = os.path.join(self.root_path, f['file_path'])
            self.image_paths.append(f_path)
            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
            
            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255  # [H, W, 3/4]
            self.poses.append(pose)
            self.images.append(image)
            
            #lidar
            pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]
            # Read pano/range image [None, intensity, depth]
            f_lidar_path = os.path.join(self.root_path, f["lidar_file_path"])
            
            # channel1 None, channel2 intensity , channel3 depth
            pc = np.load(f_lidar_path) 
            
            # ground truth raydrop mask
            ray_drop = np.where(pc.reshape(-1, 3)[:, 2] == 0.0, 0.0, 1.0).reshape( 
                self.H_lidar, self.W_lidar, 1
            )
            
            # create lidar image for ground truth [raydrop, intensity, depth]
            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale],
                axis=-1,
            )

            #compute temporal
            time = np.asarray((f['frame_id']-frame_start)/(frame_end-frame_start))
            
            #append data
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)
            self.times.append(time)
            self.frame_ids.append(f["frame_id"])

            #image helper for camera intrinsics
            # if not self.training:
            pc = convert.pano_to_lidar(pc[:, :, 2], lidar_K=self.intrinsics_lidar, lidar_K_hoz=self.intrinsics_hoz_lidar)
            pts_2d = dataset_utils.lidar2points2d(pc, self.intrinsics, np.linalg.inv(pose) @ pose_lidar)
            image_depth = dataset_utils.get_lidar_depth_image(pts_2d, img_shape=(self.H, self.W))
            self.image_depths.append(image_depth)
                    
        # Create 3d [N, 4, 4] vector of lidar2world and cam2world
        self.poses_lidar = np.stack(self.poses_lidar, axis=0) 
        self.poses = np.stack(self.poses, axis=0)
        if len(self.images_lidar)> 0 : self.images_lidar = np.stack(self.images_lidar, axis=0) # [N, H, W, C]
        if len(self.images)> 0 : self.images = np.stack(self.images, axis=0) # [N, H, W, C]
        if len(self.image_depths)> 0 : self.image_depths = np.stack(self.image_depths, axis=0) # [N, H, W, C]
        
        # Load extra stuffs such as 3d annotations
        self._load_renderings()
        
        # change lidar position, orientation and resolution to desired values during testing or validation
        if not self.training and any([np.any(self.delta_orientation), 
                                      np.any(self.delta_position), 
                                      self.H_lidar_new != 0, 
                                      self.W_lidar_new != 0,
                                      np.any(self.intrinsics_lidar_new),
                                      np.any(self.intrinsics_hoz_lidar_new),
                                      np.any(self.delta_orient_camera), 
                                      np.any(self.delta_pos_camera),
                                      self.H_new != 0,
                                      self.W_new != 0,
                                      ]):
            # Transform poses from lidar_new -> world frame
            R_mat_w2wr = Rotation.from_euler('xyz', self.delta_orientation, degrees=True).as_matrix() # R matrix for delta angles
            T_mat_w2wr = np.row_stack((np.column_stack((R_mat_w2wr, self.delta_position)), [0., 0., 0., 1.])) #T matrix for delta changes
            self.poses_lidar = np.dot(self.poses_lidar, T_mat_w2wr).astype(np.float32) # transform poses for delta changes in world

            # Desired lidar horizontal and vertical resolution            
            if self.H_lidar_new != 0:
                self.H_lidar = int(self.H_lidar_new +2) # 2 for beams at extreme ends
            if self.W_lidar_new != 0:
                self.W_lidar = int(self.W_lidar_new)
            
            #change lidar intrinsics
            if np.any(self.intrinsics_lidar_new):
                self.intrinsics_lidar = self.intrinsics_lidar_new
            if np.any(self.intrinsics_hoz_lidar_new):
                self.intrinsics_hoz_lidar = self.intrinsics_hoz_lidar_new
            
            def hlpr_fn(x):
                """world to camera x, y, z (front, left, up) --> -y, -z, x (right, down, front)"""
                x = np.array(x)
                x[[1, 2]] *= -1 #flip x, y 
                x = x[[1,2,0]] #swap
                return x

            R_mat_w2wr = Rotation.from_euler('xyz', hlpr_fn(self.delta_orient_camera), degrees=True).as_matrix() # R matrix for delta angles
            T_mat_w2wr = np.row_stack((np.column_stack((R_mat_w2wr, hlpr_fn(self.delta_pos_camera))), [0., 0., 0., 1.])) #T matrix for delta changes
            self.poses = np.dot(self.poses, T_mat_w2wr).astype(np.float32) # transform poses for delta changes in world
                
            # Change height, width and intrinsics of camera
            if self.H_new != 0 or self.W_new !=0:
                scale_x = self.W_new / self.W if self.W_new != 0 else 1
                scale_y = self.H_new / self.H if self.H_new != 0 else 1
                self.intrinsics[0, 2] *= scale_x # new cx
                self.intrinsics[1, 2] *= scale_y # new cy
                # scale focal length to match new image size. 
                # scale = np.clip(min(scale_y, scale_x), min(scale_y, scale_x), 1) 
                # self.intrinsics[1, 1] *= scale  # new fy
                # self.intrinsics[0, 0] *= scale  # new fx
                self.H = self.H_new # new image height
                self.W = self.W_new # new image width

            # Disable validation
            self.images_lidar = None 
            self.images = None 
            self.image_depths = None

            print(f"[WARN] Sensor parameters are changed")

        # Offset frames to the center of the scene and scale for aabb frame
        self.poses_lidar[:, :3, -1] = (self.poses_lidar[:, :3, -1] - self.offset) * self.scale
        self.poses[:, :3, -1] = (self.poses[:, :3, -1] - self.offset) * self.scale

        #convert to tensor from numpy
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]
        self.poses = torch.from_numpy(self.poses)  # [N, 4, 4]
        self.times = torch.from_numpy(np.asarray(self.times, dtype=np.float32)).view(-1, 1) # [N, 1]
        self.frame_ids = torch.from_numpy(np.asarray(self.frame_ids, dtype=np.float32)).view(-1, 1) # [N, 1]
        if self.images_lidar is not None: self.images_lidar = torch.from_numpy(self.images_lidar).float()  # [N, H, W, C]
        if self.images is not None: self.images = torch.from_numpy(self.images)  # [N, H, W, C]
        if self.image_depths is not None: self.image_depths = torch.from_numpy(self.image_depths).float()  # [N, H, W, C]        
        
        if self.training:
            # Initialize error_map with ones[B, H, W]
            # if scale is float then inds will have one pxl error due to round off
            self.error_map = torch.ones([self.num_frames, int(self.H_lidar/2), int(self.W_lidar/2)], dtype=torch.float) #33x515
            self.error_map_rgb = torch.ones([self.num_frames, int(self.H/4), int(self.W/4)], dtype=torch.float) #94x352
            
            # Initializing pano sampled with zeros [B, 5, H, W]
            # channel = [pxl_smpl_all, pxl_smpl_now, error_map]
            self.pano_sampled = np.zeros([self.num_frames, 3, self.H_lidar, self.W_lidar],  dtype=np.float16)
            
            # Initializing camera image sampled with zeros [B, 3, H, W]
            # channel = [pxl_smpl_all, pxl_smpl_now, error_map]
            self.rgb_sampled = np.zeros([self.num_frames, 3, self.H, self.W],  dtype=np.float16)
        else:
            self.error_map  = None
            self.pano_sampled = None
            self.error_map_rgb  = None
            self.rgb_sampled = None

        # Load data to GPU
        if self.preload:
            #camera
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            
            #lidar
            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.images_lidar is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16:
                    dtype = torch.half #float16
                else:
                    dtype = torch.float #float32
                self.images_lidar = self.images_lidar.to(dtype).to(self.device)                
            
            #Other
            self.times = self.times.to(self.device)    
            self.frame_ids = self.frame_ids.to(self.device)  

            #error map
            if self.error_map is not None: self.error_map = self.error_map.to(self.device)
            if self.error_map_rgb is not None: self.error_map_rgb = self.error_map_rgb.to(self.device)
            # self.pano_sampled = self.pano_sampled.to(self.device)
        
        # [debug] calculate mean radius of all camera poses
        # self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        # print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # [debug] view all training poses.
        # dataset_utils.visualize_poses(self.poses.numpy())

    @abc.abstractmethod
    def _load_renderings(self):
        """Load extra renderings data"""

    def collate(self, index):
        """Collate function for dataloader"""

        B = len(index)  # a list of length 1

        # Get sample position and direction for rays belongs to positions
        results = {}

        results["index"]= index #data index list
        results["time"]= self.times[index].to(self.device) # [B, 1]
        results["frame_id"]= self.frame_ids[index].to(self.device) #[B, 1]
        results["3d_annotation"] = self.annotations[index[0]]
        error_map = None if self.error_map is None else self.error_map[index].to(self.device)
        error_map_rgb = None if self.error_map_rgb is None else self.error_map_rgb[index].to(self.device)

        #camera
        if self.enable_rgb:
            poses = self.poses[index].to(self.device)  # [B, 4, 4]
            #get camera rays
            rays = dataset_utils.get_rays(
                poses,
                self.intrinsics, 
                self.H, 
                self.W,
                self.num_rays, 
                self.patch_size,
                error_map_rgb,
                self.use_error_map
            )
            #add data for training 
            results.update({
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                "rays_rgb_inds" : rays["inds"], #[B, N]
                'pose': self.poses[index],  # for normal
                'intrinsic_cam': self.intrinsics
            })
        
        #lidar
        if self.enable_lidar:            
            poses_lidar = self.poses_lidar[index].to(self.device) #[B, 4, 4] 
            #get lidar rays
            rays_lidar = dataset_utils.get_lidar_rays(
                poses_lidar,
                self.intrinsics_lidar,
                self.intrinsics_hoz_lidar,
                self.H_lidar,
                self.W_lidar,
                self.num_rays_lidar,
                self.patch_size_lidar,
                error_map,
                self.use_error_map
            )
            #add data for training 
            results.update(
                {
                    "H_lidar": self.H_lidar,
                    "W_lidar": self.W_lidar,
                    "rays_o_lidar": rays_lidar["rays_o"], #[B, N]
                    "rays_d_lidar": rays_lidar["rays_d"], #[B, N]
                    "rays_pano_inds" :rays_lidar["inds"], #[B, N]
                    "poses_lidar": poses_lidar, #[B, 4, 4]                    
                }
            )
        
        #camera
        if self.images is not None and self.enable_rgb:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            image_depths = self.image_depths[index].to(self.device)
            if self.training:
                #gather pixels of rgb image for rays
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 
                                      1,
                                      torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
                #gather pixels of depth image for rays
                image_depths = torch.gather(image_depths.view(B, -1, 1), 
                                            1,
                                            torch.stack(1 * [rays['inds']], -1))  # [B, N, 1]
            # elif self.image_depths is not None:
            #     image_depths = self.image_depths[index].to(self.device)
                # results['image_depths'] = image_depths
            results['images'] = images
            results['image_depths'] = image_depths
            results["image_frame"] = self.images[index] 
        
        #lidar
        if self.images_lidar is not None and self.enable_lidar:
            images_lidar = self.images_lidar[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                #gather pixels of lidar pano image for rays
                C = images_lidar.shape[-1]
                images_lidar = images_lidar.view(B, -1, C) #view to [B, H*W, C]
                inds = torch.stack(tensors= [rays_lidar["inds"]] * C, dim= -1) #[B, H*W, C]
                
                #gather pixels of lidar range image for rays
                images_lidar = torch.gather(input= images_lidar, 
                                            dim= 1, 
                                            index= inds)  # [B, N, 3/4]
            results["images_lidar"] = images_lidar
            results["pano_frame"] = self.images_lidar[index] 

        return results

    def dataloader(self):
        """Load the dataset"""
        
        size = len(self.poses_lidar) if self.enable_lidar else len(self.poses)
        
        #create dataloader object
        loader = DataLoader(
            list(range(size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )

        #TODO: Make it better, for accessing error_map, poses ...
        loader._data = self 

        # check if range image gt is available
        loader.has_gt = self.images_lidar is not None if self.enable_lidar else self.images is not None
        return loader

    def __len__(self):
        """Returns number of frames in this dataset."""
        num_frames = len(self.poses_lidar)
        return num_frames
