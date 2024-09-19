import glob
import os
import random
import time
import json
import cv2
import imageio
import mcubes
import numpy as np
import torch
import trimesh
from rich.console import Console
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import copy
from typing import Union, Literal
import open3d as o3d
from nvsf.lib import tools
from nvsf.nerf.dataset import dataset_utils
from nvsf.lib import convert
from nvsf.preprocess.generate_rangeview import LiDAR_2_Pano

def is_ali_cluster():
    import socket

    hostname = socket.gethostname()
    return "auto-drive" in hostname


@torch.jit.script
def linear_to_srgb(x):
    """convert between linear RGB to standard sRGB color spaces using the standard conversions"""

    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    """convert between standard sRGB color space to linear RGB using the standard conversions"""

    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def filter_bbox_dataset(pc, OBB_local):
    bbox_mask = np.isnan(pc[:, 0])
    z_min, z_max = min(OBB_local[:, 2]), max(OBB_local[:, 2])
    for i, (c1, c2) in enumerate(zip(pc[:, 2] <= z_max, pc[:, 2] >= z_min)):
        bbox_mask[i] = c1 and c2
    pc = pc[bbox_mask]
    OBB_local = sorted(OBB_local, key=lambda p: p[2])
    OBB_2D = np.array(OBB_local)[:4, :2]
    pc = filter_poly(pc, OBB_2D)
    return pc


def filter_poly(pcs, OBB_2D):
    OBB_2D = sort_quadrilateral(OBB_2D)
    mask = []
    for pc in pcs:
        mask.append(is_in_poly(pc[0], pc[1], OBB_2D))
    return pcs[mask]


def sort_quadrilateral(points):
    points = points.tolist()
    top_left = min(points, key=lambda p: p[0] + p[1])
    bottom_right = max(points, key=lambda p: p[0] + p[1])
    points.remove(top_left)
    points.remove(bottom_right)
    bottom_left, top_right = points
    if bottom_left[1] > top_right[1]:
        bottom_left, top_right = top_right, bottom_left
    return [top_left, top_right, bottom_right, bottom_left]


def is_in_poly(px, py, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def estimate_plane(xyz, normalize=True):
    """
    :param xyz:  3*3 array
    x1 y1 z1
    x2 y2 z2
    x3 y3 z3
    :return: a b c d
      model_coefficients.resize (4);
      model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
      model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
      model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
      model_coefficients[3] = 0;
      // Normalize
      model_coefficients.normalize ();
      // ... + d = 0
      model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));
    """
    vector1 = xyz[1,:] - xyz[0,:]
    vector2 = xyz[2,:] - xyz[0,:]

    if not np.all(vector1):
        # print('will divide by zero..')
        return None
    dy1dy2 = vector2 / vector1

    if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
        return None

    a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
    b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
    c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
    # normalize
    if normalize:
        r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
    # return a,b,c,d
    return np.array([a,b,c,d])

def my_ransac(data,
              distance_threshold=0.3,
              P=0.99,
              sample_size=3,
              max_iterations=1000,
              ):
    """
    :param data:
    :param sample_size:
    :param P :
    :param distance_threshold:
    :param max_iterations:
    :return:
    """
    max_point_num = -999
    i = 0
    K = 10
    L_data = len(data)
    R_L = range(L_data)

    while i < K:
        s3 = random.sample(R_L, sample_size)

        if abs(data[s3[0],1] - data[s3[1],1]) < 3:
            continue

        coeffs = estimate_plane(data[s3,:], normalize=False)
        if coeffs is None:
            continue

        r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
        d = np.divide(np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) , r)
        d_filt = np.array(d < distance_threshold)
        near_point_num = np.sum(d_filt,axis=0)

        if near_point_num > max_point_num:
            max_point_num = near_point_num

            best_model = coeffs
            best_filt = d_filt

            w = near_point_num / L_data

            wn = np.power(w, 3)
            p_no_outliers = 1.0 - wn

            K = (np.log(1-P) / np.log(p_no_outliers))

        i += 1
        if i > max_iterations:
            print(' RANSAC reached the maximum number of trials.')
            break

    return np.argwhere(best_filt).flatten(), best_model

def range_filter(
    pcd:np.ndarray, 
    dist_min:float=1, 
    dist_max:float=50, 
    z_limit:list[float]=[-2.5, 4]
    ) -> np.ndarray:
    """Filter points based on distance and ego vehicles surrounding"""
    
    #point distance from the sensor
    dist = np.sqrt(np.sum(pcd[:, :3] ** 2, axis = 1)) 
    
    #ego vehicle mask
    ego_mask = np.asarray(pcd[:,0]>-2) & np.asarray(pcd[:,0]<2) \
               & np.asarray(pcd[:,1]>-1) & np.asarray(pcd[:,1]<1) \
               & np.asarray(pcd[:,2]>-2) & np.asarray(pcd[:,2]<2) 
    
    #Combines mask for distance and ego vehicle
    mask = np.asarray(dist >= dist_min) & np.asarray(dist <= dist_max) \
           & np.asarray(pcd[:,2]>z_limit[0]) & np.asarray(pcd[:,2]<z_limit[1]) \
           & ~ego_mask
    
    #Filtered points
    pcd = pcd[mask]
    return pcd

def point_removal(
    pc_raw:np.ndarray, 
    dist_min:float=1, 
    dist_max:float=50, 
    z_limit:list[float]=[-2.5, 4]
    ) -> np.ndarray:
    """Remove ground points and outliers using RANSAC and statistical outlier removal"""
    
    #Filter points based on distance and ego vehicle surrounding
    pc_rm = range_filter(pc_raw, dist_min, dist_max, z_limit)

    #remove outliers and noisy points
    pcd_rm = o3d.geometry.PointCloud()
    pcd_rm.points = o3d.utility.Vector3dVector(pc_rm[:,:3])
    pcd_rm, ind = pcd_rm.remove_statistical_outlier(64, 3.0) #nb_neighbors, std_ratio
    pc_rm = np.asarray(pcd_rm.points)
    
    #remove ground points
    indices, _ = my_ransac(pc_rm[:, :3], distance_threshold=0.15)
    index_total = indices
    for i in range(5):
        indices, _ = my_ransac(pc_rm[:, :3], distance_threshold=0.15)
        index_total = np.unique(np.concatenate((index_total, indices)))
    indices = index_total
    
    indices = indices[pc_rm[indices, 2] < -1]    
    pc_ground = pc_rm[indices].copy()
    
    pc_rm[indices] = 999 + 1
    pc_rm = pc_rm[pc_rm[:, 2] <= 999]

    #Again remove outliers and noisy points
    pcd_rm = o3d.geometry.PointCloud()
    pcd_rm.points = o3d.utility.Vector3dVector(pc_rm[:,:3])
    pcd_rm, ind = pcd_rm.remove_statistical_outlier(64, 3.0) #nb_neighbors, std_ratio
    pc_rm = np.asarray(pcd_rm.points)

    return pc_rm, pc_ground

def torch_vis_2d(x, renormalize=False):
    """Visualize the lidar pano image

    Args:
        - x (array/tensor): Pano image of shape [3, H, W] or [1, H, W] or [H, W]
        - renormalize (bool, optional): Normalize the values b/w [0,1]. Defaults to False.
    """

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f"[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}")

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (
            x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8
        )

    plt.imshow(x)
    plt.show()

def extract_fields(bound_min, bound_max, xyz_res, query_func, S=128):
    """Sample a function over a bounded space.

    Args:
        bound_min (ndarray): The minimum bounds of the space. Shape (D,) where D is the space dimensionality.
        bound_max (ndarray): The maximum bounds of the space. Shape (D,).
        resolution (int): The number of samples to generate per dimension.
        query_func (callable): A function that takes an ndarray of shape (N, D) and returns values of shape (N, V). 
        S (int, optional): The chunk size to split sampling into. Default 128.

    Returns:
        samples (ndarray): The sampled points of shape (N, D) where N = resolution ** D.
        values (ndarray): The values returned by query_func for each sample. Shape (N, V).
    """
    #generate x,y,z coordinates and split in batches
    X = torch.linspace(bound_min[0], bound_max[0], xyz_res[0]).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], xyz_res[1]).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], xyz_res[2]).split(S)
    
    #create empty grid for density
    u = np.zeros([xyz_res[0], xyz_res[1], xyz_res[2]], dtype=np.float32)
    pnts_w_sigma = np.zeros([xyz_res[0]*xyz_res[1]*xyz_res[2], 4], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = dataset_utils.custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), 
                                     yy.reshape(-1, 1), 
                                     zz.reshape(-1, 1)],
                                     dim=-1)  # [S, 3]

                    #Query density from MLP_sigma
                    val = query_func(pts)
                    val_u = (val.reshape(len(xs), len(ys), len(zs))
                             .detach()
                             .cpu()
                             .numpy())  # [S, 1] --> [x, y, z]

                    #fill the density value in the grid array
                    u[xi * S : xi * S + len(xs),
                      yi * S : yi * S + len(ys),
                      zi * S : zi * S + len(zs),] = val_u
                    
                    #fill pcd data
                    # pcd_ = np.column_stack((pts.detach().cpu().numpy(), torch.sigmoid(val).detach().cpu().numpy()))
                    pcd_ = np.column_stack((pts.detach().cpu().numpy(), val.detach().cpu().numpy()))
                    # np.savetxt('points.txt', pcd_, delimiter= ' ', fmt='%f')
                    pnts_w_sigma[xi * S : xi * S + len(pts), : ] = pcd_

    return u, pnts_w_sigma


def extract_geometry(bound_min:list, bound_max:list, xyz_res:list, threshold:int, query_func, smoothing:bool=False):
    """Extract a mesh from a volumetric field using marching cubes.

    Args:
        - bound_min (torch.Tensor): Minimum coordinates of the volume boundary.
        - bound_max (torch.Tensor): Maximum coordinates of the volume boundary.
        - resolution (int): Resolution of the sampling grid.
        - threshold (float): Isovalue to use for mesh extraction. 
            Any grid point where u >= threshold is considered "inside" the isosurface,
            Any point where u < threshold is "outside" the isosurface.
        - query_func (callable): Function to query the volumetric field.
        - smoothing (bool): Laplacian smoothing to the density field before mesh extraction. Default False.

    Returns:
        - np.ndarray: Vertices of the extracted mesh.
        - np.ndarray: Triangles of the extracted mesh.
    """
    # print(f"threshold: {threshold}")
    
    #get density values of each voxel points
    u, pnts_w_sigma = extract_fields(bound_min, bound_max, xyz_res, query_func)
    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    #Create mesh from density values using marching cubes
    if smoothing:
        u = mcubes.smooth(u)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = (
        vertices / (np.array(xyz_res) -1.0) * (b_max_np - b_min_np)[None, :]
        + b_min_np[None, :]
    )
    return vertices, triangles, pnts_w_sigma

class cal_pred_errmat:
    """
    save evaluation result in json file
    print message
    Return:
        - json file with evaluation results
    """
    def __init__(self, lidar_metrics, camera_metrics, path, **kwargs):
        kwargs = tools.dict_to_cls(**kwargs)
        #lidar
        self.lidar_err_mat= {
            'pcd_error':dict(),
            'range_error':dict(),
            'intensity_error':dict(),
            'raydrop_error': dict()
        }
        if len(lidar_metrics) > 0 and kwargs.opt.enable_lidar:
            self.lidar_err_mat['pcd_error']['CD'] = round(float(lidar_metrics['point'].measure()[0]), 3)
            self.lidar_err_mat['pcd_error']['f_score'] = round(float(lidar_metrics['point'].measure()[1]), 3)
            self.lidar_err_mat['range_error']['RMSE'] = round(float(lidar_metrics['depth'].measure()[0]), 3)
            self.lidar_err_mat['range_error']['MedAE'] = round(float(lidar_metrics['depth'].measure()[1]), 3)
            self.lidar_err_mat['range_error']['LPIPS'] = round(float(lidar_metrics['depth'].measure()[2]), 3)
            self.lidar_err_mat['range_error']['SSIM'] = round(float(lidar_metrics['depth'].measure()[3]), 3)
            self.lidar_err_mat['range_error']['PSNR'] = round(float(lidar_metrics['depth'].measure()[4]), 3)
            self.lidar_err_mat['intensity_error']['RMSE'] = round(float(lidar_metrics['intensity'].measure()[0]), 3)
            self.lidar_err_mat['intensity_error']['MedAE'] = round(float(lidar_metrics['intensity'].measure()[1]), 3)
            self.lidar_err_mat['intensity_error']['LPIPS'] = round(float(lidar_metrics['intensity'].measure()[2]), 3)
            self.lidar_err_mat['intensity_error']['SSIM'] = round(float(lidar_metrics['intensity'].measure()[3]), 3)
            self.lidar_err_mat['intensity_error']['PSNR'] = round(float(lidar_metrics['intensity'].measure()[4]), 3)
            self.lidar_err_mat['raydrop_error']['RMSE'] = round(float(lidar_metrics['raydrop'].measure()[0]), 3)
            self.lidar_err_mat['raydrop_error']['Accuracy'] = round(float(lidar_metrics['raydrop'].measure()[1]), 3)
            self.lidar_err_mat['raydrop_error']['f_score'] = round(float(lidar_metrics['raydrop'].measure()[2]), 3)
        
        #camera
        self.rgb_err_mat ={
            "depth": dict(),
            "rgb":dict()
        }
        if len(camera_metrics) > 0 and kwargs.opt.enable_rgb:
            self.rgb_err_mat['depth']['RMSE'] = round(float(camera_metrics['rmse'].measure()), 3)
            self.rgb_err_mat['rgb']['LPIPS'] = round(float(camera_metrics['lpips'].measure()), 3)
            self.rgb_err_mat['rgb']['SSIM'] = round(float(camera_metrics['ssim'].measure()), 3)
            self.rgb_err_mat['rgb']['PSNR'] = round(float(camera_metrics['psnr'].measure()), 3)

        
        self.err_mat= {"lidar": self.lidar_err_mat,
                       "rgb": self.rgb_err_mat}
        self.path = path
    
    def save_json(self):
        #save in json file
        self.file_path=self.path + '_error_matrices.json'
        with open(self.file_path, 'w') as f:
            json.dump(self.err_mat, f, indent=4)
    
    def print_msg(self):
        print(f"[INFO] Results saved in json file at: {self.file_path}")

def get_pcd_bound_to_world(pred_depth, pred_intensity, loader, frame_data):
    """
    Calculate pcd using predicted depth and intensity
    Transform pcd from scene bound frame to world frame
    Args:
        - pred_depth(np.array): nd array of predicted depth [H, W]
        - pred_intensity(np.array): nd array of predicted intensity [1, H, W]
        - loader: data loader object
        - frame_data: data of frame
    return:
        - pred_pcd_world(np.array): nd array of predicted point cloud in world frame
    """
    # Save predicted range and intensities in pcd in world frame
    #1 Get pcd from pred depth and intensities in lidar_new frame + bound frame scale
    pred_pcd = convert.pano_to_lidar_with_intensities(pred_depth , 
                                            pred_intensity, 
                                            loader._data.intrinsics_lidar,
                                            loader._data.intrinsics_hoz_lidar)
    #2 Transform pcd from bound frame to lidar_new frame (rescale)
    pred_pcd[:,:3] = pred_pcd[:,:3]/loader._data.scale 
    
    #3 Get T matrix from lidar_new to world frame (bound frame -> world frame)                   
    mat_T = frame_data['poses_lidar'][0,...].detach().cpu().numpy() #convert tensor to numpy
    mat_T[:3,-1] = (mat_T[:3,-1]/loader._data.scale) + loader._data.offset # conpensation for rescale and offset
    
    #4 Transform pcd from lidar_new to world frame
    pred_pcd_ = np.column_stack((pred_pcd[:,:3], np.ones(len(pred_pcd)))) # add column of ones
    pred_pcd_world_ = np.dot(mat_T, pred_pcd_.T).T[:,:3] #Transform to world
    pred_pcd_world = np.column_stack((pred_pcd_world_, pred_pcd[:,-1])) # add intensity

    return pred_pcd_world

class UtilsTrainer:
    """Required methods for training utils
    """
    def __init__(self):
        self.local_rank = 0
        self.mute = False 
        self.console = Console()
        self.workspace = None
        self.name = None
        self.use_checkpoint = None
        self.time_stamp = None
        self.device = None
        self.fp16 = False
        self.log_ptr = None
        self.model = None
        # pass

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def workspace_prepare(self):
        """Prepare workspace and logging
        """

        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            os.makedirs(os.path.join(self.workspace, 'validation'), exist_ok=True)
            os.makedirs(os.path.join(self.workspace, 'results'), exist_ok=True)
            os.makedirs(os.path.join(self.workspace, 'export'), exist_ok=True)
            
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")


        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        # self.log(
        #     f"[INFO] #parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}"
        # )

        #Log model information
        self.log("--------------=========== MODEL START ===========--------------")
        self.log(self.model)
        self.log(f"Total parameters           :=> {sum(p.numel() for p in self.model.parameters())}")
        self.log(f"Total trainable parameters :=> {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}")
        self.log("--------------===========  MODEL END ===========--------------")

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading model from checkpoint {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
    
    def log(self, *args, **kwargs):
        """for logging the events"""

        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def export_mesh_density(self, bound_min=None, bound_max=None, save_path=None, xyz_res=[256,256,256], threshold=10, smoothing=False):
        """Export nerf to mesh

        Args:
            - save_path (str, optional): Path for saving the mesh model. Defaults to None.
            - resolution (int, optional): resolution of sampling grid . Defaults to 256.
            - threshold (int, optional): threshold for extracting geometry. Defaults to 10.

        Returns:
            - mesh (ply): mesh will be saved path
            - density field (text): txt file will be saved path
        """
        if save_path is None:
            save_path = os.path.join(
                self.workspace, "export", f"{self.name}_{self.epoch}.ply"
            )

        self.log(f"==> Saving mesh and pnts with intensity: {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Export mesh function
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))["sigma"]
            return sigma
        
        # Get minimum and maximum bound values
        if bound_min is None:
            bound_min = self.model.aabb_infer[:3]
        else:
            bound_min = torch.tensor(bound_min, device=self.model.aabb_infer.device)
        if bound_max is None:
            bound_max = self.model.aabb_infer[3:]
        else:
            bound_max = torch.tensor(bound_max, device=self.model.aabb_infer.device)
        assert abs(bound_min).max() <=1.0 and abs(bound_max).max() <=1.0, "Bound values should be within -1 to 1 range"
            
        #extract vertices and triangles from density field
        vertices, triangles, pnts_w_sigma = extract_geometry(bound_min= bound_min,
                                                             bound_max= bound_max,
                                                             xyz_res= xyz_res,
                                                             threshold= threshold,
                                                             query_func= query_func,
                                                             smoothing = smoothing)
        #Creating and saving mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)
        self.log(f"==> Finished saving mesh.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        """Save the checkpoint data

        Args:
            - name (str, optional): Name of the checkpoint. Defaults to None.
            - full (bool, optional): flag for saving full model. Defaults to False.
            - best (bool, optional): Flag for saving only the best model. Defaults to False.
            - remove_old (bool, optional): Flag for removing the old models. Defaults to True.
        """
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:
            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, 
                    # so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if "density_grid" in state["model"]:
                        del state["model"]["density_grid"]

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        """Load the checkpoint from the given checkpoints

        Args:
            - checkpoint (str, optional): Checkpoint path. Defaults to None.
            - model_only (bool, optional): Flag for model only. Defaults to False.
        """
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if model_only:
            return

        if "stats" in checkpoint_dict:
            self.stats = checkpoint_dict["stats"]
        if "epoch" in checkpoint_dict:
            self.epoch = checkpoint_dict["epoch"]
        if "global_step" in checkpoint_dict:
            self.global_step = checkpoint_dict["global_step"]
            self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def compute_object_masks(depth: Union[np.ndarray, torch.Tensor], intensity: Union[np.ndarray, torch.Tensor], data: dict, **kwargs) -> np.ndarray:
    """Calculate background and foreground object mask

    Args:
        - depth (Union[np.ndarray, torch.Tensor]): depth values of shape [H, W]
        - intensity (Union[np.ndarray, torch.Tensor]): intensity values of shape [H, W]
        - data (dict): data of current loop from dataloader

    Returns:
        - np.ndarray: static and dynamic object mask in pano format [H, W]and pcd [N, 1]
    """
    kwargs = tools.dict_to_cls(**kwargs)
                
    #convert to numpy
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.detach().cpu().numpy()
    if isinstance(data['poses_lidar'], torch.Tensor):
        T_lidar2world = data['poses_lidar'][0].detach().cpu().numpy()
    
    #aabb to world
    T_lidar2world[:3, 3] = (T_lidar2world[:3, 3] / kwargs.opt.scale) + kwargs.opt.offset

    #convert to point cloud
    pred_pcd_lidar = convert.pano_to_lidar_with_intensities(depth / kwargs.opt.scale, 
                                                            intensity[:, :, None], 
                                                            kwargs.opt.intrinsics_lidar,
                                                            kwargs.opt.intrinsics_hoz_lidar)

    #get background and foreground objects mask
    dyna_obj_mask = []
    hull_points = [] 
    for ann in data['3d_annotation']:
        #transfrom vertices from world frame to lidar frame
        ver_3dbbox = ann['vertices'] 
        ver_3dbbox = np.column_stack((ver_3dbbox, np.ones(ver_3dbbox.shape[0])))
        ver_3dbbox = np.matmul(np.linalg.inv(T_lidar2world),  ver_3dbbox.T).T[:,:3] 
        
        #check points in convex hull
        pcd_filtered, inhull_mask = tools.check_in_hull(pred_pcd_lidar, ver_3dbbox)
        hull_points.append(pcd_filtered)
        dyna_obj_mask.append(inhull_mask)
    hull_points = np.vstack(hull_points)
    dyna_obj_mask = np.bitwise_or.reduce(dyna_obj_mask, axis=0) #[N, 1]
    static_obj_mask = ~dyna_obj_mask #[N, 1]

    dyna_obj_pano_mask = LiDAR_2_Pano(                    #[H, W, 1]
        local_points_with_intensities= np.column_stack([pred_pcd_lidar[:, :3], dyna_obj_mask]),
        lidar_H= data['H_lidar'],
        lidar_W= data['W_lidar'],
        intrinsics= kwargs.opt.intrinsics_lidar,
        intrinsics_hoz= kwargs.opt.intrinsics_hoz_lidar,
        max_depth= kwargs.opt.lidar_max_depth / kwargs.opt.scale,
        )[:,:,1]
    static_obj_pano_mask = np.where(dyna_obj_pano_mask == 0, 1, 0) #[H, W, 1]

    return static_obj_pano_mask, dyna_obj_pano_mask, static_obj_mask, static_obj_mask


def compute_object_masks_img(data: dict, **kwargs) -> np.ndarray:
    """Calculate background and foreground object mask

    Args:
        data (dict): data of current loop from dataloader

    Returns:
        np.ndarray: static and dynamic object mask in image shape [H, W]
    """
    kwargs = tools.dict_to_cls(**kwargs)
                
    #convert to numpy
    if isinstance(data['poses_lidar'], torch.Tensor):
        T_cam2world = data['pose'][0].detach().cpu().numpy()

    #compute camera K matrix
    # K_cam = np.zeros((3, 3))
    # K_cam[0, 0] = data['intrinsic_cam'][0] #fx
    # K_cam[1, 1] = data['intrinsic_cam'][1] #fy
    # K_cam[0, 2] = data['intrinsic_cam'][2] #cx
    # K_cam[1, 2] = data['intrinsic_cam'][3] #cy
    # K_cam[2, 2] = 1
    K_cam = data['intrinsic_cam']
    
    #computer camera to world T matrix from aabb
    T_cam2world[:3, 3] = (T_cam2world[:3, 3] / kwargs.opt.scale) + kwargs.opt.offset

    #compute corner points of 2d bounding boxes in image frame
    ver_2dbboxes = []
    pixels_2dbboxes = []
    for ann in data['3d_annotation']:
        ver_3dbbox = ann['vertices'] #world frame
        ver_3dbbox = np.column_stack((ver_3dbbox, np.ones(ver_3dbbox.shape[0])))
        ver_3dbbox_cam = np.matmul(np.linalg.inv(T_cam2world),  ver_3dbbox.T).T[:,:3]
        ver_2dbbox = np.matmul(K_cam, ver_3dbbox_cam.T).T
        if np.all(ver_2dbbox[:,2] > 0): 
            ver_2dbbox = (ver_2dbbox/ ver_2dbbox[:, [2]])[:,:2] #normalize by depth
            
            #computer 2d bbox corner points
            x_min = max(0, int(ver_2dbbox[:,0].min()))
            y_min = max(0, int(ver_2dbbox[:,1].min()))
            x_max = min(data['W']-1, int(ver_2dbbox[:,0].max()))
            y_max = min(data['H']-1, int(ver_2dbbox[:,1].max()))
            # w, h = x_max - x_min, y_max - y_min
            bbox_2d = np.array([[x_min, y_min], [x_max, y_min], 
                                [x_max, y_max], [x_min, y_max]])
            ver_2dbboxes.append(bbox_2d)
            
            #pixel coordinates within 2d bbox
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    pixels_2dbboxes.append((y, x))
    
    if len(pixels_2dbboxes) > 0:
        pixels_2dbboxes = np.vstack(pixels_2dbboxes)
        #compute 2d bbox mask for image
        static_img_mask = np.ones([data['H'],  data['W']], dtype=bool)
        static_img_mask[pixels_2dbboxes[:,0], pixels_2dbboxes[:,1]] = False
        dynamic_img_mask = ~static_img_mask
    else:
        static_img_mask = np.ones([data['H'], data['W']], dtype = bool)
        dynamic_img_mask = ~static_img_mask

    return static_img_mask, dynamic_img_mask

def vis_training(index, sampled_img, save_path=None, show=False, fig_name='Image pixels sampling'):
    """Visualize training plots image sampled pixels"""       
    
    plt.figure(fig_name)
    plt.clf() #clear
    plot_rows = 3
    plot_cals = 1
    # Create custom colormap
    white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])

    # error map 
    plt.subplot(plot_rows,plot_cals,1) 
    plt.title(f"Error map '{index}'")
    # plt.imshow(self.sampled_img[index][4]/(np.linalg.norm(self.sampled_img[index][4]) + 1e-16), 
    plt.imshow(sampled_img[index][2], cmap=white_red) # gray, gray_r, Reds, viridis
    plt.xlabel('Width')
    plt.ylabel('Height')

    #pixels sampled so far during training
    plt.subplot(plot_rows,plot_cals,2) 
    plt.title(f"Pixels sampled so far '{index}'")
    plt.imshow(sampled_img[index][0], cmap='gray')
    plt.xlabel('Width')
    plt.ylabel('Height')

    #pixels sampled in this iteration
    plt.subplot(plot_rows,plot_cals,3) 
    plt.title(f"Pixels sampled now '{index}'")
    plt.imshow(sampled_img[index][1], cmap='gray')
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.subplots_adjust(left=None, 
                        bottom=None,  
                        right=None, 
                        top=None, 
                        wspace=None, 
                        hspace=None)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    
    if show:
        plt.show(block=False)
        plt.pause(0.00001)
