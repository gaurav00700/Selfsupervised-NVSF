import glob
from locale import ABMON_10
import os
import random
from re import X
import time
import json
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorboardX
import torch, torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch_ema import ExponentialMovingAverage
from PIL import Image
import copy
from typing import Union, Literal
from nvsf.lib import tools, convert
from nvsf.lib import error_matrices as err_mat
from nvsf.nerf import utils
from nvsf.nerf.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from nvsf.preprocess.generate_rangeview import LiDAR_2_Pano

class Trainer(utils.UtilsTrainer):
    """Object class for Nerf training, validation, testing and helper functions"""
    def __init__(
        self,
        name,  # name of this experiment
        opt,  # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
            metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        depth_metrics={}, # error matrices 
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=1,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):
        super(Trainer, self).__init__()
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.depth_metrics = depth_metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (device
            if device is not None
            else torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        self.bce_fn = torch.nn.BCELoss()
        self.cham_fn = chamfer_3DDist()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        # if self.opt.patch_size > 1:
        #     import lpips
        #     self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=0.001,
                                        weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                            lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(),
                                                decay=ema_decay)
        else:
            self.ema = None
        
        # mix precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        #lidar metrics for foreground and background objects
        self.depth_metrics_static = copy.deepcopy(self.depth_metrics)
        self.depth_metrics_dynamic = copy.deepcopy(self.depth_metrics)

        #camera metrics for foreground and background objects
        self.metrics_static = copy.deepcopy(self.metrics)
        self.metrics_dynamic = copy.deepcopy(self.metrics)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {"loss": [], 
                      "valid_loss": [], 
                      "results": [],  # metrics[0], or valid_loss
                      "results_rgb": [],  # metrics[0], or valid_loss
                      "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
                      "best_result": None}

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"
        
        #prepare workspace for logging and saving data
        self.workspace_prepare()

    def adjust_activate_levels(self, epoch, total_epochs):
        if epoch <= int(total_epochs * 0.2):
            self.opt.activate_levels = 16
        else:
            self.opt.activate_levels = 8
    
    def train_step(self, data, pbar_train):
        """Training of Nerf model for one epoch step."""

        # Initialize all returned values
        pred_rgb = None
        gt_rgb = None
        image_pred_depth = None 
        image_depths = None
        pred_raydrop = None
        gt_raydrop = None
        pred_intensity = None
        gt_intensity = None
        pred_depth = None
        gt_depth = None
        loss = 0

        #Temporal information
        time_ego = data['time'] # [B, 1]
        
        # Lidar
        loss_d = 0
        loss_i = 0
        loss_rd = 0
        lidar_loss = 0
        chamfer_loss = 0
        flow_loss = 0
        los_loss = 0
        loss_sr = 0
        if self.opt.enable_lidar:
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            images_lidar = data["images_lidar"]  # [B, N, 3/4], ground truth
            B_lidar, N_lidar, C_lidar = images_lidar.shape
            
            # Get ground truth of raydrop mask, intensity and depth [B, N]
            gt_raydrop = images_lidar[:, :, 0]
            gt_intensity = images_lidar[:, :, 1] * gt_raydrop
            gt_depth = images_lidar[:, :, 2] * gt_raydrop
            
            #Render lidar model
            outputs_lidar = self.model.render(rays_o_lidar,
                                              rays_d_lidar,
                                              time_ego,
                                              cal_lidar_color=True,
                                              staged=False,
                                              perturb=True,
                                              force_all_rays=False if self.opt.patch_size == 1 else True,
                                              **vars(self.opt))
            
            # Prediction using raydrop mask [B, N]
            pred_raydrop = outputs_lidar["image_lidar"][:, :, 0]
            pred_raydrop_mask = torch.where(pred_raydrop > self.opt.raydrop_thres, 1, 0)
            pred_intensity = outputs_lidar["image_lidar"][:, :, 1] * gt_raydrop
            pred_depth = outputs_lidar["depth_lidar"] * gt_raydrop
            
            if self.opt.raydrop_loss == 'bce':
                pred_raydrop = F.sigmoid(pred_raydrop)
            
            # label smoothing for ray-drop
            smooth = self.opt.smooth_factor # 0.2
            gt_raydrop_smooth = gt_raydrop.clamp(smooth, 1-smooth)
            
            # Training loss calculation
            loss_d = self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth)
            loss_rd = self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop_smooth)
            loss_i = self.opt.alpha_i * self.criterion["intensity"](pred_intensity, gt_intensity)
            lidar_loss = loss_d + loss_rd + loss_i
            
            #Expand dimensions [B, N] --> [B, N, 1]
            pred_raydrop = pred_raydrop.unsqueeze(-1)
            pred_intensity = pred_intensity.unsqueeze(-1)
            pred_depth = pred_depth.unsqueeze(-1)
            gt_raydrop = gt_raydrop.unsqueeze(-1)
            gt_intensity = gt_intensity.unsqueeze(-1)
            gt_depth = gt_depth.unsqueeze(-1)

            # CD Loss
            pred_lidar = rays_d_lidar * pred_depth / self.opt.scale
            gt_lidar = rays_d_lidar * gt_depth / self.opt.scale
            dist1, dist2, _, _ = self.cham_fn(pred_lidar, gt_lidar)
            chamfer_loss = (dist1 + dist2).mean() * 0.5
            # loss = loss + chamfer_loss

            #Scene flow loss
            if self.opt.flow_loss:
                frame_idx = int(time_ego * (self.opt.num_frames - 1)) 
                pc = self.pc_list[f"{frame_idx}"]
                pc = torch.from_numpy(pc).cuda().float().contiguous()
                
                # Query scene flow mlp
                pred_flow = self.model.flow(pc, time_ego)
                pred_flow_forward = pred_flow["flow_forward"]
                pred_flow_backward = pred_flow["flow_backward"]

                #Forward flow loss
                if f"{frame_idx+1}" in self.pc_list.keys():
                    pc_pred = pc + pred_flow_forward
                    pc_forward = self.pc_list[f"{frame_idx+1}"]
                    pc_forward = torch.from_numpy(pc_forward).cuda().float().contiguous()                
                    dist1, dist2, _, _ = self.cham_fn(pc_pred.unsqueeze(0), pc_forward.unsqueeze(0))
                    chamfer_dist = (dist1.sum() + dist2.sum()) * 0.5 #CD loss
                    # loss = loss + 0.01 * chamfer_dist
                    # flow_loss += 0.01 * chamfer_dist + 0.01 * pred_flow_forward.abs().mean()
                    flow_loss += chamfer_dist + pred_flow_forward.abs().mean()
                
                #Backward flow loss
                if f"{frame_idx-1}" in self.pc_list.keys():
                    pc_pred = pc + pred_flow_backward
                    pc_backward = self.pc_list[f"{frame_idx-1}"]
                    pc_backward = torch.from_numpy(pc_backward).cuda().float().contiguous()
                    dist1, dist2, _, _ = self.cham_fn(pc_pred.unsqueeze(0), pc_backward.unsqueeze(0))
                    chamfer_dist = (dist1.sum() + dist2.sum()) * 0.5
                    # loss = loss + 0.01 * chamfer_dist
                    # flow_loss += 0.01 * chamfer_dist + 0.01 * pred_flow_backward.abs().mean()
                    flow_loss += chamfer_dist + pred_flow_backward.abs().mean()
                
                # regularize flow on the ground
                # ground = self.pc_ground_list[f"{frame_idx}"]
                # ground = torch.from_numpy(ground).cuda().float().contiguous()
                # zero_flow = self.model.flow(ground, torch.rand(1).to(time_ego))
                # flow_loss += (zero_flow["flow_forward"].abs().sum() + zero_flow["flow_backward"].abs().sum())
            
            # line-of-sight loss for Lidar in Urban Radiance Fields
            if self.opt.use_urf_loss:
                eps = 0.02 * 0.1 ** min(self.global_step / self.opt.iters, 1)
                # gt_depth [B, N]
                weights = outputs_lidar["weights"] # [B*N, T]
                z_vals = outputs_lidar["z_vals"]

                depth_mask = gt_depth.reshape(z_vals.shape[0], 1) > 0.0
                mask_empty = (z_vals < (gt_depth.reshape(z_vals.shape[0], 1) - eps)) | (z_vals > (gt_depth.reshape(z_vals.shape[0], 1) + eps))
                loss_empty = ((mask_empty * weights) ** 2).sum() / depth_mask.sum()
                los_loss += 0.1 * loss_empty

                mask_near = (z_vals > (gt_depth.reshape(z_vals.shape[0], 1) - eps)) & (z_vals < (gt_depth.reshape(z_vals.shape[0], 1) + eps))
                distance = mask_near * (z_vals - gt_depth.reshape(z_vals.shape[0], 1))
                sigma = eps / 3.
                distr = 1.0 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(-(distance ** 2 / (2 * sigma ** 2)))
                distr /= distr.max()
                distr *= mask_near
                loss_near = ((mask_near * weights - distr) ** 2).sum() / depth_mask.sum()
                los_loss += 0.1 * loss_near

            # Structural regularization
            # Getting the patch value (x,y)
            if isinstance(self.opt.patch_size_lidar, int):
                patch_size_h, patch_size_w = self.opt.patch_size_lidar, self.opt.patch_size_lidar
            elif len(self.opt.patch_size_lidar) == 1:
                patch_size_h, patch_size_w = self.opt.patch_size_lidar[0], self.opt.patch_size_lidar[0]
            else:
                patch_size_h, patch_size_w = self.opt.patch_size_lidar
            
            # Apply structural regularization
            if patch_size_h > 1:
                pred_depth_sr = (pred_depth.view(-1, patch_size_h, patch_size_w, 1) # [B, 1, patch_size_h, patch_size_w]
                                .permute(0, 3, 1, 2)
                                .contiguous() 
                                / self.opt.scale)
                #number of patches
                num_patch = pred_depth_sr.shape[0]

                #Compute x and y gradients for predicted depth map
                #using sobel filter [num_patch, 1, patch_size_h, patch_size_w]
                if self.opt.sobel_grad:
                    pred_grad_x = F.conv2d(pred_depth_sr,
                                        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                                        .unsqueeze(0)
                                        .unsqueeze(0)
                                        .to(self.device),
                                        padding=1)#[..., :-1]
                    pred_grad_y = F.conv2d(pred_depth_sr,
                                        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                                        .unsqueeze(0)
                                        .unsqueeze(0)
                                        .to(self.device),
                                        padding=1)#[..., :-1, :]
                #manual [num_patch, 1, patch_size_h, patch_size_w] 
                else:
                    pred_grad_x = pred_depth_sr[:, :, :, :-1] - pred_depth_sr[:, :, :, 1:]
                    pred_grad_x = torch.cat((pred_grad_x, pred_grad_x[:,:,:,-1:]), dim=3) #padding
                    pred_grad_y = pred_depth_sr[:, :, :-1, :] - pred_depth_sr[:, :, 1:, :]
                    pred_grad_y = torch.cat((pred_grad_y, pred_grad_y[:,:,-1:,:]), dim=2) #padding
                
                # Edge-aware losses
                if self.opt.grad_norm_smooth:
                    grad_norm = torch.exp(-pred_grad_x.abs()) + torch.exp(-pred_grad_y.abs()) #[num_patch,1,patch_size_h, patch_size_w]
                    # print('grad_norm:', grad_norm)
                    loss_sr = loss_sr + self.opt.alpha_grad_norm * grad_norm 

                if self.opt.spatial_smooth:
                    spatial_loss = pred_grad_x**2 + pred_grad_y**2 #[num_patch,1,patch_size_h, patch_size_w]
                    # print('spatial_loss:', spatial_loss)
                    loss_sr = loss_sr + self.opt.alpha_spatial * spatial_loss 

                if self.opt.tv_loss:
                    tv_loss = pred_grad_x.abs() + pred_grad_y.abs() #[num_patch,1,patch_size_h, patch_size_w]
                    # print('tv_loss:', tv_loss)
                    loss_sr = loss_sr + self.opt.alpha_tv * tv_loss
                
                #Calc for gradient based loss (need depth grad for pred and gt)
                if self.opt.grad_loss:
                    #initialize gradient losses
                    grad_loss_x = 0
                    grad_loss_y = 0
                    gt_depth_sr = (gt_depth.view(-1, patch_size_h, patch_size_w, 1)
                                .permute(0, 3, 1, 2)
                                .contiguous()
                                /self.opt.scale)
                    gt_raydrop_sr = (gt_raydrop.view(-1, patch_size_h, patch_size_w, 1)
                                .permute(0, 3, 1, 2)
                                .contiguous())

                    #compute x and y gradients of ground truth depth
                    #using sobel filter [num_patch, 1, patch_size_h, patch_size_w]
                    if self.opt.sobel_grad: 
                        gt_grad_y = F.conv2d(gt_depth_sr,
                                            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                                            .unsqueeze(0)
                                            .unsqueeze(0)
                                            .to(self.device),
                                            padding=1)

                        gt_grad_x = F.conv2d(gt_depth_sr,
                                            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                                            .unsqueeze(0)
                                            .unsqueeze(0)
                                            .to(self.device),
                                            padding=1)
                    #manual [num_patch, 1, patch_size_h, patch_size_w]
                    else:                    
                        gt_grad_x = gt_depth_sr[:, :, :, :-1] - gt_depth_sr[:, :, :, 1:] #[num_patch, 1, patch_size_h, patch_size_w-1]
                        gt_grad_x = torch.cat((gt_grad_x, gt_grad_x[:,:,:,-1:]), dim=3) #padding
                        gt_grad_y = gt_depth_sr[:, :, :-1, :] - gt_depth_sr[:, :, 1:, :] #[num_patch, 1, patch_size_h-1, patch_size_w]
                        gt_grad_y = torch.cat((gt_grad_y, gt_grad_y[:,:,-1:,:]), dim=2) #padding
                    
                    # get 2d pixel coordinates of patches
                    patch_pxls_h = ((data['rays_pano_inds'] // data['W_lidar']) #[H]
                                    .reshape(-1, patch_size_h, patch_size_w, 1)
                                    .permute(0, 3, 1, 2)
                                    .contiguous()) 
                    patch_pxls_w = ((data['rays_pano_inds'] % data['W_lidar']) #[W]
                                    .reshape(-1, patch_size_h, patch_size_w, 1)
                                    .permute(0, 3, 1, 2)
                                    .contiguous())
                    
                    #Compute gradients of gt depth [H_lidar, W_lidar]
                    #single gradients
                    gt_pano_grad_x = (data['pano_frame'][0, ..., 2][:,:-1]    
                                            - data['pano_frame'][0, ..., 2][:,1:])/self.opt.scale
                    gt_pano_grad_x = torch.cat((gt_pano_grad_x, gt_pano_grad_x[:,-1:]), dim=1) #padding
                    gt_pano_grad_y = (data['pano_frame'][0, ..., 2][:-1,:]
                                            - data['pano_frame'][0, ..., 2][1:,:])/self.opt.scale
                    gt_pano_grad_y = torch.cat((gt_pano_grad_y, gt_pano_grad_y[-1:,:]), dim=0) #padding
                    #double gradients  [H_lidar, W_lidar]
                    gt_pano_grad_xx = gt_pano_grad_x[:, :-1].abs() - gt_pano_grad_x[:, 1:].abs()
                    gt_pano_grad_xx = torch.cat((gt_pano_grad_xx, gt_pano_grad_xx[:, -1:]), dim=1) #padding
                    gt_pano_grad_yy = gt_pano_grad_y[:-1, :].abs() - gt_pano_grad_y[1:, :].abs()
                    gt_pano_grad_yy = torch.cat((gt_pano_grad_yy, gt_pano_grad_yy[-1:, :]), dim=0) #padding
                    
                    #gradient threshold value
                    # patch_pxls_clip_x = 0.05 #1cm
                    # patch_pxls_clip_y = 0.05 #1cm
                    
                    #compute gradients masks [num_patch, 1, patch_size_h, patch_size_w]
                    #1 using gradients clip from pano image
                    # grad_mask_x = torch.where(gt_grad_x.abs() < patch_pxls_clip_x, 1, 0)
                    # grad_mask_y = torch.where(gt_grad_y.abs() < patch_pxls_clip_y, 1, 0)
                    
                    #2 using gradients clip
                    #gather double grads values of gt_pano for patches using patch pixel indices
                    patch_pxls_grad_xx = torch.gather(gt_pano_grad_xx.expand(num_patch, 1, -1, -1),  #[num_patch, 1, patch_size_h, W_lidar]
                                                    2, 
                                                    patch_pxls_h[:,:,:,:1].repeat(1,1,1,gt_pano_grad_xx.shape[1]))
                    patch_pxls_grad_xx = torch.gather(patch_pxls_grad_xx, 3, patch_pxls_w) #[num_patch, 1, patch_size_h, patch_size_w]

                    patch_pxls_grad_yy = torch.gather(gt_pano_grad_yy.expand(num_patch, 1, -1, -1),  ##[num_patch, 1, H_lidar, patch_size_w]
                                                    3, 
                                                    patch_pxls_w[:,:,:1,:].repeat(1,1,gt_pano_grad_xx.shape[0],1))
                    patch_pxls_grad_yy = torch.gather(patch_pxls_grad_yy, 2, patch_pxls_h) #[num_patch, 1, patch_size_h, patch_size_w]

                    #compute gradients masks [num_patch, 1, patch_size_h, patch_size_w]
                    grad_mask_x = torch.where(patch_pxls_grad_xx.abs() < 0.05, 1, 0) #TODO: hard coded, develop logic
                    grad_mask_y = torch.where(patch_pxls_grad_yy.abs() < 0.05, 1, 0)

                    #Compute final mask using gt raydrop and gradient mask
                    mask_dx = gt_raydrop_sr * grad_mask_x
                    mask_dy = gt_raydrop_sr * grad_mask_y
                    
                    #Compute gradient loss between pred and gt depth gradients
                    if self.opt.depth_grad_loss == "cos":
                        grad_loss_x = self.criterion["grad"]((pred_grad_x * mask_dx).reshape(num_patch, -1), #[num_patch, patch_h*(patch_w-1)]
                                                        (gt_grad_x * mask_dx).reshape(num_patch, -1))
                        grad_loss_y = self.criterion["grad"]((pred_grad_y * mask_dy).reshape(num_patch, -1), #[num_patch, (patch_h-1)*patch_w]
                                                        (gt_grad_y * mask_dy).reshape(num_patch, -1))                    
                        grad_loss_x = ((1 - grad_loss_x)
                                    .reshape(num_patch, 1, 1, 1)
                                    .expand(num_patch, 1, patch_size_h, patch_size_w)) #[num_patch, 1, patch_h, patch_w-1]
                        grad_loss_y = ((1 - grad_loss_y)
                                    .reshape(num_patch, 1, 1, 1)
                                    .expand(num_patch, 1, patch_size_h, patch_size_w)) #[num_patch, 1, patch_h-1, patch_w]
                    else:
                        grad_loss_x = self.criterion["grad"](pred_grad_x*mask_dx, gt_grad_x*mask_dx) #grad_x loss
                        grad_loss_y = self.criterion["grad"](pred_grad_y*mask_dy, gt_grad_y*mask_dy) #grad_y loss
                    
                    # total grad loss
                    grad_loss = self.opt.alpha_grad * (grad_loss_x + grad_loss_y) #[num_patch, 1, patch_h, patch_w]
                    
                    #total sr loss
                    # loss_sr = (loss_sr + grad_loss).reshape(lidar_loss.shape) # [B, N]
                    loss_sr = loss_sr + grad_loss.sum() # [B, N]
        
        
        # Camera
        rgb_loss = 0
        rgb_depth_loss = 0
        if self.opt.enable_rgb:
            rays_o = data['rays_o']  # [B, N, 3]
            rays_d = data['rays_d']  # [B, N, 3]
            images = data['images']  # [B, N, 3/4]

            B, N, C = images.shape

            if self.opt.color_space == 'linear':
                images[..., :3] = utils.srgb_to_linear(images[..., :3])

            if C == 3 or self.model.bg_radius > 0:
                bg_color = 1
            # train with random background color if not using a bg model and has alpha channel.
            else:
                bg_color = torch.rand_like(images[..., :3])  # [N, 3], pixel-wise random.

            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                    1 - images[..., 3:])
            else:
                gt_rgb = images

            #Render camera model
            outputs = self.model.render(
                rays_o,
                rays_d,
                time_ego,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                force_all_rays=False if self.opt.patch_size == 1 else True,
                **vars(self.opt))
            
            #colour supervision
            pred_rgb = outputs['image']
            rgb_loss = self.opt.alpha_rgb * self.criterion['rgb'](pred_rgb, gt_rgb) #[B, N, 3]

            #depth supervision
            if self.opt.use_rgbd_loss and 'image_depths' in data:
                image_gt_depths = data['image_depths'].squeeze(-1)
                image_pred_depth = outputs['depth']

                max_depth = 80 * self.opt.scale
                image_gt_depths *= self.opt.scale
                image_gt_depths[image_gt_depths > max_depth] = max_depth
                image_pred_depth[image_pred_depth > max_depth] = max_depth
                mask = image_gt_depths > 0
                rgb_depth_loss = self.criterion['rgb_depth'](image_pred_depth * mask, 
                                                             image_gt_depths * mask)
                # rgb_depth_loss = self.opt.alpha_rd * rgb_depth_loss.mean()
                rgb_depth_loss = self.opt.alpha_rd * rgb_depth_loss
        
        #Other losses
        if 'ref_gt_points' in data.keys():
            ref_gt_points = data['ref_gt_points']
            distill_loss = self.model.distill(ref_gt_points)
            loss = loss + distill_loss.mean(0)

        if self.opt.enable_rgb and 'distill_loss' in outputs:
            # print(outputs['distill_loss'])
            loss = loss + outputs['distill_loss']

        if self.opt.enable_lidar and 'distill_loss' in outputs_lidar:
            # print(outputs['distill_loss'])
            loss = loss + outputs_lidar['distill_loss']

        if self.opt.enable_rgb and 'eikonal_loss' in outputs:
            loss = loss + outputs['eikonal_loss'] * self.opt.eikonal_loss_weight

        if self.opt.enable_lidar and 'eikonal_loss' in outputs_lidar:
            loss = loss + outputs_lidar['eikonal_loss'] * self.opt.eikonal_loss_weight
        
        # special case for CCNeRF's rank-residual training
        # if len(loss.shape) == 3:  # [K, B, N]
        #     loss = loss.mean(0)

        #Total training loss
        helper_loss = lambda x: x.sum() if isinstance(x, torch.Tensor) else 0
        loss += helper_loss(lidar_loss) + chamfer_loss + flow_loss + los_loss + \
                loss_sr + helper_loss(rgb_loss) + helper_loss(rgb_depth_loss)
        
        loss[torch.isnan(loss)] = 0.0
        loss[torch.isinf(loss)] = 1e5

        if self.opt.enable_lidar:
            #Update error map for heuristic pixel sampling
            index = data['index'] # [B]
            pixel_1d_inds = data['rays_pano_inds'] # [B, N] 

            #add loss for plotting
            H_i = (pixel_1d_inds // data['W_lidar']).detach().cpu().numpy() #y [N]
            W_i = (pixel_1d_inds % data['W_lidar']).detach().cpu().numpy() #x [N]
            self.pano_sampled[index, 2, H_i, W_i] = lidar_loss.detach().cpu().numpy()

            # update error_map for pixel sampling
            # get pano error map of current training batch
            error_map = self.error_map[index] # [B, H_ * W_], height & width may be different from lidar pano image
            B, error_map_H, error_map_W = error_map.shape

            #get loss of current training pixels
            error = lidar_loss.detach().to(error_map.device) # [B, N]

            #normalize values between [0,1]
            # error[error<0] = error[error<0] - error.min() #handling negative values
            error = (error-error.min())/(error.max()-error.min() + torch.finfo().eps)
            
            #scale the values
            min_range, max_range = 1, 1e3 #range
            error = error*(max_range - min_range) + min_range
            
            #update error map 
            #get error_map pixel inds corresponding to pano inds
            scale_h , scale_w = error_map_H/data['H_lidar'], error_map_W/data['W_lidar']
            err_pxl_H = (pixel_1d_inds // data['W_lidar']*scale_h).long() 
            err_pxl_W = (pixel_1d_inds % data['W_lidar']*scale_w ).long()

            # ema update
            ema_error = 0.1 * error_map[0, err_pxl_H, err_pxl_W] + 0.9 * error #use torch.gather
            # update error map of current training batch
            error_map[0, err_pxl_H, err_pxl_W] = ema_error #use torch.scatter
            # put back error map
            self.error_map[index] = error_map
        
        if self.opt.enable_rgb:
            #Update error map for heuristic pixel sampling
            index = data['index'] # [B]
            pixel_1d_inds = data['rays_rgb_inds'] # [B, N] 

            #add loss for plotting
            H_i = (pixel_1d_inds // data['W']).detach().cpu().numpy() #y [N]
            W_i = (pixel_1d_inds % data['W']).detach().cpu().numpy() #x [N]
            # self.rgb_sampled[index, 2, H_i, W_i] = rgb_loss.detach().cpu().numpy()
            self.rgb_sampled[index, 2, H_i, W_i] = rgb_loss.sum(dim=2).detach().cpu().numpy() # [B, N, 3] -> [B, N]

            # update error_map for pixel sampling
            # get pano error map of current training batch
            error_map = self.error_map_rgb[index] # [B, H_ * W_], height & width may be different from lidar pano image
            B, error_map_H, error_map_W = error_map.shape

            #get loss of current training pixels
            # error = rgb_loss.detach().sum(dim=2).to(error_map.device) # [B, N]
            error = rgb_loss.sum(dim=2).detach().to(error_map.device) # [B, N, 3] -> [B, N]

            #normalize values between [0,1]
            # error[error<0] = error[error<0] - error.min() #handling negative values
            error = (error-error.min())/(error.max()-error.min() + torch.finfo().eps)
            
            #scale the values
            min_range, max_range = 1, 1e3 #range
            error = error*(max_range - min_range) + min_range
            
            #update error map 
            #get error_map pixel inds corresponding to pano inds
            scale_h , scale_w = error_map_H/data['H'], error_map_W/data['W']
            err_pxl_H = (pixel_1d_inds // data['W']*scale_h).long() 
            err_pxl_W = (pixel_1d_inds % data['W']*scale_w ).long()

            # ema update
            ema_error = 0.1 * error_map[0, err_pxl_H, err_pxl_W] + 0.9 * error #use torch.gather
            # update error map of current training batch
            error_map[0, err_pxl_H, err_pxl_W] = ema_error #use torch.scatter
            # put back error map
            self.error_map_rgb[index] = error_map

        #calc loss_sr contribution
        helper_pbar = lambda x: x.sum().item() if isinstance(x, torch.Tensor) else 0
        pbar_train.set_description(
            # f"Loss_d: {loss_d.mean().item() if isinstance(loss_d, torch.Tensor) else 0:.3f} | "
            f"Loss_d: {helper_pbar(loss_d):.3f} | "
            f"Loss_i: {helper_pbar(loss_i):.3f} | "
            f"Loss_rd: {helper_pbar(loss_rd):.3f} | "
            f"Loss_sr: {helper_pbar(loss_sr):.3f} | "
            f"Loss_cd: {helper_pbar(chamfer_loss):.3f} | "
            f"Loss_flow: {helper_pbar(flow_loss):.3f} | "
            f"Loss_rgb: {helper_pbar(rgb_loss):.3f} | "
            f"Loss_rgb_d: {helper_pbar(rgb_depth_loss):.3f} "
        )
        
        return (pred_rgb, #[B, N, 3]
                gt_rgb,
                image_pred_depth,
                image_depths,
                pred_raydrop,
                gt_raydrop,
                pred_intensity, #[B, N, 1]
                gt_intensity,
                pred_depth,
                gt_depth,
                loss)

    def eval_step(self, data):
        """Evaluation of Nerf model for evaluation data."""

        pred_intensity = None
        pred_depth = None
        pred_depth_crop = None
        pred_raydrop = None
        gt_intensity = None
        gt_depth = None
        gt_depth_crop = None
        gt_raydrop = None
        pred_rgb = None
        pred_rgb_depth = None
        gt_image_depth = None
        gt_rgb = None
        loss = 0
        
        time_ego = data['time'] #[B, 1]

        #Lidar
        lidar_loss = 0
        if self.opt.enable_lidar:
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            images_lidar = data["images_lidar"]  # [B, H, W, 3/4] , C=[raydrop, intensity, depth]
            gt_raydrop = images_lidar[:, :, :, 0] # [B, H, W]

            if self.opt.dataloader == "nerf_mvl":
                valid_crop = gt_raydrop != -1
                valid_crop_idx = torch.nonzero(valid_crop)
                crop_h, crop_w = (max(valid_crop_idx[:, 1]) - min(valid_crop_idx[:, 1]) + 1,
                                  max(valid_crop_idx[:, 2]) - min(valid_crop_idx[:, 2]) + 1,)

                valid_mask = torch.where(gt_raydrop == -1, 0, 1)
                gt_raydrop = gt_raydrop * valid_mask
            
            #drop intensity and depth using raydrop mask [B, H, W]
            gt_intensity = images_lidar[:, :, :, 1] * gt_raydrop
            gt_depth = images_lidar[:, :, :, 2] * gt_raydrop
            B_lidar, H_lidar, W_lidar, C_lidar = images_lidar.shape

            #prediction from network, dict{'image_lidar', 'depth_lidar'}
            outputs_lidar = self.model.render(rays_o_lidar, 
                                              rays_d_lidar,
                                              time_ego,
                                              cal_lidar_color=True,
                                              staged=True,
                                              perturb=False,
                                              **vars(self.opt))
            
            # Reshape as per like lidar pano image [B, H, W, 2], C=[raydrop, intensity]
            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(
                B_lidar, H_lidar, W_lidar, 2
            )

            # Get prediction raydrop, intensity and depth [B, H, W]
            pred_raydrop = pred_rgb_lidar[:, :, :, 0] #raydrop
            pred_intensity = pred_rgb_lidar[:, :, :, 1] # intensity
            pred_depth = outputs_lidar["depth_lidar"].reshape(B_lidar, H_lidar, W_lidar) # depth
            
            #update raydrop
            if self.opt.raydrop_loss == 'bce':
                pred_raydrop = F.sigmoid(pred_raydrop)
            if self.use_refine:
                pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
                pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
            
            #compute raydrop mask
            raydrop_mask = torch.where(pred_raydrop > self.opt.raydrop_thres, 1, 0)            
            if self.opt.dataloader == "nerf_mvl":
                raydrop_mask = raydrop_mask * valid_mask            
            
            # update intensity and depth using raydrop mask [B, H, W] 
            # if self.opt.alpha_r > 0 and (not torch.all(raydrop_mask == 0)):
            pred_intensity = pred_intensity * raydrop_mask
            pred_depth = pred_depth * raydrop_mask
            
            # Calculate total loss
            lidar_loss = (
                self.opt.alpha_d * self.criterion["depth"](pred_depth, gt_depth).mean()
                + self.opt.alpha_r * self.criterion["raydrop"](pred_raydrop, gt_raydrop).mean()
                + self.opt.alpha_i * self.criterion["intensity"](pred_intensity, gt_intensity).mean()
            )

            if self.opt.dataloader == "nerf_mvl":
                pred_intensity = pred_intensity[valid_crop].reshape(B_lidar, crop_h, crop_w)
                gt_intensity = gt_intensity[valid_crop].reshape(B_lidar, crop_h, crop_w)
                pred_depth_crop = pred_depth[valid_crop].reshape(B_lidar, crop_h, crop_w)
                gt_depth_crop = gt_depth[valid_crop].reshape(B_lidar, crop_h, crop_w)

            #Expand dimension [B, H, W] --> [B, H, W, 1] 
            pred_intensity = pred_intensity.unsqueeze(-1)
            pred_raydrop = pred_raydrop.unsqueeze(-1)
            gt_intensity = gt_intensity.unsqueeze(-1)
            gt_raydrop = gt_raydrop.unsqueeze(-1)

        # Camera
        rgb_loss = 0
        if self.opt.enable_rgb:
            rays_o = data['rays_o']  # [B, N, 3]
            rays_d = data['rays_d']  # [B, N, 3]
            images = data['images']  # [B, H, W, 3/4]

            if 'image_depths' in data:
                gt_image_depth = data['image_depths'].squeeze(-1)

            B, H, W, C = images.shape
            masks = None
            if 'masks' in data:
                masks = data['masks'].reshape(B, H, W, 1)

            if self.opt.color_space == 'linear':
                images[..., :3] = utils.srgb_to_linear(images[..., :3])

            # eval with fixed background color
            bg_color = 1

            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                    1 - images[..., 3:])
            else:
                gt_rgb = images

            outputs = self.model.render(rays_o,
                                        rays_d,
                                        time_ego,
                                        staged=True,
                                        bg_color=bg_color,
                                        perturb=False,
                                        **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3)
            pred_rgb_depth = outputs['depth'].reshape(B, H, W)

            if masks is not None:
                gt_rgb = gt_rgb * masks
                pred_rgb = pred_rgb * masks

            rgb_loss = self.opt.alpha_rgb * self.criterion['rgb'](
                pred_rgb, gt_rgb).mean()
        
        loss = lidar_loss + rgb_loss

        return (
            pred_rgb,
            pred_rgb_depth,
            pred_intensity, #[B, H, W, 1] 
            pred_depth,
            pred_depth_crop,
            pred_raydrop,
            gt_rgb,
            gt_image_depth,
            gt_intensity,
            gt_depth,
            gt_depth_crop,
            gt_raydrop,
            loss,
        )

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):
        """Get predictions from trained model. """

        pred_raydrop = None
        pred_intensity = None
        pred_depth = None

        time_ego = data['time'] #[B, 1]
        
        #Lidar
        if self.opt.enable_lidar:
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            H_lidar, W_lidar = data["H_lidar"], data["W_lidar"]
            
            masks_lidar = None
            if 'masks_lidar' in data:
                masks_lidar = data['masks_lidar'].reshape(-1, H_lidar, W_lidar)

            # Get prediction lidar_image and range from model using ray origin and direction
            outputs_lidar = self.model.render(rays_o_lidar,
                                              rays_d_lidar,
                                              time_ego,
                                              cal_lidar_color=True,
                                              staged=True,
                                              perturb=perturb,
                                              **vars(self.opt))
            # Reshape raydrop and intensity  [B, H, W, 2]
            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(
                -1, H_lidar, W_lidar, 2
            )
            # Get prediction raydrop, intensity and depth [B, H, W]
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]            
            pred_intensity = pred_rgb_lidar[:, :, :, 1]
            pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)
            
            #update raydrop
            if self.opt.raydrop_loss == 'bce':
                pred_raydrop = F.sigmoid(pred_raydrop)
            if self.use_refine:
                pred_raydrop = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
                pred_raydrop = self.model.unet(pred_raydrop).squeeze(0)
            
            #compute raydrop mask
            raydrop_mask = torch.where(pred_raydrop > self.opt.raydrop_thres, 1, 0)   

            #apply raydrop mask on predictions 
            if self.opt.alpha_r > 0:
                pred_intensity = pred_intensity * raydrop_mask
                pred_depth = pred_depth * raydrop_mask
            if masks_lidar is not None:
                pred_depth = pred_depth * masks_lidar
                pred_raydrop = pred_raydrop * masks_lidar
                pred_intensity = pred_intensity * masks_lidar

        # RGB
        pred_rgb = None
        pred_rgb_depth = None
        if self.opt.enable_rgb:
            rays_o = data['rays_o']  # [B, N, 3]
            rays_d = data['rays_d']  # [B, N, 3]
            H, W = data['H'], data['W']
            B = rays_o.shape[0]
            masks = None
            if 'masks' in data:
                masks = data['masks'].reshape(B, H, W, 1)

            if bg_color is not None:
                bg_color = bg_color.to(self.device)

            outputs = self.model.render(rays_o,
                                        rays_d,
                                        time_ego,
                                        staged=True,
                                        bg_color=bg_color,
                                        perturb=perturb,
                                        **vars(self.opt))
            pred_rgb = outputs['image'].reshape(-1, H, W, 3)
            pred_rgb_depth = outputs['depth'].reshape(-1, H, W)
            if masks is not None:
                pred_rgb = pred_rgb * masks

        return (pred_rgb, 
                pred_rgb_depth, 
                pred_raydrop, 
                pred_intensity, 
                pred_depth)

    def refine(self, loader):
        """Refine raydrop predictions using Unet"""

        #Change train dataloader to validation mode. Other way is to use separate dataloader.
        #But it will have high memory usage. So, we can reuse the dataloader.
        loader._data.training = False
        loader._data.num_rays_lidar = -1
        loader._data.num_rays = -1
        if self.ema is not None:
            self.ema.copy_to() # load ema model weights
            self.ema = None    # no need for final model weights

        self.model.eval()

        unet_input_list = []
        gt_list = []

        self.log("[INFO] Preparing for Raydrop Refinement ...")
        for i, data in tqdm.tqdm(iterable=enumerate(loader), total=len(loader)):
            rays_o_lidar = data["rays_o_lidar"]  # [B, N, 3]
            rays_d_lidar = data["rays_d_lidar"]  # [B, N, 3]
            time_ego = data['time']
            B_lidar, H_lidar, W_lidar, C_lidar = data["images_lidar"].shape
            gt_list.append(data["images_lidar"])
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                with torch.no_grad():
                    outputs_lidar = self.model.render(
                        rays_o_lidar,
                        rays_d_lidar,
                        time_ego,
                        cal_lidar_color=True,
                        staged=True,
                        perturb=False,
                        **vars(self.opt),
                    )

            pred_rgb_lidar = outputs_lidar["image_lidar"].reshape(B_lidar, H_lidar, W_lidar, 2)
            pred_raydrop = pred_rgb_lidar[:, :, :, 0]
            pred_intensity = pred_rgb_lidar[:, :, :, 1]
            pred_depth = outputs_lidar["depth_lidar"].reshape(-1, H_lidar, W_lidar)

            unet_input = torch.cat([pred_raydrop, pred_intensity, pred_depth], dim=0).unsqueeze(0)
            unet_input_list.append(unet_input)

            # if i % 10 == 0:
            #     print(f"{i+1}/{len(loader)}")

        unet_input = torch.cat(unet_input_list, dim=0).cuda().float().contiguous()   # [B, 3, H, W]
        gt_lidar = torch.cat(gt_list, dim=0).cuda().float().permute(0,3,1,2).contiguous()  # [B, 3, H, W] (C=raydrop, intensity, depth)
        raydrop_gt = gt_lidar[:, 0:1, :, :]   # [B, 1, H, W]
        
        self.model.unet.train()

        loss_total = []

        refine_bs = None # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
        refine_epoch = 1000

        optimizer = torch.optim.Adam(self.model.unet.parameters(), lr=0.001, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_epoch)

        self.log("[INFO] Start UNet Optimization ...")
        for i in range(refine_epoch):
            optimizer.zero_grad()

            if refine_bs is not None:
                idx = np.random.choice(unet_input.shape[0], refine_bs, replace=False)
                input = unet_input[idx,...]
                gt = raydrop_gt[idx,...]                
            else:
                input = unet_input
                gt = raydrop_gt
            
            # random mask for dropping our regions in input data like data augmentation
            # forcing the network to learn from incomplete data and potentially improve generalization
            mask = torch.ones_like(input).to(input.device)
            box_num_max = 32
            box_size_y_max = int(0.1 * input.shape[2])
            box_size_x_max = int(0.1 * input.shape[3])
            for j in range(np.random.randint(box_num_max)):
                box_size_y = np.random.randint(1, box_size_y_max)
                box_size_x = np.random.randint(1, box_size_x_max)
                yi = np.random.randint(input.shape[2]-box_size_y)
                xi = np.random.randint(input.shape[3]-box_size_x)
                mask[:, :, yi:yi+box_size_y, xi:xi+box_size_x] = 0.
            input = input * mask
            
            #model prediction
            raydrop_refine = self.model.unet(input)
            
            #calculate loss
            bce_loss = self.bce_fn(raydrop_refine, gt)
            loss = bce_loss

            loss.backward()

            loss_total.append(loss.item())
    
            if i % 50 == 0:
                log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.log(f"[{log_time}] iter:{i}, lr:{optimizer.param_groups[0]['lr']:.6f}, raydrop loss:{loss.item()}")

            optimizer.step()
            scheduler.step()

        #Save model checkpoint
        file_path = f"{self.name}_ep{self.epoch:04d}_refine"
        self.save_checkpoint(name= file_path, full=False, best=False, remove_old=False)

        #Revert dataloader to training mode
        loader._data.training = True
        loader._data.num_rays_lidar = self.opt.num_rays_lidar
        loader._data.num_rays = self.opt.num_rays

    def train(self, train_loader, valid_loader, max_epochs):
        """Training , validate and save chekpoints"""

        if self.use_tensorboardX and self.local_rank == 0:
            if utils.is_ali_cluster() and self.opt.cluster_summary_path is not None:
                summary_path = self.opt.cluster_summary_path
            else:
                # summary_path = os.path.join(self.workspace, "run", self.name)
                summary_path = os.path.join(self.workspace, "run")
            self.writer = tensorboardX.SummaryWriter(summary_path)
        
        #Process point cloud by remove background points
        if self.opt.flow_loss:
            self.process_pointcloud(train_loader)
        
        #Flag for changing patch size of structural regularization
        if self.opt.change_patch_size_lidar[0] > 1:
            change_dataloader = True
        else:
            change_dataloader = False
        
        #Initialize pixel sampler
        self.pixel_sampler = 'random'
        
        #Get a ref to loader data
        self.error_map = train_loader._data.error_map
        self.pano_sampled = train_loader._data.pano_sampled
        self.error_map_rgb = train_loader._data.error_map_rgb
        self.rgb_sampled = train_loader._data.rgb_sampled
        
        # Training and validation epochs
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            #change patch size
            if change_dataloader:
                if self.epoch % self.opt.change_patch_size_epoch == 0:
                    self.pixel_sampler = 'patch' 
                    train_loader._data.patch_size_lidar = self.opt.change_patch_size_lidar #pixel sampling
                    self.opt.patch_size_lidar = self.opt.change_patch_size_lidar #structural regularization loss
                    train_loader._data.patch_size = self.opt.patch_size
                else: 
                    self.pixel_sampler = 'random'
                    train_loader._data.patch_size_lidar = 1
                    self.opt.patch_size_lidar = 1
                    train_loader._data.patch_size = 1

            if self.pixel_sampler == 'random':
                train_loader._data.use_error_map = False
            else:
                train_loader._data.use_error_map = self.opt.use_error_map
            
            self.train_one_epoch(train_loader)        

            if self.epoch % self.eval_interval == 0:
                #save checkpoints
                self.save_checkpoint(full=True, best=False)

                #plot training error plots
                index = len(train_loader)//2
                if self.opt.enable_lidar: 
                    save_path = os.path.join(self.workspace,"validation",f"{self.name}_train_subplots_lidar_{self.epoch}_{index}.png")
                    utils.vis_training(index, self.pano_sampled, save_path, False, 'Lidar image sampling')
                    self.log(f"[INFO] Plot saved at: {save_path}")
                if self.opt.enable_rgb: 
                    save_path = os.path.join(self.workspace,"validation",f"{self.name}_train_subplots_camera_{self.epoch}_{index}.png")
                    utils.vis_training(index, self.rgb_sampled, save_path, False, 'Camera image sampling')
                    self.log(f"[INFO] Plot saved at: {save_path}")

                #evaluation epoch
                self.use_refine = False
                self.evaluate_one_epoch(valid_loader)
        
        #Refine raydrop using UNET
        self.refine(train_loader)
        self.log("[INFO] Evaluating after raydrop refinement")
        self.use_refine = True
        self.evaluate_one_epoch(valid_loader, name= f"{self.name}_ep{self.epoch:04d}_refined")
        
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None, use_refine=True):
        """validate the results of the evaluation dataset"""

        self.use_refine = use_refine
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, use_refine=True):
        """testing the trained model"""
        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, saving results to '{save_path}'")
    
        self.use_refine = use_refine

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval() #change model mode to evaluation

        # For range and intensity pano video
        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_rgb = []
            all_preds_rgb_depth = []
        

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    # Get prediction
                    (preds_rgb,  # [B, H, W, C]
                     preds_rgb_depth,  # [B, H, W]
                     preds_raydrop,  # [B, H, W]
                     preds_intensity, 
                     preds_depth) = self.test_step(data)

                # RGB
                if self.opt.enable_rgb:
                    if self.opt.color_space == 'linear':
                        preds_rgb = utils.linear_to_srgb(preds_rgb)
                    
                    #convert tensor to numpy
                    pred_rgb = preds_rgb[0].detach().cpu().numpy()
                    pred_rgb_depth = preds_rgb_depth[0].detach().cpu().numpy()
                    
                    #convert to image
                    img_color_rgb_pred = (pred_rgb * 255).astype(np.uint8)
                    img_depth_rgb_pred = (pred_rgb_depth * 255).astype(np.uint8)

                    #save depth prediction in npy file
                    # np.save(os.path.join(save_path, f'test_{name}_{i:04d}_depth_rgb.npy'),
                    #         pred_rgb_depth)                    

                    if write_video:
                        all_preds_rgb.append(img_color_rgb_pred)
                        all_preds_rgb_depth.append(img_depth_rgb_pred)
                    else:
                        # cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(
                            os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                            cv2.cvtColor(img_color_rgb_pred, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(
                            os.path.join(save_path, f'{name}_{i:04d}_rgb_depth.png'),
                            # cv2.applyColorMap(img_depth_rgb_pred, 20))  #colormap options: 20, 9
                            img_depth_rgb_pred)
                
                #Lidar
                if self.opt.enable_lidar:
                    
                    # Convert tensor to numpy
                    pred_raydrop = preds_raydrop[0].detach().cpu().numpy()
                    pred_intensity = preds_intensity[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()

                    # Calculate ray drop mask using prediction [H, W]
                    pred_raydrop_mask = np.where(pred_raydrop > self.opt.raydrop_thres, 1.0, 0.0)

                    #Create images [H, W, 1] 
                    #raydrop mask is already applied during test_step on intensity and depth predictions
                    img_raydrop_pred = (pred_raydrop * 255).astype(np.uint8)
                    img_intensity_pred = (pred_intensity * 255).astype(np.uint8)
                    img_depth_pred = (pred_depth * 255).astype(np.uint8)
                    # pred_depth = (pred_depth / self.opt.scale).astype(np.uint8)
                    img_raydrop_masked_pred = (pred_raydrop_mask * 255).astype(np.uint8)
                    
                    # Get pcd with intensity in world frame
                    pred_pcd_world = utils.get_pcd_bound_to_world(pred_depth, img_intensity_pred, loader, data)
                    
                    # Get pcd with intensity in lidar frame
                    #1 Get pcd from pred depth and intensities in lidar_new frame + bound frame scale
                    pred_pcd_lidar = convert.pano_to_lidar_with_intensities(pred_depth, 
                                                                            img_intensity_pred, 
                                                                            loader._data.intrinsics_lidar,
                                                                            loader._data.intrinsics_hoz_lidar,)
                    #2 Transform pcd from bound frame to lidar frame (rescale)
                    pred_pcd_lidar[:,:3] = pred_pcd_lidar[:,:3]/loader._data.scale 
                    # pred_pcd_lidar = pred_pcd_lidar[pred_pcd_lidar[:,2] <= 1.0] # clip pcd using Z range

                    # # Save pcd in world frame [x,y,z,i]
                    np.savetxt(os.path.join(save_path, f"test_{name}_{i:04d}_pcd_world.txt"),
                               pred_pcd_world, delimiter=' ', fmt="%f")
                    
                    # Save pcd in lidar frame [x,y,z,i]
                    np.savetxt(
                        os.path.join(save_path, f"test_{name}_{i:04d}_pcd_lidar.txt"),
                        pred_pcd_lidar, delimiter=' ', fmt="%f"
                    )

                    # Save pcd in lidar frame of nuScene format and in .bin files [x,y,z,i,i]
                    # pred_pcd_nus = np.column_stack((pred_pcd_lidar, pred_pcd_lidar[:,-1])) # [n,5]
                    # pred_pcd_nus.tofile(os.path.join(save_path, f"test_{name}_{i:04d}_pcd_lidar_ns.bin"), 
                    #                     sep='', 
                    #                     format='%f')
                    
                    # Save pcd in pcd format
                    _ = tools.convert_to_o3dpcd( 
                        pred_pcd_lidar, 
                        os.path.join(save_path, f"test_{name}_{i:04d}_pcd_lidar.pcd"), 
                        ) 

                    
                    # Save predicted lidar pano image in numpy file
                    pred_lidar = convert.pano_to_lidar(pred_depth / self.opt.scale, 
                                                       loader._data.intrinsics_lidar,
                                                       loader._data.intrinsics_hoz_lidar)
                    
                    if self.opt.dataloader == "nerf_mvl":
                        pred_lidar = utils.filter_bbox_dataset(pred_lidar, data["OBB_local"][:, :3])

                    # np.save(os.path.join(save_path, f"test_{name}_{i:04d}_depth_lidar.npy"),
                    #         pred_lidar)

                    #Apply colour maps to images
                    img_raydrop_masked_pred = cv2.cvtColor(img_raydrop_masked_pred, cv2.COLOR_GRAY2BGR)
                    img_intensity_pred = cv2.applyColorMap(img_intensity_pred, 1)
                    img_depth_pred = cv2.applyColorMap(img_depth_pred, 20) #colormap options: 20, 9

                    #Save images
                    if write_video:
                        all_preds.append(img_intensity_pred)
                        all_preds_depth.append(img_depth_pred) 
                    else:                       
                        img_pred = cv2.vconcat([img_raydrop_masked_pred, img_intensity_pred, img_depth_pred])
                        cv2.imwrite(os.path.join(save_path, f"test_{name}_{i:04d}.png"), img_pred)

                pbar.update(loader.batch_size)

        if write_video:
            #Lidar
            if self.opt.enable_lidar:
                all_preds = np.stack(all_preds, axis=0)
                all_preds_depth = np.stack(all_preds_depth, axis=0)
                imageio.mimwrite(os.path.join(save_path, f"{name}_lidar_rgb.mp4"),
                                 all_preds, fps=25, quality=8, macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, f"{name}_depth.mp4"),
                                 all_preds_depth, fps=25, quality=8, macro_block_size=1,)
            # RGB
            if self.opt.enable_rgb:
                all_preds_rgb = np.stack(all_preds_rgb, axis=0)
                all_preds_rgb_depth = np.stack(all_preds_rgb_depth, axis=0)
                imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'),
                                    all_preds_rgb,
                                    fps=25,
                                    quality=8,
                                    macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path,
                                                f'{name}_rgb_depth.mp4'),
                                    all_preds_rgb_depth,
                                    fps=25,
                                    quality=8,
                                    macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        self.log(f"Pixel sampler: '{self.pixel_sampler}'")
        
        # torch.cuda.empty_cache()
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
            for metric in self.depth_metrics.values():
                metric.clear()

        self.model.train() #change model mode to training

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            
            pbar_train = tqdm.tqdm( bar_format="{desc}")
        
        # Training loop
        self.local_step = 0
        for data in loader:
            self.local_step += 1
            self.global_step += 1
            
            # Perform forward pass
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                (pred_rgb, #[B, N, 3]
                 gt_rgb,
                 pred_depth_rgb, 
                 gt_depth_rgb,
                 pred_raydrop, # [B, N, 1]
                 gt_raydrop,
                 pred_intensity,
                 gt_intensity,
                 pred_depth,
                 gt_depth,
                 loss) = self.train_step(data, pbar_train) 
            
            # Perform backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update learning rate
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    #camera
                    if self.opt.enable_rgb:
                        for metric, func in self.metrics.items():
                            if metric in ('rmse') and gt_depth_rgb is not None:
                                func.update(pred_depth_rgb / self.opt.scale, gt_depth_rgb)
                            elif metric in ('pnsr', 'lpips', 'ssim'):
                                func.update(pred_rgb, gt_rgb)
                    #lidar
                    if self.opt.enable_lidar:
                        for metric, func in self.depth_metrics.items():
                            if metric in ('point', 'depth'):
                                func.update(pred_depth, gt_depth) # for depth/range and pcd
                            elif metric == 'intensity':  #for intensity error)
                                func.update(pred_intensity[...,0], gt_intensity[...,0])
                            elif metric == 'raydrop': #raydrop
                                func.update(pred_raydrop, gt_raydrop)
                            else:
                                raise ValueError(f"Unknown metric: {metric}")

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr",
                                           self.optimizer.param_groups[0]["lr"],
                                           self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)
            
            #Update pano_sampled data 
            #Lidar
            if self.opt.enable_lidar:
                # Get batch index for accessing data for corresponding frame
                index = data['index'][0]
                pano_pxl_1d = data['rays_pano_inds'][0].detach().cpu().numpy()

                #convert pano pixels from 1d to 2d image
                H_i = pano_pxl_1d // loader._data.W_lidar #y [N]
                W_i = pano_pxl_1d % loader._data.W_lidar #x [N]
                
                #Fill pixel sampled for the epoch
                self.pano_sampled[index, 0, H_i, W_i] += 1 #increase the counter
                
                #Fill pixel sampled this training step
                self.pano_sampled[index, 1] = 0 #reset all pixels
                self.pano_sampled[index, 1, H_i, W_i] = 1 #fill sampled pixels
            
            #Camera 
            if self.opt.enable_rgb:
                # Get batch index for accessing data for corresponding frame
                index = data['index'][0]
                pano_pxl_1d = data['rays_rgb_inds'][0].detach().cpu().numpy()

                #convert pano pixels from 1d to 2d image
                H_i = pano_pxl_1d // loader._data.W #y [N]
                W_i = pano_pxl_1d % loader._data.W #x [N]
                
                #Fill pixel sampled for the epoch
                self.rgb_sampled[index, 0, H_i, W_i] += 1 #increase the counter
                
                #Fill pixel sampled this training step
                self.rgb_sampled[index, 1] = 0 #reset all pixels
                self.rgb_sampled[index, 1, H_i, W_i] = 1 #fill sampled pixels
        
            #visualize training
            if self.opt.vis_training:
                if self.opt.enable_lidar:
                    utils.vis_training(index, self.pano_sampled, show=True, fig_name='Lidar image sampling')
                if self.opt.enable_rgb:
                    utils.vis_training(index, self.rgb_sampled, show=True, fig_name='Camera image sampling')

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        self.log(f"average_loss: {average_loss}.")

        if self.local_rank == 0:
            pbar.close()
            pbar_train.close()
            if self.report_metric_at_train:
                #camera
                if self.opt.enable_rgb:
                    #TODO: update is required
                    for metric in self.metrics:
                        self.log(metric.report(), style="green")
                        if self.use_tensorboardX:
                            metric.write(self.writer,
                                         self.epoch,
                                         prefix="RGB_train")
                        metric.clear()
                #lidar
                if self.opt.enable_lidar:
                    for metric in self.depth_metrics.values():
                        self.log(metric.report(), style="red")
                        if self.use_tensorboardX:
                            metric.write(self.writer, self.epoch, prefix="LiDAR_train")
                        metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
        # torch.cuda.empty_cache()

    def evaluate_one_epoch(self, loader, name=None):
        """Evaluate the data, save the prediction data, calculate error matrices and other functions."""

        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics.values():
                metric.clear()
            for metric in self.depth_metrics.values():
                metric.clear()

        self.model.eval() #change model mode to evaluation

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        # Calculate predictions
        with torch.no_grad():
            self.local_step = 0

            for i, data in enumerate(loader):
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    (preds_rgb, 
                     preds_depth_rgb,
                     preds_intensity, #[B, H, W, 1]
                     preds_depth,  #[B, H, W]
                     preds_depth_crop,
                     preds_raydrop, #[B, H, W, 1]
                     gt_rgb, 
                     gt_image_depth,
                     gt_intensity,
                     gt_depth,
                     gt_depth_crop,
                     gt_raydrop,
                     loss) = self.eval_step(data) 

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device)
                                  for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device)
                                        for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device)
                                   for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # Calculate error matrices and save prediction data
                if self.local_rank == 0:

                    #helper function for masks
                    hlpr_fn_mask = lambda x: [torch.from_numpy(i[None,...]) \
                                                .to(preds_depth.dtype) \
                                                .to(self.device) \
                                                    for i in x] #[H, W] -> [1, H, W]
                    #camera
                    if self.opt.enable_rgb:
                        for metric, func in self.metrics.items():
                            if metric in ('rmse') and gt_image_depth is not None:
                                func.update(preds_depth_rgb / self.opt.scale, gt_image_depth)
                            elif metric in ('psnr', 'lpips', 'ssim'):
                                func.update(preds_rgb, gt_rgb)
                        
                        if len(data['3d_annotation']) > 0:
                            static_img_mask, dynamic_img_mask = hlpr_fn_mask(
                                utils.compute_object_masks_img(data, **vars(self))
                            )                        
                        else:
                            static_img_mask =  hlpr_fn_mask(np.ones(preds_depth_rgb.shape))[0]
                            dynamic_img_mask = hlpr_fn_mask(np.zeros(preds_depth_rgb.shape))[0]
                        
                        #update error metrics for background
                        for metric, func in self.metrics_static.items():
                            if metric in ('rmse') and gt_image_depth is not None:
                                func.update(preds_depth_rgb / self.opt.scale * static_img_mask, 
                                            gt_image_depth * static_img_mask)
                            elif metric in ('psnr', 'lpips', 'ssim'):
                                func.update(preds_rgb * static_img_mask[..., None], 
                                            gt_rgb * static_img_mask[..., None])

                        #update error metrics for foreground
                        for metric, func in self.metrics_dynamic.items():
                            if metric in ('rmse') and gt_image_depth is not None:
                                func.update(preds_depth_rgb / self.opt.scale * dynamic_img_mask, 
                                            gt_image_depth * dynamic_img_mask)
                            elif metric in ('psnr', 'lpips', 'ssim'):
                                func.update(preds_rgb * dynamic_img_mask[..., None], 
                                            gt_rgb * dynamic_img_mask[..., None])

                    #lidar
                    if self.opt.enable_lidar:
                        for metric, func in self.depth_metrics.items(): 
                            if metric in ('point', 'depth'): #for Depth and PCD error
                                if self.opt.dataloader == "nerf_mvl" : 
                                    func.update(preds_depth_crop, gt_depth_crop)
                                else:
                                    func.update(preds_depth, gt_depth)
                            elif metric == 'intensity':  #for intensity error)
                                func.update(preds_intensity[...,0], gt_intensity[...,0])
                            elif metric == 'raydrop': #raydrop
                                func.update(preds_raydrop, gt_raydrop)
                            else:
                                raise ValueError(f"Unknown metric: {metric}")
                            
                        #compute masks for background and foreground objects [H, W]
                        if len(data['3d_annotation']) > 0:
                            (static_obj_mask_pred, 
                             dyna_obj_mask_pred, 
                             _, _) = hlpr_fn_mask(utils.compute_object_masks(preds_depth[0],
                                                                             preds_intensity[0, :, :, 0], 
                                                                             data, 
                                                                             **vars(self)))
                            (static_obj_mask_gt, 
                             dyna_obj_mask_gt, 
                             _, _) = hlpr_fn_mask(utils.compute_object_masks(gt_depth[0],
                                                                             gt_intensity[0, :, :, 0],
                                                                             data,
                                                                             **vars(self)))
                        else:
                            static_obj_mask_pred = static_obj_mask_gt = hlpr_fn_mask(np.ones(preds_depth.shape))[0]
                            dyna_obj_mask_pred = dyna_obj_mask_gt = hlpr_fn_mask(np.zeros(preds_depth.shape))[0]
                        
                        #update error metrices for background
                        for metric, func in self.depth_metrics_static.items(): 
                            if metric in ('point', 'depth'): #for Depth and PCD error
                                func.update(preds_depth * static_obj_mask_pred, 
                                            gt_depth * static_obj_mask_gt)
                            elif metric == 'intensity':  #for intensity error)
                                func.update(preds_intensity[...,0]*static_obj_mask_pred, 
                                            gt_intensity[...,0] * static_obj_mask_gt)
                            elif metric == 'raydrop': #raydrop
                                func.update(preds_raydrop*static_obj_mask_pred[...,None], 
                                            gt_raydrop * static_obj_mask_gt[...,None])
                        
                        #update error metrices for foreground
                        for metric, func in self.depth_metrics_dynamic.items(): 
                            if metric in ('point', 'depth'): #for Depth and PCD error
                                func.update(preds_depth * dyna_obj_mask_pred, 
                                            gt_depth * dyna_obj_mask_gt)
                            elif metric == 'intensity':  #for intensity error)
                                func.update(preds_intensity[...,0]*dyna_obj_mask_pred, 
                                            gt_intensity[...,0] * dyna_obj_mask_gt)
                            elif metric == 'raydrop': #raydrop
                                func.update(preds_raydrop*dyna_obj_mask_pred[...,None], 
                                            gt_raydrop *dyna_obj_mask_gt[...,None])
                    
                    # Save prediction data
                    if self.opt.enable_lidar:
                        #file path for  lidar raydrop pano image
                        save_path_lidar_val = os.path.join(self.workspace,
                                                         "validation",
                                                         f"{name}_{self.local_step:04d}.png")
                        #file path for  lidar raydrop pano image
                        save_path_raydrop = os.path.join(self.workspace,
                                                         "validation",
                                                         f"{name}_{self.local_step:04d}_raydrop.png")

                        #file path for  lidar intensity pano image
                        save_path_intensity = os.path.join(self.workspace,
                                                           "validation",
                                                           f"{name}_{self.local_step:04d}_intensity.png",)

                        #file path for  lidar depth/range pano image
                        save_path_depth = os.path.join(self.workspace,
                                                       "validation",
                                                       f"{name}_{self.local_step:04d}_depth.png",)

                        os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)
                        
                        # Convert from tensor to numpy
                        pred_raydrop = preds_raydrop[0].detach().cpu().numpy() #[B, H, W] --> [H, W, 1]
                        pred_intensity = preds_intensity[0].detach().cpu().numpy() #[B, H, W] --> [H, W, 1]
                        pred_depth = preds_depth[0].detach().cpu().numpy() #[B, H, W] -->  [H, W]
                        gt_raydrop = gt_raydrop[0].detach().cpu().numpy() #[B, H, W] --> [H, W, 1]
                        gt_intensity = gt_intensity[0].detach().cpu().numpy() #[B, H, W] --> [H, W, 1]
                        gt_depth = gt_depth[0].detach().cpu().numpy() #[B, H, W] -->  [H, W]

                        #compute raydrop mask [H, W, 1]
                        pred_raydrop_mask = np.where(pred_raydrop > self.opt.raydrop_thres, 1.0, 0.0)

                        #Create images [H, W, 1]
                        #raydrop mask is already applied during eval_step on intensity and depth predictions
                        img_raydrop_gt = (gt_raydrop * 255).astype(np.uint8)
                        img_intensity_gt = (gt_intensity * 255).astype(np.uint8)      
                        img_depth_gt = (gt_depth[..., None] * 255).astype(np.uint8)
                        img_raydrop_pred = (pred_raydrop * 255).astype(np.uint8)
                        img_raydrop_masked_pred = (pred_raydrop * pred_raydrop_mask * 255).astype(np.uint8)
                        img_intensity_pred = (pred_intensity * 255).astype(np.uint8)      
                        img_depth_pred = (pred_depth[..., None] * 255).astype(np.uint8)
                        # img_intensity_masked_pred = (pred_intensity * pred_raydrop_mask * 255).astype(np.uint8)
                        # img_depth_masked_pred = (pred_depth[..., None] * pred_raydrop_mask * 255).astype(np.uint8)                 

                        # Get predicted pcd in lidar frame [x,y,z,i]
                        pred_pcd_lidar = convert.pano_to_lidar_with_intensities(pred_depth / self.opt.scale, 
                                                                                img_intensity_pred, 
                                                                                loader._data.intrinsics_lidar,
                                                                                loader._data.intrinsics_hoz_lidar)
                        
                        # Get predicted pcd in world frame [x,y,z,i]
                        pred_pcd_world = utils.get_pcd_bound_to_world(pred_depth, img_intensity_pred, loader, data)
                        
                        # Compute depth/range error pano in meters
                        depth_error_pano = err_mat.depth_error_ratio(gt_depth/self.opt.scale, pred_depth/self.opt.scale)
                        
                        # Get depth error [x,y,z,error]
                        # world frame
                        pcd_depth_err = utils.get_pcd_bound_to_world(pred_depth, depth_error_pano, loader, data)
                        # lidar frame 
                        # pcd_depth_err = convert.pano_to_lidar_with_intensities(pred_depth, 
                        #                                                        depth_error_pano.reshape(loader._data.H_lidar, loader._data.W_lidar),
                        #                                                        loader._data.intrinsics_lidar, 
                        #                                                        loader._data.intrinsics_hoz_lidar) 

                        # Save pcd with depth/range error [x,y,z,error]
                        np.savetxt(
                            os.path.join(self.workspace, "validation", f"{name}_{self.local_step:04d}_pcd_error_world.txt"),
                            pcd_depth_err, delimiter=' ', fmt="%f"
                        )

                        # Save pcd data in world frame [x,y,z,i]
                        np.savetxt(
                            os.path.join(self.workspace, "validation", f"{name}_{self.local_step:04d}_pcd_world.txt"),
                            pred_pcd_world, delimiter=' ', fmt="%f"
                        )

                        # Save pcd data in lidar frame [x,y,z,i]
                        # np.savetxt(
                        #     os.path.join(self.workspace, "validation", f"{name}_{self.local_step:04d}_pcd_lidar.txt"),
                        #     pred_pcd_lidar, delimiter=' ', fmt="%f"
                        # )

                        # Save predicted lidar pano/range image in .npy file
                        # np.save(
                        #     os.path.join(self.workspace, "validation", f"{name}_{self.local_step:04d}_lidar.npy",),
                        #     pred_pcd_lidar
                        # )             

                        #Apply colour maps to images
                        img_raydrop_gt = cv2.cvtColor(img_raydrop_gt, cv2.COLOR_GRAY2BGR)
                        img_intensity_gt = cv2.applyColorMap(img_intensity_gt, 1)
                        img_depth_gt = cv2.applyColorMap(img_depth_gt, 20) #colormap options: 20, 9
                        img_raydrop_pred = cv2.cvtColor(img_raydrop_pred, cv2.COLOR_GRAY2BGR)
                        img_raydrop_masked_pred = cv2.cvtColor(img_raydrop_masked_pred, cv2.COLOR_GRAY2BGR)
                        img_intensity_pred = cv2.applyColorMap(img_intensity_pred, 1)
                        img_depth_pred = cv2.applyColorMap(img_depth_pred, 20) #colormap options: 20, 9
                        # img_intensity_masked_pred = cv2.applyColorMap(img_intensity_masked_pred, 1)
                        # img_depth_masked_pred = cv2.applyColorMap(img_depth_masked_pred, 20)
                         
                        #Save stacked validation image
                        img_pred = cv2.vconcat([img_raydrop_gt, img_intensity_gt, img_depth_gt, 
                                                img_raydrop_masked_pred, img_intensity_pred, img_depth_pred])
                        cv2.imwrite(save_path_lidar_val, img_pred)
                    
                    # RGB
                    if self.opt.enable_rgb:
                        # save image
                        save_path_rgb_val = os.path.join(self.workspace, 'validation',f'{name}_{self.local_step:04d}.png')
                        save_path_rgb = os.path.join(self.workspace, 'validation',f'{name}_{self.local_step:04d}_rgb.png')
                        save_path_depth_rgb = os.path.join(self.workspace, 'validation',f'{name}_{self.local_step:04d}_rgb_depth.png')

                        #self.log(f"==> Saving validation image to {save_path}")
                        os.makedirs(os.path.dirname(save_path_depth_rgb), exist_ok=True)

                        if self.opt.color_space == 'linear':
                            preds_rgb = utils.linear_to_srgb(preds_rgb)
                        
                        # Convert from tensor to numpy
                        pred_rgb = preds_rgb[0].detach().cpu().numpy()
                        pred_depth_rgb = preds_depth_rgb[0].detach().cpu().numpy()

                        # Convert to image
                        img_rgb_pred = (pred_rgb * 255).astype(np.uint8)
                        img_rgb_depth_pred = (pred_depth_rgb * 255).astype(np.uint8)

                        cv2.imwrite(save_path_rgb, cv2.cvtColor(img_rgb_pred, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_depth_rgb, img_rgb_depth_pred)
                        # img_rgb_val = cv2.vconcat([cv2.cvtColor(img_rgb_pred, cv2.COLOR_RGB2BGR), img_rgb_depth_pred])
                        # cv2.imwrite(save_path_rgb_val, img_rgb_val)

                    # update progress bars
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            # save error metric in json file
            for type, depth_metrices, matrices in zip(
                ('all', 'static', 'dynamic'),
                (self.depth_metrics, self.depth_metrics_static, self.depth_metrics_dynamic), 
                (self.metrics, self.metrics_static, self.metrics_dynamic)
                ):
                file_path=os.path.join(self.workspace, 'validation', f"{name}_{type}")
                preds_json = utils.cal_pred_errmat(depth_metrices, matrices, file_path, **vars(self)) 
                preds_json.save_json()
                # preds_json.print_msg()
                self.log(f"[INFO] Results saved in json file at: {preds_json.file_path}")

            # logging and tensorboard
            #lidar
            if self.opt.enable_lidar:
                if len(self.depth_metrics) > 0:
                    result = self.depth_metrics["point"].measure()[0]  #for PointMeter
                    self.stats["results"].append(result if self.best_mode == "min" else -result)  # if max mode, use -result
                else:
                    self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

                # logging and tensorboard  
                self.log("================Lidar metrics (Combined)================")
                for metric in self.depth_metrics.values():
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="LiDAR_evaluate(Combine)")
                    metric.clear()
                self.log("================Lidar metrics (Background)================")
                for metric in self.depth_metrics_static.values():
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="LiDAR_evaluate(Background)")
                    metric.clear()
                self.log("================Lidar metrics (Foreground)================")
                for metric in self.depth_metrics_dynamic.values():
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="LiDAR_evaluate(Foreground)")
                    metric.clear()
                
            #camera 
            if self.opt.enable_rgb:
                if len(self.metrics) > 0 :
                    result_rgb = self.metrics["psnr"].measure()  #for PSNR
                    self.stats["results_rgb"].append(result_rgb if self.best_mode == "min" else -result)
                else:
                    self.stats["results_rgb"].append(average_loss)  # if no metric, choose best by min loss
                
                # logging and tensorboard  
                self.log("================Camera metrics (Combined)================")
                for metric in self.metrics.values():
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="RGB_evaluate(Combined)")
                    metric.clear()

                self.log("================Camera metrics (Background)================")
                for metric in self.metrics_static.values():
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="RGB_evaluate(Background)")
                    metric.clear()

                self.log("================Camera metrics (Foreground)================")
                for metric in self.metrics_dynamic.values():
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="RGB_evaluate(Foreground)")
                    metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
    
    def process_pointcloud(self, loader):
        """Process point cloud by removing background points Flow MLP"""

        #Change train dataloader to validation mode. Other way is to use separate dataloader.
        #But it will have high memory usage. So, we can reuse the same dataloader.
        loader._data.enable_lidar = True
        loader._data.training = False
        loader._data.num_rays_lidar = -1
        loader._data.num_rays = -1

        self.log("[INFO] Preparing Point Clouds for Scene flow...")
        self.pc_list = {}
        self.pc_ground_list = {}
        for i, data in tqdm.tqdm(iterable=enumerate(loader), total=len(loader)):
            # pano to lidar
            images_lidar = data["images_lidar"]
            gt_raydrop = images_lidar[:, :, :, 0]
            gt_depth = images_lidar[:, :, :, 2] * gt_raydrop
            gt_lidar = convert.pano_to_lidar(
                gt_depth.squeeze(0).clone().detach().cpu().numpy() / self.opt.scale, 
                loader._data.intrinsics_lidar,
                loader._data.intrinsics_hoz_lidar,
            )

            # remove ground points and outliers
            points, ground = utils.point_removal(
                pc_raw= gt_lidar,
                dist_min= 1,
                dist_max= 0.75*self.opt.lidar_max_depth/self.opt.scale,
                z_limit= [-3.5, 4] if self.opt.dataloader=='daas' else [-2.5, 4],
                )
            
            #capture points of 3d annotation. 
            #currently only dynamic pcd is considered
            # points = []
            # for ann in data['3d_annotation']:
            #     pcd_filtered, inhull_mask = tools.check_in_hull(gt_lidar, ann['vertices'])
            #     points.append(pcd_filtered)
            # points = np.concatenate(points, axis=0)

            # transform filtered points to world frame
            pose = data["poses_lidar"].squeeze(0)
            pose = pose.clone().detach().cpu().numpy()
            # pose[:3, 3]=(pose[:3,3]+self.opt.offset)/self.opt.scale #For debugging
            points = points * self.opt.scale
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            points = (pose @ points.T).T[:,:3]
            
            # transform ground points to world frame
            ground = ground * self.opt.scale
            ground = np.hstack((ground, np.ones((ground.shape[0], 1))))
            ground = (pose @ ground.T).T[:,:3]

            time_lidar = data["time"]
            frame_idx = int(time_lidar * (self.opt.num_frames - 1))
            self.pc_list[f"{frame_idx}"] = points
            self.pc_ground_list[f"{frame_idx}"] = ground
            # if i % 10 == 0:
            #     print(f"{i+1}/{len(loader)}")
        
        #Revert dataloader to training mode
        loader._data.enable_lidar = self.opt.enable_lidar
        loader._data.training = True
        loader._data.num_rays_lidar = self.opt.num_rays_lidar
        loader._data.num_rays = self.opt.num_rays
    
    def extract_embaddings(self, image):
        """Function for extracting embeddings from image"""
        # Load pre-trained VGG16 model
        model = torch.models.vgg16(pretrained=True).features.to(self.device)
        
        # Load and preprocess image
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),  # Resize image to expected input size (224 for VGG16)
            torchvision.transforms.CenterCrop(224),  # Crop central region of the resized image
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load and preprocess image
        # image = Image.fromarray(next(iter(valid_loader))['images'][0].detach().cpu().numpy().astype(np.uint8))
        image = Image.fromarray(image.detach().cpu().numpy().astype(np.uint8))
        image = preprocess(image)  # Load and convert to RGB
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        
        # Extract features (embeddings)
        with torch.no_grad(): embeddings = model(image) 
        embeddings = embeddings.view(embeddings.size(0), -1)  #Flatten
        
        return embeddings
    