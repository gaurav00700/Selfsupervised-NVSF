import torch
import configargparse
import os
import time
from nvsf.nerf.models.network_dynamic import NeRFNetwork
from nvsf.nerf.utils import seed_everything
from nvsf.nerf.trainer import Trainer
from nvsf.lib import error_matrices as err_mat
from nvsf.nerf.dataset import kitti360_dataset
    
def get_arg_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, required=True, help="config file path",)
    parser.add_argument("--path", type=str, required=True, help="path of pano image and configs json files")
    parser.add_argument("--name", type=str, default="nvsf", help="experiment name")
    parser.add_argument("-L", action="store_true", help="equals --fp16 --preloadynamic")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--test_eval", action="store_true", help="test and eval mode")
    parser.add_argument("--workspace", type=str, default="nvsf/log", help="path of workspace")
    parser.add_argument("--cluster_summary_path", type=str, default="/summary", help="Overwrite default summary path if on cluster",)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preload",action="store_true",help="preload all data into GPU, accelerate training but use more GPU memory",)
    
    #dataset
    parser.add_argument("--dataloader", type=str, required=True, choices=("kitti360", "dgt", "daas"), help="Dataset name")
    parser.add_argument("--sequence_id", type=str,  required=True, help="Sequence name as per dataset")
    parser.add_argument("--min_near", type=float, default=1.0, help="minimum near distance for camera")
    parser.add_argument("--min_near_lidar",type=float,default=1.0,help="minimum near distance for lidar",)
    parser.add_argument("--lidar_max_depth", type=float, required=True, help="max distance for lidar")
    parser.add_argument("--intrinsics_lidar", nargs='+', type=float, required=True, help="Vertical Lidar intrinsics parameters in degrees [fov_up, fov]")
    parser.add_argument("--intrinsics_hoz_lidar", nargs='+', type=float, required=True, help="Horizontal Lidar intrinsics parameters in degrees [fov_up, fov]")
    parser.add_argument("--offset",type=float, nargs="*", default=[0, 0, 0], help="offset of sensor location after recenter for aabb",)
    parser.add_argument("--scale",type=float, default=0.01, help="scale sensor location into box[-bound, bound]^3",)
    parser.add_argument("--bound",type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, ""if > 1, will invoke adaptive ray marching.",)
    parser.add_argument("--num_frames", type=int, required=True, help="total number of sequence frames for temporal modeling")
    parser.add_argument("--color_space", type=str, default="srgb", help="Color space, supports (linear, srgb)",)
    parser.add_argument("--active_sensor", action="store_true", help="enable volume rendering for active sensor.")

    # Network configuration
    parser.add_argument("--refine", action="store_true", help="refine mode")
    parser.add_argument("--use_refine", action="store_true", help="whether to use refined raydrop",)
    parser.add_argument("--fp16", action="store_true", help="use amp mixed precision training")
    parser.add_argument("--min_resolution", type=int, default=32, help="minimum resolution for planes")
    parser.add_argument("--base_resolution", type=int, default=512, help="minimum/coarse resolution for hash grid at the largest scale")
    parser.add_argument("--max_resolution", type=int, default=32768, help="maximum/finest resolution for hash grid at the smallest scale")
    parser.add_argument("--time_resolution", type=int, default=8, help="temporal resolution")
    parser.add_argument("--n_levels_plane", type=int, default=4, help="n_levels for planes")
    parser.add_argument("--n_features_per_level_plane", type=int, default=8, help="n_features_per_level for planes")
    parser.add_argument("--n_levels_hash", type=int, default=8, help="n_levels for hash grid")
    parser.add_argument("--n_features_per_level_hash", type=int, default=4, help="n_features_per_level for hash grid (dimension F)")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="Hash table size T, 2^n")
    parser.add_argument("--num_layers_flow", type=int, default=3, help="num_layers of flownet")
    parser.add_argument("--hidden_dim_flow", type=int, default=64, help="hidden_dim of flownet")
    parser.add_argument("--num_layers_sigma", type=int, default=2, help="num_layers of sigmanet")
    parser.add_argument("--hidden_dim_sigma", type=int, default=64, help="hidden_dim of sigmanet")
    parser.add_argument("--geo_feat_dim", type=int, default=15, help="geo_feat_dim of sigmanet")
    parser.add_argument("--num_layers_color", type=int, default=3, help="num_layers of color/intensity/raydrop network")
    parser.add_argument("--hidden_dim_lidar", type=int, default=64, help="hidden_dim of intensity/raydrop")
    parser.add_argument("--out_lidar_dim", type=int, default=2, help="output dim for lidar intensity/raydrop")

    # training options
    parser.add_argument("--eval_interval", type=int, default=100, help="eval interval")
    parser.add_argument("--activate_levels", type=int, default=0, help="Hash levels to activate for multimodality fusion[coarse to fine]")
    parser.add_argument('--enable_rgb', action='store_true', help="Enable rgb.")
    parser.add_argument("--enable_lidar", action="store_true", help="Enable lidar.")
    parser.add_argument("--epochs",type=int, default=500, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default="latest", help= "choose from: scratch, latest, latest_model, best, ckpt_path")
    parser.add_argument("--num_rays",type=int,default=2048,help="num rays sampled per image for each training step",)
    parser.add_argument("--num_rays_lidar",type=int,default=2048,help="num rays sampled per image for each training step",) #total: 1030x66=67980
    parser.add_argument("--num_steps", type=int, default=768, help="num steps sampled per ray")
    parser.add_argument("--upsample_steps", type=int, default=64, help="num steps up-sampled per ray")
    parser.add_argument("--max_ray_batch",type=int,default=4096,help="batch size of rays at inference to avoid OOM)",)  
    parser.add_argument("--raydrop_thres", type=float, default=0.5, help="threshold for raydrop mask")
    parser.add_argument("--smooth_factor", type=float, default=0.0, help="label smoothing for ray-drop in training step")
    parser.add_argument("--density_scale", type=float, default=1,  help="scale factor for density during training in renderer")
    parser.add_argument("--ema_decay", type=float, default=0.95, help="use ema during training")  
    parser.add_argument("--use_error_map", action="store_true", help="use error map for pixel sampling")  
    parser.add_argument("--vis_training", action="store_true", help="visualize training plots")  

    # Loss functions
    parser.add_argument('--rgb_loss', type=str, default='mse', help="l1, bce, mse, huber")
    parser.add_argument("--rgb_depth_loss", type=str, default="l1", help="l1, bce, mse, huber")
    parser.add_argument("--depth_loss", type=str, default="l1", help="l1, bce, mse, huber")
    parser.add_argument("--depth_grad_loss", type=str, default="l1", help="l1, mse, huber, cos")
    parser.add_argument("--intensity_loss", type=str, default="mse", help="l1, bce, mse, huber")
    parser.add_argument("--raydrop_loss", type=str, default="mse", help="l1, bce, mse, huber")
    parser.add_argument("--flow_loss", action="store_true", help="Flag of scene flow loss for spatio-temporal")
    parser.add_argument("--grad_loss", action="store_true", help="Flag of gradient loss for geometry regularization")
    parser.add_argument('--use_rgbd_loss', action="store_true", help="use rgb depth(from lidar) loss for camera")
    parser.add_argument("--use_urf_loss", action="store_true", help="enable line-of-sight loss for Lidar like in URF.")
    parser.add_argument("--alpha_d", type=float, default=1, help="weight for depth loss")
    parser.add_argument("--alpha_i", type=float, default=0.1, help="weight for intensity loss")
    parser.add_argument("--alpha_r", type=float, default=0.01, help="weight for raydrop loss")
    parser.add_argument('--alpha_rgb', type=float, default=1, help="weight for camera rgb loss")
    parser.add_argument('--alpha_rd', type=float, default=1, help="weight for camera depth loss")

    # Structural regularization
    parser.add_argument("--alpha_grad_norm", type=float, default=0.1, help="weight of gradient normal loss for geometry regularization")
    parser.add_argument("--alpha_spatial", type=float, default=0.1, help="weight of smoothness loss for geometry regularization")
    parser.add_argument("--alpha_tv", type=float, default=0.1, help="weight of TV loss for geometry regularization")
    parser.add_argument("--alpha_grad", type=float, default=0.1, help="weight of gradient loss for geometry regularization")
    parser.add_argument("--grad_norm_smooth", action="store_true", help="Encourage smoothness and penalize sharp edges in predicted depth maps")
    parser.add_argument("--spatial_smooth", action="store_true", help="Encourages spatial consistency in depth values")
    parser.add_argument("--tv_loss", action="store_true", help="Promotes smoothness and reduces noise")
    parser.add_argument("--sobel_grad", action="store_true", help="Compute gradients and penalize blurry or inaccurate edges")
    parser.add_argument("--patch_size", type=int, default=1, help="render patches in training, so as to apply. LPIPS loss, 1 means disabled, use [64, 32, 16] to enable",)
    parser.add_argument("--patch_size_lidar", type=int, default=1, help="patch size for lidar, 1 means disabled, use [64, 32, 16] to enable",)
    parser.add_argument("--change_patch_size_lidar", nargs="+", type=int, default=[2, 8], help="1 means disabled, use [64, 32, 16] to enable, change during training",)
    parser.add_argument("--change_patch_size_epoch", type=int, default=2, help="change patch_size interval",)
    parser.add_argument("--intensity_inv_scale", type=float, default=1, help="for intensity meter")
    parser.add_argument("--raydrop_ratio", type=float, default=0.5, help="for intensity meter")
    
    # (the default value is for the fox dataset)
    parser.add_argument("--dt_gamma",type=float,default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)",)
    parser.add_argument("--density_thresh",type=float,default=10, help="threshold for density grid to be occupied",)
    parser.add_argument("--bg_radius",type=float,default=-1, help="if positive, use a background model at sphere(bg_radius)",)

    # Lidar artifacts configurations for inference
    parser.add_argument("--delta_position", nargs='+', type=float, default=[0., 0., 0.], help="change position (dx, dy, dz) of Lidar. '+' , '-' => up , down")
    parser.add_argument("--delta_orientation", nargs='+', type=float, default=[0., 0., 0.], help="change orientation in degree (roll, pitch, yaw) of Lidar. '+' , '-' => up , down")
    parser.add_argument("--intrinsics_lidar_new", nargs='+', type=float, default=[0., 0.], help="Changed lidar horizontal intrinsics parameters in degrees [fov_up, fov]")
    parser.add_argument("--intrinsics_hoz_lidar_new", nargs='+', type=float, default=[0., 0.], help="Changed lidar vertical intrinsics parameters in degrees [fov_up, fov]")
    parser.add_argument("--V_lidar_ch", type=int, default=0, help="New vertical channels beams  of Lidar sensor, 0 means unchanged")
    parser.add_argument("--H_lidar_ch", type=int, default=0, help="New horizontal channel beams of Lidar sensor, 0 means unchanged")
    parser.add_argument("--H_new", type=int, default=0, help="New image height of camera, 0 means unchanged")
    parser.add_argument("--W_new", type=int, default=0, help="New image width of camera, 0 means unchanged")
    parser.add_argument("--delta_pos_camera", nargs='+', type=float, default=[0., 0., 0.], help="change position (dx, dy, dz) of camera. '+' , '-' => up , down")
    parser.add_argument("--delta_orient_camera", nargs='+', type=float, default=[0., 0., 0.], help="change orientation in degree (roll, pitch, yaw) of camera. '+' , '-' => up , down")

    return parser

def main():
    parser = get_arg_parser()
    opt = parser.parse_args()
    print(opt)
    seed_everything(opt.seed)

    #select dataloader
    dataloader = {
        "kitti360": kitti360_dataset.KITTI360Dataset,
        # "daas": daas_dataset.DaaSDataset,
        # "dgt": dgt_dataset.DGTDataset,
    }

    # Create workspace directory
    os.makedirs(opt.workspace, exist_ok=True)

    # Save args to file
    f = os.path.join(opt.workspace, f"args_{opt.name}.txt")
    with open(f, "w") as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write("{} = {}\n".format(arg, attr))

    if opt.L:
        opt.fp16 = True
        opt.preload = True

    if opt.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert (opt.num_rays % (opt.patch_size**2) == 0), "patch_size ** 2 should be dividable by num_rays."
    
    #Scale near and far values of ray sampling of sensors
    opt.min_near *= opt.scale  
    opt.min_near_lidar *= opt.scale
    opt.lidar_max_depth *= opt.scale

    assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
    assert opt.enable_lidar or opt.enable_rgb, "At least one of Lidar or Camera should be enabled"
    
    # Create framework network model
    model = NeRFNetwork(
            min_resolution=opt.min_resolution,
            base_resolution=opt.base_resolution,
            max_resolution=opt.max_resolution,
            time_resolution=opt.time_resolution,
            n_levels_plane=opt.n_levels_plane,
            n_features_per_level_plane=opt.n_features_per_level_plane,
            n_levels_hash=opt.n_levels_hash,
            n_features_per_level_hash=opt.n_features_per_level_hash,
            log2_hashmap_size=opt.log2_hashmap_size,
            num_layers_flow=opt.num_layers_flow,
            hidden_dim_flow=opt.hidden_dim_flow,
            num_layers_sigma=opt.num_layers_sigma,
            hidden_dim_sigma=opt.hidden_dim_sigma,
            geo_feat_dim=opt.geo_feat_dim,
            num_layers_lidar=opt.num_layers_color, #for raydrop/intensity/rgb_color
            hidden_dim_lidar=opt.hidden_dim_lidar,
            num_frames=opt.num_frames, #scene frames
            bound=opt.bound,
            min_near=opt.min_near, #camera
            min_near_lidar=opt.min_near_lidar, #lidar
            lidar_max_depth=opt.lidar_max_depth, #lidar
            density_scale=opt.density_scale,
            active_sensor=opt.active_sensor,
            )
    
    # Device options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Available Loss functions from pytorch
    loss_dict = {
        "mse": torch.nn.MSELoss(reduction="none"),
        "l1": torch.nn.L1Loss(reduction="none"),
        "smoothl1": torch.nn.SmoothL1Loss(reduction="none", beta=0.1),
        "huber": torch.nn.HuberLoss(reduction="none", delta=0.2 * opt.scale),
        "bce": torch.nn.BCEWithLogitsLoss(reduction="none"),
        "cos": torch.nn.CosineSimilarity(),
    }

    # Loss functions selection for depth, raydrop, intensity and gradient regularization
    criterion = {
        'rgb': loss_dict[opt.rgb_loss],
        "depth": loss_dict[opt.depth_loss],
        "rgb_depth": loss_dict[opt.rgb_depth_loss],
        "raydrop": loss_dict[opt.raydrop_loss],
        "intensity": loss_dict[opt.intensity_loss],
        "grad": loss_dict[opt.depth_grad_loss],
    }
    #lidar
    if opt.enable_lidar:
        depth_metrics = {
            "point": err_mat.PointsMeter(scale=opt.scale, intrinsics=opt.intrinsics_lidar, intrinsics_hoz=opt.intrinsics_hoz_lidar), # for pcd and raydrop
            "depth": err_mat.DepthMeter_L4D(scale=opt.scale), # for depth
            "intensity": err_mat.IntensityMeter_L4D(scale=opt.intensity_inv_scale), # for intensity
            "raydrop": err_mat.RaydropMeter(ratio=opt.raydrop_ratio), # for raydrop
        }
    else:
        depth_metrics = {}
    #camera
    if opt.enable_rgb:
        metrics = {
            "rmse": err_mat.RMSEMeter(rgb_metric=True),
            "psnr": err_mat.PSNRMeter(),
            "lpips": err_mat.LPIPSMeter(device=device),
            "ssim": err_mat.SSIMMeter(),
        }
    else:
        metrics = {}
    
    # Do testing and evaluation
    if opt.test or opt.test_eval:
        # Data Loader for testing
        test_loader = dataloader[opt.dataloader](
            device=device,
            split="test", # train, val, test, all
            training=False,
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size=opt.patch_size,
            patch_size_lidar=opt.patch_size_lidar,
            enable_lidar=opt.enable_lidar,
            enable_rgb=opt.enable_rgb,
            color_space=opt.color_space,
            num_rays=opt.num_rays,
            num_rays_lidar=opt.num_rays_lidar,
            delta_position=opt.delta_position,
            delta_orientation=opt.delta_orientation,
            H_lidar_new=opt.V_lidar_ch, # 64
            W_lidar_new=opt.H_lidar_ch, # 1030
            intrinsics_lidar=opt.intrinsics_lidar,
            intrinsics_hoz_lidar=opt.intrinsics_hoz_lidar,
            intrinsics_lidar_new=opt.intrinsics_lidar_new,
            intrinsics_hoz_lidar_new=opt.intrinsics_hoz_lidar_new,
            delta_pos_camera=opt.delta_pos_camera,
            delta_orient_camera=opt.delta_orient_camera,
            H_new = opt.H_new,
            W_new = opt.W_new,
        ).dataloader()

        # Create test object from trainer class
        trainer = Trainer(
            opt.name,
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            metrics=metrics,
            depth_metrics=depth_metrics,
            use_checkpoint=opt.ckpt,
        )
        # Testing and evaluation 
        if test_loader.has_gt and opt.test_eval:
            trainer.evaluate(test_loader, use_refine=opt.use_refine) 
        trainer.test(test_loader, write_video=False, use_refine=opt.use_refine)
        
        # Save mesh model 
        trainer.export_mesh_density(bound_min=[-0.5, -0.5, 0.06],  #should be within [-1, 0] range
                                    bound_max=[0.5, 0.5, 0.09], #should be within [0, 1] range
                                    xyz_res=[500, 500, 50], #number of pnts b/w bounds
                                    threshold=10)
    
    #training and validation pipeline
    else:
        # DataLoader for training 
        train_loader = dataloader[opt.dataloader](
            device=device,
            split="train", # train, val, test, all
            training=True,
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size=opt.patch_size,
            patch_size_lidar=opt.patch_size_lidar,
            use_error_map=opt.use_error_map,
            enable_lidar=opt.enable_lidar,
            enable_rgb= opt.enable_rgb,
            color_space=opt.color_space,
            num_rays=opt.num_rays,
            num_rays_lidar=opt.num_rays_lidar,
            intrinsics_lidar=opt.intrinsics_lidar,
            intrinsics_hoz_lidar=opt.intrinsics_hoz_lidar
        ).dataloader()

        # DataLoader for validation 
        valid_loader = dataloader[opt.dataloader](
            device=device,
            split="val", # train, val, test, all
            training=False,
            root_path=opt.path,
            sequence_id=opt.sequence_id,
            preload=opt.preload,
            scale=opt.scale,
            offset=opt.offset,
            fp16=opt.fp16,
            patch_size=opt.patch_size,
            patch_size_lidar=opt.patch_size_lidar,
            enable_lidar=opt.enable_lidar,
            enable_rgb=opt.enable_rgb,
            num_rays=opt.num_rays,
            color_space=opt.color_space,
            num_rays_lidar=opt.num_rays_lidar,
            intrinsics_lidar=opt.intrinsics_lidar,
            intrinsics_hoz_lidar=opt.intrinsics_hoz_lidar,
        ).dataloader()

        # optimizer
        optimizer = lambda model: torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
        )

        # Set total numbers of iterations
        opt.iters = int(opt.epochs * len(train_loader))
        print(f"[INFO] Max epochs: {opt.iters}")

        # Learning rate scheduler
        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
        )
        
        #trainer object
        trainer = Trainer(
            opt.name,
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            optimizer=optimizer,
            criterion=criterion,
            ema_decay=opt.ema_decay,
            fp16=opt.fp16,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            depth_metrics=depth_metrics,
            metrics=metrics,
            use_checkpoint=opt.ckpt,
            eval_interval=opt.eval_interval,
        )        

        # Start training
        time_start = time.time()
        trainer.train(train_loader, valid_loader, opt.epochs)
        print(f'==> Training finished in: {round((time.time() - time_start)/60, 2)} minutes')

if __name__ == "__main__":
    main()
