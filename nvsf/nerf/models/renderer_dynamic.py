import math
import trimesh
import torch
import torch.nn as nn
from nvsf.nerf.raymarching import raymarching


def sample_pdf(bins, weights, n_samples, det=False):
    """
    Hierarchical sampling with invert CDF
    Args:
        bins: tensor of shape [B, T], old_z_vals, B: batch size and T: number of bins in PDF.
        weights: tensor of shape [B, T - 1], weights of each bin in the PDF, Bin weights
        n_samples: number of samples to generate.
        det: flag indicating whether to use deterministic sampling
    Return:
        [B, n_samples], new_z_vals

    """

    # computing the CDF of the PDF
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) #weights tensor normalized to sum to 1
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) # concatenate zero at first index of each ray [B, len(bins)]
    
    # Take uniform samples from the interval [0, 1]
    if det: #deterministically
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples
        ).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # expand fines samples for all rays of the batch [B, n_samples]
    else: #randonmly
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True) #find the indices of the bins in the CDF that the uniform samples fall into
    below = torch.max(torch.zeros_like(inds - 1), inds - 1) #get below indices of bins and greater than zero
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds) #get above indices of bins and smallter than 1
    inds_g = torch.stack([below, above], -1)  # get below and above index of cdf bin for each fine sample points [B, n_samples, 2]

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # [B, n_samples, T]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) #cdf data/bins for inds_g [B, n_samples, 2]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) #coarse zvals/bins for inds_g [B, n_samples, 2]

    denom = cdf_g[..., 1] - cdf_g[..., 0] # difference between CDF values of adjacent bins [B, n_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom) # cap the denom (cdf_g difference) between [1, 0]
    t = (u - cdf_g[..., 0]) / denom # normalize the distance between fine sample (n_samples) and lower CDF value [B, n_samples]
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]) #compute samples by linearly interpolating between two bin values

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print("[visualize points]", pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(
        self,
        bound=1,
        density_scale=1,  # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
        min_near=0.01, #minimum distance to sample points for camera
        min_near_lidar=0.01, #near distance to sample points for lidar
        lidar_max_depth=0.81, #far distance to sample points for lidar
        density_thresh=0.01,
        bg_radius=-1,
        active_sensor=False
    ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.min_near_lidar = min_near_lidar
        self.lidar_max_depth = lidar_max_depth
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius  # radius of the background sphere.
        self.active_sensor = active_sensor

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        rays_o,
        rays_d,
        time,
        cal_lidar_color=False,
        num_steps=768,
        upsample_steps=128,
        bg_color=None,
        perturb=False,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        if cal_lidar_color:
            self.out_dim = self.out_lidar_color_dim
        else:
            self.out_dim = self.out_color_dim

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose Axis-Aligned Bounding Box (aabb) bounds parameter
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps in world frame
        if cal_lidar_color:
            nears = (
                torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.min_near_lidar
            )
            fars = (
                torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.lidar_max_depth
            ) 
        else:
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)

        nears.unsqueeze_(-1) # Added dimension
        fars.unsqueeze_(-1)  # Added dimension

        # print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')
        # Get sampling distance with in bounds for all rays
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]
        
        # delta/step size [N,1]
        sample_dist = (fars - nears) / num_steps 
        
        # perturb z_vals during training (stratified sampling)
        if perturb:
            z_vals = (z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist)
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # get coordinates of coarse sample(end) points for rays (o + d*t)
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip within bounds

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB (dict) from MLP
        density_outputs = self.density(xyzs.reshape(-1, 3), time, cal_lidar_color, **kwargs)

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1) # reshape to [N, T]

        # Calculate delta/stepsize for new sample points of ray
        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        
        # new alphas: light contributions by ray segment [N, T+t+1]
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        
        if self.active_sensor: #lidar
        # if cal_lidar_color: #lidar
            alphas = 1 - torch.exp(-2 * deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
            
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]

        # new weights: light is blocked earlier along ray segment [N, T+t]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]

        # Reshape rays directions as xyzs [N, T+t, 3]
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        # Query lidar colour values (raydrop and intensity) from MLP
        mask = weights > 1e-4  #TODO: hard coded
        rgbs = self.color(
            xyzs.reshape(-1, 3),
            dirs.reshape(-1, 3),
            cal_lidar_color=cal_lidar_color,
            mask=mask.reshape(-1),
            **density_outputs
        )

        rgbs = rgbs.view(N, -1, self.out_dim)  # [N, T+t, 3]

        # print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate predicted depth/range  Note: not real depth!!
        # ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        # depth = torch.sum(weights * ori_z_vals, dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate lidar intensity color and raydrop probability 
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(
                rays_o, rays_d, self.bg_radius
            )  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        if not cal_lidar_color:
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, self.out_dim)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        #lidar rendering
        if cal_lidar_color: 
            return {
                'depth_lidar': depth,
                'image_lidar': image,
                'weights_sum_lidar': weights_sum,
                "weights": weights,
                "z_vals": z_vals,
            }
        
        #camera rendering
        else:
            return {
                'depth': depth,
                'image': image,
                'weights_sum': weights_sum,
                "weights": weights,
                "z_vals": z_vals,
            }

    def render(
        self,
        rays_o,
        rays_d,
        time,
        cal_lidar_color=False,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        
        #Batch querying for evaluation
        if staged:
            if cal_lidar_color:
                out_dim = self.out_lidar_color_dim
                res_keys = ["depth_lidar", "image_lidar"]
            else:
                out_dim = self.out_color_dim
                res_keys = ['depth', 'image']
            # Create empty tensors for depth and image predictions
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, out_dim), device=device)

            # Preform predictions in batches of rays
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b : b + 1, head:tail],
                                    rays_d[b : b + 1, head:tail],
                                    time[b:b+1],
                                    cal_lidar_color=cal_lidar_color,
                                    **kwargs)
                    # Fill depth and image predictions
                    depth[b : b + 1, head:tail] = results_[res_keys[0]]
                    image[b : b + 1, head:tail] = results_[res_keys[1]]
                    
                    # update ray head for slicing
                    head += max_ray_batch 
                    
            results = {}
            results[res_keys[0]] = depth
            results[res_keys[1]] = image
        
        #for training
        else: 
            results = _run(rays_o, 
                           rays_d, 
                           time,
                           cal_lidar_color=cal_lidar_color, 
                           **kwargs)

        return results
