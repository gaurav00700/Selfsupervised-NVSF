import torch
import numpy as np
import os, sys
import tinycudann as tcnn
from nvsf.nerf.activation import trunc_exp
from nvsf.nerf.models.renderer_dynamic import NeRFRenderer
from nvsf.nerf.models.planes_field import Planes4D
from nvsf.nerf.models.hash_field import HashGrid4D
from nvsf.nerf.models.flow_field import FlowField
from nvsf.nerf.models.unet import UNet

class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        min_resolution=32, #K-planes
        base_resolution=512, #Multires hash encoding (tcnn)
        max_resolution=32768, # Multires hash encoding (tcnn)
        time_resolution=25, #K-planes,  Multires hash encoding (tcnn)
        n_levels_plane=4, #K-planes
        n_features_per_level_plane=8, #K-planes
        n_levels_hash=8, #Multires hash encoding (tcnn)
        n_features_per_level_hash=4, #Multires hash encoding (tcnn)
        log2_hashmap_size=19, #Multires hash encoding (tcnn)
        num_layers_flow=3, #Flow network
        hidden_dim_flow=64, # ""
        num_layers_sigma=2, #Sigma network
        hidden_dim_sigma=64, # Sigma network
        geo_feat_dim=15, # Sigma network
        num_layers_lidar=3, # Intensity and Raydrop networks
        hidden_dim_lidar=64, # Intensity and Raydrop networks
        num_layers_color=3, #Camera colour network *
        hidden_dim_color=64, #Camera colour network *
        out_color_dim=3, #Camera colour network *
        out_lidar_color_dim=2, #Lidar colour network *
        num_frames=51, # Flow and sigma networks
        bound=1, # *
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.out_color_dim = out_color_dim
        self.out_lidar_color_dim = out_lidar_color_dim
        self.num_frames = num_frames
        self.n_features_per_level_hash = n_features_per_level_hash

        # Space and time encoder using 6 planes to represent 4D volume (xyzt)
        self.planes_encoder = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=n_features_per_level_plane,
            resolution=[min_resolution] * 3 + [time_resolution],
            multiscale_res=[2**(n) for n in range(n_levels_plane)],
            # concat_features=True,
            # decompose=True,
        )
        
        #Hybrid encoder for space and time using Muti-resolution Hash encoding and Kplanes
        self.hash_encoder = HashGrid4D(
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels=n_levels_hash,
            n_features_per_level=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
        )
        # Space and time encoder using 6 planes to represent 4D volume (xyzt)
        self.planes_encoder_lidar = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=n_features_per_level_plane,
            resolution=[min_resolution] * 3 + [time_resolution],
            multiscale_res=[2**(n) for n in range(n_levels_plane)],
            concat_ms_feat=True,
            decompose=True,
        )
        
        #Hybrid encoder for space and time using Muti-resolution Hash encoding and Kplanes
        self.hash_encoder_lidar = HashGrid4D(
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels=n_levels_hash,
            n_features_per_level=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
        )
        # Space and time encoder using 6 planes to represent 4D volume (xyzt)
        self.planes_encoder_camera = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=n_features_per_level_plane,
            resolution=[min_resolution] * 3 + [time_resolution],
            multiscale_res=[2**(n) for n in range(n_levels_plane)],
            concat_ms_feat=True,
            decompose=True,
        )
        
        #Hybrid encoder for space and time using Muti-resolution Hash encoding and Kplanes
        self.hash_encoder_camera = HashGrid4D(
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels=n_levels_hash,
            n_features_per_level=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
        )

        #Direction encoder for Lidar
        self.view_encoder_lidar = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 12,
            },
        )
        #Temporal neural field for time conditioning
        #Ref: https://arxiv.org/pdf/2303.15126.pdf
        self.flow_net = FlowField(
            input_dim=4,
            num_layers=num_layers_flow,
            hidden_dim=hidden_dim_flow,
            use_grid=True,
        )

        #Density network
        self.sigma_net = tcnn.Network(
            n_input_dims=(self.planes_encoder_lidar.n_output_dims + self.hash_encoder_lidar.n_output_dims)*1,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_sigma,
                "n_hidden_layers": num_layers_sigma - 1,
            },
        )
        
        #Intensity network for Lidar
        self.intensity_net = tcnn.Network(
            n_input_dims=self.view_encoder_lidar.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )
        
        #Raydrop network for Lidar
        self.raydrop_net = tcnn.Network(
            n_input_dims=self.view_encoder_lidar.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )
        
        #Camera colour Network
        # Direction encoder for Camera
        self.view_encoder_camera = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "SphericalHarmonics",
                             "degree": 4,
                             },
                        )
        # self.view_encoder_camera = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "degree": 12,
        #     },
        # )
        
        # Create color network
        self.color_net = tcnn.Network(n_input_dims=self.view_encoder_camera.n_output_dims + geo_feat_dim,
                                      n_output_dims=self.out_color_dim,
                                      network_config={
                                          "otype": "FullyFusedMLP",
                                          "activation": "ReLU",
                                          "output_activation": "None",
                                          "n_neurons": hidden_dim_color,
                                          "n_hidden_layers": num_layers_color - 1,
                                            },
                                    )
    
        #Raydrop refinement network    
        self.unet = UNet(in_channels=3, out_channels=1)

    def forward(self, x, d):
        pass

    def flow(self, x, t):
        # x: [N, 3] in [-bound, bound] for point clouds
        x = (x + self.bound) / (2 * self.bound)
        # frame_idx = int(t * (self.num_frames - 1))

        if t.shape[0] == 1:
            t = t.repeat(x.shape[0], 1)
        xt = torch.cat([x, t], dim=-1)

        flow = self.flow_net(xt)

        return {
            "flow_forward": flow[:, :3], # flow_forward,
            "flow_backward": flow[:, 3:], #flow_backward,
        }
    
    def density(self, x, t=None, cal_lidar_color=False, **kwargs):
        """Density prediction using separate 4D Multi-res Hash planes encoding for Lidar and Camera"""
        
        # x: [N, 3], in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        frame_idx = int(t * (self.num_frames - 1))

        #encode space and space+time for static and dynamic features using MultiRes hash grid
        if cal_lidar_color:
            hash_feat_s, hash_feat_d = self.hash_encoder_lidar(x, t) #[N, 32], [N, 24]
        else:
            hash_feat_s, hash_feat_d = self.hash_encoder_camera(x, t) #[N, 32], [N, 32]

        if t.shape[0] == 1:
            t = t.repeat(x.shape[0], 1)            
        xt = torch.cat([x, t], dim=-1) #[N, 4]
        
        #Dynamic scene encoding using Kplanes
        if cal_lidar_color:
            plane_feat_s, plane_feat_d = self.planes_encoder_lidar(xt) #[N, 32], [N, 32]
        else:
            plane_feat_s, plane_feat_d = self.planes_encoder_camera(xt) #[N, 32], [N, 32]

        # integrate neighboring dynamic features
        flow = self.flow_net(xt) #[N, 6] 3->forward and 3->backward
        hash_feat_1 = hash_feat_2 = hash_feat_d #dynamic features hash
        plane_feat_1 = plane_feat_2 = plane_feat_d #dynamic features planes
        
        #forward time
        if frame_idx < self.num_frames - 1:
            x1 = x + flow[:, :3]
            t1 = torch.tensor((frame_idx + 1) / self.num_frames)
            with torch.no_grad():
                if cal_lidar_color:
                    hash_feat_1 = self.hash_encoder_lidar.forward_dynamic(x1, t1)
                else:
                    hash_feat_1 = self.hash_encoder_camera.forward_dynamic(x1, t1)
            t1 = t1.repeat(x1.shape[0], 1).to(x1.device)
            xt1 = torch.cat([x1, t1], dim=-1)
            if cal_lidar_color:
                plane_feat_1 = self.planes_encoder_lidar.forward_dynamic(xt1)
            else:
                plane_feat_1 = self.planes_encoder_camera.forward_dynamic(xt1)
        
        #backward time
        if frame_idx > 0:
            x2 = x + flow[:, 3:]
            t2 = torch.tensor((frame_idx - 1) / self.num_frames)
            with torch.no_grad():
                if cal_lidar_color:
                    hash_feat_2 = self.hash_encoder_lidar.forward_dynamic(x2, t2)
                else:
                    hash_feat_2 = self.hash_encoder_camera.forward_dynamic(x2, t2)
            t2 = t2.repeat(x2.shape[0], 1).to(x2.device)
            xt2 = torch.cat([x2, t2], dim=-1)
            if cal_lidar_color:
                plane_feat_2 = self.planes_encoder_lidar.forward_dynamic(xt2)
            else:
                plane_feat_2 = self.planes_encoder_camera.forward_dynamic(xt2)

        plane_feat_d = 0.5 * plane_feat_d + 0.25 * (plane_feat_1 + plane_feat_2) #[N,32]
        hash_feat_d = 0.5 * hash_feat_d + 0.25 * (hash_feat_1 + hash_feat_2) #[N,24]

        features = torch.cat([plane_feat_s, plane_feat_d, hash_feat_s, hash_feat_d], dim=-1) #[N, 120]

        h = self.sigma_net(features)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }

    # allow masked inference
    def color(self, x, d, cal_lidar_color=False, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        
        #where mask is of compositing weight above some threshold
        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], self.out_dim, dtype=x.dtype, device=x.device
            )  # [N, 3]
            
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        if cal_lidar_color: #for lidar
            d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.view_encoder_lidar(d) #encode direction
            logits = torch.cat([d, geo_feat], dim=-1) #concat direction and geometry features
            intensity = self.intensity_net(logits) #query intensity net
            # intensity = torch.sigmoid(intensity) #apply sigmoid 
            raydrop = self.raydrop_net(logits) #query raydrop net
            # raydrop = torch.sigmoid(raydrop) #apply sigmoid 
            h = torch.cat([raydrop, intensity], dim=-1) #concat intensity and raydrop
        else: #for camera
            d = (d + 1) / 2
            d = self.view_encoder_camera(d)
            logits = torch.cat([d, geo_feat], dim=-1)
            h = self.color_net(logits)

        #apply sigmoid activation on attribute logits
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # optimizer utils
    def get_params(self, lr):
        params = [
            # {"params": self.planes_encoder.parameters(), "lr": lr}, #scene 4D volume
            # {"params": self.hash_encoder.parameters(), "lr": lr}, #position
            {"params": self.planes_encoder_lidar.parameters(), "lr": lr}, #scene 4D volume
            {"params": self.hash_encoder_lidar.parameters(), "lr": lr}, #position
            {"params": self.planes_encoder_camera.parameters(), "lr": lr}, #scene 4D volume
            {"params": self.hash_encoder_camera.parameters(), "lr": lr}, #position
            {"params": self.view_encoder_lidar.parameters(), "lr": lr}, #lidar direction
            {"params": self.view_encoder_camera.parameters(), "lr": lr}, #camera direction
            {"params": self.flow_net.parameters(), "lr": 0.1 * lr}, #time flow net
            {"params": self.sigma_net.parameters(), "lr": lr}, #density net
            # {"params": self.encoder_lidar_dir.parameters(), "lr": lr},
            {"params": self.intensity_net.parameters(), "lr": 0.1 * lr}, #intensity net
            {"params": self.raydrop_net.parameters(), "lr": 0.1 * lr}, #raydrop net
            {"params": self.color_net.parameters(), "lr": lr},
            # {"params": self.lidar_color_net.parameters(), "lr": lr},
        ]
        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})

        return params
    
if __name__ == "__main__":
    model = NeRFNetwork().cuda()
    x = torch.rand(100, 3).cuda()
    t = torch.tensor([0.2]).cuda()
    result = model.density(x, t)
    print(result)