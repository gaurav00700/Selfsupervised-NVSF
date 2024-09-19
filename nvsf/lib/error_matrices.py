from matplotlib import rcParamsDefault
import torch
import numpy as np
import os
from skimage.metrics import structural_similarity
from torchmetrics.functional import structural_similarity_index_measure
import lpips
from scipy.spatial import KDTree
from nvsf.nerf.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from nvsf.lib.convert import pano_to_lidar

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean
    # distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

class PSNRMeter:
    """Calculate Peak signal to noise ratio
    """
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2) + 1e-8)

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / (self.N + 1e-8)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.3f}"

class RMSEMeter:
    """Calculate Root mean square error
    """

    def __init__(self, rgb_metric=False):
        self.V = 0
        self.N = 0
        self.rgb_metric = rgb_metric

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        if self.rgb_metric:
            zero_mask = np.where(truths == 0, 0, 1)
            preds = preds * zero_mask
            max_depth = 80
            preds[preds > max_depth] = max_depth
            truths[truths > max_depth] = max_depth

        delta_sq = (truths - preds) ** 2
        rmse = np.sqrt(delta_sq.mean())

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / (self.N + 1e-8)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "RMSE"), self.measure(), global_step)

    def report(self):
        if self.rgb_metric:
            return f"RMSE = {self.measure():.3f}"
        else:
            return f"RMSE_intensity = {self.measure():.3f}"

class MAEMeter:
    """Calculate Mean absolute error
    """

    def __init__(self, intensity_inv_scale=1.0):
        self.V = 0
        self.N = 0
        self.intensity_inv_scale = intensity_inv_scale

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # Mean Absolute Error
        mae = np.abs(
            truths * self.intensity_inv_scale - preds * self.intensity_inv_scale
        ).mean()

        self.V += mae
        self.N += 1

    def measure(self):
        return self.V / (self.N + 1e-8)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "MAE"), self.measure(), global_step)

    def report(self):
        return f"MAE_intensity = {self.measure():.3f}"

class IntensityMeter_L4D:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        intensity_error = self.compute_intensity_errors(truths, preds)

        intensity_error = list(intensity_error)
        self.V.append(intensity_error)
        self.N += 1

    def compute_intensity_errors(
        self, gt, pred, min_intensity=1e-6, max_intensity=1.0,
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_intensity**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"intensity error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Intensity_error (RMSE, MedAE, LPIPS, SSIM, PNSR) = {self.measure()}"

class DepthMeter_L4D:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error = self.compute_depth_errors(truths, preds)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1

    def compute_depth_errors(
        self, gt, pred, min_depth=1e-6, max_depth=80,
    ):  
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth
        
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_depth**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"depth error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error (RMSE, MedAE, LPIPS, SSIM, PNSR) = {self.measure()}"
    
class PointsMeter:
    """Calculate chamfer distance and F-score of all points.
    """
    def __init__(self, scale:float, intrinsics:list[float], intrinsics_hoz:list[float]=[180.0, 360.0]):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics
        self.intrinsics_hoz = intrinsics_hoz

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        chamLoss = chamfer_3DDist()
        pred_lidar = pano_to_lidar(preds[0], self.intrinsics, self.intrinsics_hoz)
        gt_lidar = pano_to_lidar(truths[0], self.intrinsics, self.intrinsics_hoz)

        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(pred_lidar[None, ...]).cuda(),
            torch.FloatTensor(gt_lidar[None, ...]).cuda(),
        )
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V) , "prediction and gt should should be equal"
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "Point error (CD)"), self.measure()[0], global_step)

    def report(self):
        cd, f_score = self.measure()
        # return f"Points_error(CD, F-score) = {[round(cd, 3), round(f_score*100, 3)]}"
        return f"Points_error(CD, F-score) = {[round(cd, 3), round(f_score, 3)]}"


class RaydropMeter:
    def __init__(self, ratio=0.5):
        self.V = []
        self.N = 0
        self.ratio = ratio

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        results = []

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())
        results.append(rmse)

        preds_mask = np.where(preds > self.ratio, 1, 0)
        acc = (preds_mask==truths).mean()
        results.append(acc)

        TP = np.sum((truths == 1) & (preds_mask == 1))
        FP = np.sum((truths == 0) & (preds_mask == 1))
        TN = np.sum((truths == 0) & (preds_mask == 0))
        FN = np.sum((truths == 1) & (preds_mask == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append(f1)

        self.V.append(results)
        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "raydrop error"), self.measure()[0], global_step)

    def report(self):
        return f"Rdrop_error (RMSE, Accuracy, F_score) = {self.measure()}"

class SSIMMeter:
    """Calculate Structural similarity
    """
    
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    # def prepare_inputs(self, *inputs):
    #     outputs = []
    #     for i, inp in enumerate(inputs):
    #         inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
    #         if torch.is_tensor(inp):
    #             inp = inp.detach().cpu().numpy()
    #         outputs.append(inp)

    #     return outputs

    def update(self, preds, truths):
        # preds, truths = self.prepare_inputs(preds, truths)
        # ssim = structural_similarity(preds.squeeze(0).squeeze(-1), truths.squeeze(0).squeeze(-1))

        preds, truths = self.prepare_inputs(
            preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)
        if torch.isnan(ssim): ssim = 0.0
        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / (self.N + 1e-8)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f"SSIM = {self.measure():.3f}"


class LPIPSMeter:
    def __init__(self, net="alex", device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / (self.N + 1e-8)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(
            os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step
        )

    def report(self):
        return f"LPIPS ({self.net}) = {self.measure():.3f}"

def depth_error_ratio(gt, pred, min_depth=1e-3, max_depth=80.0):
    """Calculate error in range/depth wrt ground truth values

    Args:
        - gt (ndarray/tensor): ground truth values in meters
        - pred (ndarray/tensor): predicted values in meters
        - min_depth (float, optional): Lidar minimum range in meters. Defaults to 1e-3.
        - max_depth (int, optional): Lidar maximum range in meters. Defaults to 80.

    Returns:
        - range_error (ndarray): Range error np.ndarray
    """
    
    assert gt.shape == pred.shape, "Shape of both pcds should be same"

    # convert gt and pred tensor to numpy arrays
    helper_fn = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    gt, pred = helper_fn(gt), helper_fn(pred)

    # Keeping all values withing  min and max lidar range
    pred[pred < min_depth] = min_depth 
    pred[pred > max_depth] = max_depth
    gt[gt < min_depth] = min_depth
    gt[gt > max_depth] = max_depth
    
    # Calculate depth/range  error
    range_error = gt - pred
    # range_error_norm = (range_error - range_error.mean()) / (range_error.max() - range_error.min())*100
    
    return range_error

def chamfer_dist(pc1, pc2, sample_size=None):
    """Calculate the chamfer distance of two point clouds

    Args:
        pc1 (nd.array): Point cloud [n,3]
        pc2 (nd.array): Point cloud [n,3]
        sample_size (int, optional): To reduce computation, if point cloud are very large. Defaults to None.

    Returns:
        float: chamfer distance between point clouds pc1 and pc2
    """
    # Remove intensity values if available
    if pc1.shape[1] == 4:
        pc1 =pc1[:,:3]
    if pc2.shape[1] == 4:
        pc2 =pc2[:,:3]

    # Randomly sample subset of points 
    if sample_size:  
        idx1 = np.random.randint(0, pc1.shape[0], sample_size)
        idx2 = np.random.randint(0, pc2.shape[0], sample_size)
        pc1_sampled = pc1[idx1, :]
        pc2_sampled = pc2[idx2, :]
    else:
        pc1_sampled = pc1
        pc2_sampled = pc2

    # Build KDTree for pc2
    tree = KDTree(pc2_sampled)

    # Query tree to find nearest neighbor and distance
    dists1, idx2 = tree.query(pc1_sampled, k=1) 

    # Compute approximations of min distances
    dists2 = np.linalg.norm(pc2_sampled[:,None] - pc1_sampled, axis=-1).min(axis=0)

    # Average and sum distances
    return np.mean(dists1) + np.mean(dists2)

def hausdorff_distance_loss(pred, target):
    """
    Calculates the Hausdorff distance loss for tensors of shape (batch_size, 1, 2, 8).
    """

    batch_size, _, height, width = pred.shape

    # Reshape tensors to 2D matrices for distance calculation
    pred_flat = pred.view(batch_size, -1)  # (256, 16)
    target_flat = target.view(batch_size, -1)  # (256, 16)

    dist_matrix = torch.cdist(pred_flat, target_flat)

    # Find the maximum distances in both directions
    forward_hausdorff = torch.max(torch.min(dist_matrix, dim=1)[0], dim=0)[0]
    backward_hausdorff = torch.max(torch.min(dist_matrix, dim=0)[0], dim=0)[0]

    # Combine both directions for the final Hausdorff distance
    hausdorff_dist = torch.max(forward_hausdorff, backward_hausdorff)

    # Optional: Average over batch elements for final loss
    loss = torch.mean(hausdorff_dist)

    return loss